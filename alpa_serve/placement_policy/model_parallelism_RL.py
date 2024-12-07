"""Selective replication with model parallelism."""
from collections import namedtuple, OrderedDict
from functools import partial
import logging
import math
import multiprocessing
import time
from typing import List, Tuple, Dict, Optional
from itertools import product
import itertools
import argparse
import os
import datetime

import numpy as np
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelData, ClusterEnv, ModelPlacement,
    PlacementEvaluator, gen_train_workload,
    replica_placement_round_robin,
    replica_placement_fast_greedy, replica_placement_beam_search,
    replica_placement_on_last_group, evolutionary_search, ModelPlacementWithReplacement)
from alpa_serve.simulator.controller import simulate_one_case, approximate_one_case_one_placement
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.simulator.monitor import Monitor
from alpa_serve.trace import Trace
from alpa_serve.util import GB, write_tsv, ServingCase, inf, eps
from alpa_serve.util import (
    get_factors, get_partitions, get2tok, decompose2tok, all_node_combinations,
    ServingCase, eps)
from osdi23_artifact.general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite

import random
import gym
from gym import spaces
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import alpa_serve.placement_policy.rl_utils as rl_utils
from alpa_serve.placement_policy import MyModelParallelismILP


def get_placement(action, group_configs, num_models):
    '''
    根据动作获取放置策略
    '''
    num_groups = len(group_configs)
    # action转为二维数组action_2d[i][j]，0表示不放置，1表示放置
    action_2d = action.reshape(num_groups, num_models)
    group_models = [[] for _ in range(num_groups)]
    for i in range(num_groups):
        for j in range(num_models):
            if action_2d[i][j] == 1:
                group_models[i].append(j)

    sol = ModelPlacement(group_configs, group_models)
    return sol


class GoodputMaximizationEnv(gym.Env):
    def __init__(self, placement, model_names, prof_ress, model_ids, slos, arrivals, cluster_env,
                 max_steps=100, window_size=200):
        super(GoodputMaximizationEnv, self).__init__()
        
        self.placement = placement  # 集群放置情况，包含每个设备组的放置情况
        self.ori_placement = placement
        self.model_names = model_names  # 请求对应的模型名称
        self.prof_ress = prof_ress  # 模型profile信息
        self.model_ids = model_ids  # 请求对应的模型ID
        self.slos = slos  # 请求对应的SLO
        self.arrivals = arrivals  # 请求到达时间
        self.cluster_env = cluster_env  # 集群环境
        
        self.num_groups = len(self.placement.group_models)  # 设备组的数量
        self.num_models = len(self.model_names)  # 模型的数量

        self.max_steps = max_steps  # 最大步数
        self.current_step = 0  # 当前时间步

        # 初始化状态，假设集群有多个节点，状态可以包括集群资源和模型放置情况
        self.state = self.initialize_state(self.ori_placement)
        self.previous_goodput = 0
        self.start_time = self.arrivals[0]  # 起始时间
        self.window_size = window_size  # 窗口大小
        
        # 动作空间：每个模型在每个组上是否放置（布尔值表示是否放置）
        self.action_space = spaces.MultiBinary(self.num_groups * self.num_models)
        
        # 状态空间：
        # （1）模型维度：每个模型的请求率、请求成功率、每个模型的理论吞吐能力（这里可以尝试替换为模型负载）、每个模型的资源需求
        # （2）组维度：每个组的请求率、请求成功率、资源使用情况
        state_dim = self.num_models * 4 + self.num_groups * 3
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(state_dim,), dtype=np.float32
        )
        # self.observation_space = spaces.Box(
        #     low=0, high=np.inf, shape=(state_dim,), dtype=np.float32
        # )

    def initialize_state(self, placement):
        # 初始化状态：每个模型的请求率、成功率、吞吐能力，每组资源使用情况等
        # 假设每个模型放置在资源的起始位置
        model_info = np.zeros(self.num_models * 4)  # 每个模型的请求率、成功率、吞吐能力
        group_info = np.zeros(self.num_groups * 3)  # 每组资源使用情况
        state = np.concatenate((model_info, group_info))
        return state

    def reset(self):
        # 重置环境，返回初始状态
        self.state = self.initialize_state(self.ori_placement)
        self.previous_goodput = 0  # 记录上一个time step的goodput
        self.current_step = 0  # 重置时间步
        self.start_time = self.arrivals[0]
        return self.state

    def step(self, action):
        # 执行动作并更新状态，计算奖励
        new_state, reward, done = self.take_action(action)
        self.state = new_state
        self.current_step += 1
        return new_state, reward, done, {}

    def take_action(self, action):  
        # 得到新的放置策略
        new_placement = get_placement(action, self.placement.group_configs, self.num_models)

        # 判断放置是否符合资源约束
        if new_placement.check_(self.prof_ress, self.cluster_env):
            # 放置成功，更新状态
            self.placement = new_placement
            (goodput, state) = self.calculate_matrix()
            reward = self.calculate_reward(goodput)
             # 更新状态
            new_state = self.update_state(state)
        else:
            reward = -1  # 内存不满足时的惩罚
            new_state = self.state

        # 终止条件：达到了最大步数
        done = self.current_step >= self.max_steps

        return new_state, reward, done
    

    def update_state(self, state):
        """
        更新状态向量。
        """
        return state


    def calculate_matrix(self):
        # 计算当前时间步的goodput，基于资源和任务的处理能力
        start_i = np.where(self.arrivals == self.start_time)[0][0]

        # 非顺序执行，通过随机数生成start_i和end_i
        # start_i = random.randint(0, len(self.arrivals) - self.window_size)
        # 随机生成window_size，范围在200-1000之间
        window_size = random.randint(200, int(len(self.arrivals)/100))
        end_i = start_i + window_size
        interval_time = self.arrivals[end_i] - self.arrivals[start_i]

        (start, finish, good,
            model_num_requests, model_num_good_requests,
            group_num_requests, group_num_good_requests,
            receive_request_model_ids, replacement_time, monitor) = approximate_one_case_one_placement(
                        self.placement, self.model_names, self.prof_ress, 
                        self.model_ids[start_i:end_i], self.slos[start_i:end_i], self.arrivals[start_i:end_i])
        goodput = np.mean(good)

        # 计算每个模型的请求率、请求成功率
        model_requests_rate = model_num_requests / interval_time
        model_goodput_rate = model_num_good_requests / (model_num_requests + 1e-6)

        # 计算每个模型的理论吞吐能力（这里可以尝试替换为模型负载）、每个模型的资源需求
        monitor = Monitor(self.placement, self.model_names, self.prof_ress)
        model_capability, _ = monitor.analyse_model_capability()
        
        model_mem_usage = [0] * self.num_models
        for i in range(len(self.placement.group_configs)):
            for model_id in self.placement.group_models[i]:
                model_mem_usage[model_id] += monitor.cal_model_memory_usage(model_id, self.placement.group_configs[i])

        # 计算每个组的请求率、请求成功率
        group_requests_rate = group_num_requests / interval_time
        group_goodput_rate = group_num_good_requests / (group_num_requests + 1e-6)
        # 归一化
        # group_requests_rate = [rate / max(group_requests_rate) for rate in group_requests_rate]

        group_mem_usage = monitor.cal_group_memory_usage()

        # 归一化
        model_requests_rate = [rate / (max(model_requests_rate) + 1e-6) for rate in model_requests_rate]
        model_capability = [cap / (max(model_capability) + 1e-6)for cap in model_capability]
        model_mem_usage = [mem / (max(model_mem_usage) + 1e-6)for mem in model_mem_usage]
        group_requests_rate = [rate / (max(group_requests_rate) + 1e-6)for rate in group_requests_rate]
        group_mem_usage = [mem / (max(group_mem_usage) + 1e-6)for mem in group_mem_usage]

        # 更新起始时间
        self.start_time = self.arrivals[end_i]

        return (goodput, np.concatenate((model_requests_rate, model_goodput_rate, model_capability, model_mem_usage, 
                                        group_requests_rate, group_goodput_rate, group_mem_usage)))

    def calculate_reward(self, goodput):
        # 计算基于goodput的奖励
        # 可以根据current_goodput与之前的goodput比较，计算增益
        # 奖励可以是goodput的增量或相对于上一个goodput的增益
        previous_goodput = self.previous_goodput  # 记录上一个time step的goodput
        # reward = goodput - previous_goodput
        reward = goodput
        self.previous_goodput = goodput  # 更新为当前goodput
        return reward 


class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)


class DQN:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.total_count = 0  # 用于使用衰减的贪婪算法
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        self.total_count += 1
        # if np.random.random() < self.epsilon:
        if np.random.random() < 1 / self.total_count:
            action = np.random.randint(0, 2, size=self.action_dim).astype(int)
        else:
            state = np.array(state)  # 先将列表转换为 numpy.ndarray
            state = torch.tensor(state, dtype=torch.float).to(self.device)  # 然后转换为 torch.tensor
            q_values = self.q_net(state)  # 获取动作的Q值
            # 将每个动作的Q值转换为布尔值（Q值 > 0时选择动作）
            action = (q_values > 0).cpu().numpy().astype(int).flatten()
        return action
    
    def inference(self, state):
        '''
        训练好agent后，用于推理
        '''
        state = np.array(state)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        q_values = self.q_net(state)
        action = (q_values > 0).cpu().numpy().astype(int).flatten()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)

        actions_array = np.array(transition_dict['actions'])
        actions = torch.tensor(actions_array, dtype=torch.int).to(self.device)

        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # q_values = self.q_net(states).gather(1, actions)  # Q值
        # # 下个状态的最大Q值
        # max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        # 计算当前Q值
        q_values = self.q_net(states)
        # 只选择当前动作对应的Q值
        q_values = (q_values * actions).sum(dim=1, keepdim=True)

        # 计算下个状态的最大Q值
        with torch.no_grad():
            max_next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]

        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1


def get_rl_agent_one_case(placement, model_names, prof_ress, model_ids, slos, 
                        arrivals, rl_kwargs, mixed = True, enable_batching = False,
                        unique_type2model_ids = None, scheduling_policy = 'load_balance',
                        replacement = False):
    
    # 训练DQN模型
    lr = 2e-3
    num_episodes = 500
    hidden_dim = 256
    gamma = 0.02  # 0.98
    epsilon = 0.01
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # env_name = 'CartPole-v0'
    # env = gym.make(env_name)
    # 创建环境实例
    group_configs = placement.group_configs
    group_models = [[] for _ in range(len(group_configs))]
    placement = ModelPlacement(group_configs, group_models)
    # placement = ModelPlacement([ParallelConfig(1, 1, 4)] * 2, [[] for _ in range(2)])
    # placement = self.monitor.placement

    env = GoodputMaximizationEnv(placement, model_names, prof_ress, model_ids, slos, arrivals, rl_kwargs.get('cluster_env', None),
                                 max_steps=100, window_size=1000)
    random.seed(0)
    np.random.seed(0)
    # env.seed(0)
    torch.manual_seed(0)
    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    # 获取本文件所在的绝对路径
    current_path = os.path.dirname(os.path.abspath(__file__))

    if rl_kwargs['rl_stage'] == 'test':
        # 检查本地是否有已经训练好的模型
        if os.path.exists(os.path.join(current_path, 'dqn_model.pth')):
            agent.q_net.load_state_dict(torch.load(os.path.join(current_path, 'dqn_model.pth')))
            agent.target_q_net.load_state_dict(torch.load(os.path.join(current_path, 'dqn_model.pth')))
            # 切换到推理模式
            agent.q_net.eval()
            agent.target_q_net.eval()
            print('Load test model successfully:', os.path.join(current_path, 'dqn_model.pth'))
            return agent
        else:
            assert False, "No trained dnq-model found!"
    # 训练DQN
    elif 'train' in rl_kwargs['rl_stage']:
        # 是否增量学习
        if rl_kwargs['incre_learning'] and os.path.exists(os.path.join(current_path, 'dqn_model.pth')):
            agent.q_net.load_state_dict(torch.load(os.path.join(current_path, 'dqn_model.pth')))
            agent.target_q_net.load_state_dict(torch.load(os.path.join(current_path, 'dqn_model.pth')))
            
            # 切换到训练模式
            agent.q_net.train()
            agent.target_q_net.train()
            print('Load increment learning model successfully:', os.path.join(current_path, 'dqn_model.pth'))

        return_list = rl_utils.train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size)

        episodes_list = list(range(len(return_list)))
        plt.figure()
        plt.plot(episodes_list, return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format("GoodputMaximizationEnv"))
        plt.show()
        # 保存图片到本文件所在的文件夹
        plt.savefig(os.path.join(current_path, 'DQN_on_GoodputMaximizationEnv.png'))
        print('图表已保存至:', os.path.join(current_path, 'DQN_on_GoodputMaximizationEnv.png'))

        mv_return = rl_utils.moving_average(return_list, 9)
        plt.figure()
        plt.plot(episodes_list, mv_return)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on {}'.format("GoodputMaximizationEnv"))
        plt.show()
        # 保存图片
        plt.savefig(os.path.join(current_path, 'DQN_on_GoodputMaximizationEnv_moving_average.png'))
        print('图表已保存至:', os.path.join(current_path, 'DQN_on_GoodputMaximizationEnv_moving_average.png'))

        # 保存模型
        if rl_kwargs['save_model']:
            torch.save(agent.q_net.state_dict(), os.path.join(current_path, 'dqn_model.pth'))
            print('模型已保存至:', os.path.join(current_path, 'dqn_model.pth'))

    return agent


class MyModelParallelismDQNReplacement(MyModelParallelismILP):
    def __init__(self, replacement_interval: int = -1,
                 use_evo_search: bool = False, 
                 dynamic_replacement: bool = False,
                 replacement_time: int = 0,
                 monitor: Monitor = None,
                 agent: DQN = None,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.replacement_interval = replacement_interval
        self.use_evo_search = use_evo_search
        self.dynamic_replacement =  dynamic_replacement
        self.replacement_time = replacement_time
        self.monitor = monitor
        self.agent = agent

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # 在初始阶段，用ILP求解模型放置
        if self.agent is None:
            sol, _ = super().solve_placement(model_datas, cluster_env, train_workload)
            return ModelPlacement(sol.group_configs, sol.group_models), None

        # 使用训练好的DQN模型进行模型放置
        assert self.monitor is not None, "monitor is required for DQN placement"
        assert self.agent is not None, "agent is required for DQN placement"

        # 从monitor中获取数据
        # （1）模型维度：每个模型的请求率、请求成功率、每个模型的理论吞吐能力（这里可以尝试替换为模型负载）、每个模型的资源需求
        # （2）组维度：每个组的请求率、请求成功率、资源使用情况
        state = self.monitor.state

        # 使用DQN进行推理
        action = self.agent.inference(state)
        # 得到新的放置策略
        new_placement = get_placement(action, self.monitor.placement.group_configs, len(model_datas))

        # 判断放置是否符合资源约束, 不符合则返回旧的放置策略
        if new_placement.check_(self.monitor.prof_ress, cluster_env):
            return new_placement, None
        else:
            return self.monitor.placement, None



