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

import numpy as np
import ray

from alpa_serve.profiling import ParallelConfig
from alpa_serve.placement_policy.base_policy import (
    BasePlacementPolicy, ModelData, ClusterEnv, ModelPlacement,
    PlacementEvaluator, gen_train_workload,
    replica_placement_round_robin,
    replica_placement_fast_greedy, replica_placement_beam_search,
    replica_placement_on_last_group, evolutionary_search, ModelPlacementWithReplacement)
from alpa_serve.simulator.controller import simulate_one_case
from alpa_serve.simulator.executable import Executable
from alpa_serve.simulator.workload import Workload, GammaProcess
from alpa_serve.simulator.monitor import Monitor
from alpa_serve.trace import Trace
from alpa_serve.util import (
    get_factors, get_partitions, get2tok, decompose2tok, all_node_combinations,
    ServingCase, eps)


def compute_capability(model_data, parallel_config, max_bs):
    slo = model_data.slo
    latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

    if latency_mem is None:
        return 0

    num_stages = parallel_config.pp
    max_cap = 0
    for b, ls in latency_mem.latency.items():
        if b > max_bs:
            continue

        # slo = sum(ls) + (n-1) * max(ls)
        # so, n = ceil((slo - sum(ls)) / max(ls)) + 1
        max_cap = max(max_cap, (slo - sum(ls)) // max(ls) + 1)

    return max_cap * (0.99 ** num_stages)

class MyModelParallelismILP(BasePlacementPolicy):
    def __init__(self, verbose: int = 0, model_groups: Optional[List[List[str]]] = None):
        super().__init__(verbose)

        self.time_limit = 30
        self.lamda = 0.1  # trade-off
        self.max_bs = 1
        self.model_groups = model_groups

        # Hard coded for now. Expose this as parameters later
        self.group_configs = [
            ParallelConfig(1, 1, 1), ParallelConfig(1, 1, 2),
            ParallelConfig(1, 2, 1), ParallelConfig(1, 2, 2),
            ParallelConfig(1, 4, 1), ParallelConfig(1, 1, 4),
            ParallelConfig(1, 2, 4), ParallelConfig(1, 4, 2),
            ParallelConfig(1, 1, 8), ParallelConfig(1, 8, 1)]
        self.group_sizes = [
            np.prod(x) for x in self.group_configs
        ]

    def compute_max_stage_mem(self, model_data, parallel_config, mem_budget):
        latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return mem_budget * 2

        return max(latency_mem.weight_mem)

    def compute_device_group(self, cluster_env):
        '''
        对每个节点，遍历其所有可能的设备组合，设备组的大小为2的幂，比如节点的设备数为4，则可能的设备组合为[2, 2]，[4]，[1, 1, 2]
        '''
        device_counts = [cluster_env.num_devices_per_node] * int(cluster_env.num_devices // cluster_env.num_devices_per_node)
        if cluster_env.num_devices % cluster_env.num_devices_per_node:
            device_counts.append(cluster_env.num_devices % cluster_env.num_devices_per_node)
        return all_node_combinations(device_counts)
    
    def solve_one_device_group_NIP(self, model_datas, cluster_env, device_group, model_requests):
        import numpy as np
        from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint

        # 集群参数
        N = cluster_env.num_devices_per_node
        M = cluster_env.num_devices / N
        E = cluster_env.mem_budget

        # 模型参数
        C = len(model_datas)
        w = np.array([model_requests[model_data.name] for model_data in model_datas])
        # 对w进行归一化
        w = w.astype(float)
        w /= np.sum(w)

        # 模型并行参数
        K = len(self.group_configs)
        model_compute_cap = np.zeros((C, K))
        model_weight_mem = np.zeros((C, K))
        for i in range(C):
            model_data = model_datas[i]
            for k in range(K):
                parallel_config = self.group_configs[k]
                model_compute_cap[i][k] = compute_capability(model_data, parallel_config, self.max_bs)
                model_weight_mem[i][k] = self.compute_max_stage_mem(model_data, parallel_config, cluster_env.mem_budget)

        # 设备组参数
        S = len(device_group)

        # 1. 创建变量
        # 将x, p, z平铺成一维数组进行优化
        # x_size = C * S
        # p_size = S * K
        z_size = C * S * K

        # 2. 目标函数
        def objective(vars):
            # x = vars[:x_size].reshape(C, S)
            # p = vars[x_size:x_size + p_size].reshape(S, K)
            # cap = np.sum(np.dot(x, model_compute_cap) * w[:, None], axis=0)
            z = vars.reshape(C, S, K)
            cap = np.zeros(C)
            for j in range(S):
                cap += np.sum(np.multiply(z[:, j, :], model_compute_cap), axis=1)
            obj = np.sum(cap * w)
            return -obj  # 最小化负目标函数

        # 3. 约束条件
        constraints = []

        # (a). 每个GPU的内存约束
        for j in range(S):
            def memory_constraint(vars, j=j):
                z = vars.reshape(C, S, K)
                p = z[:, j, :]
                return E - np.sum(np.multiply(p, model_weight_mem))
            constraints.append({'type': 'ineq', 'fun': memory_constraint})

        # (b). 每个device group, 选择的并行策略需满足容量约束
        # for j in range(S):
        #     def capacity_constraint(vars, j=j):
        #         z = vars.reshape(C, S, K)
        #         p = np.sum(z[:, j, :], axis=0)
        #         return device_group[j] - np.sum(p[j, :] * self.group_sizes)
        #     constraints.append({'type': 'eq', 'fun': capacity_constraint})

        # (c). 每个设备组只能选择一个并行策略
        # for j in range(S):
        #     def single_parallel_strategy(vars, j=j):
        #         z = vars.reshape(C, S, K)
        #         p = np.sum(z[:, j, :], axis=0)
        #         return 1 - np.sum(p[j, :])
        #     constraints.append({'type': 'eq', 'fun': single_parallel_strategy})

        # (d). 线性化约束
        # 线性化z的约束（需要适当调整）
        for i in range(C):
            for j in range(S):
                for k in range(K):
                    def linearization(vars, i=i, j=j, k=k):
                        z = vars.reshape(C, S, K)
                        p = np.sum(z[:, j, :], axis=0)
                        x = np.sum(z[i, :, :], axis=0)
                        return z[i, j, k] - (x + p - 1)
                    constraints.append({'type': 'ineq', 'fun': linearization})

        # 4. 初始值和边界
        initial_guess = np.zeros(z_size)
        bounds = [(0, 1)] * z_size  # 假设变量的边界在 [0, 1] 之间

        # 5. 求解
        result = minimize(objective, initial_guess, constraints=constraints, bounds=bounds)

        if result.success:
            status = "Optimization succeeded"
            objective_value = -result.fun  # 因为目标函数是负的
        else:
            status = "Optimization failed"
            objective_value = -1.0

        print(f"Status: {status}\tObjective: {objective_value}")

        return result.x.reshape(C, S, K), objective_value

    
    def solve_one_device_group_ILP(self, model_datas, cluster_env, device_group, model_requests):
        import pulp
        from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus, LpMinimize, value
        from math import sqrt
        from scipy.optimize import minimize

        # 集群参数
        N = cluster_env.num_devices_per_node
        M = cluster_env.num_devices / N
        E = cluster_env.mem_budget
        
        # 模型参数
        C = len(model_datas)
        w = [model_requests[model_data.name] for model_data in model_datas]
        # 对w进行归一化
        w = [w[i] / sum(w) for i in range(C)]

        # 模型并行参数
        K = len(self.group_configs)
        model_compute_cap = np.zeros((C, K))
        # model_weight_mem = np.zeros((C, K, np.max(self.group_sizes)))
        model_weight_mem = np.zeros((C, K))
        for i in range(C):
            model_data = model_datas[i]
            for k in range(K):
                parallel_config = self.group_configs[k]
                model_compute_cap[i][k] = compute_capability(model_data, parallel_config, self.max_bs)
                # weight_mem = model_data.profiling_result.para_dict.get(parallel_config, None).weight_mem
                # model_weight_mem[i][k][:len(weight_mem)] = weight_mem
                model_weight_mem[i][k] = self.compute_max_stage_mem(model_data, parallel_config, cluster_env.mem_budget)
        # 对model_compute_cap进行归一化，除以最小值
        # model_compute_cap = model_compute_cap / np.min(model_compute_cap)

        # 设备组参数
        S = len(device_group)

        # 1. 创建变量
        # 模型i是否放在设备组j上
        x = LpVariable.matrix("x", (range(C), range(S)), cat="Binary")
        # 模型i在设备组j上的并行方案，先假设一个设备组只有一个并行方案
        p = LpVariable.matrix("p", (range(S), range(K)), cat="Binary")
        z = LpVariable.matrix("z", (range(C), range(S), range(K)), cat="Binary")

        # 2. 目标函数
        prob = LpProblem("myProblem", LpMaximize)
        cap = [None] * C
        for i in range(C):
            cap[i] = lpSum(z[i][j][k] * model_compute_cap[i][k] for j in range(S) for k in range(K))
         # 计算w的模(L2范数)
        # w_norm = sqrt(sum(w[i] * w[i] for i in range(C)))
        obj = lpSum(w[i] * cap[i] for i in range(C))
        prob += obj

        # 3. 约束
        # (a). 每个GPU的内存约束， 这里暂时没有把act_mem考虑进去
        for j in range(S):
            prob += lpSum(z[i][j][k] * model_weight_mem[i][k] for i in range(C) for k in range(K)) <= E
        # (b). 对于每个device group, 选择的并行策略需满足容量约束
        for j in range(S):
            # p[j][k]为1，说明选择了并行策略k, 根据group_sizes[k]<=device_group[j]，可以保证容量约束
            prob += lpSum(p[j][k] * self.group_sizes[k] for k in range(K)) == device_group[j]
        # (*c). 每个设备组只能选择一个并行策略
        for j in range(S):
            prob += lpSum(p[j][k] for k in range(K)) == 1
        # (d). 线性化约束
        # 添加约束，确保 z[i][j][k] 为 1 当且仅当 x[i][j] 和 p[j][k] 都为 1
        for i in range(C):
            for j in range(S):
                for k in range(K):
                    prob += z[i][j][k] <= x[i][j]  # 如果模型在设备组上，才能选择并行策略
                    prob += z[i][j][k] <= p[j][k]  # 如果选择了并行策略，模型必须在设备组上
                    prob += z[i][j][k] >= x[i][j] + p[j][k] - 1
        
        # 4. 求解
        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")
        
        solver = pulp.PULP_CBC_CMD(mip=True, msg=False, timeLimit=self.time_limit, threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if self.verbose >= 2:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}")
        
        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError("Cannot run the function under the given memory budget. Please increase the memory budget.")
        return x, p, objective


    def solve_placement_one_model_group(self, 
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None,
                        interval: int = 0,
                        replacement_time: int = -1):
        import pulp

        # 计算train_workload中每个模型的请求率
        model_requests = OrderedDict()
        for model_data in model_datas:
            model_requests[model_data.name] = 1  # 默认请求率都相等
        # 若是动态替换，计算在replacement_time前100s内每个模型的请求率
        if replacement_time > 0:
            cal_time = 100
            for request in train_workload.requests:
                arrival_time = train_workload.arrivals[request.idx]
                if arrival_time > replacement_time: break
                elif arrival_time >= max(replacement_time - cal_time, 0) and arrival_time < replacement_time:
                    model_requests[request.model_name] += 1
            for model_name in model_requests:
                model_requests[model_name] = model_requests[model_name] / cal_time
        # 若是间隔固定时间重新放置模型，计算在这段时间内每个模型的请求率
        elif interval > 0:
            for request in train_workload.requests:
                model_requests[request.model_name] += 1
            for model_name in model_requests:
                model_requests[model_name] = model_requests[model_name] / interval
        # 若都不是，则计算全局的模型请求率
        else:
            for request in train_workload.requests:
                model_requests[request.model_name] += 1
            for model_name in model_requests:
                model_requests[model_name] = model_requests[model_name] / len(train_workload.requests)

        device_groups = self.compute_device_group(cluster_env)
        # device_groups = [[4]]
        max_objective = -1
        for device_group in device_groups:
            x_, p_, objective = self.solve_one_device_group_ILP(model_datas, cluster_env, device_group, model_requests)
            if objective > max_objective:
                max_objective = objective
                x = x_
                p = p_
                best_device_group = device_group

        C = len(model_datas)
        S = len(best_device_group)
        K = len(self.group_configs)

        # 设备组选择
        p_res = []
        for j in range(S):
            assert sum(pulp.value(p[j][k]) for k in range(K)) == 1
            for k in range(K):
                if pulp.value(p[j][k]):
                    p_res.append(k)
        
        # 模型放置
        x_res = np.zeros((C, S), dtype=np.int8)
        for i in range(C):
            for j in range(S):
                if pulp.value(x[i][j]):
                    x_res[i][j] = 1
        
        group_configs = []
        group_models = []
        for j in range(S):
            config_id = p_res[j]
            if self.group_sizes[config_id]:
                tmp = []
                for i in range(C):
                    if x_res[i][j]:
                        tmp.append(i)
                group_configs.append(self.group_configs[config_id])
                group_models.append(tmp)
        
        return ModelPlacement(group_configs, group_models), {"objective": objective}
    
    
    def solve_placement(self, 
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None,
                        interval: int = 0,
                        replacement_time: int = -1):
        
        tic = time.time()
        if self.model_groups is None:
            placement, objective = self.solve_placement_one_model_group(model_datas, cluster_env, train_workload, interval, replacement_time)
            print(f"slove Time: {time.time() - tic}")
            return placement, objective
        else:
            # 根据每个组模型的请求量，分配计算资源
            model_requests = OrderedDict()
            for request in train_workload.requests:
                model_requests[request.model_name] = model_requests.get(request.model_name, 0) + 1  # 使用 get 方法简化计数

            model_groups_requests = []
            for model_group in self.model_groups:
                group_requests = sum(model_requests[model_name] for model_name in model_group)  # 使用 sum 函数简化求和
                model_groups_requests.append(group_requests)

            model_groups_requests = np.array(model_groups_requests)
            model_groups_requests = model_groups_requests / np.sum(model_groups_requests)  # 归一化

            cluster_devices = cluster_env.num_devices
            # 初步分配设备
            initial_device_allocation = np.floor(model_groups_requests * cluster_devices).astype(int)
            # 计算已分配的设备总数
            total_allocated = np.sum(initial_device_allocation)
            # 计算剩余设备数量
            remaining_devices = cluster_devices - total_allocated
            # 确保每个设备都被分配，优先分配给请求量较大的组
            while remaining_devices > 0:
                # 找到当前剩余请求最多的组
                max_index = np.argmax(model_groups_requests * cluster_devices - initial_device_allocation)
                initial_device_allocation[max_index] += 1
                remaining_devices -= 1

            # 最终的设备分配
            model_group_devices = initial_device_allocation.tolist()
                
            group_configs = []
            group_models = []
            for i, model_group in enumerate(self.model_groups):
                sub_model_datas = []
                for model in model_datas:
                    if model.name in model_group:
                        sub_model_datas.append(model)
                sub_cluster_env = ClusterEnv(model_group_devices[i], cluster_env.mem_budget, cluster_env.num_devices_per_node)
                sub_sol, _ = self.solve_placement_one_model_group(sub_model_datas, sub_cluster_env, train_workload, interval, replacement_time)
                group_configs += sub_sol.group_configs
                # 将子组的模型索引转换为全局模型索引
                group_models += [[model_datas.index(sub_model_datas[model_id]) for model_id in group] for group in sub_sol.group_models]
            
            print(f"slove Time: {time.time() - tic}")
            return ModelPlacement(group_configs, group_models), None

class MyModelParallelismHeuReplacement(MyModelParallelismILP):
    def __init__(self, replacement_interval: int = -1,
                 use_evo_search: bool = False, 
                 dynamic_replacement: bool = False,
                 replacement_time: int = 0,
                 monitor: Monitor = None,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.replacement_interval = replacement_interval
        self.use_evo_search = use_evo_search
        self.dynamic_replacement =  dynamic_replacement
        self.replacement_time = replacement_time
        self.monitor = monitor

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # 在初始阶段，用ILP求解模型放置
        if self.monitor is None:
            sol, _ = super().solve_placement(model_datas, cluster_env, train_workload)
            return ModelPlacement(sol.group_configs, sol.group_models), None

        ori_placement = self.monitor.placement
        scale_up_model = self.monitor.scale_up_model
        scale_down_model = self.monitor.scale_down_model

        # 对于每个需要扩容的模型，找到需要缩容的模型，进行替换，并检查是否符合内存约束
        while len(scale_up_model) > 0:
            ori_group_configs, ori_group_models = ori_placement.group_configs, ori_placement.group_models
            successful_replacement = False  # 标记是否成功替换

            # 遍历尝试替换的缩容模型组合
            for num_replace in range(0, len(scale_down_model) + 1):
                # 尝试组合替换缩容模型
                for replace_model_indice in itertools.combinations(scale_down_model, num_replace):
                    new_placement = ori_placement.copy()
                    new_group_configs, new_group_models = new_placement.group_configs, new_placement.group_models

                    # 遍历所有放置组，进行替换
                    for g in range(len(new_group_models)):
                        if all(model_indice in new_group_models[g] for model_indice in replace_model_indice):
                            # 移除缩容模型
                            for model_indice in replace_model_indice:
                                new_group_models[g].remove(model_indice)

                            # 逐步添加扩容模型
                            for scale_up_model_indice in scale_up_model[:]:  # 使用切片避免修改列表时出错
                                new_group_models[g].append(scale_up_model_indice)

                                # 创建新的放置方案并验证内存约束
                                sol = ModelPlacement(new_group_configs, new_group_models)
                                sol = sol.normalize()
                                if sol.check(model_datas, cluster_env):  # 内存验证
                                    # 更新放置方案
                                    ori_placement = sol
                                    # 删除已替换的模型
                                    scale_down_model = [x for x in scale_down_model if x not in replace_model_indice]
                                    scale_up_model.remove(scale_up_model_indice)  # 删除已放置的扩容模型
                                    successful_replacement = True
                                    print(f"Successful replacement: {replace_model_indice} -> {scale_up_model_indice}")
                                    break
                                else:
                                    # 如果内存不满足，回滚放置，跳出扩容模型添加循环
                                    new_group_models[g] = ori_group_models[g]
                                    break

                            # 如果成功替换，退出当前循环
                            if successful_replacement:
                                break

                    # 如果成功替换，退出内部循环
                    if successful_replacement:
                        break
                
                # 如果成功替换，退出外层循环
                if successful_replacement:
                    ori_placement = ModelPlacement(new_group_configs, new_group_models)
                    ori_placement = ori_placement.normalize()
                    break

            # 如果所有需要扩容的模型都已经得到满足，退出外层循环
            if len(scale_up_model) == 0:
                break

            # 如果没有可用的缩容模型进行替换，退出
            if not successful_replacement:
                break

        # 输出没有被成功scale_up的模型
        if len(scale_up_model) > 0:
            print(f"Failed to scale up models: {scale_up_model}")
        # 返回最终的模型放置方案
        return ori_placement, None


class MyModelParallelismILPReplacement(MyModelParallelismILP):
    def __init__(self, replacement_interval: int = -1,
                 use_evo_search: bool = False, 
                 dynamic_replacement: bool = False,
                 replacement_time: int = 0,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.replacement_interval = replacement_interval
        self.use_evo_search = use_evo_search
        self.dynamic_replacement =  dynamic_replacement
        self.replacement_time = replacement_time

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        if self.replacement_interval > 0:
            ws = train_workload.split_time_interval(self.replacement_interval)

            start_times = []
            placements = []
            for i in range(len(ws)):
                sol, _ = super().solve_placement(model_datas, cluster_env, ws[i], interval=self.replacement_interval)
                start_times.append(ws[i].arrivals[0])
                placements.append(sol)

            return ModelPlacementWithReplacement(start_times, placements), None
        
        elif self.dynamic_replacement:
            print("Replacement Time: ", self.replacement_time)
            sol, _ = super().solve_placement(model_datas, cluster_env, train_workload, 
                                             replacement_time=self.replacement_time)
            return ModelPlacement(sol.group_configs, sol.group_models), None

class ModelParallelismILP(BasePlacementPolicy):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.time_limit = 30
        self.sum_k = 1e-4
        self.max_bs = 1

        # Hard coded for now. Expose this as parameters later
        self.group_configs = [
            ParallelConfig(0, 0, 0),
            ParallelConfig(1, 1, 1),
            ParallelConfig(1, 1, 2),
            ParallelConfig(1, 2, 1),
            ParallelConfig(1, 2, 2),
            # ParallelConfig(1, 4, 1),
            # ParallelConfig(1, 1, 4),
            # ParallelConfig(1, 1, 8),
        ]
        self.group_sizes = [
            np.prod(x) for x in self.group_configs
        ]

    def compute_max_stage_mem(self, model_data, parallel_config, mem_budget):
        latency_mem = model_data.profiling_result.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return mem_budget * 2

        return max(latency_mem.weight_mem)

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None,
                        interval: int = -1):
        import pulp
        from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus

        tic = time.time()

        # Load constants
        N = len(model_datas)
        M = cluster_env.num_devices
        C = cluster_env.mem_budget
        if interval == -1:
            a = [x.rate for x in model_datas]
        else:
            model_requests = OrderedDict()
            for model_data in model_datas:
                model_requests[model_data.name] = 0
            for request in train_workload.requests:
                model_requests[request.model_name] += 1
            a = [model_requests[model_data.name] / interval for model_data in model_datas] 
        c = [x.profiling_result.para_dict[ParallelConfig(1, 1, 1)].weight_mem[0]
             for x in model_datas]

        G = cluster_env.num_devices
        K = len(self.group_configs)
        g = self.group_sizes
        f = np.zeros((N, K))
        d = np.zeros((N, K))
        for i in range(N):
            model_data = model_datas[i]
            for k in range(K):
                parallel_config = self.group_configs[k]
                f[i][k] = compute_capability(model_data, parallel_config, self.max_bs)
                d[i][k] = self.compute_max_stage_mem(
                    model_data, parallel_config, cluster_env.mem_budget)

        # 1. Create variables
        p = LpVariable.matrix("p", (range(N), range(G)), cat="Binary")
        cap = [None] * N
        min_tolerance = LpVariable("min_tolerance", lowBound=0)
        sum_tolerance = LpVariable("sum_tolerance", lowBound=0)
        s = LpVariable.matrix("s", (range(G), range(K)), cat="Binary")
        pxs = LpVariable.matrix("pxs", (range(N), range(G), range(K)), cat="Binary")

        # 2. Objective
        prob = LpProblem("myProblem", LpMaximize)
        obj = min_tolerance + self.sum_k * sum_tolerance
        prob += obj

        # 3. Constraints
        # (a). memory budget on each GPU
        # for j in range(G):
        #     prob += (lpSum(p[i][j] * (c[i] / C) for i in range(N)) <=
        #              lpSum(s[j][k] * g[k] for k in range(K)))

        ## A more precise version, not used right now
        for j in range(G):
           prob += (lpSum(pxs[i][j][k] * (d[i][k] / C)
                          for i in range(N) for k in range(K)) <= 1)

        # (b). capability
        for i in range(N):
            cap[i] = lpSum(pxs[i][j][k] * f[i][k]
                           for j in range(G) for k in range(K))

        # (c). min tolerance and sum tolerance
        for i in range(N):
            prob += min_tolerance <= cap[i] / a[i]

        prob += sum_tolerance == lpSum(cap[i] / a[i] for i in range(N))

        # (d). group size
        prob += lpSum(s[j][k] * g[k] for j in range(G) for k in range(K)) == M

        # (e). only one configuration
        for j in range(G):
            prob += lpSum(s[j][k] for k in range(K)) == 1

        # (f). linearization
        for i in range(N):
            for j in range(G):
                for k in range(K):
                    prob += pxs[i][j][k] <= p[i][j]
                    prob += pxs[i][j][k] <= s[j][k]
                    prob += pxs[i][j][k] >= p[i][j] + s[j][k] - 1

        assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")

        solver = pulp.PULP_CBC_CMD(mip=True,
                                   msg=False,
                                   timeLimit=self.time_limit,
                                   threads=multiprocessing.cpu_count())
        prob.solve(solver)

        status = prob.status
        objective = pulp.value(prob.objective)
        objective = float(objective) if objective is not None else -1.0
        if self.verbose >= 2:
            print(f"ILP Status: {LpStatus[status]}\tObjective: {objective}\t"
                  f"Time: {time.time() - tic}")

        if prob.status in [pulp.LpStatusInfeasible]:
            raise RuntimeError(
                "Cannot run the function under the given memory budget. "
                "Please increase the memory budget.")

        # Group configuration selection
        s_res = []
        for j in range(G):
            assert sum(pulp.value(s[j][k]) for k in range(K)) == 1
            for k in range(K):
                if pulp.value(s[j][k]):
                    s_res.append(k)

        # Placement
        p_res = np.zeros((N, G), dtype=np.int8)
        for i in range(N):
            for j in range(G):
                if pulp.value(p[i][j]):
                    p_res[i][j] = 1

        group_configs = []
        group_models = []
        for j in range(G):
            config_id = s_res[j]
            if self.group_sizes[config_id]:
                tmp = []
                for i in range(N):
                    if p_res[i][j]:
                        tmp.append(i)
                group_configs.append(self.group_configs[config_id])
                group_models.append(tmp)

        return ModelPlacement(group_configs, group_models), {"objective": objective}

class ModelParallelismILPReplacement(ModelParallelismILP):
    def __init__(self, replacement_interval: int,
                 use_evo_search: bool = False, verbose: int = 0):
        super().__init__(verbose=verbose)

        self.replacement_interval = replacement_interval
        self.use_evo_search = use_evo_search 

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        ws = train_workload.split_time_interval(self.replacement_interval)

        start_times = []
        placements = []
        for i in range(len(ws)):
            sol, _ = super().solve_placement(model_datas, cluster_env, ws[i], self.replacement_interval)
            start_times.append(ws[i].arrivals[0])
            placements.append(sol)

        return ModelPlacementWithReplacement(start_times, placements), None


class ModelParallelismGreedy(BasePlacementPolicy):

    def __init__(self, group_size: int = 2,
                 use_evo_search: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.group_size = group_size
        self.use_evo_search = use_evo_search

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        # Run greedy placement
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                                       "fast_simulator", False)

        assert cluster_env.num_devices % self.group_size == 0
        num_groups = cluster_env.num_devices // self.group_size
        sol = ModelPlacement([ParallelConfig(1,1,self.group_size)] * num_groups,
                             [[] for _ in range(num_groups)])
        sol = replica_placement_fast_greedy(
            sol, model_datas, cluster_env, train_workload,
            evaluator, self.verbose)

        if self.use_evo_search:
            sol = evolutionary_search([sol], model_datas, cluster_env,
                                      evaluator, 200, self.verbose)
        return sol, None


def solve_separation_placement(self,
                               eco_separation: List[Tuple[List[ModelData], ClusterEnv]],
                               model_id_map,
                               train_workload: Workload):
    sol = ModelPlacement([],[])
    for i, eco in enumerate(eco_separation):
        sub_model_datas, sub_cluster_env = eco
        eco_sol, _ = self.solve_placement_one_eco(sub_model_datas, sub_cluster_env, train_workload)
        sol.group_configs += eco_sol.group_configs
        sol.group_models += [[model_id_map[(i, model_id)] for model_id in group]
                             for group in eco_sol.group_models]
    return sol


class ModelParallelismRR(BasePlacementPolicy):

    def __init__(self,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op

        self.evaluator_method = "fast_simulator"


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        # parallel config (dp = 1, op = 1, pp = 4)
        num_reg_groups = cluster_env.num_devices // 4
        quo_groups = decompose2tok(cluster_env.num_devices % 4)
        init_sol = ModelPlacement([ParallelConfig(1, 1, 4)] * num_reg_groups +
                                  [ParallelConfig(1, 1, s) for s in quo_groups],
                                  [[] for _ in range(num_reg_groups + len(quo_groups))])

        sol = replica_placement_round_robin(
                   init_sol, model_datas, cluster_env, train_workload, self.verbose)

        return sol, {}


class ModelParallelismSearch(BasePlacementPolicy):

    def __init__(self,
                 max_bs: int = 1,
                 max_pp: int = 8,
                 max_op: int = 4,
                 use_evo_search: bool = False,
                 use_separation: bool = False,
                 verbose: int = 0):
        super().__init__(verbose=verbose)

        self.max_bs = max_bs
        self.max_pp = max_pp
        self.max_op = max_op
        self.n_iter = 1
        self.seed = 0
        self.beam_size = 3
        self.use_evo_search = use_evo_search
        self.use_separation = use_separation

        self.evaluator_method = "fast_simulator"
        self.parallel_evaluator = False
        self.parallel_initial_placement = False

        if ((self.parallel_evaluator or self.parallel_initial_placement)
            and not ray.is_initialized()):
            ray.init(address="auto", ignore_reinit_error=True)


    def solve_placement_one_eco(self,
                                model_datas: List[ModelData],
                                cluster_env: ClusterEnv,
                                train_workload: Workload = None):
        evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
            self.evaluator_method, self.parallel_evaluator)

        # Get initial solutions
        initial_sols = self.enumerate_group_configs_uneven(cluster_env)

        if self.parallel_initial_placement:
            func = ray.remote(replica_placement_fast_greedy).remote
            for i in range(len(initial_sols)):
                initial_sols[i] = func(
                    initial_sols[i], model_datas, cluster_env, train_workload, None,
                    self.verbose)
            initial_sols = ray.get(initial_sols)
        else:
            for i in range(len(initial_sols)):
                initial_sols[i] = replica_placement_fast_greedy(
                    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                    self.verbose)
                #initial_sols[i] = replica_placement_beam_search(
                #    initial_sols[i], model_datas, cluster_env, train_workload, evaluator,
                #     self.beam_size, self.verbose)

        scores = evaluator.get_scores(initial_sols)
        best_idx = np.argmax(scores)
        best_sol = initial_sols[best_idx]

        return best_sol, {}


    def enumerate_separations(self,
                              model_datas: List[ModelData],
                              cluster_env: ClusterEnv):
        same_model_threshold = 0.38
        # 这里是将模型进行分组，分组的依据是模型的latency
        model_id_map = {}  # (cluster_id, model_id) -> model_id
        eco_model_datas = []  # List[List[ModelData]]每个cluster中的模型
        cluster_latencies = []  # 每个cluster的latency
        for model_id, model_data in enumerate(model_datas):
            cur_latency = max(model_data.profiling_result. \
                          para_dict[ParallelConfig(1, 1, 1)].latency[1])
            flag = False
            for i, cluster in enumerate(eco_model_datas):
                cluster_latency = max(cluster[0].profiling_result. \
                                  para_dict[ParallelConfig(1, 1, 1)].latency[1])
                if math.fabs(cur_latency - cluster_latency) / cluster_latency < same_model_threshold:
                    model_id_map[(i, len(cluster))] = model_id
                    cluster.append(model_data)
                    flag = True
                    break
            if not flag:
                model_id_map[(len(eco_model_datas), 0)] = model_id
                eco_model_datas.append([model_data])
                cluster_latencies.append(cur_latency)

        # List[List[(List[ModelData], ClusterEnv)]]
        # 将集群设备分为len(eco_model_datas)个部分，生成分区
        partitions = get_partitions(cluster_env.num_devices, len(eco_model_datas))

        ## reduce num partitions
        # 计算每个model组的请求率占比
        ratio = np.empty(len(eco_model_datas), dtype=np.float32)
        for i, eco_model_data in enumerate(eco_model_datas):
            ratio[i] = sum(x.rate for x in eco_model_data)
        ratio = ratio / np.sum(ratio)   # q/s

        for threshold in [1.0, 0.5, 0.3, 0.2, 0.1]:
            reduced_partitions = []
            for partition in partitions:
                throughputs = [x / l for x, l in zip(partition, cluster_latencies)]   # q/s
                norm_throughputs = np.array(throughputs) / sum(throughputs)
                dis = np.max(np.abs(ratio - norm_throughputs))
                if dis < threshold:
                    reduced_partitions.append(partition)

            if len(reduced_partitions) < 100:
                break

        print(f"original: {len(partitions)}  reduced: {len(reduced_partitions)}")

        separations = [[(eco_model_datas[i], ClusterEnv(device_cnt, cluster_env.mem_budget)) \
                        for i, device_cnt in enumerate(partition)] \
                       for partition in reduced_partitions]

        return separations, model_id_map


    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        # Generate workloads
        if train_workload is None:
            train_workload = gen_train_workload(model_datas)

        best_sol, _ = self.solve_placement_one_eco(model_datas, cluster_env, train_workload)

        # Separate unequal model
        if self.use_separation:
            eco_separations, model_id_map = self.enumerate_separations(model_datas, cluster_env)
            print("number of combinations: ", len(eco_separations))

            parallel = False
            if parallel:
                func = ray.remote(solve_separation_placement).remote
            else:
                func = solve_separation_placement

            sols = []
            for eco_separation in eco_separations:
                sols.append(func(self, eco_separation, model_id_map, train_workload))

            if parallel:
                sols = ray.get(sols)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            scores = evaluator.get_scores(sols)
            best_idx = np.argmax(scores)

            evaluator = PlacementEvaluator(model_datas, cluster_env, train_workload,
                self.evaluator_method, self.parallel_evaluator)
            score_mixed = evaluator.get_scores([best_sol])[0]

            print(f"score_mixed: {score_mixed:.3f}, score_separate: {scores[best_idx]:.3f}")
            if scores[best_idx] > score_mixed:
                print("Separation is better.")
                best_sol = sols[best_idx]
            else:
                print("Mixed is better.")

        if self.use_evo_search:
            best_sol = evolutionary_search(
                [best_sol], model_datas, cluster_env,
                evaluator, 200, self.verbose)
        return best_sol, {}


    def enumerate_group_configs_uneven(self, cluster_env: ClusterEnv):
        '''
        遍历获得所有可能的group配置，group_size为2的倍数
        '''
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get2tok(num_devices):
            # 这里新增了一个判断条件，group_size不能大于num_devices_per_node
            if group_size > num_devices_per_node:
                continue
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue
            num_reg_groups = num_devices // group_size
            quo_groups = decompose2tok(num_devices % group_size)

            for pp in get_factors(group_size):
                op = group_size // pp

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_reg_groups +
                                           [ParallelConfig(1, 1, s) for s in quo_groups],
                                           [[] for _ in range(num_reg_groups + len(quo_groups))]))
        return sols


    def enumerate_group_configs(self, cluster_env):
        sols = []
        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        for group_size in get_factors(num_devices):
            if group_size > num_devices_per_node and group_size % num_devices_per_node != 0:
                continue

            for pp in get_factors(group_size):
                op = group_size // pp
                num_groups = num_devices // group_size

                if pp > self.max_pp or op > self.max_op:
                    continue

                sols.append(ModelPlacement([ParallelConfig(1, op, pp)] * num_groups,
                                           [[] for _ in range(num_groups)]))
        return sols

    def greedy_group_configs(self,
                             model_datas: List[ModelData],
                             cluster_env: ClusterEnv,
                             train_workload: Workload,
                             evaluator: PlacementEvaluator,
                             beam_size = 3):

        assert beam_size >= 1, "beam size should >= 1."

        num_devices = cluster_env.num_devices
        num_devices_per_node = cluster_env.num_devices_per_node

        beam_sols = [[ModelPlacement([], [])]]

        for cur_num in range(1, num_devices + 1):
            ## solve sols[cur_num]
            next_sols = []
            for last_group_size in range(1, (cur_num - 1) % num_devices_per_node + 1 + 1):
                ## solve from sols[cur_num - last_group_size]
                # print("last_group_size ", last_group_size)
                for pp in get_factors(last_group_size):
                    op = last_group_size // pp
                    if pp > self.max_pp or op > self.max_op:
                        continue

                    for sol in beam_sols[cur_num - last_group_size]:
                        pre_sol = sol.copy()
                        pre_sol.group_configs.append(ParallelConfig(1, op, pp))
                        pre_sol.group_models = [[] for _ in range(len(pre_sol.group_configs))]

                        #new_sol = replica_placement_on_last_group(
                        #new_sol = replica_placement_beam_search(
                        #              pre_sol, model_datas, cluster_env, train_workload,
                        #              evaluator, self.beam_size, self.verbose)
                        new_sol = replica_placement_fast_greedy(
                                      pre_sol, model_datas, cluster_env, train_workload,
                                      evaluator, self.verbose)
 
                        next_sols.append(new_sol)
            scores = evaluator.get_scores(next_sols)
            next_indices = np.argsort(scores)[::-1][:beam_size]
            beam_sols.append([])
            for i in range(len(next_indices)):
                beam_sols[cur_num].append(next_sols[next_indices[i]])

        return beam_sols[num_devices]

class ModelParallelismEqual(BasePlacementPolicy):
    
    def __init__(self, pp, op, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.pp = pp
        self.op = op

    def solve_placement(self,
                        model_datas: List[ModelData],
                        cluster_env: ClusterEnv,
                        train_workload: Workload = None):
        group_size = self.op * self.pp
        num_groups = cluster_env.num_devices // group_size
        sol = ModelPlacement([ParallelConfig(1, self.op, self.pp)] * num_groups, [[i] for i in range(num_groups)])

        return sol, None
