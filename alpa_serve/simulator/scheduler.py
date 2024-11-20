from typing import Callable, Deque, Dict, Iterable, List, Optional
from collections import deque
from alpa_serve.profiling import ProfilingDatabase, ParallelConfig
from alpa_serve.profiling import ProfilingResult
import numpy as np


class Request:
    def __init__(self, id: int, model_id: int, group_id: int, arrival_time, latency, slo):
        self.id = id
        self.model_id = model_id
        self.group_id = group_id
        self.arrival_time = arrival_time
        self.latency = latency
        self.slo = slo
        self.return_time = -1
        self.finish_one_stage_time = 0


# 一个设备组有一个请求队列，对于队列中的请求，若一个请求的latency在比前
# 一个请求的latency更短，则判断是否可以交换这两个请求的位置，交换前后都能满足slo要求,
# 这样可以减缓车队效应，提高整体的服务质量
class Group_Controller:
    def __init__(self, group_config: ParallelConfig, group_models: List[str], 
                 model_names: List[str], prof_ress: Dict[str, ProfilingResult],
                 slo_scale: Optional[float] = 5.0, latency_tolerance_ratio: Optional[float] = 2):
        self.group_config = group_config
        self.group_models = group_models
        self.model_names = model_names
        
        self.requests_deque: Deque[Request] = deque()
        self.slo_scale = slo_scale
        self.latency_tolerance_ratio = latency_tolerance_ratio
        self.prof_ress = prof_ress
        # self.prof_database = ProfilingDatabase("/home/zy/python_project/mms/alpa_serve/syn_profiling_result.pkl")

        self.model_latency = self.get_model_latency()
        self.model_stage_latency = self.get_model_stage_latency()

        self.group_executor = Group_Executor(self.group_config, self.group_models, self.model_stage_latency)

        self.swap_num = 0
    
    def get_final_return_time(self):
        if self.requests_deque:
            return self.requests_deque[-1].return_time
        else:
            return 0
    
    def get_model_latency(self):
        model_latency = {}
        for model_id in self.group_models:
            model_sum_latency = np.empty(self.group_config.pp, dtype=np.float32)
            latency = self.prof_ress[model_id].para_dict[self.group_config].latency[1]
            for i in range(len(latency)):
                model_sum_latency[i] = latency[i]
            penalty = 0.009 * len(model_sum_latency)
            model_latency[model_id] = np.float32(sum(model_sum_latency) * (1 + penalty))
        return model_latency
    
    def get_model_stage_latency(self):
        model_stage_latency = {}
        for model_id in self.group_models:
            stage_latency = np.empty(self.group_config.pp, dtype=np.float32)
            latency = self.prof_ress[model_id].para_dict[self.group_config].latency[1]
            penalty = 0.009 * len(latency)
            for i in range(len(latency)):
                stage_latency[i] = latency[i] * (1 + penalty)
            model_stage_latency[model_id] = stage_latency
        return model_stage_latency
    
    def get_group_models(self):
        return self.group_models
    
    def check_slo_and_return_time(self, request: Request):
        # 判断是否满足 SLO，返回是否和预期返回时间
        fixed_overhead = 0.011
        if self.requests_deque:
            last_request = self.requests_deque[-1]
            expect_start_time = max(request.arrival_time, last_request.finish_one_stage_time)
            expect_finish_one_stage_time = expect_start_time + self.model_stage_latency[request.model_id][0]
            expect_return_time = expect_finish_one_stage_time + fixed_overhead
            for i in range(1, len(self.model_stage_latency[request.model_id])):
                expect_return_time += self.model_stage_latency[request.model_id][i]
            if request.arrival_time + request.slo >= expect_return_time:
                return True, expect_finish_one_stage_time, expect_return_time
            else:
                # print(f"Request {request.id} cannot meet SLO, reject from checking!")
                return False, expect_finish_one_stage_time, expect_return_time
        else:
            expect_finish_one_stage_time = request.arrival_time + self.model_stage_latency[request.model_id][0]
            expect_return_time = expect_finish_one_stage_time + fixed_overhead
            for i in range(1, len(self.model_stage_latency[request.model_id])):
                expect_return_time += self.model_stage_latency[request.model_id][i]
            if request.arrival_time + request.slo >= expect_return_time:
                return True, expect_finish_one_stage_time, expect_return_time
            else:
                return False, expect_finish_one_stage_time, expect_return_time

    def add_request(self, id: int, model_id: int, group_id: int, arrival_time, slo) -> bool:
        if model_id not in self.group_models:
            return False
        
        latency = self.model_latency[model_id]
        new_request = Request(id, model_id, group_id, arrival_time, latency, slo)
        flag, expect_finish_one_stage_time, expect_return_time = self.check_slo_and_return_time(new_request)
        
        if flag:
            new_request.finish_one_stage_time = expect_finish_one_stage_time
            new_request.return_time = expect_return_time
            self.requests_deque.append(new_request)
        else:
            return False

        # 检查是否需要交换以减少车队效应
        if len(self.requests_deque) > 1:
            last_request = self.requests_deque[-1]
            second_last_request = self.requests_deque[-2]
            
            if second_last_request.latency / last_request.latency > self.latency_tolerance_ratio:
                # print(f"Swap request {self.requests_deque[-1].id} and {self.requests_deque[-2].id}!")
                
                # 先保存原来的 return_time，用于后续比较
                original_last_return_time = last_request.return_time
                
                # 尝试交换
                self.swap_requests(-1, -2)
                
                # 检查交换后请求的 SLO
                swapped_request = self.requests_deque.pop()  # 暂时移除最后一个请求
                flag, new_expect_finish_one_stage_time, new_expect_return_time = self.check_slo_and_return_time(swapped_request)
                
                if flag and original_last_return_time >= new_expect_return_time:                    
                    # 更新交换后请求的时间
                    swapped_request.finish_one_stage_time = new_expect_finish_one_stage_time
                    swapped_request.return_time = new_expect_return_time
                    self.requests_deque.append(swapped_request)
                    print(f"Swap request {self.requests_deque[-1].id} and {self.requests_deque[-2].id} successfully!")
                    self.swap_num += 1

                else:
                    # 不满足 SLO，将请求放回原位置并恢复顺序
                    self.requests_deque.append(swapped_request)
                    self.swap_requests(-1, -2)
                    
        return True



    def swap_requests(self, i: int, j: int):
        fixed_overhead = 0.011
        # 将请求 i 与请求 j 的位置交换
        self.requests_deque[i], self.requests_deque[j] = self.requests_deque[j], self.requests_deque[i]
        # 更新交换后的请求的 finish_one_stage_time, return_time
        if i > j:
            i, j = j, i
        # 更新交换后请求的 finish_one_stage_time 和 return_time
        for idx in [i, j]:
            request = self.requests_deque[idx]
            model_id = request.model_id
            stage_latencies = self.model_stage_latency[model_id]
            
            # 确定第一个阶段的预计开始时间
            if idx == 0:
                expect_start_time = request.arrival_time
            else:
                last_request = self.requests_deque[idx - 1]
                expect_start_time = max(request.arrival_time, last_request.finish_one_stage_time)
            
            # 计算第一个阶段的 finish_one_stage_time
            request.finish_one_stage_time = expect_start_time + stage_latencies[0]
            
            # 计算包含所有阶段和固定额外时间的 return_time
            expect_return_time = expect_start_time + fixed_overhead
            for stage_latency in stage_latencies:
                expect_return_time += stage_latency
            request.return_time = expect_return_time


    def get_requests(self) -> List[Request]:
        return list(self.requests_deque)
    

    def process_requests(self, clear: bool = False):
        request_ids = []
        good = []
        finish = []
        model_num_good_requests = []
        group_num_good_requests = []

        # 模拟请求的处理, 处理队列中的请求
        if clear:
            num = 0
        else:
            num = 5
        while len(self.requests_deque) > num:
            request = self.requests_deque.popleft()
            return_time, tmp_time = self.group_executor.process_requests(request)
            if return_time <= request.arrival_time + request.slo:
                self.final_return_time = return_time
                self.group_executor.update_device_clock(tmp_time)
            else:
                # print(f"Request {request.id} cannot meet SLO, reject from processing!")
                pass
            
            request_ids.append(request.id)
            good.append(return_time <= request.arrival_time + request.slo)
            finish.append(return_time)
            model_num_good_requests.append(request.model_id)
            group_num_good_requests.append(request.group_id)
        
        return request_ids, good, finish, model_num_good_requests, group_num_good_requests


class Group_Executor:
    def __init__(self, group_config: ParallelConfig, group_models: List[str], model_stage_latency: Dict[str, List[float]]):
        self.group_config = group_config
        self.group_models = group_models
        self.model_stage_latency = model_stage_latency
        
        self.device_clock = np.zeros(group_config.pp, dtype=np.float64)
    
    def update_device_clock(self, tmp_time):
        for i in range(len(tmp_time)):
            self.device_clock[i] = tmp_time[i]
    
    # 当队列中有请求时，处理请求
    def process_requests(self, request: Request) -> float:
        fixed_overhead = 0.011
        t = request.arrival_time
        tmp_time = np.zeros(self.group_config.pp, dtype=np.float64)
        for i in range(len(self.model_stage_latency[request.model_id])):
            t = max(t, self.device_clock[i]) + self.model_stage_latency[request.model_id][i]
            tmp_time[i] = t
        return t + fixed_overhead, tmp_time


# 集群调度器，包含多个组调度器，集群调度器负责将请求分发给各个组调度器，以实现负载均衡
class Cluster_Controller:
    def __init__(self, placement, model_names: List[str],
                 prof_ress: Dict[str, ProfilingResult], num_requests: int,
                 slo_scale: Optional[float] = 5.0, latency_tolerance_ratio: Optional[float] = 1.2):
        from alpa_serve.placement_policy.base_policy import ModelPlacement
        assert isinstance(placement, ModelPlacement)
        self.group_configs = placement.group_configs
        self.groups_models = placement.group_models
        self.model_names = model_names
        self.group_num = len(self.group_configs)

        self.group_controllers = [Group_Controller(self.group_configs[i], self.groups_models[i], 
                                                   self.model_names, prof_ress, 
                                                   slo_scale, latency_tolerance_ratio) for i in range(self.group_num)]
        
        self.num_models = len(self.model_names)
        self.num_groups = len(self.group_configs)
        self.finish = np.empty(num_requests, dtype=np.float64)
        self.good = np.empty(num_requests, dtype=bool)
        self.model_num_requests = np.zeros(self.num_models, dtype=np.int32)
        self.model_num_good_requests = np.zeros(self.num_models, dtype=np.int32)
        self.group_num_requests = np.zeros(self.num_groups, dtype=np.int32)
        self.group_num_good_requests = np.zeros(self.num_groups, dtype=np.int32)


    def add_request(self, id: int, model_id: int, arrival_time, slo,
                    scheduler_policy: Optional[str] = "load_balance"):
        # 检查模型是否存在self.groups_models,这是一个二维列表
        self.model_num_requests[model_id] += 1
        flag = False
        for group_models in self.groups_models:
            if model_id in group_models:
                # 根据调度策略选择一个组调度器
                if scheduler_policy == "load_balance":
                    group_id = self.load_balance(model_id)
                elif scheduler_policy == "busiest_device":
                    group_id = self.busiest_device(model_id)
                else:
                    raise ValueError(f"Invalid scheduler policy: {scheduler_policy}")
                # 将请求添加到对应的组调度器
                flag = self.group_controllers[group_id].add_request(id, model_id, group_id, arrival_time, slo)
                self.group_num_requests[group_id] += 1
                break

        if flag == False:
            self.finish[id] = arrival_time
            self.good[id] = False


    def load_balance(self, model_id: str) -> int:
        # 简单的负载均衡策略，选择队列中最后一个请求返回时间最早的组
        min_group_id = -10086
        min_return_time = float("inf")
        for group_id in range(self.group_num):
            if self.group_controllers[group_id].get_final_return_time() < min_return_time and model_id in self.group_controllers[group_id].get_group_models():
                min_group_id = group_id
                min_return_time = self.group_controllers[group_id].get_final_return_time()
        return min_group_id
    

    def busiest_device(self, model_id: str) -> int:
        # 选择队列中最后一个请求返回时间最晚的组
        max_group_id = 0
        max_return_time = self.group_controllers[0].get_final_return_time()
        for group_id in range(1, self.group_num):
            if self.group_controllers[group_id].get_final_return_time() > max_return_time and model_id in self.group_controllers[group_id].get_group_models():
                max_group_id = group_id
                max_return_time = self.group_controllers[group_id].get_final_return_time()
        return max_group_id
    
    def process_requests(self, clear: bool = False):
        for group_controller in self.group_controllers:
            request_ids, good, finish, model_num_good_requests, group_num_good_requests = group_controller.process_requests(clear)
            for i in range(len(request_ids)):
                self.finish[request_ids[i]] = finish[i]
                self.good[request_ids[i]] = good[i]
                self.model_num_good_requests[model_num_good_requests[i]] += 1
                self.group_num_good_requests[group_num_good_requests[i]] += 1

    def get_results(self):
        return self.model_num_requests, self.model_num_good_requests, self.group_num_requests,  \
            self.group_num_good_requests, self.finish, self.good
    
    def get_swap_num(self):
        swap_num = 0
        for group_controller in self.group_controllers:
            swap_num += group_controller.swap_num
        return swap_num
        


