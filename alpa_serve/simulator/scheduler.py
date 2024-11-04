from typing import Callable, Deque, Dict, Iterable, List, Optional
from collections import deque
from alpa_serve.profiling import ProfilingDatabase, ParallelConfig

class Request:
    def __init__(self, model_name: str, arrival_time: float, 
                 latency: float, slo: float):
        self.model_name = model_name
        self.arrival_time = arrival_time
        self.latency = latency
        self.slo = slo
        self.return_time = 0

# 调度器需满足功能：自动维护请求队列，对于队列中的请求，若一个请求的latency在比前
# 一个请求的latency更短，则判断是否可以交换这两个请求的位置，交换前后都能满足slo要求,
# 这样可以减缓车队效应，提高整体的服务质量
class Scheduler:
    def __init__(self, slo_threshold: float, prof_database_path=Optional[str],
                 slo_scale: Optional[float] = 5.0, latency_tolerance_ratio: Optional[float] = 2):
        self.slo_threshold = slo_threshold
        self.requests_deque: Deque[Request] = deque()
        self.model_latency: Dict[str, float] = {}
        self.slo_scale = slo_scale
        self.latency_tolerance_ratio = latency_tolerance_ratio
        if prof_database_path:
            self.prof_database = ProfilingDatabase(prof_database_path)
        else:
            self.prof_database = ProfilingDatabase("/home/zhangy/python_project/mms/alpa_serve/syn_profiling_result.pkl")
    
    def add_request(self, model_name: str, arrival_time: float):
        # 从prof_database中获取模型的latency
        request_latency = self.model_latency.get(model_name, 0)
        slo = self.slo_scale * sum(self.prof_database.get(model_name).para_dict[ParallelConfig(1,1,1)
        ].latency[1])
        new_request = Request(model_name, arrival_time, request_latency, slo)
        self.requests_deque.append(new_request)
        # 获取队列尾最后一个请求的latency
        if self.requests_deque:
            last_request = self.requests_deque[-1]
            if last_request.latency / new_request.latency > self.latency_tolerance_ratio:
                # 交换位置
                self.swap_requests(-1, -2)
    

    def swap_requests(self, i: int, j: int):
        self.requests[i], self.requests[j] = self.requests[j], self.requests[i]
    
    
    def get_requests(self) -> List[Request]:
        return list(self.requests_deque)
