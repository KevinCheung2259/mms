import argparse
import datetime
from collections import namedtuple
import os
import numpy as np
import ray
import time
from typing import Callable, List, Dict, Optional, Tuple

import sys
sys.path.append('/home/zy/python_project/mms')

from alpa_serve.simulator.controller import (Controller, DummyController,
    approximate_one_case_one_placement, approximate_scheduler_one_case_one_placement)
from alpa_serve.simulator.workload import (Workload, GammaProcess, UniformMMPP, DEFAULT_WARMUP, 
                                           StatsResult, PerModelStatsResult)
from alpa_serve.profiling import ProfilingDatabase, ParallelConfig
from alpa_serve.placement_policy import (ClusterEnv, ModelData,
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationReplacement, SelectiveReplicationUniform,
    ModelParallelismILP, ModelParallelismGreedy, ModelParallelismRR,
    ModelParallelismSearch, MyModelParallelismILP, MyModelParallelismILPReplacement, ModelParallelismILPReplacement,
    MyModelParallelismHeuReplacement, MySelectiveReplicationReplacement, ModelParallelismSearchReplacement)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.trace import Trace, report_group_stats
from alpa_serve.util import GB, write_tsv, ServingCase, inf, eps
from collections import OrderedDict
from alpa_serve.trace import Trace, TraceReplay
from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case
from alpa_serve.placement_policy.base_policy import ModelPlacement
from monitor.divide_models import divide_models
from monitor.my_general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite

GeneralModelCase = namedtuple("GeneralModelCase", [
    "exp_name", "num_devices", "num_devices_per_node", "mem_budget", "model_types", "model_names",
    "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs",
    "slo_scale", "duration", "policy_name"])

def get_general_model_serving_case(case, prof_database=None, model_groups_num: Optional[int] = 1):
    assert isinstance(case, GeneralModelCase), "not GeneralModelCase"
    if prof_database is None:
        prof_database = ProfilingDatabase("/home/zy/python_project/mms/alpa_serve/syn_profiling_result.pkl")

    (exp_name, num_devices, num_devices_per_node, mem_budget, model_types, model_names,
     total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
     slo_scale, duration, policy_name) = case

    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget, num_devices_per_node=num_devices_per_node)
    assert len(model_names) == len(model_types)
    num_models = len(model_names)
    single_latency = {
        model_type: sum(prof_database.get(model_type).para_dict[ParallelConfig(1,1,1)
        ].latency[1]) for model_type in set(model_types)}
    slos = [single_latency[model_type] * slo_scale for model_type in model_types]

    if rate_distribution == "uniform":
        rates = [total_rate / num_models] * num_models
    elif rate_distribution == "power_law":
        alpha = 0.5
        s = sum((x+1)**(-alpha) for x in range(num_models))
        base = total_rate / s
        rates = [base * ((x+1) ** (-alpha)) for x in range(num_models)]
    elif rate_distribution is None:
        pass
    else:
        raise ValueError(f"Invalid rate distribution: {rate_distribution}")

    train_workload = None
    if arrival_process == "gamma":
        seed = 0
        arrival_processes = []
        peak_rate = 10  # 峰值期间的请求率
        base_rate = 0.2  # 其他时间的基础请求率
        interval_seconds = arrival_process_kwargs.get("interval_seconds", 600)
        num_intervals = int(duration // interval_seconds)

        peak_times_per_model = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]
        
        # 根据分配的高峰时间为每个模型构建到达过程
        for i in range(num_models):
            np.random.seed(seed)
            peak_times = peak_times_per_model[i]
            seed += 1  # 增加种子以确保模型之间的随机性
            distribution = []
            for t in range(num_intervals):
                if t in peak_times:
                    rate = peak_rate * rates[i]
                else:
                    rate = base_rate * rates[i]
                # 假设 GammaProcess 接受 (rate, 系数变异) 作为参数
                distribution.append(GammaProcess(rate, arrival_process_kwargs["cv"]))
            arrival_processes.append(distribution)
        
        replays = OrderedDict()
        start_time = "0.0.0"
        end_time = f"0.{int(duration / 3600)}.0"
        start_d, start_h, start_m = Trace.timestr_to_dhm(start_time)
        end_d, end_h, end_m = Trace.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        for m in range(len(arrival_processes)):
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(arrival_processes[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                # if DEBUG:
                #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                #     self.visualize_inter_arrival(np.array(arrivals), "test")
                arrival_distribution_params.append(distribution.params())
                seed += 1
            replays[model_names[m]] = TraceReplay(model_names[m],
                                    np.array(arrivals),
                                    "synthetic",
                                    start_time,
                                    end_time,
                                    interval_seconds,
                                    arrival_distribution="power_law",
                                    arrival_distribution_params=arrival_distribution_params)
        
        # 画出每个模型的trace
        from monitor.util import plot_model_traces
        plot_model_traces(replays, model_names, duration, 60)  # 每分钟一个时间段
        
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        # for debugging:
        for m in replays:
            replays[m].report_stats()

        report_group_stats(list(replays.values()))
        arrival_processes = [replays[model_name] for model_name in model_names]
    elif arrival_process == "mixed":
        mixed_ratio = arrival_process_kwargs.get("mixed_ratio", 0.5)
        # 1. 生成 Synthetic (gamma) 到达过程，缩放因子为 0.5
        seed = 0
        synthetic_arrival_processes = []
        synthetic_peak_rate = 10 * mixed_ratio  # 缩放峰值到达率
        synthetic_base_rate = 0.2 * mixed_ratio # 缩放基础到达率

        interval_seconds = arrival_process_kwargs.get("interval_seconds", 600)
        num_intervals = int(duration // interval_seconds)

        peak_times_per_model = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]

        for i in range(num_models):
            np.random.seed(seed)
            peak_times = peak_times_per_model[i]
            seed += 1  # 增加种子以确保模型之间的随机性
            distribution = []
            for t in range(num_intervals):
                if t in peak_times:
                    rate = synthetic_peak_rate * rates[i]
                else:
                    rate = synthetic_base_rate * rates[i]
                # 假设 GammaProcess 接受 (rate, 系数变异) 作为参数
                distribution.append(GammaProcess(rate, arrival_process_kwargs["cv"]))
            synthetic_arrival_processes.append(distribution)
        
        # 生成 Synthetic 的 Replays
        synthetic_replays = OrderedDict()
        start_time = "0.0.0"
        end_time = f"0.{int(duration / 3600)}.0"
        start_d, start_h, start_m = Trace.timestr_to_dhm(start_time)
        end_d, end_h, end_m = Trace.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        for m in range(len(synthetic_arrival_processes)):
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(synthetic_arrival_processes[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                arrival_distribution_params.append(distribution.params())
                seed += 1
            synthetic_replays[model_names[m]] = TraceReplay(
                model_names[m],
                np.array(arrivals),
                "synthetic",
                start_time,
                end_time,
                interval_seconds,
                arrival_distribution="gamma",
                arrival_distribution_params=arrival_distribution_params
            )
        
        # 2. 生成 Azure_v1 到达过程
        azure_v1_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        azure_v1_replays = azure_v1_trace.replay(
            model_names,
            model_mapping_strategy="stripe",
            arrival_distribution="gamma",
            start_time="0.0.0",
            end_time="0.{0}.0".format(int(duration / 3600)),
            interval_seconds=interval_seconds,
            rate_scale_factor=arrival_process_kwargs["rate_scale"] * (1-mixed_ratio),  # 缩放到达率
            cv_scale_factor=arrival_process_kwargs["cv_scale"]
        )
        del azure_v1_trace
        
        # 3. 合并 Synthetic 和 Azure_v1 的到达过程
        replays = OrderedDict()
        for model_name in model_names:
            synthetic_arrivals = synthetic_replays[model_name].arrivals
            azure_v1_arrivals = azure_v1_replays[model_name].arrivals
            combined_arrivals = np.concatenate((synthetic_arrivals, azure_v1_arrivals))
            combined_arrivals = np.sort(combined_arrivals)  # 确保到达时间有序

            # 合并 arrival_distribution_params
            combined_params = synthetic_replays[model_name].arrival_distribution_params + azure_v1_replays[model_name].arrival_distribution_params

            # 创建新的 TraceReplay
            replays[model_name] = TraceReplay(
                model_name,
                combined_arrivals,
                "mixed",
                start_time,
                end_time,
                interval_seconds,
                arrival_distribution="gamma",
                arrival_distribution_params=combined_params
            )
        
        # 4. 绘制混合后的 trace
        from monitor.util import plot_model_traces
        plot_model_traces(replays, model_names, duration, interval_seconds)
        
        # 5. 合并 workloads
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        # 6. 调试信息
        for m in replays:
            replays[m].report_stats()

        report_group_stats(list(replays.values()))
        arrival_processes = [replays[model_name] for model_name in model_names]
        
    elif arrival_process == "uniform_mmpp":
        arrival_processes = [
            UniformMMPP(**arrival_process_kwargs)
            for _ in range(num_models)
        ]
    elif arrival_process == "azure_v2":
        azure_v2_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v2_trace = Trace("azure_v2", azure_v2_trace_dir)
        train_replays = azure_v2_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time='13.0.0',
                                              end_time='13.1.0',
                                              # end_time='13.1.0',
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        test_replays = azure_v2_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time='13.1.0',
                                              end_time='13.2.0',
                                              # end_time='13.1.0',
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        
        from monitor.util import plot_model_traces, plot_all_models_requests
        plot_model_traces(test_replays, model_names, duration, 60)
        plot_all_models_requests(test_replays, model_names, duration, 60)

        del azure_v2_trace

        train_ws, test_ws = [], []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            train_ws.append(train_replays[model_name].to_workload(slo))
            test_ws.append(test_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*train_ws)
        test_workload = Workload.merge(*test_ws)

        # for debugging:
        for m in test_replays:
            test_replays[m].report_stats()

        report_group_stats(list(test_replays.values()))
        arrival_processes = [test_replays[model_name] for model_name in model_names]
        replays = test_replays
    elif arrival_process == "azure_v1":
        azure_v1_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        train_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time="0.0.0",
                                              end_time="0.1.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        test_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time="0.1.0",
                                              end_time="0.2.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        
        from monitor.util import plot_model_traces, plot_all_models_requests
        plot_model_traces(test_replays, model_names, duration, 60)
        plot_all_models_requests(test_replays, model_names, duration, 60)
        
        del azure_v1_trace

        train_ws, test_ws = [], []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            train_ws.append(train_replays[model_name].to_workload(slo))
            test_ws.append(test_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*train_ws)
        test_workload = Workload.merge(*test_ws)

        # for debugging:
        for m in test_replays:
            # 打印每个模型的trace统计信息
            test_replays[m].report_stats()
        # 打印全部的trace统计信息
        report_group_stats(list(test_replays.values()))
        arrival_processes = [test_replays[model_name] for model_name in model_names]
        replays = test_replays
    elif arrival_process == "online":
        # 生成训练集
        azure_v1_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        train_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time="0.0.0",
                                              end_time="0.1.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        del azure_v1_trace
        ws = []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        # 生成测试集
        seed = 0
        arrival_processes = []
        peak_rate = 10  # 峰值期间的请求率
        base_rate = 0.5  # 其他时间的基础请求率
        interval_seconds = arrival_process_kwargs.get("interval_seconds", 600)
        num_intervals = int(duration // interval_seconds)

        peak_times_per_model = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]]
        
        # 根据分配的高峰时间为每个模型构建到达过程
        for i in range(num_models):
            np.random.seed(seed)
            peak_times = peak_times_per_model[i]
            seed += 1  # 增加种子以确保模型之间的随机性
            distribution = []
            for t in range(num_intervals):
                if t in peak_times:
                    rate = peak_rate * rates[i]
                else:
                    rate = base_rate * rates[i]
                # 假设 GammaProcess 接受 (rate, 系数变异) 作为参数
                distribution.append(GammaProcess(rate, arrival_process_kwargs["cv"]))
            arrival_processes.append(distribution)
        
        replays = OrderedDict()
        start_time = "0.0.0"
        end_time = f"0.{int(duration / 3600)}.0"
        start_d, start_h, start_m = Trace.timestr_to_dhm(start_time)
        end_d, end_h, end_m = Trace.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        for m in range(len(arrival_processes)):
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(arrival_processes[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                # if DEBUG:
                #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                #     self.visualize_inter_arrival(np.array(arrivals), "test")
                arrival_distribution_params.append(distribution.params())
                seed += 1
            replays[model_names[m]] = TraceReplay(model_names[m],
                                    np.array(arrivals),
                                    "synthetic",
                                    start_time,
                                    end_time,
                                    interval_seconds,
                                    arrival_distribution="power_law",
                                    arrival_distribution_params=arrival_distribution_params)
        
        # 画出每个模型的trace
        from monitor.util import plot_model_traces
        plot_model_traces(replays, model_names, duration, 60)  # 每分钟一个时间段

        test_ws = []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            test_ws.append(replays[model_name].to_workload(slo))
        test_workload = Workload.merge(*test_ws)

        # for debugging:
        for m in replays:
            replays[m].report_stats()

        report_group_stats(list(replays.values()))
        arrival_processes = [replays[model_name] for model_name in model_names]

    elif arrival_process == "whole_process":
        # 生成训练集和测试集
        azure_v1_trace_dir = arrival_process_kwargs["trace_dir"]
        azure_v1_trace = Trace("azure_v1", azure_v1_trace_dir)
        train_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time="0.0.0",
                                              end_time="0.1.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        
        test_replays = azure_v1_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time="0.1.0",
                                              end_time="0.2.0",
                                              interval_seconds=60,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        del azure_v1_trace
        ws = []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        seed = 0
        arrival_processes = []
        peak_rate = 10  # 峰值期间的请求率
        base_rate = 0.5  # 其他时间的基础请求率
        interval_seconds = arrival_process_kwargs.get("interval_seconds", 600)
        num_intervals = int((duration*2/3) // interval_seconds)

        peak_times_per_model = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
            [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            [40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119]]
        
        # 根据分配的高峰时间为每个模型构建到达过程
        for i in range(num_models):
            np.random.seed(seed)
            peak_times = peak_times_per_model[i]
            seed += 1  # 增加种子以确保模型之间的随机性
            distribution = []
            for t in range(num_intervals):
                if t in peak_times:
                    rate = peak_rate * rates[i]
                else:
                    rate = base_rate * rates[i]
                # 假设 GammaProcess 接受 (rate, 系数变异) 作为参数
                distribution.append(GammaProcess(rate, arrival_process_kwargs["cv"]))
            arrival_processes.append(distribution)
        
        synthetic_replays = OrderedDict()
        start_time = "0.2.0"
        end_time = f"0.{2 + int((duration*2/3) / 3600)}.0"
        start_d, start_h, start_m = Trace.timestr_to_dhm(start_time)
        end_d, end_h, end_m = Trace.timestr_to_dhm(end_time)
        start_timestamp_seconds = start_d * 24 * 60 * 60 + start_h * 60 * 60 + start_m * 60
        end_timestamp_seconds = end_d * 24 * 60 * 60 + end_h * 60 * 60 + end_m * 60
        for m in range(len(arrival_processes)):
            arrivals = []
            arrival_distribution_params = []
            for i, distribution in enumerate(arrival_processes[m]):
                if distribution is None:
                    arrival_distribution_params.append(None)
                    continue
                start = i * interval_seconds + start_timestamp_seconds
                arrivals.extend(distribution.generate_arrivals(start, interval_seconds, seed))
                # if DEBUG:
                #     arrivals.extend(distribution.generate_arrivals(0, 1.0e9, seed))
                #     self.visualize_inter_arrival(np.array(arrivals), "test")
                arrival_distribution_params.append(distribution.params())
                seed += 1
            synthetic_replays[model_names[m]] = TraceReplay(model_names[m],
                                    np.array(arrivals),
                                    "synthetic",
                                    start_time,
                                    end_time,
                                    interval_seconds,
                                    arrival_distribution="power_law",
                                    arrival_distribution_params=arrival_distribution_params)
        
        # 合并test_replays和synthetic_replays的到达过程
        replays = OrderedDict()
        for model_name in model_names:
            synthetic_arrivals = synthetic_replays[model_name].arrivals
            test_arrivals = test_replays[model_name].arrivals
            combined_arrivals = np.concatenate((test_arrivals, synthetic_arrivals))
            combined_arrivals = np.sort(combined_arrivals)

            # 合并 arrival_distribution_params
            combined_params = test_replays[model_name].arrival_distribution_params + synthetic_replays[model_name].arrival_distribution_params

            # 创建新的 TraceReplay
            replays[model_name] = TraceReplay(
                model_name,
                combined_arrivals,
                "mixed",
                "0.1.0",
                f"0.{2 + int((duration*2/3) / 3600)}.0",
                interval_seconds,
                arrival_distribution="gamma",
                arrival_distribution_params=combined_params
            )

        # 画出每个模型的trace
        from monitor.util import plot_model_traces
        plot_model_traces(replays, model_names, duration, 60)  # 每分钟一个时间段

        test_ws = []
        for model_name, slo in zip(model_names, slos):
            # 将每个模型的trace转换成workload
            test_ws.append(replays[model_name].to_workload(slo))
        test_workload = Workload.merge(*test_ws)

        # for debugging:
        for m in replays:
            replays[m].report_stats()

        report_group_stats(list(replays.values()))
        arrival_processes = [replays[model_name] for model_name in model_names]

    else:
        raise ValueError("Invalid arrival process: {arrival_process}")

    # 每个模型的arrival_process的rate和cv
    rates = [a.rate() for a in arrival_processes]
    cvs = [a.cv() for a in arrival_processes]

    if model_groups_num > 1 and "groups" in policy_name:
        model_groups = divide_models(model_names, duration=duration, replays=replays, group_num=model_groups_num)
    else:
        model_groups = None

    def register_models(controller):
        is_simulator = isinstance(controller, (Controller, DummyController))
        # 注册每个模型到controller
        for model_name, model_type in zip(model_names, model_types):
            controller.register_model.remote(
                model_name, get_model_def(model_type, is_simulator,
                                          prof_database))
        return

    def generate_workload(start=0):
        '''
        生成整个集群即将到来的工作负载，注意这里每个模型的arrival_processes是在之前已经生成好了的，
        这里只不过是结合了模型各自的slo水平并转换成workload的形式，最后再汇总
        '''
        w = Workload.empty()
        for i in range(num_models):
            w += arrival_processes[i].to_workload(slos[i])
            # if "azure" in arrival_process:
            #     w += arrival_processes[i].to_workload(slos[i])
            # else:
            #     w += arrival_processes[i].generate_workload(model_names[i], start,
            #                                                 duration, slo=slos[i], seed=i)
        return w

    def place_models(controller, placement=None, replacement_time=0, monitor=None, agent=None):
        num_models = len(model_names)
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(model_names[i], slos[i], rates[i], cvs[i],
                                         prof_database.get(model_types[i])))

        if policy_name == "sr-ilp":
            policy = SelectiveReplicationILP(verbose=1)
        elif policy_name == "sr-greedy":
            policy = SelectiveReplicationGreedy(verbose=1)
        elif "sr-replace-dynamic" in policy_name:
            policy = MySelectiveReplicationReplacement(verbose=1, dynamic_replacement=True,
                                                      replacement_time=replacement_time, monitor=monitor)
        elif "sr-replace" in policy_name:
            interval = int(policy_name.split("-")[2])
            policy = SelectiveReplicationReplacement(verbose=1,
                 replacement_interval=interval)
        elif "my-mp-ilp-replace" in policy_name:
            interval = int(policy_name.split("-")[-1])
            policy = MyModelParallelismILPReplacement(verbose=2, replacement_interval=interval)
        elif policy_name == "my-mp-ilp-dynamic":
            policy = MyModelParallelismILPReplacement(verbose=2, dynamic_replacement=True,
                                                      replacement_time=replacement_time)
        elif policy_name == "heuristic-dynamic":
            policy = MyModelParallelismHeuReplacement(verbose=2, dynamic_replacement=True,
                                                      replacement_time=replacement_time, monitor=monitor)
        elif policy_name == "dqn-dynamic":
            from alpa_serve.placement_policy import MyModelParallelismDQNReplacement
            policy = MyModelParallelismDQNReplacement(verbose=2, dynamic_replacement=True,
                                                      replacement_time=replacement_time, monitor=monitor, agent=agent)
        elif "mp-ilp-replace" in policy_name:
            interval = int(policy_name.split("-")[-1])
            policy = ModelParallelismILPReplacement(verbose=2, replacement_interval=interval)
        elif policy_name == "mp-ilp":
            policy = ModelParallelismILP(verbose=1)
        elif policy_name == "my-mp-ilp":   
            policy = MyModelParallelismILP(verbose=2)
        elif policy_name == "my-mp-ilp-model-groups":
            policy = MyModelParallelismILP(verbose=2, model_groups=model_groups)
        elif policy_name == "mp-round-robin":
            policy = ModelParallelismRR(verbose=0)
        elif policy_name in ["mp-search", "mp-search-evo", "mp-search-sep"]:
            use_evo_search = "evo" in policy_name
            use_separation = "sep" in policy_name
            policy = ModelParallelismSearch(
                use_evo_search=use_evo_search, use_separation=use_separation, verbose=2)
        elif "mp-search-sep-replace" in policy_name:
            use_evo_search = "evo" in policy_name
            use_separation = "sep" in policy_name
            interval = int(policy_name.split("-")[-1])
            policy = ModelParallelismSearchReplacement(
                 replacement_interval=interval, use_evo_search=use_evo_search, use_separation=use_separation,verbose=2)
        elif "mp-greedy" in policy_name:
            group_size = int(policy_name.split("-")[2])
            use_evo_search = "evo" in policy_name
            policy = ModelParallelismGreedy(
                use_evo_search=use_evo_search,
                group_size=group_size, verbose=1)
        elif policy_name == "sr-uniform":
            policy = SelectiveReplicationUniform(verbose=1)
        else:
            raise ValueError(f"Invalid placement policy: {policy_name}")

        if placement is not None:
            policy.place_models_impl(controller, cluster_env, model_datas, placement)
        else:
            if "azure" in arrival_process:
                placement = policy.place_models(controller, cluster_env, model_datas, train_workload, test_workload)    
            elif "gamma" in arrival_process:
                placement = policy.place_models(controller, cluster_env, model_datas, train_workload, test_workload)  
            elif "mixed" in arrival_process:
                placement = policy.place_models(controller, cluster_env, model_datas, train_workload, test_workload)  
            elif "online" in arrival_process or "whole_process" in arrival_process:
                placement = policy.place_models(controller, cluster_env, model_datas, train_workload, test_workload)  
            else:
                placement = policy.place_models(controller, cluster_env, model_datas)

        return placement

    return ServingCase(register_models, generate_workload, place_models)


