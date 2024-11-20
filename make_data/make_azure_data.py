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
    ModelParallelismSearch, MyModelParallelismILP, MyModelParallelismILPReplacement, ModelParallelismILPReplacement)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.trace import Trace, report_group_stats
from alpa_serve.util import GB, write_tsv, ServingCase, inf, eps
from collections import OrderedDict
from alpa_serve.trace import Trace, TraceReplay
from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case
from osdi23_artifact.general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite
from alpa_serve.placement_policy.base_policy import ModelPlacement

GeneralModelCase = namedtuple("GeneralModelCase", [
    "exp_name", "num_devices", "num_devices_per_node", "mem_budget", "model_types", "model_names",
    "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs",
    "slo_scale", "duration", "policy_name"])

def approximate_one_case(case: ServingCase,
                         duration: int = 3600,
                         seed: int = 0,
                         warmup: int = DEFAULT_WARMUP,
                         debug: bool = False,
                         fast_stats: bool = False,
                         enable_batching: bool = False,
                         placement: Optional[ModelPlacement] = None,
                         model_mapping_strategy: Optional[str] = None,
                         scheduling_policy: Optional[str] = 'load_balance',
                         dynamic_placement: Optional[bool] = False) -> Tuple[StatsResult, ModelPlacement]:
    """A fast simulator that only simulates one stage for a pipeline."""
    from alpa_serve.placement_policy.base_policy import (
        ModelPlacement, ModelPlacementWithReplacement)

    solver_time = 0
    register_models, generate_workload, place_models = case

    workload = generate_workload()


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
        peak_rate = 2  # 峰值期间的请求率
        base_rate = 0.2  # 其他时间的基础请求率
        interval_seconds = arrival_process_kwargs.get("interval_seconds", 600)
        for i in range(num_models):
            np.random.seed(seed)
            # 设置波峰的时间段（例如：100-200秒和300-400秒）
            num_intervals = duration // interval_seconds
            # 随机选择波峰的时间段
            peak_times = np.random.choice(num_intervals, 2, replace=False)
            seed += 1
            distribution = []
            for t in range(duration // interval_seconds):
                if t in peak_times:
                    rate = peak_rate * rates[i]
                else:
                    rate = base_rate * rates[i]
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
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

        # for debugging:
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
                                              end_time='13.23.60',
                                              # end_time='13.1.0',
                                              interval_seconds=5400,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        test_replays = azure_v2_trace.replay(model_names,
                                              model_mapping_strategy="stripe",
                                              arrival_distribution="gamma",
                                              start_time='13.0.0',
                                              end_time='13.23.60',
                                              # end_time='13.1.0',
                                              interval_seconds=5400,
                                              rate_scale_factor=arrival_process_kwargs["rate_scale"],
                                              cv_scale_factor=arrival_process_kwargs["cv_scale"])
        del azure_v2_trace
        ws = []
        for model_name, slo in zip(model_names, slos):
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)

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
        # for debugging:
        for m in test_replays:
            # 打印每个模型的trace统计信息
            test_replays[m].report_stats()
        # 打印全部的trace统计信息
        report_group_stats(list(test_replays.values()))
        arrival_processes = [test_replays[model_name] for model_name in model_names]
        replays = test_replays
    else:
        raise ValueError("Invalid arrival process: {arrival_process}")

    # 每个模型的arrival_process的rate和cv
    rates = [a.rate() for a in arrival_processes]
    cvs = [a.cv() for a in arrival_processes]

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


_DATA_HEADS = ("exp_name", "num_models", "model_groups_num",
               "num_devices", "num_devices_per_node", "mem_budget", 
               "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs", "slo_scale", "duration", "policy_name", 
               "placement", "goodput", "mode", "solver_time")

def run_one_general_model_case(case, mode, output_file=None, prof_database=None,
                               debug=False, monitor_kwargs=None, model_groups_num: Optional[int]=1):
    serving_case = get_general_model_serving_case(case, prof_database, model_groups_num=model_groups_num)
    if "dynamic" in case.policy_name:
        dynamic_placement = True
    else :
        dynamic_placement = False

    if mode == "simulate":
        stats, placement, solver_time = approximate_one_case(serving_case, debug=debug, 
                                                dynamic_placement=dynamic_placement, duration=case.duration)
    else:
        stats, placement = run_one_case(serving_case, debug=debug)

    #Workload.print_stats(stats)
    print(f"group #req: {stats.group_num_requests}")

    print(f"latency_mean: {stats.latency_mean}")
    latency_p99 = []
    for m in stats.per_model_stats:
        latency_p99.append(m.latency_p99)
    print(f"latency_p99: {latency_p99}")


    (exp_name, num_devices, num_devices_per_node, mem_budget, model_types, model_names,
    total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
    slo_scale, duration, policy_name) = case

    case_info = (num_devices, num_devices_per_node, mem_budget, 
                 total_rate, rate_distribution, arrival_process,
                 arrival_process_kwargs, slo_scale,
                 duration, policy_name)
    res = (placement, round(stats.goodput, 3), mode, round(solver_time, 3))
    values = (exp_name, len(model_types), model_groups_num) + case_info + res

    if output_file is not None:
        write_tsv(_DATA_HEADS, values, output_file)

    if monitor_kwargs.get("monitor", False):
        # placement = ModelPlacement(group_configs=(ParallelConfig(dp=1, op=1, pp=2), ParallelConfig(dp=1, op=1, pp=2)), group_models=((0, 2, 5, 8), (1, 3, 4, 9, 11)))
        from monitor import run_monitor_one_general_model_case
        monitor_kwargs["output_file"] = output_file
        run_monitor_one_general_model_case(case, serving_case, mode, placement=placement, monitor_kwargs=monitor_kwargs)
        # approximate_one_case(serving_case, debug=debug, duration=duration, 
        #                     placement=placement, model_mapping_strategy=model_mapping_strategy,
        #                     scheduling_policy=scheduling_policy)
    
    return values

def run_general_model_cases(cases, output_file=None,
                            mode="simulate", debug_tstamp=False, parallel=False, 
                            monitor_kwargs=None, model_groups_num: Optional[int]=1,
                            duration: Optional[int]=3600):
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd(), "excludes": ["backup"]})

    if parallel:
        run_one_case_ = ray.remote(num_cpus=2)(run_one_general_model_case).remote
    else:
        run_one_case_ = run_one_general_model_case

    results = []
    for case in cases:
        results.append(run_one_case_(case, mode,
            output_file=output_file, debug=debug_tstamp, monitor_kwargs=monitor_kwargs, 
            model_groups_num=model_groups_num))

    if parallel:
        results = ray.get(results)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="default")
    parser.add_argument("--output", type=str, default="res_general_model_cases.tsv")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--mode", choices=["simulate", "run"],
                        default="simulate")
    parser.add_argument("--trace-dir", type=str, default="~/azure_v2.pkl")
    parser.add_argument("--exp-ids", type=str, default="all",
                        choices=["all", "goodput_vs_num_devices", "goodput_vs_num_models",
                              "goodput_vs_slo", "goodput_vs_rate", "goodput_vs_cv",
                              "device_vs_model"])
    '''
    集群设置
    '''
    parser.add_argument("--num-devices", type=int, default=4)
    parser.add_argument("--mem-budget", type=int, default=13)  # default 13 GB
    parser.add_argument("--num_devices_per_node", type=int, default=2)  # default 4 devices per node
    '''
    请求设置
    '''
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"], default="power_law")
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--rate_scale", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=3600)
    parser.add_argument("--interval_seconds", type=int, default=600)
    '''
    模型设置
    '''
    parser.add_argument("--model-type", type=str, default="mixed",
                        choices=["all_transformers", "mixed", "synthetic"])
    parser.add_argument("--fixed_num_modelset", type=int, default=1)
    parser.add_argument("--policy", type=str, default="mp-search-sep")
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--large-models", action="store_true")
    parser.add_argument("--model_groups_num", type=int, default=1)
    '''
    监控设置
    '''
    parser.add_argument("--monitor", action="store_true", default=False)
    parser.add_argument("--model_mapping_strategy", type=str,
                        choices=["stripe", "round_robin", "specify_model_type_stripe", "specify_model_type_round_robin"],
                        default="stripe")
    parser.add_argument("--scheduling_policy", type=str, default="load_balance",
                        choices=["load_balance", "busiest_device"])
    parser.add_argument("--detail", type=str, default=True, 
                        help="Whether to output complete cluster operation process data")
    parser.add_argument("--plot_single_save_path", type=str, help="输出单个模型的图片保存路径",
                        default="monitor/monitor_exp_instance_busiest_device/plot_res_monitor_general_model_cases_single_model")
    parser.add_argument("--plot_cluster_save_path", type=str, help="输出集群的图片保存路径",
                        default="monitor/monitor_exp_instance_busiest_device")
    
    args = parser.parse_args()

    monitor_kwargs = {"monitor": args.monitor, "duration": args.duration, 
                    "model_mapping_strategy": args.model_mapping_strategy, 
                    "scheduling_policy": args.scheduling_policy,
                    "detail": args.detail}
    # choices: {"sr-greedy", "sr-ilp", "mp-ilp",
    #           "mp-round-robin", "mp-greedy-2", "mp-greedy-8", "mp-search", "mp-search-sep"}
    if "azure_v1" in args.trace_dir or "azure_v1" in args.workload:
        trace_dir = "/home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl"
    elif "azure_v2" in args.trace_dir or "azure_v2" in args.workload:
        trace_dir = "/home/zy/data/datasets/azure_v2.pkl"
    else:
        trace_dir = args.trace_dir
    num_devices_per_node = args.num_devices_per_node
    num_devices = args.num_devices
    if args.policy:
        policies = [args.policy]
    else:
        if args.workload == "azure_v1":
            policies = ["sr-greedy", "sr-replace-60", "mp-search-sep"]
        else:
            policies = ["sr-greedy", "sr-replace-21600", "mp-search-sep"]
    mem_budget = args.mem_budget * GB
    model_type = args.model_type

    # multi-model config
    if args.model_type == "mixed":
        #model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b", "moe-1.3b", "moe-2.4b", "moe-5.3b"] # 39.2 GB
        model_set = ["bert-6.7b", "moe-5.3b", "bert-2.6b", "moe-2.4b", "bert-1.3b", "moe-1.3b"] # 39.2 GB
    elif args.model_type == "all_transformers":
        #model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b"] # 21.2 G
        #model_set = ["bert-6.7b", "bert-1.3b"]
        model_set = ["bert-6.7b", "bert-2.6b", "bert-1.3b"]
    elif args.model_type == "synthetic":
        model_set = ["bert-6.7b", "moe-5.3b", "bert-2.6b", "moe-2.4b", "bert-1.3b", "moe-1.3b",
                    "bert-6.7b-a", "moe-5.3b-a", "bert-2.6b-a", "moe-2.4b-a", "bert-1.3b-a", "moe-1.3b-a"]
                    #  "bert-6.7b-b", "moe-5.3b-b", "bert-2.6b-b", "moe-2.4b-b", "bert-1.3b-b", "moe-1.3b-b",
                    #  "bert-6.7b-c", "moe-5.3b-c", "bert-2.6b-c", "moe-2.4b-c", "bert-1.3b-c", "moe-1.3b-c"]
    
    # workload config 
    if args.workload == "synthetic":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv, "interval_seconds": args.interval_seconds}

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite["mixed"]
    elif args.workload == "azure_v1":
        rate_distribution = None
        total_rate = -1
        duration = 3600

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite["mixed"]  

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale ,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl"}
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = 86400

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite["mixed"]

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zy/data/datasets/azure_v2.pkl"}
    else:
        raise ValueError("Unsupported workload!")

    # default models to be served
    fixed_num_modelset = args.fixed_num_modelset
    model_types = model_set * fixed_num_modelset
    model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(fixed_num_modelset)], [])

    if args.output.endswith(".tsv"):
        output_file_name = args.output
    else:
        output_file_name = args.output + ".tsv"

    # 回到当前文件所在的文件夹
    # current_dir = os.path.dirname(os.path.realpath(__file__))  # 获取当前文件的目录
    # os.chdir(current_dir)  # 切换到当前文件所在的目录

    if args.exp_name:
        exp_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.exp_name)
        os.makedirs(exp_name, exist_ok=True)
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   exp_name, output_file_name)
    else:
        output_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   output_folder, output_file_name)

    # parse exp ids:
    if args.exp_ids == "all":
        experiments = ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                       "goodput_vs_rate", "goodput_vs_cv"]
    else:
        assert args.exp_ids in ["goodput_vs_num_devices", "goodput_vs_num_models", "goodput_vs_slo",
                       "goodput_vs_rate", "goodput_vs_cv"]
        experiments = [args.exp_ids]
    
    if args.ablation:
        experiments = ["goodput_vs_rate", "goodput_vs_cv"]

    cases = []

    num_devices_list = [4]

    ##### goodput vs num_devices #####
    # total_rate = 5
    # num_devices_list = [4]  # 4, 8, 12, 16, 20
    # policies = ["my-mp-ilp", "my-mp-ilp-replace-600", "my-mp-ilp-dynamic", "mp-search-sep", "sr-greedy", "sr-replace-600"]
    if "goodput_vs_num_devices" in experiments:
        print("=== Running goodput vs. #devices ===")
        exp_name = "goodput_vs_num_devices"
        for num_devices in num_devices_list:
            for policy_name in policies:
                cases.append(GeneralModelCase(exp_name,
                    num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))

   
    ##### goodput vs num_models #####
    # if "goodput_vs_num_models" in experiments:
    #     print("=== Running goodput vs. #models ===")
    #     exp_name = "goodput_vs_num_models"
    #     for num_modelset in num_modelset_list:
    #         for policy_name in policies:
    #             new_model_types = model_set * num_modelset
    #             new_model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(num_modelset)], [])
    #             if args.workload == "synthetic":
    #                  cases.append(GeneralModelCase(exp_name,
    #                     fixed_num_devices, mem_budget, new_model_types, new_model_names,
    #                     total_rate * num_modelset / fixed_num_modelset, rate_distribution,
    #                     arrival_process, arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name))
    #             else:
    #                 new_arrival_process_kwargs = {"rate_scale": num_modelset / fixed_num_modelset,
    #                                              "cv_scale": fixed_cv_scale,
    #                                              "trace_dir": args.trace_dir}
    #                 cases.append(GeneralModelCase(exp_name,
    #                     fixed_num_devices, mem_budget, new_model_types, new_model_names,
    #                     total_rate, rate_distribution,
    #                     arrival_process, arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name))

    ##### goodput vs slo #####
    # fixed_num_devices = 4
    # total_rate = 5
    # slo_scales = [1, 2.5, 5, 7.5, 10, 12.5, 15]
    if "goodput_vs_slo" in experiments:
        print("=== Running goodput vs. SLO ===")
        exp_name = "goodput_vs_slo"
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(GeneralModelCase(exp_name,
                    fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))

    # rate_list = [7.5, 15, 25, 30]
    if args.policy:
        policies = [args.policy]
    else:
        policies = ["my-mp-ilp", "my-mp-ilp-replace-600", "my-mp-ilp-dynamic", "mp-search-sep", "sr-greedy", "sr-replace-600"]
    fixed_num_devices = num_devices
    ##### goodput vs rate/rate_scale #####
    if "goodput_vs_rate" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. rate ===")
            exp_name = "goodput_vs_rate"
            for new_rate in rate_list:
                for policy_name in policies:
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                        new_rate, rate_distribution,
                        arrival_process, arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
        else:
            print("=== Running goodput vs. rate_scale ===")
            exp_name = "goodput_vs_rate_scale"
            for rate_scale in rate_scales:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": rate_scale,
                                                  "cv_scale": fixed_cv_scale,
                                                  "trace_dir": trace_dir}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))


    ##### goodput vs cv/cv_scale #####
    # cv_list = [1, 3, 5, 7, 9, 11, 13, 15]
    cv_list = [3]
    if "goodput_vs_cv" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. cv ===")
            exp_name = "goodput_vs_cv"
            for new_cv in cv_list:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"cv": new_cv}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
        else:
            print("=== Running goodput vs. cv_scale ===")
            exp_name = "goodput_vs_cv_scale"
            cv_scales = [3]
            fixed_rate_scale = 0.01
            for cv_scale in cv_scales:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                                  "cv_scale": cv_scale,
                                                  "trace_dir": trace_dir}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))

    if args.single:
        cases = [cases[0]]
        args.parallel = False

    n_cases = len(cases)
    M = 8
    n_case_each_run = (n_cases + M - 1) // M
    for i in range(M):
        start_case = i * n_case_each_run
        end_case = (i + 1) * n_case_each_run  if (i + 1) * n_case_each_run < n_cases else n_cases
        run_general_model_cases(cases[start_case:end_case],
                                output_file=output_file,
                                mode=args.mode, parallel=args.parallel,
                                monitor_kwargs=monitor_kwargs,
                                model_groups_num=args.model_groups_num)