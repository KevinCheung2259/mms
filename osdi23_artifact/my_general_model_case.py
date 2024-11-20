import argparse
import datetime
from collections import namedtuple
import os
import numpy as np
import ray

import sys
sys.path.append('/home/zy/python_project/mms')

from alpa_serve.simulator.controller import (Controller, DummyController,
    simulate_one_case, approximate_one_case)
from alpa_serve.simulator.workload import Workload, GammaProcess, UniformMMPP
from alpa_serve.profiling import ProfilingDatabase, ParallelConfig
from alpa_serve.placement_policy import (ClusterEnv, ModelData,
    SelectiveReplicationILP, SelectiveReplicationGreedy,
    SelectiveReplicationReplacement, SelectiveReplicationUniform,
    ModelParallelismILP, ModelParallelismGreedy, ModelParallelismRR,
    ModelParallelismSearch)
from alpa_serve.profiling import ProfilingDatabase
from alpa_serve.trace import Trace, report_group_stats
from alpa_serve.util import GB, write_tsv, ServingCase
from collections import OrderedDict
from alpa_serve.trace import Trace, TraceReplay
from benchmarks.alpa.util import get_model_def
from benchmarks.alpa.run_one_case import run_one_case
from osdi23_artifact.general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite


GeneralModelCase = namedtuple("GeneralModelCase", [
    "exp_name", "num_devices", "mem_budget", "model_types", "model_names",
    "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs",
    "slo_scale", "duration", "policy_name"])


def get_general_model_serving_case(case, prof_database=None):
    assert isinstance(case, GeneralModelCase), "not GeneralModelCase"
    if prof_database is None:
        prof_database = ProfilingDatabase("/home/zy/python_project/mms/alpa_serve/syn_profiling_result.pkl")

    (exp_name, num_devices, mem_budget, model_types, model_names,
     total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
     slo_scale, duration, policy_name) = case

    cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget)
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
            # 设置波峰的时间段（例如：100-200秒和300-400秒）
            num_intervals = duration // interval_seconds
            # 随机选择波峰的时间段
            peak_times = np.random.choice(num_intervals, 2, replace=False)
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
    else:
        raise ValueError("Invalid arrival process: {arrival_process}")

    # 每个模型的arrival_process的rate和cv
    rates = [a.rate() for a in arrival_processes]
    cvs = [a.cv() for a in arrival_processes]

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

    def place_models(controller):
        num_models = len(model_names)
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(model_names[i], slos[i], rates[i], cvs[i],
                                         prof_database.get(model_types[i])))

        if policy_name == "sr-ilp":
            policy = SelectiveReplicationILP(verbose=1)
        elif policy_name == "sr-greedy":
            policy = SelectiveReplicationGreedy(verbose=1)
        elif "sr-replace" in policy_name:
            interval = int(policy_name.split("-")[2])
            policy = SelectiveReplicationReplacement(verbose=1,
                 replacement_interval=interval)
        elif policy_name == "mp-ilp":
            policy = ModelParallelismILP(verbose=1)
        elif policy_name == "mp-round-robin":
            policy = ModelParallelismRR(verbose=0)
        elif policy_name in ["mp-search", "mp-search-evo", "mp-search-sep"]:
            use_evo_search = "evo" in policy_name
            use_separation = "sep" in policy_name
            policy = ModelParallelismSearch(
                use_evo_search=use_evo_search, use_separation=use_separation, verbose=2)
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

        if "azure" in arrival_process:
            placement = policy.place_models(controller, cluster_env, model_datas, train_workload)
        else:
            placement = policy.place_models(controller, cluster_env, model_datas)

        return placement

    return ServingCase(register_models, generate_workload, place_models)


_DATA_HEADS = ("exp_name", "num_models",
               "num_devices", "mem_budget", "total_rate", "rate_distribution",
               "arrival_process", "arrival_process_kwargs", "slo_scale", "duration",
               "policy_name", "placement", "goodput", "mode")

_SINGLE_MODEL_DATA_HEADS = ("exp_name", "num_models",
               "num_devices", "mem_budget", "total_rate", "rate_distribution",
               "arrival_process", "arrival_process_kwargs", "slo_scale", "duration",
               "policy_name", "placement", "goodput", "mode", 
               "model_name", "num_requests", "goodput", "throughput", "avg_latency", 
               "latency_std", "latency_p90", "latency_p99") 

_SINGLE_MODEL_DATA_HEADS_DETAIL_TIME_WINDOW = (
               "model_name", "model_is_running", "model_received_requests", "model_returned_requests", "model_dropped_requests")

def run_one_general_model_case(case, mode,
                               output_file=None, prof_database=None,
                               debug=False):
    serving_case = get_general_model_serving_case(case, prof_database)

    if mode == "simulate":
        stats, placement = approximate_one_case(serving_case, debug=debug)
    else:
        stats, placement = run_one_case(serving_case, debug=debug)

    #Workload.print_stats(stats)
    print(f"group #req: {stats.group_num_requests}")

    (exp_name, num_devices, mem_budget, model_types, model_names,
    total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
    slo_scale, duration, policy_name) = case

    case_info = (num_devices, mem_budget, total_rate,
                 rate_distribution, arrival_process,
                 arrival_process_kwargs, slo_scale,
                 duration, policy_name)
    res = (placement, round(stats.goodput, 3), mode)
    values = (exp_name, len(model_types)) + case_info + res

    if output_file is not None:
        write_tsv(_DATA_HEADS, values, output_file)

    # 打印每个模型在每个时间窗口的运行情况，并写入tsv文件中
    output_file_single_model = f"{output_file.split('.')[0]}_single_model.tsv"
    for model_stats in stats.per_model_stats:
        values = (exp_name, len(model_types)) + case_info + res + (model_stats.name, 
                model_stats.num_requests, model_stats.goodput, model_stats.throughput, model_stats.latency_mean, 
                model_stats.latency_std, model_stats.latency_p90, model_stats.latency_p99)

        # 将每个模型在每个时间窗口的运行情况写入csv文件中
        write_tsv(_SINGLE_MODEL_DATA_HEADS, values, output_file_single_model, print_line=False)
    
    # 将每个模型的"model_is_running", "model_received_requests", "model_returned_requests", "model_dropped_requests"写入tsv文件
    output_file_single_model_dir = output_file_single_model.split('.')[0] + "_dir"
    os.makedirs(output_file_single_model_dir, exist_ok=True)
    for model_stats in stats.per_model_stats:
        output_file_single_model_detail = os.path.join(output_file_single_model_dir, f"{model_stats.name}.tsv")
        for i in range(len(model_stats.model_is_running)):
            detail_values = (model_stats.name, model_stats.model_is_running[i], model_stats.model_received_requests[i], model_stats.model_returned_requests[i], model_stats.model_dropped_requests[i])
            write_tsv(_SINGLE_MODEL_DATA_HEADS_DETAIL_TIME_WINDOW, detail_values, output_file_single_model_detail, print_line=False)

    return values

def run_general_model_cases(cases, output_file=None,
                            mode="simulate", debug_tstamp=False, parallel=False):
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd(), "excludes": ["backup"]})

    if parallel:
        run_one_case_ = ray.remote(num_cpus=2)(run_one_general_model_case).remote
    else:
        run_one_case_ = run_one_general_model_case

    results = []
    for case in cases:
        results.append(run_one_case_(case, mode,
            output_file=output_file, debug=debug_tstamp))

    if parallel:
        results = ray.get(results)

    return results


def read_general_model_case_tsv(filename):
    rows = []  # List[dict]

    for line in open(filename):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        (exp_name, num_models,
         num_devices, mem_budget,
         total_rate, rate_distribution,
         arrival_process, arrival_process_kwargs,
         slo_scale, duration, policy_name,
         placement, goodput, mode) = line.split("\t")

        num_devices = int(num_devices)
        num_models = int(num_models)
        total_rate = float(total_rate)
        arrival_process_kwargs = eval(arrival_process_kwargs)
        slo_scale = float(slo_scale)
        duration = float(duration)
        goodput = float(goodput)

        values = locals()
        row = {
            key: values[key]
            for key in _DATA_HEADS
        }
        rows.append(row)

    return rows

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
    parser.add_argument("--mem-budget", type=int, default=13)  # default 13 GB
    '''
    请求设置
    '''
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"], default="power_law")
    parser.add_argument("--rate", type=float, default=30)
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

    args = parser.parse_args()

    # choices: {"sr-greedy", "sr-ilp", "mp-ilp",
    #           "mp-round-robin", "mp-greedy-2", "mp-greedy-8", "mp-search", "mp-search-sep"}
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
        duration = -1

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite[model_type]

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale ,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = -1

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite[model_type]

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": args.trace_dir}
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

    if args.exp_name:
        os.makedirs(args.exp_name, exist_ok=True)
        output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                   args.exp_name, output_file_name)
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

    num_devices_list = [16]

    print("打印配置信息")
    print("workload: ", args.workload)
    print("model_type: ", args.model_type)
    print("policy: ", args.policy)
    print("num_devices_list: ", num_devices_list)
    print("mem_budget: ", mem_budget)
    print("model_types: ", model_types)
    print("model_names: ", model_names)

    ##### goodput vs num_devices #####
    if "goodput_vs_num_devices" in experiments:
        print("=== Running goodput vs. #devices ===")
        exp_name = "goodput_vs_num_devices"
        for num_devices in num_devices_list:
            for policy_name in policies:
                cases.append(GeneralModelCase(exp_name,
                    num_devices, mem_budget, model_types, model_names,
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
    if "goodput_vs_slo" in experiments:
        print("=== Running goodput vs. SLO ===")
        exp_name = "goodput_vs_slo"
        for slo_scale in slo_scales:
            for policy_name in policies:
                cases.append(GeneralModelCase(exp_name,
                    fixed_num_devices, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, arrival_process_kwargs,
                    slo_scale, duration, policy_name))

    ##### goodput vs rate/rate_scale #####
    if "goodput_vs_rate" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. rate ===")
            exp_name = "goodput_vs_rate"
            for new_rate in rate_list:
                for policy_name in policies:
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_types, model_names,
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
                                                  "trace_dir": args.trace_dir}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))


    ##### goodput vs cv/cv_scale #####
    if "goodput_vs_cv" in experiments:
        if args.workload == "synthetic":
            print("=== Running goodput vs. cv ===")
            exp_name = "goodput_vs_cv"
            for new_cv in cv_list:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"cv": new_cv}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_types, model_names,
                        total_rate, rate_distribution,
                        arrival_process, new_arrival_process_kwargs,
                        fixed_slo_scale, duration, policy_name))
        else:
            print("=== Running goodput vs. cv_scale ===")
            exp_name = "goodput_vs_cv_scale"
            for cv_scale in cv_scales:
                for policy_name in policies:
                    new_arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                                  "cv_scale": cv_scale,
                                                  "trace_dir": args.trace_dir}
                    cases.append(GeneralModelCase(exp_name,
                        fixed_num_devices, mem_budget, model_types, model_names,
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
                                mode=args.mode, parallel=args.parallel)