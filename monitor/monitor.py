'''
监控系统，给定请求到来情况和模型放置方案，得到在服务过程中单模型、设备组、整个集群的服务情况。
监控指标包括: 每0.1秒请求到来/返回/抛弃的个数、排队长度、是否正在运行、吞吐、goodput、利用率、内存使用情况
'''
import numpy as np
import ray
import os
import datetime
import itertools
from alpa_serve.placement_policy.base_policy import ModelPlacement, BasePlacementPolicy
from alpa_serve.profiling import ProfilingResult, ParallelConfig, ProfilingDatabase
import argparse
from alpa_serve.util import GB, write_tsv, ServingCase
from benchmarks.alpa.run_one_case import run_one_case
from osdi23_artifact.general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite
from benchmarks.alpa.general_model_case import GeneralModelCase, get_general_model_serving_case
from my_general_model_case import approximate_one_case
from request_inspect import plot_inspect_single_model_request, plot_inspect_cluster_request
from memory_usage_inspect import plot_inspect_group_memroy

_SINGLE_MODEL_DATA_HEADS = ("model_name", "num_requests", "goodput", "throughput", "avg_latency", 
               "latency_std", "latency_p90", "latency_p99") 

_SINGLE_MODEL_DATA_HEADS_DETAIL_TIME_WINDOW = (
               "model_name", "model_is_running", "model_received_requests", "model_returned_requests", "model_dropped_requests")

def run_monitor_one_general_model_case(case, serving_case, mode, debug=False, placement=None, monitor_kwargs=None):
    model_mapping_strategy = monitor_kwargs.get("model_mapping_strategy", None)
    scheduling_policy = monitor_kwargs.get("scheduling_policy", None)
    duration = monitor_kwargs.get("duration", 3600)
    detail = monitor_kwargs.get("detail", False)
    output_file = monitor_kwargs.get("output_file", None)
    if mode == "simulate":
        stats, placement = approximate_one_case(serving_case, duration=duration, debug=debug,
                                                placement=placement, model_mapping_strategy=model_mapping_strategy,
                                                scheduling_policy=scheduling_policy)
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
    res = (placement, round(stats.goodput, 3), round(stats.latency_mean, 3), round(stats.request_rate, 3), mode)
    values = (exp_name, len(model_types)) + case_info + res
    print(
        "========== Results ==========\n"
        f"Experiment Name: {exp_name}\n"
        f"Number of Models: {len(model_types)}\n"
        f"Number of Devices: {num_devices}\n"
        f"Memory Budget: {mem_budget}\n"
        f"Total Rate: {total_rate}\n"
        f"Rate Distribution: {rate_distribution}\n"
        f"Arrival Process: {arrival_process}\n"
        f"Arrival Process Arguments: {arrival_process_kwargs}\n"
        f"SLO Scale: {slo_scale}\n"
        f"Duration: {duration}\n"
        f"Policy Name: {policy_name}\n"
        f"Placement: {placement}\n"
        f"Goodput: {round(stats.goodput, 3)}\n"
        f"Lantency Mean: {round(stats.latency_mean, 3)}\n"
        f"Request Rate: {round(stats.request_rate, 3)}\n"
        f"Mode: {mode}")

    if detail:
        # 打印每个模型在每个时间窗口的运行情况，并写入tsv文件中
        output_file_single_model = f"{output_file.split('.')[0]}_single_model.tsv"
        for model_stats in stats.per_model_stats:
            values =  (model_stats.name, model_stats.num_requests, model_stats.goodput, model_stats.throughput, model_stats.latency_mean, 
                    model_stats.latency_std, model_stats.latency_p90, model_stats.latency_p99)

            # 将每个模型在每个时间窗口的运行情况写入csv文件中
            write_tsv(_SINGLE_MODEL_DATA_HEADS, values, output_file_single_model, print_line=True)
        
        # 将每个模型的"model_is_running", "model_received_requests", "model_returned_requests", "model_dropped_requests"写入tsv文件
        output_file_single_model_dir = output_file_single_model.split('.')[0] + "_dir"
        os.makedirs(output_file_single_model_dir, exist_ok=True)
        for model_stats in stats.per_model_stats:
            output_file_single_model_detail = os.path.join(output_file_single_model_dir, f"{model_stats.name}.tsv")
            for i in range(len(model_stats.model_is_running)):
                detail_values = (model_stats.name, model_stats.model_is_running[i], model_stats.model_received_requests[i], 
                                    model_stats.model_returned_requests[i], model_stats.model_dropped_requests[i])
                write_tsv(_SINGLE_MODEL_DATA_HEADS_DETAIL_TIME_WINDOW, detail_values, output_file_single_model_detail, print_line=False)
        
        # 画出request的情况，若没有plot_save_path，则创建
        cluster_output_file_dir = os.path.dirname(output_file)
        plot_single_save_path = output_file_single_model.split('.')[0] + "_plot"
        if not os.path.exists(plot_single_save_path):
            os.makedirs(plot_single_save_path)
        plot_inspect_single_model_request(output_file_single_model_dir, plot_single_save_path)
        plot_inspect_cluster_request(output_file_single_model_dir, cluster_output_file_dir)

        # 画出内存使用情况
        plot_inspect_group_memroy(output_file_single_model_dir, cluster_output_file_dir, model_placement=placement,
                            model_types=model_types, model_names=model_names)

    return values

def run_monitor_general_model_cases(cases, output_file=None, duration=3600,
                            mode="simulate", debug_tstamp=False, parallel=False, detail=False,
                            placement=None, model_mapping_strategy=None, scheduling_policy='load_balance'):
    if not ray.is_initialized():
        ray.init(address="auto", runtime_env={"working_dir": os.getcwd(), "excludes": ["backup"]})

    if parallel:
        run_one_case_ = ray.remote(num_cpus=2)(run_monitor_one_general_model_case).remote
    else:
        run_one_case_ = run_monitor_one_general_model_case

    results = []
    for case in cases:
        results.append(run_one_case_(case, mode, duration=duration, output_file=output_file, 
                                     debug=debug_tstamp, detail=detail, placement=placement, 
                                     model_mapping_strategy=model_mapping_strategy, scheduling_policy=scheduling_policy))

    if parallel:
        results = ray.get(results)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    '''
    实验配置参数
    '''
    parser.add_argument("--exp-name", type=str, default="monitor_exp_busiest_device_1",)
    # 只有指定了output参数，才会将结果写入文件
    parser.add_argument("--output", type=str, default="res_monitor_general_model_cases.tsv", 
                        help="Output file name default: res_monitor_general_model_cases.tsv")
    parser.add_argument("--detail", type=str, default=True, 
                        help="Whether to output complete cluster operation process data")
    parser.add_argument("--parallel", action="store_true", default=True)
    parser.add_argument("--mode", choices=["simulate", "run"], default="simulate")
    parser.add_argument("--single", action="store_true", default=True)
    '''
    请求参数
    '''
    parser.add_argument("--trace-dir", type=str, 
                    default="/home/zhangy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl")
    parser.add_argument("--workload", type=str, default="azure_v1",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="power_law")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--rate_scale", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)
    parser.add_argument("--model_mapping_strategy", type=str,
                        choices=["stripe", "round_robin", "specify_model_type_stripe", "specify_model_type_round_robin"],
                        default="specify_model_type_stripe")
    parser.add_argument("--scheduling_policy", type=str, default="busiest_device",
                        choices=["load_balance", "busiest_device"])
    '''
    模型参数
    '''
    parser.add_argument("--mem-budget", type=int, default=13)
    parser.add_argument("--model-type", type=str, default="mixed",
                        choices=["all_transformers", "mixed"])
    parser.add_argument("--policy", type=str, default="mp-search-sep")
    parser.add_argument("--large-models", action="store_true")
    args = parser.parse_args()

    '''
    必须指定的参数：
    model_placement: 模型放置方案
    model_set: 模型集合
    fixed_num_modelset: 每个模型的数量
    '''
    
    # model_placement = ModelPlacement(group_configs=[ParallelConfig(dp=1, op=4, pp=1), 
    #                           ParallelConfig(dp=1, op=4, pp=1), 
    #                           ParallelConfig(dp=1, op=1, pp=8), 
    #                           ParallelConfig(dp=1, op=1, pp=8), 
    #                           ParallelConfig(dp=1, op=1, pp=8)], 
    #                           group_models=[[0, 18, 48, 60], [12, 24, 42, 54], 
    #                                         [1, 4, 8, 9, 11, 14, 29, 33, 35, 37, 39, 41, 43, 44, 50, 52, 53, 61, 71], 
    #                                         [2, 3, 7, 10, 13, 15, 16, 17, 21, 27, 28, 31, 38, 40, 45, 46, 51, 59, 63, 65, 70], 
    #                                         [5, 16, 19, 20, 22, 23, 25, 26, 32, 34, 47, 49, 55, 56, 57, 58, 62, 64, 69]])
    model_placement = ModelPlacement(group_configs=[ParallelConfig(dp=1, op=1, pp=2), ParallelConfig(dp=1, op=1, pp=2), 
                                                    ParallelConfig(dp=1, op=1, pp=2), ParallelConfig(dp=1, op=1, pp=2),
                                                    ParallelConfig(dp=1, op=1, pp=2), ParallelConfig(dp=1, op=1, pp=2), 
                                                    ParallelConfig(dp=1, op=1, pp=2), ParallelConfig(dp=1, op=1, pp=2)], 
                                    group_models=[[0], [1, 4, 5], [1, 4, 5], [1, 4, 5], [2, 3, 4, 5], [2, 3, 4, 5], 
                                                  [2, 3, 4, 5], [2, 3, 4, 5]])
    
    model_set = ["bert-6.7b", "moe-5.3b", "bert-2.6b", "moe-2.4b", "bert-1.3b", "moe-1.3b"]
    fixed_num_modelset = 1
    
    # 找出group_models中每种模型的数量
    model_types_num = [0] * len(model_set)
    for models in range(len(model_placement.group_models)):
        for m in range(len(model_placement.group_models[models])):
            model_types_num[model_placement.group_models[models][m]] += 1

    if fixed_num_modelset == 1:
        fixed_num_modelset = max(model_types_num)

    model_types = model_set * fixed_num_modelset
    model_names = sum([[f"{model_type}-{i}" for model_type in model_set] for i in range(fixed_num_modelset)], [])

    # 如果model_placement.group_models这个二维列表中有重复的元素, 则更新group_models将重复的元素替换为不重复的元素
    group_models_flatten = list(itertools.chain.from_iterable(model_placement.group_models))
    if len(group_models_flatten) != len(set(group_models_flatten)):
        group_models = model_placement.group_models
        model_types_num = [-1] * len(model_set)
        for models in range(len(model_placement.group_models)):
            for m in range(len(model_placement.group_models[models])):
                model_types_num[model_placement.group_models[models][m]] += 1
                group_models[models][m] = model_types_num[model_placement.group_models[models][m]] * len(model_set) + model_placement.group_models[models][m]
        model_placement.group_models = group_models

    num_devices = 0
    # 获取每个group_configs的设备数
    for i in range(len(model_placement.group_configs)):
        num_devices += model_placement.group_configs[i].dp * model_placement.group_configs[i].op * model_placement.group_configs[i].pp
    
    mem_budget = args.mem_budget * GB
    
    model_type = args.model_type
    # workload config
    if args.workload == "synthetic":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        arrival_process = "gamma"
        arrival_process_kwargs = {"cv": args.cv}

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite[model_type]
    elif args.workload == "azure_v1":
        rate_distribution = None
        total_rate = -1
        duration = 3600

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite[model_type]

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale * args.rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zhangy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl"}
    elif args.workload == "azure_v2":
        # real trace does not need these config
        rate_distribution = None
        total_rate = -1
        duration = 86400

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v2_suite[model_type]

        arrival_process = "azure_v2"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zhangy/data/datasets/azure_v2.pkl"}
    else:
        raise ValueError("Unsupported workload!")

    if args.output is not None:
        if args.output.endswith(".tsv"):
            output_file_name = args.output
        else:
            output_file_name = args.output + ".tsv"

        if args.exp_name:
            # os.makedirs(args.exp_name, exist_ok=True)
            output_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), args.exp_name)
            os.makedirs(output_folder, exist_ok=True)

            output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    args.exp_name, output_file_name)
        else:
            output_folder = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print(f"Output folder: {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    output_folder, output_file_name)
        # 如果文件不存在，创建文件
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                pass  # 不写入任何信息
            print(f"File '{output_file}' has been created successfully.")
    else:
        output_file = None

    cases = []

    print("=== Running monitor cluster with placement ===")

    # slo_scales = [0.75, 1, 2, 3, 4, 5, 7.5, 10, 15, 25]
    # exp_name = "monitor_goodput_vs_slo_scales"
    # for slo_scale in slo_scales:
    #     cases.append(GeneralModelCase(exp_name,
    #         num_devices, mem_budget, model_types, model_names,
    #         total_rate, rate_distribution,
    #         arrival_process, arrival_process_kwargs,
    #         slo_scale, duration, args.policy))
    
    cases.append(GeneralModelCase(args.exp_name,
        num_devices, mem_budget, model_types, model_names,
        total_rate, rate_distribution,
        arrival_process, arrival_process_kwargs,
        fixed_slo_scale, duration, args.policy))

    if args.single:
        cases = [cases[0]]
        args.parallel = False

    n_cases = len(cases)
    M = 8
    n_case_each_run = (n_cases + M - 1) // M

    for i in range(M):
        start_case = i * n_case_each_run
        end_case = (i + 1) * n_case_each_run  if (i + 1) * n_case_each_run < n_cases else n_cases
        run_monitor_general_model_cases(cases[start_case:end_case],
                                output_file=output_file, duration=duration,
                                mode=args.mode, parallel=args.parallel,
                                detail=args.detail, placement=model_placement,
                                model_mapping_strategy=args.model_mapping_strategy,
                                scheduling_policy=args.scheduling_policy)