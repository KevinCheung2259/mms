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
from general_model_serving_case import GeneralModelCase, get_general_model_serving_case


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
                         dynamic_placement: Optional[bool] = False,
                         rl_kwargs: Optional[dict]=None) -> Tuple[StatsResult, ModelPlacement]:
    """A fast simulator that only simulates one stage for a pipeline."""
    from alpa_serve.placement_policy.base_policy import (
        ModelPlacement, ModelPlacementWithReplacement)

    solver_time = 0
    register_models, generate_workload, place_models = case

    workload = generate_workload()

    if workload.enable_simulator_cache and workload.cached_data:
        model_ids, slos, model_names, prof_ress = workload.cached_data
        placement = place_models(None)
    else:
        # Launch the controller
        controller = DummyController()
        register_models(controller)
        if placement:
            placement = place_models(controller, placement=placement)
        else:
            start_placement_time = time.time()
            placement = place_models(controller)
            solver_time = time.time() - start_placement_time

        # Note: assume the model registration order is the same as the model id order in group_models
        model_names, prof_ress = zip(*controller.name2profiling.items())
        unique_model_types = np.sort(list(set(["-".join(m.split("-")[:3]) for m in model_names])))
        unique_model_types = list(unique_model_types)

        name2model_id = {m: i for i, m in enumerate(model_names)}
        unique_type2model_ids = None

        model_ids = np.array([name2model_id.get(r.model_name, -1) for r in workload.requests], dtype=np.int32)
        slos = np.array([r.slo for r in workload.requests], dtype=np.float32)

        if workload.enable_simulator_cache:
            workload.cached_data = (model_ids, slos, model_names, prof_ress)
    
    # 获取RL模型
    if rl_kwargs["rl_stage"] is not None:
        from alpa_serve.placement_policy.model_parallelism_RL import get_rl_agent_one_case
        agent = get_rl_agent_one_case(
                placement, model_names, prof_ress, model_ids, slos, workload.arrivals, rl_kwargs)

    if isinstance(placement, ModelPlacement):
        if dynamic_placement == False:
            (start, finish, good, 
            model_num_requests, model_num_good_requests, group_num_requests, group_num_good_requests,
            receive_request_model_ids, _, _) = approximate_one_case_one_placement(
                placement, model_names, prof_ress, model_ids, slos, workload.arrivals, 
                enable_batching=enable_batching, unique_type2model_ids=unique_type2model_ids,
                scheduling_policy=scheduling_policy)
            
        elif rl_kwargs.get("rl_stage", False):
            window_size = 200
            arrivals = workload.arrivals
            ori_i = 0
            start_i = 0
            end_i = start_i + window_size
            pt = 0
            start_list, finish_list, good_list = [], [], []
            model_num_requests_list, model_num_good_requests_list = [], []
            group_num_requests_list, group_num_good_requests_list = [], []
            placement_list = [placement]
            change_time = [arrivals[0]]

            while start_i < len(arrivals):
                # end_i不得超过边界
                end_i = min(len(arrivals), end_i)
                (start, finish, good, 
                    model_num_requests, model_num_good_requests, group_num_requests, group_num_good_requests,
                    receive_request_model_ids, replacement_time, monitor) = approximate_one_case_one_placement(
                        placement, model_names, prof_ress, 
                        model_ids[start_i:end_i], slos[start_i:end_i], workload.arrivals[start_i:end_i], 
                        enable_batching=enable_batching, unique_type2model_ids=unique_type2model_ids,
                        scheduling_policy=scheduling_policy, replacement=True, return_monitor=True)
               
                start_list.append(start)
                finish_list.append(finish)
                good_list.append(good)
                group_num_requests_list.append(group_num_requests)
                group_num_good_requests_list.append(group_num_good_requests)
                model_num_requests_list.append(model_num_requests)
                model_num_good_requests_list.append(model_num_good_requests)

                start_placement_time = time.time()
                replacement_time = arrivals[min(end_i, len(arrivals)-1)]
                print(f"try to solve placement at time: {replacement_time}")
                new_placement = place_models(controller, replacement_time=replacement_time, monitor=monitor, agent=agent)
                solver_time = time.time() - start_placement_time
                # 判断两次placement是否相等
                if new_placement.group_models != placement.group_models:
                    # print("replace at time: ", change_time[-1])
                    change_time.append(replacement_time)
                    placement_list.append(new_placement)
                    placement = new_placement
                    print(f"placement changed at time: {replacement_time}")
                else:
                    print("new_placement equal to placement, no changed!")
                pt += 1
                start_i = end_i
                end_i = start_i + window_size
            
            start = np.concatenate(start_list)
            finish = np.concatenate(finish_list)
            good = np.concatenate(good_list)
            try:
                group_num_requests = np.sum(group_num_requests_list, axis=0)
                group_num_good_requests = np.sum(group_num_good_requests_list, axis=0)
            except:  # 若每个时间段的分组数量不一样，不能直接合并
                group_num_requests = [item for sublist in group_num_requests_list for item in sublist]
                group_num_good_requests = [item for sublist in group_num_good_requests_list for item in sublist]
            model_num_requests = np.sum(model_num_requests_list, axis=0)
            model_num_good_requests = np.sum(model_num_good_requests_list, axis=0)
            # 打印每个阶段的placement
            for i in range(len(placement_list)):
                print(f"start_time: {change_time[i]}, placement {i}: {placement_list[i]}")
            placement = ModelPlacementWithReplacement(change_time, placement_list)
            assert isinstance(placement, ModelPlacementWithReplacement)

        
        else: # 这一部分是动态放置dynamic用到的代码
            arrivals = workload.arrivals
            start_time = arrivals[0]
            ori_i = 0
            start_i = 0
            pt = 0
            start_list, finish_list, good_list = [], [], []
            model_num_requests_list, model_num_good_requests_list = [], []
            group_num_requests_list, group_num_good_requests_list = [], []
            placement_list = [placement]
            change_time = [arrivals[0]]

            for i in range(len(arrivals)):
                if arrivals[i] > start_time:
                    (start, finish, good, 
                    model_num_requests, model_num_good_requests, group_num_requests, group_num_good_requests,
                    receive_request_model_ids, replacement_time, monitor) = approximate_one_case_one_placement(
                        placement, model_names, prof_ress, 
                        model_ids[start_i:], slos[start_i:], workload.arrivals[start_i:], 
                        enable_batching=enable_batching, unique_type2model_ids=unique_type2model_ids,
                        scheduling_policy=scheduling_policy, replacement=True)

                    if replacement_time is None:
                        start_list.append(start)
                        finish_list.append(finish)
                        good_list.append(good)
                        group_num_requests_list.append(group_num_requests)
                        group_num_good_requests_list.append(group_num_good_requests)
                        model_num_requests_list.append(model_num_requests)
                        model_num_good_requests_list.append(model_num_good_requests)
                        break
                    
                    start_i = np.where(arrivals == replacement_time)[0][0]
                    start_time = replacement_time
                    
                    start_list.append(start[:start_i-ori_i])
                    finish_list.append(finish[:start_i-ori_i])
                    good_list.append(good[:start_i-ori_i])
                    ori_i = start_i

                    group_num_requests_list.append(group_num_requests)
                    group_num_good_requests_list.append(group_num_good_requests)
                    model_num_requests_list.append(model_num_requests)
                    model_num_good_requests_list.append(model_num_good_requests)

                    start_placement_time = time.time()
                    print(f"try to solve placement at time: {replacement_time}")
                    if rl_kwargs.get("rl_stage", False):
                        new_placement = place_models(controller, replacement_time=replacement_time, monitor=monitor, agent=agent)
                    else:
                        new_placement = place_models(controller, replacement_time=replacement_time, monitor=monitor)
                    solver_time = time.time() - start_placement_time
                    # 判断两次placement是否相等
                    if new_placement.group_models != placement.group_models:
                        # print("replace at time: ", change_time[-1])
                        change_time.append(start_time)
                        placement_list.append(new_placement)
                        placement = new_placement
                        print(f"placement changed at time: {start_time}")
                    else:
                        print("new_placement equal to placement, no changed!")
                    pt += 1
            
            start = np.concatenate(start_list)
            finish = np.concatenate(finish_list)
            good = np.concatenate(good_list)
            try:
                group_num_requests = np.sum(group_num_requests_list, axis=0)
                group_num_good_requests = np.sum(group_num_good_requests_list, axis=0)
            except:  # 若每个时间段的分组数量不一样，不能直接合并
                group_num_requests = [item for sublist in group_num_requests_list for item in sublist]
                group_num_good_requests = [item for sublist in group_num_good_requests_list for item in sublist]
            model_num_requests = np.sum(model_num_requests_list, axis=0)
            model_num_good_requests = np.sum(model_num_good_requests_list, axis=0)
            # 打印每个阶段的placement
            for i in range(len(placement_list)):
                print(f"start_time: {change_time[i]}, placement {i}: {placement_list[i]}")
            placement = ModelPlacementWithReplacement(change_time, placement_list)
            assert isinstance(placement, ModelPlacementWithReplacement)

    # 这一部分是sr-replace-xxx用到的代码
    elif isinstance(placement, ModelPlacementWithReplacement):
        arrivals = workload.arrivals
        change_times = placement.start_times[1:] + [inf]

        start_list, finish_list, good_list = [], [], []
        model_num_requests_list, model_num_good_requests_list = [], []
        group_num_requests_list, group_num_good_requests_list = [], []

        start_i = 0
        pt = 0

        for i in range(len(arrivals)):
            if arrivals[i] > change_times[pt]:
                (start, finish, good, 
                 model_num_requests, model_num_good_requests, group_num_requests, group_num_good_requests, 
                 receive_request_model_ids, _, _) = approximate_one_case_one_placement(
                     placement.placements[pt], model_names, prof_ress,
                     model_ids[start_i:i], slos[start_i:i], arrivals[start_i:i], enable_batching=enable_batching)
                start_list.append(start)
                finish_list.append(finish)
                good_list.append(good)
                group_num_requests_list.append(group_num_requests)
                group_num_good_requests_list.append(group_num_good_requests)
                model_num_requests_list.append(model_num_requests)
                model_num_good_requests_list.append(model_num_good_requests)

                start_i = i
                pt += 1

        (start, finish, good, 
         model_num_requests, model_num_good_requests, group_num_requests, group_num_good_requests,
         receive_request_model_ids, _, _) = approximate_one_case_one_placement(
             placement.placements[pt], model_names, prof_ress,
             model_ids[start_i:], slos[start_i:], arrivals[start_i:], enable_batching=enable_batching)
        start_list.append(start)
        finish_list.append(finish)
        good_list.append(good)
        group_num_requests_list.append(group_num_requests)
        group_num_good_requests_list.append(group_num_good_requests)
        model_num_requests_list.append(model_num_requests)
        model_num_good_requests_list.append(model_num_good_requests)

        start = np.concatenate(start_list)
        finish = np.concatenate(finish_list)
        good = np.concatenate(good_list)
        try:
            group_num_requests = np.sum(group_num_requests_list, axis=0)
            group_num_good_requests = np.sum(group_num_good_requests_list, axis=0)
        except:  # 若每个时间段的分组数量不一样，不能直接合并
            group_num_requests = [item for sublist in group_num_requests_list for item in sublist]
            group_num_good_requests = [item for sublist in group_num_good_requests_list for item in sublist]
        model_num_requests = np.sum(model_num_requests_list, axis=0)
        model_num_good_requests = np.sum(model_num_good_requests_list, axis=0)
        # 打印每个阶段的placement
        for i in range(len(placement.placements)):
            print(f"placement {i}: {placement.placements[i]}")

    if fast_stats:
        # Note: no warmup
        interval = start[-1] - start[0]
        per_model_stats = [PerModelStatsResult(
            model_names[i], model_num_requests[i],
            model_num_good_requests[i] / (model_num_requests[i] + eps),
            model_num_requests[i] / interval,
            0, 0, 0, 0, [], [], [], [], [], [], []) for i in range(len(model_names))]
        stats = StatsResult(per_model_stats, tuple(group_num_requests),
                            np.mean(good), np.mean(finish - start),
                            len(start), len(start) / interval)
    else:
        if receive_request_model_ids is not None:
            receive_request_model = set([m for m in receive_request_model_ids if m >= 0])
            print("receive_request_model_num: ", len(list(receive_request_model)))
            # 打印每个模型的goodput, 保留三位小数
            for i in range(len(unique_model_types)):
                model_type = unique_model_types[i]
                model_goodput = model_num_good_requests[i] / (model_num_requests[i])
                print(f"model_type: {model_type}, goodput: {round(model_goodput, 3)}")
            stats = workload.compute_stats(start=start, finish=finish, good=good, warmup=warmup, 
                                        receive_request_model_ids=receive_request_model_ids, 
                                        unique_model_types=unique_model_types,
                                        model_names=model_names, duration=duration)
        else:
            stats = workload.compute_stats(start=start, finish=finish, good=good, warmup=warmup, duration=duration)
        stats.group_num_requests = tuple(group_num_requests)
    
    return stats, placement, solver_time



_DATA_HEADS = ("exp_name", "num_models", "model_groups_num",
               "num_devices", "num_devices_per_node", "mem_budget", 
               "total_rate", "rate_distribution", "arrival_process", "arrival_process_kwargs", "slo_scale", "duration", "policy_name", 
               "placement", "goodput", "mode", "solver_time")

def run_one_general_model_case(case, mode, output_file=None, prof_database=None,
                               debug=False, monitor_kwargs=None, model_groups_num: Optional[int]=1,
                               rl_kwargs: Optional[dict]=None):
    serving_case = get_general_model_serving_case(case, prof_database, model_groups_num=model_groups_num)

    (exp_name, num_devices, num_devices_per_node, mem_budget, model_types, model_names,
    total_rate, rate_distribution, arrival_process, arrival_process_kwargs,
    slo_scale, duration, policy_name) = case
    if rl_kwargs["rl_stage"] is not None:
        cluster_env = ClusterEnv(num_devices=num_devices, mem_budget=mem_budget, num_devices_per_node=num_devices_per_node)
        rl_kwargs["cluster_env"] = cluster_env

    if "dynamic" in case.policy_name:
        dynamic_placement = True
    else:
        dynamic_placement = False

    if mode == "simulate":
        stats, placement, solver_time = approximate_one_case(serving_case, debug=debug, 
                                                dynamic_placement=dynamic_placement, duration=case.duration, 
                                                rl_kwargs=rl_kwargs)
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
                            duration: Optional[int]=3600, rl_kwargs: Optional[dict]=None):
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
            model_groups_num=model_groups_num, rl_kwargs=rl_kwargs))

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
                        choices=["synthetic", "azure_v1", "azure_v2", "mixed", "online", "whole_process"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"], default="power_law")
    parser.add_argument("--rate", type=float, default=10)
    parser.add_argument("--rate_scale", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=3600)
    parser.add_argument("--interval_seconds", type=int, default=60)
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
    parser.add_argument("--model_groups_num", type=int, default=2)
    '''
    强化学习RL设置
    '''
    parser.add_argument("--rl_stage", type=str, default=None, choices=["train", "test", "train_test", None])
    parser.add_argument("--incre_learning", type=bool, default=False)
    parser.add_argument("--rl_policy", type=str, default="dqn", choices=["dqn", "ppo"]) # 现在只支持dqn
    parser.add_argument("--save_model", type=bool, default=False)
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
    
    rl_kwargs = {"rl_stage": args.rl_stage, "incre_learning": args.incre_learning, 
                 "rl_policy": args.rl_policy, "save_model": args.save_model}

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
    elif args.workload == "mixed" or args.workload == "online" or args.workload == "whole_process":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        if args.workload == "whole_process":
            duration = args.duration * 3  # 1h for azure_v1, 2h for sythetic
        else:
            duration = args.duration

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite["mixed"]

        _, _, _, \
        fixed_rate_scale, fixed_cv_scale, \
        _, _, _, \
        _, _, _, _ = azure_v1_suite["mixed"]

        arrival_process = args.workload
        arrival_process_kwargs = {"cv": args.cv, 
                                  "interval_seconds": args.interval_seconds, 
                                  "rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl",
                                  "mixed_ratio": 0.5}
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
    # fixed_num_modelset = args.fixed_num_modelset
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

    ##### goodput vs num_devices #####
    # total_rate = 5
    num_devices_list = [12]  # 4, 8, 12, 16, 20
    policies = ["mp-search-sep", 'sr-replace-600'] # "heuristic-dynamic", 'mp-search-sep', 'sr-greedy', 'sr-replace-600', 'my-mp-ilp-replace-600', 'my-mp-ilp'
    # "mp-search-sep", 'sr-replace-600'
    if policies == ['dqn-dynamic']:
        rl_kwargs['rl_stage'] = 'train'
        # rl_kwargs['incre_learning'] = True
    
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
                                model_groups_num=args.model_groups_num,
                                rl_kwargs=rl_kwargs)