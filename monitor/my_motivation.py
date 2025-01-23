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
from divide_models import divide_models
from my_general_model_suite import synthetic_suite, azure_v1_suite, azure_v2_suite
from my_general_model_case import run_general_model_cases, GeneralModelCase


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
                        choices=["synthetic", "azure_v1", "azure_v2", "mixed"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"], default="power_law")
    parser.add_argument("--rate", type=float, default=5)
    parser.add_argument("--rate_scale", type=float, default=1.0)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument("--cv_scale", type=float, default=4)
    parser.add_argument('--duration', type=float, default=3600)
    parser.add_argument("--interval_seconds", type=int, default=60)
    parser.add_argument("--mixed_ratio", type=float, default=0.5)
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
    elif args.workload == "mixed":
        rate_distribution = args.rate_distribution
        total_rate = args.rate
        duration = args.duration

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = synthetic_suite["mixed"]

        _, _, _, \
        fixed_rate_scale, fixed_cv_scale, \
        _, _, _, \
        _, _, _, _ = azure_v1_suite["mixed"]

        arrival_process = "mixed"
        arrival_process_kwargs = {"cv": args.cv, 
                                  "interval_seconds": args.interval_seconds, 
                                  "rate_scale": fixed_rate_scale,
                                  "cv_scale": fixed_cv_scale,
                                  "trace_dir": "/home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl",
                                  "mixed_ratio": args.mixed_ratio}
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
    num_devices_list = [args.num_devices]  # 4, 8, 12, 16, 20, 24
    cv_scales = [args.cv_scale] # 1, 5, 9, 13, 17
    policies = [args.policy] # 'mp-search-sep', 'mp-search-sep-replace-60', 'sr-replace-60', 'heuristic-dynamic', 'dqn-dynamic'
    # 'mp-search-sep', 'mp-search-sep-replace-60', 'sr-replace-60', 
    if policies == ['dqn-dynamic']:
        rl_kwargs['rl_stage'] = 'train'

    print("=== Running goodput vs. #devices and cv ===")
    exp_name = "motivation_goodput_vs_num_devices_cv"
    for num_devices in num_devices_list:
        for cv_scale in cv_scales:
            for policy_name in policies:
                if args.workload == "azure_v1":
                    new_arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
                                                    "cv_scale": cv_scale,
                                                    "trace_dir": trace_dir}
                elif args.workload == "synthetic":
                    new_arrival_process_kwargs = {"cv": cv_scale, 
                                                  "interval_seconds": args.interval_seconds}
                elif args.workload == "mixed":
                    new_arrival_process_kwargs = {"cv": args.cv, 
                                                "interval_seconds": args.interval_seconds, 
                                                "rate_scale": fixed_rate_scale,
                                                "cv_scale": cv_scale,
                                                "trace_dir": "/home/zy/data/datasets/azurefunctions-dataset2019/azure_v1.pkl",
                                                "mixed_ratio": args.mixed_ratio}
                cases.append(GeneralModelCase(exp_name,
                    num_devices, num_devices_per_node, mem_budget, model_types, model_names,
                    total_rate, rate_distribution,
                    arrival_process, new_arrival_process_kwargs,
                    fixed_slo_scale, duration, policy_name))
    

    # ##### goodput vs cv/cv_scale #####
    # # cv_list = [1, 3, 5, 7, 9, 11, 13, 15]
    # if "goodput_vs_cv" in experiments:
    #     if args.workload == "synthetic":
    #         print("=== Running goodput vs. cv ===")
    #         exp_name = "goodput_vs_cv"
    #         for new_cv in cv_list:
    #             for policy_name in policies:
    #                 new_arrival_process_kwargs = {"cv": new_cv}
    #                 cases.append(GeneralModelCase(exp_name,
    #                     fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
    #                     total_rate, rate_distribution,
    #                     arrival_process, new_arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name))
    #     else:
    #         print("=== Running goodput vs. cv_scale ===")
    #         exp_name = "goodput_vs_cv_scale"
    #         for cv_scale in cv_scales:
    #             for policy_name in policies:
    #                 new_arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
    #                                               "cv_scale": cv_scale,
    #                                               "trace_dir": trace_dir}
    #                 cases.append(GeneralModelCase(exp_name,
    #                     fixed_num_devices, num_devices_per_node, mem_budget, model_types, model_names,
    #                     total_rate, rate_distribution,
    #                     arrival_process, new_arrival_process_kwargs,
    #                     fixed_slo_scale, duration, policy_name))

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