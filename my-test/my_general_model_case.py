import argparse
import datetime
from collections import namedtuple
import os

import ray

import sys
sys.path.append('/home/zhangy/python_project/mms')

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
        prof_database = ProfilingDatabase("/home/zhangy/python_project/mms/alpa_serve/profiling_result.pkl")

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
        arrival_processes = [
            GammaProcess(rates[i], arrival_process_kwargs["cv"])
            for i in range(num_models)
        ]
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
            ws.append(train_replays[model_name].to_workload(slo))
        train_workload = Workload.merge(*ws)
        # for debugging:
        for m in test_replays:
            test_replays[m].report_stats()
        report_group_stats(list(test_replays.values()))
        arrival_processes = [test_replays[model_name] for model_name in model_names]
    else:
        raise ValueError("Invalid arrival process: {arrival_process}")

    rates = [a.rate() for a in arrival_processes]
    cvs = [a.cv() for a in arrival_processes]

    def register_models(controller):
        is_simulator = isinstance(controller, (Controller, DummyController))

        for model_name, model_type in zip(model_names, model_types):
            controller.register_model.remote(
                model_name, get_model_def(model_type, is_simulator,
                                          prof_database))

    def generate_workload(start=0):
        w = Workload.empty()
        for i in range(num_models):
            if "azure" in arrival_process:
                w += arrival_processes[i].to_workload(slos[i])
            else:
                w += arrival_processes[i].generate_workload(model_names[i], start,
                                                            duration, slo=slos[i], seed=i)
        return w

    def place_models(controller):
        num_models = len(model_names)
        model_datas = []
        for i in range(num_models):
            model_datas.append(ModelData(model_names[i], slos[i], rates[i], cvs[i],
                                         prof_database.get(model_types[i])))

        if policy_name == "sr-ilp":
            policy = SelectiveReplicationILP(verbose=1)
        # selective replication greedy
        elif policy_name == "sr-greedy":
            policy = SelectiveReplicationGreedy(verbose=1)
        # 这是clock++的
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
    parser.add_argument("--mem-budget", type=int, default=13)
    parser.add_argument("--workload", type=str, default="synthetic",
                        choices=["synthetic", "azure_v1", "azure_v2"])
    parser.add_argument("--rate-distribution", choices=["uniform", "power_law"],
                        default="power_law")
    parser.add_argument("--rate", type=float, default=64)
    parser.add_argument("--cv", type=float, default=4)
    parser.add_argument('--duration', type=float, default=200)
    parser.add_argument("--model-type", type=str, default="all_transformers",
                        choices=["all_transformers", "mixed"])
    parser.add_argument("--policy", type=str)
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
    else:
        #model_set = ["bert-1.3b", "bert-2.6b", "bert-6.7b"] # 21.2 G
        #model_set = ["bert-6.7b", "bert-1.3b"]
        model_set = ["bert-6.7b", "bert-2.6b", "bert-1.3b"]
    
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
        duration = -1

        fixed_num_devices, fixed_num_modelset, fixed_slo_scale, \
        fixed_rate_scale, fixed_cv_scale, \
        num_devices_list, num_modelset_list, slo_scales, \
        rate_list, cv_list, rate_scales, cv_scales = azure_v1_suite[model_type]

        arrival_process = "azure_v1"
        arrival_process_kwargs = {"rate_scale": fixed_rate_scale,
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

    num_devices_list = [32]

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