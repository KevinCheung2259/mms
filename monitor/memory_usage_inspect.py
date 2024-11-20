import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib.pyplot as plt
from alpa_serve.profiling import ProfilingDatabase
import itertools
from alpa_serve.placement_policy.base_policy import ModelPlacement, ParallelConfig
from alpa_serve.util import GB
# 我想导入同一个文件夹下utils.py中的函数，该怎么做？
from util import smooth_list

def plot_inspect_group_memory(single_model_dir_path, plot_save_path, prof_database=None, model_placement=None, 
                              model_types=None, model_names=None, duration=3600):
    # 1. 获取每个group中的模型名称、并行方式
    # 2. 读取每个模型的内存使用情况
    if prof_database is None:
        prof_database = ProfilingDatabase("/home/zy/python_project/mms/alpa_serve/syn_profiling_result.pkl")

    # 获取文件夹中的所有文件
    files = os.listdir(single_model_dir_path)
    all_group_act_mem = []
    
    for group_id in range(len(model_placement.group_models)):
        group_models = model_placement.group_models[group_id]
        group_parallel_config = model_placement.group_configs[group_id]
        # 长度为duration的列表，记录每个时间点的内存使用情况
        group_act_mem = [0] * duration

        for model_id in group_models:
            model_name = model_names[model_id]
            model_type = model_types[model_id]
            model_act_mem = sum(prof_database.get(model_type).para_dict[group_parallel_config].act_mem[1])

            # 读取每个模型的运行情况
            model_is_running = []
            full_path = os.path.join(single_model_dir_path, model_name + '.tsv')
            with open(full_path, 'r') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    model_is_running.append(line.split('\t')[1].strip().lower() == 'true')
            model_is_running = [int(x) for x in model_is_running]

            # 计算每个时间点的内存使用情况
            for i in range(duration):
                if model_is_running[i]:
                    group_act_mem[i] += model_act_mem * 1.0 / GB
        all_group_act_mem.append(group_act_mem)
    
    for group_id in range(len(model_placement.group_models)):
        all_group_act_mem[group_id] = smooth_list(all_group_act_mem[group_id], interval=36)
        
    
    # 画图
    plt.figure(figsize=(12, 6))
    for group_id in range(len(model_placement.group_models)):
        plt.plot(all_group_act_mem[group_id], label=f"group_{group_id}")
        print(f"group_{group_id} memory usage: max={max(all_group_act_mem[group_id])}, min={min(all_group_act_mem[group_id])}, mean={sum(all_group_act_mem[group_id]) / duration}")
    plt.xlabel('Time(s)')
    plt.ylabel('Memory Usage(GB)')
    plt.title('Memory Usage of Each Group')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_save_path + '/every_group_memory_usage.png')
    plt.close()
    print('图表已保存至', plot_save_path + '/every_group_memory_usage.png')

def plot_inspect_cluster_memory(single_model_dir_path, plot_save_path, interval=1):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--single_model_dir_path", type=str, help="整个集群模型运行信息的文件夹路径",
                        default="monitor/monitor_exp_busiest_device/res_monitor_general_model_cases_single_model_dir")
    parser.add_argument("--plot_cluster_save_path", type=str, help="输出集群的图片保存路径",
                        default="monitor/monitor_exp_busiest_device")
    args = parser.parse_args()
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

    
    plot_inspect_group_memory(args.single_model_dir_path, args.plot_cluster_save_path, model_placement=model_placement,
                              model_types=model_types, model_names=model_names)
    
