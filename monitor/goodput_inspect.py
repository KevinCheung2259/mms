import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def read_file(file_path):
    '''
    读取文件，返回mixed_single_model_dict
    '''
    # 读取文件
    mixed_single_model_dict = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            model_name = line.split('\t')[0]
            num_requests = line.split('\t')[1]
            goodput = line.split('\t')[2]
            throughput = line.split('\t')[3]
            avg_latency = line.split('\t')[4]
            latency_std = line.split('\t')[5]
            latency_p90 = line.split('\t')[6]
            latency_p99 = line.split('\t')[7]
            if model_name not in mixed_single_model_dict:
                mixed_single_model_dict[model_name] = {}
            mixed_single_model_dict[model_name]["num_requests"] = num_requests
            mixed_single_model_dict[model_name]["goodput"] = goodput
            mixed_single_model_dict[model_name]["throughput"] = throughput
            mixed_single_model_dict[model_name]["avg_latency"] = avg_latency
            mixed_single_model_dict[model_name]["latency_std"] = latency_std
            mixed_single_model_dict[model_name]["latency_p90"] = latency_p90
            mixed_single_model_dict[model_name]["latency_p99"] = latency_p99
    return mixed_single_model_dict

def plot_inspect_single_model_goodput(mixed_single_model_dict, plot_save_path):
    # 提取模型名称和goodput，并排序
    model_names = list(mixed_single_model_dict.keys())
    goodputs = [float(mixed_single_model_dict[model]["goodput"]) for model in model_names]
    # 对goodputs进行排序
    goodputs_sorted = sorted(goodputs, reverse=True)
    model_names_sorted = [model_names[i] for i in np.argsort(goodputs)]

    # 创建柱状图
    plt.figure(figsize=(12, 6))
    plt.bar(model_names_sorted, goodputs_sorted, color='skyblue')
    plt.xlabel('Model Name')
    plt.ylabel('Goodput')
    plt.title('Goodput of Each Model')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(plot_save_path + '/every_single_model_goodput.png')
    plt.close()
    print('图表已保存至', plot_save_path + '/every_single_model_goodput.png')

def plot_models_goodput_vs_arrival_rate(mixed_single_model_dict, plot_save_path):
    model_names = list(mixed_single_model_dict.keys())
    goodputs = [float(mixed_single_model_dict[model]["goodput"]) for model in model_names]
    arrival_rates = [int(mixed_single_model_dict[model]["num_requests"]) / 3600 for model in model_names]
    # 创建散点图
    plt.figure(figsize=(12, 6))
    plt.scatter(arrival_rates, goodputs, color='orange')
    plt.xlabel('Arrival Rate')
    plt.ylabel('Goodput')
    plt.title('Goodput and Arrival Rate of Each Model')
    plt.savefig(plot_save_path + '/models_goodput_vs_arrival_rate.png')
    plt.close()
    print('图表已保存至', plot_save_path + '/models_goodput_vs_arrival_rate.png')

def plot_model_types_goodput_vs_arrival_rate(mixed_single_model_dict, plot_save_path):
    # 列出所有的模型名字
    model_names = list(mixed_single_model_dict.keys())
    # 在第二个"-"截断
    model_names = ['-'.join(mn.split('-')[:2]) for mn in model_names]
    unique_model_catagory = list(set(model_names))
    # 提取每个模型名字里的数字为模型大小
    model_sizes = {}
    for model_name in unique_model_catagory:
        model_sizes[model_name] = float(model_name.split('-')[-1][:-1])

    # 统计每个类别的模型数量、goodput、arrival_rate
    model_catagory_count = {}
    model_catagory_goodput = {}
    model_catagory_arrival_rate = {}
    for model_catagory in unique_model_catagory:
        model_catagory_count[model_catagory] = model_names.count(model_catagory)
        model_catagory_goodput[model_catagory] = 0
        model_catagory_arrival_rate[model_catagory] = 0
    for model_name in mixed_single_model_dict:
        model_catagory = '-'.join(model_name.split('-')[:2])
        model_catagory_goodput[model_catagory] += float(mixed_single_model_dict[model_name]["goodput"])
        model_catagory_arrival_rate[model_catagory] += int(mixed_single_model_dict[model_name]["num_requests"]) / 3600

    # 计算每个类别的goodput和arrival_rate
    for model_catagory in unique_model_catagory:
        model_catagory_goodput[model_catagory] /= model_catagory_count[model_catagory]
        model_catagory_arrival_rate[model_catagory] /= model_catagory_count[model_catagory]
    
    # 创建散点图,每个点旁边标明模型类别
    plt.figure(figsize=(12, 6))
    for model_catagory in unique_model_catagory:
        plt.scatter(model_catagory_arrival_rate[model_catagory], model_catagory_goodput[model_catagory], label=model_catagory, s=model_sizes[model_catagory] * 100)
        plt.text(model_catagory_arrival_rate[model_catagory], model_catagory_goodput[model_catagory], model_catagory)
    plt.xlabel('Arrival Rate (requests/s)')
    plt.ylabel('Goodput')
    plt.title('Goodput and Arrival Rate of Each Model Catagory')
    plt.legend()
    plt.savefig(plot_save_path + '/model_types_goodput_vs_arrival_rate.png')
    plt.close()
    print('图表已保存至', plot_save_path + '/model_types_goodput_vs_arrival_rate.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--every_sing_model_data_path", type=str, help="每个模型运行信息的文件路径",
                        default="monitor/monitor_exp_load_balance/res_monitor_general_model_cases_single_model.tsv")
    parser.add_argument("--plot_save_path", type=str, help="输出图片保存路径",
                        default="monitor/monitor_exp_load_balance")
    args = parser.parse_args()
    # 若没有plot_save_path，则创建
    if not os.path.exists(args.plot_save_path):
        os.makedirs(args.plot_save_path)
    # plot_inspect_single_model_request(args.single_model_dir_path, args.plot_save_path)
    mixed_single_model_dict = read_file(args.every_sing_model_data_path)
    plot_inspect_single_model_goodput(mixed_single_model_dict, args.plot_save_path)
    plot_models_goodput_vs_arrival_rate(mixed_single_model_dict, args.plot_save_path)
    plot_model_types_goodput_vs_arrival_rate(mixed_single_model_dict, args.plot_save_path)
