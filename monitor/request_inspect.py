import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib.pyplot as plt
from util import smooth_list
import numpy as np
import pandas as pd

def draw_plot(*lists, xlabel='X-Axis', ylabel='Y-Axis', colors=None, 
              title='Line Plot', labels=None, xlim=None, ylim=None, save_path=None):
    """
    绘制折线图。
    
    参数:
    *lists: 可变数量的列表，每个列表代表一条折线
    xlabel: X轴的标签 (默认值 'X-Axis')
    ylabel: Y轴的标签 (默认值 'Y-Axis')
    title: 图表标题 (默认值 'Line Plot')
    labels: 每条折线的名称 (列表类型，默认为 None)
    xlim: x轴的范围，元组类型 (默认值 None)
    ylim: y轴的范围，元组类型 (默认值 None)
    save_path: 如果提供路径，将图表保存到指定文件路径 (默认值 None)

    示例调用
    draw_plot([1, 2, 3, 4], [4, 3, 2, 1], labels=['Dataset 1', 'Dataset 2'], xlabel='X-Axis', ylabel='Y-Axis', 
    title='Custom Line Plot', xlim=(0, 5), ylim=(0, 5))
    """
    
    # 使用Seaborn的风格
    sns.set(style="whitegrid")  # 设置背景样式为白色网格
    plt.figure(figsize=(10, 6), dpi=100)  # 设置图表大小和分辨率
    
    # 如果没有传递labels，则使用默认的Line 1, Line 2...
    if labels is None:
        labels = [f'Line {i+1}' for i in range(len(lists))]
    
    # 绘制每一条折线
    for i, data in enumerate(lists):
        plt.plot(data, label=labels[i], color=colors[i] if colors else None, linewidth=2.0, marker='o', markersize=6)
    
    # 设置标题和轴标签，字体大小
    plt.title(title, fontsize=18, pad=15)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    
    # 设置x轴和y轴的范围
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    
    # 显示图例，图例字体大小，位置设置为图表外侧右上角
    plt.legend(loc='upper left', fontsize=12)
    
    # 设置刻度大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # 调整子图周围的空白
    plt.tight_layout()

    # 保存图像 (如果指定路径)
    if save_path:
        plt.savefig(save_path, dpi=300)  # 以高分辨率保存
        print(f"图表已保存至 {save_path}")
    
    # 显示图表
    plt.show()

def plot_inspect_single_model(single_model_dir_path, plot_save_path, target):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(single_model_dir_path) if f.endswith('.tsv')]
    model_names = [f[:-4] for f in files]
    model_names = np.sort(model_names)
    num_models = len(files)
    fig, axs = plt.subplots(nrows=(num_models + 1) // 2, ncols=2, figsize=(14, 10))
    axs = axs.flatten()  # 将子图数组展平，方便迭代访问

    for i in range(len(model_names)):
        # 模型名称为文件名
        model_name = model_names[i]
        file_name = model_name + '.tsv'

        full_path = os.path.join(single_model_dir_path, file_name)
        df = pd.read_csv(full_path, sep='\t')

        model_target = list(df[target])
        # 对于model_queue，每60个数据点求和
        if target in ['model_queue', 'model_returned_requests']:
            model_target = [sum(model_target[i:i+60]) for i in range(0, len(model_target), 60)]

        # 绘制到对应子图
        axs[i].plot(model_target, label=f"{model_name} {target}")
        axs[i].set_title(f"{model_name} {target}")
        # axs[i].set_ylabel(f"#{target}")
        axs[i].set_xlabel("time")
        # axs[i].legend()

    # 移除多余的空白子图
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # 调整布局并显示图表
    plt.tight_layout()
    
    # 保存图表到为pdf
    plt.savefig(plot_save_path + '/' + target + '.pdf')
    print(f"图表已保存至 {plot_save_path}/{target}.pdf")
    plt.close()

def plot_inspect_single_model_request(single_model_dir_path, plot_save_path):
    # 获取文件夹中的所有文件
    files = [f for f in os.listdir(single_model_dir_path) if f.endswith('.tsv')]

    for file_name in files:
        # 模型名称为文件名
        model_name = file_name[:-4]
        model_received_requests = []
        model_returned_requests = []
        model_dropped_requests = []

        full_path = os.path.join(single_model_dir_path, file_name)
        df = pd.read_csv(full_path, sep='\t')

        model_received_requests = list(df['model_received_requests'])
        model_returned_requests = list(df['model_returned_requests'])
        model_dropped_requests = list(df['model_dropped_requests'])

        # 画出model_is_running, model_received_requests, model_returned_requests, model_dropped_requests的折线图
        plt.figure(figsize=(12, 6))
        plt.plot(model_received_requests, label='model_received_requests', color='orange')
        plt.plot(model_returned_requests, label='model_returned_requests', color='green')
        plt.plot(model_dropped_requests, label='model_dropped_requests', color='red')
        # 横坐标是时间，从0到3600，纵坐标是请求数
        plt.xlabel('Time(s)')
        plt.ylabel('Requests')

        plt.title(model_name)
        plt.legend()
        plt.savefig(plot_save_path + '/' + model_name + '.png')
        print(f"图表已保存至 {plot_save_path}/{model_name}.png")
        plt.close()

# def plot_inspect_single_model_returned_requests(single_model_dir_path, plot_save_path):
#     # 获取文件夹中的所有文件
#     files = [f for f in os.listdir(single_model_dir_path) if f.endswith('.tsv')]
#     model_names = [f[:-4] for f in files]
#     model_names = np.sort(model_names)
#     num_models = len(files)
#     fig, axs = plt.subplots(nrows=(num_models + 1) // 2, ncols=2, figsize=(14, 10))
#     axs = axs.flatten()  # 将子图数组展平，方便迭代访问

#     for i in range(len(model_names)):
#         # 模型名称为文件名
#         model_name = model_names[i]
#         file_name = model_name + '.tsv'

#         full_path = os.path.join(single_model_dir_path, file_name)
#         df = pd.read_csv(full_path, sep='\t')

#         model_queue = list(df['model_returned_requests'])
#         # 对于model_queue，每60个数据点求和
#         new_model_queue = [sum(model_queue[i:i+60]) for i in range(0, len(model_queue), 60)]

#         # 绘制到对应子图
#         axs[i].plot(new_model_queue, label=f"{model_name} returned requests")
#         axs[i].set_title(f"{model_name} returned requests per minute")
#         axs[i].set_ylabel("#returned requests")
#         axs[i].set_xlabel("time (minute)")
#         axs[i].legend()

#     # 移除多余的空白子图
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])

#     # 调整布局并显示图表
#     plt.tight_layout()
    
#     # 保存图表到为pdf
#     plt.savefig(plot_save_path + '/' + 'model_returned_requests.pdf')
#     print(f"图表已保存至 {plot_save_path}/model_returned_requests.pdf")
#     plt.close()

# def plot_inspect_single_model_queue(single_model_dir_path, plot_save_path):
#     # 获取文件夹中的所有文件
#     files = [f for f in os.listdir(single_model_dir_path) if f.endswith('.tsv')]
#     model_names = [f[:-4] for f in files]
#     model_names = np.sort(model_names)
#     num_models = len(files)
#     fig, axs = plt.subplots(nrows=(num_models + 1) // 2, ncols=2, figsize=(14, 10))
#     axs = axs.flatten()  # 将子图数组展平，方便迭代访问

#     for i in range(len(model_names)):
#         # 模型名称为文件名
#         model_name = model_names[i]
#         file_name = model_name + '.tsv'

#         full_path = os.path.join(single_model_dir_path, file_name)
#         df = pd.read_csv(full_path, sep='\t')

#         model_queue = list(df['model_queue'])
#         # 对于model_queue，每60个数据点求和
#         new_model_queue = [sum(model_queue[i:i+60]) for i in range(0, len(model_queue), 60)]

#         # 绘制到对应子图
#         axs[i].plot(new_model_queue, label=f"{model_name} queue")
#         axs[i].set_title(f"{model_name} queue per minute")
#         axs[i].set_ylabel("#requests")
#         axs[i].set_xlabel("time (minute)")
#         axs[i].legend()

#     # 移除多余的空白子图
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])

#     # 调整布局并显示图表
#     plt.tight_layout()
    
#     # 保存图表到为pdf
#     plt.savefig(plot_save_path + '/' + 'model_queue.pdf')
#     print(f"图表已保存至 {plot_save_path}/model_queue.pdf")
#     plt.close()


def plot_inspect_cluster_request(single_model_dir_path, plot_save_path, interval=1):
    # 获取文件夹中的所有 TSV 文件
    files = [f for f in os.listdir(single_model_dir_path) if f.endswith('.tsv')]

    total_received_requests, total_returned_requests, total_dropped_requests, total_model_queue = [], [], [], []

    for file_name in files:
        # 读取 TSV 文件
        full_path = os.path.join(single_model_dir_path, file_name)
        
        # 读取 TSV 文件内容到 DataFrame
        df = pd.read_csv(full_path, sep='\t')
        total_received_requests.append(list(df['model_received_requests']))
        total_returned_requests.append(list(df['model_returned_requests']))
        total_dropped_requests.append(list(df['model_dropped_requests']))
        total_model_queue.append(list(df['model_queue']))
    
    # 计算集群的总请求数、返回请求数、丢弃请求数、队列长度
    total_received_requests = [sum(x) for x in zip(*total_received_requests)]
    total_returned_requests = [sum(x) for x in zip(*total_returned_requests)]
    total_dropped_requests = [sum(x) for x in zip(*total_dropped_requests)]
    total_model_queue = [sum(x) for x in zip(*total_model_queue)]

    # 处理平滑数据
    total_received_requests = smooth_list(total_received_requests, interval=interval)
    total_returned_requests = smooth_list(total_returned_requests, interval=interval)
    total_dropped_requests = smooth_list(total_dropped_requests, interval=interval)
    total_model_queue = smooth_list(total_model_queue, interval=interval)

    # 绘制请求数折线图
    draw_plot(
        total_received_requests, total_returned_requests, total_dropped_requests,
        labels=['total_received_requests', 'total_returned_requests', 'total_dropped_requests'],
        colors=['orange', 'green', 'red'], xlabel='Time(s)', ylabel='Requests', title='Cluster Requests',
        save_path=os.path.join(plot_save_path, 'cluster_requests.png')
    )

    # 绘制集群队列长度折线图
    draw_plot(
        total_model_queue, labels=['total_model_queue'], colors=['blue'],
        xlabel='Time(s)', ylabel='Queue Length', title='Cluster Queue Length',
        save_path=os.path.join(plot_save_path, 'cluster_queue.png')
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser() 
    parser.add_argument("--single_model_dir_path", type=str, help="整个集群模型运行信息的文件夹路径",
                        default="monitor/monitor_exp_instance_busiest_device/res_monitor_general_model_cases_single_model_dir")
    parser.add_argument("--plot_single_save_path", type=str, help="输出单个模型的图片保存路径",
                        default="monitor/monitor_exp_instance_busiest_device/plot_res_monitor_general_model_cases_single_model")
    parser.add_argument("--plot_cluster_save_path", type=str, help="输出集群的图片保存路径",
                        default="monitor/monitor_exp_instance_busiest_device")
    args = parser.parse_args()
    # 若没有plot_save_path，则创建
    if not os.path.exists(args.plot_single_save_path):
        os.makedirs(args.plot_single_save_path)
    # plot_inspect_single_model_request(args.single_model_dir_path, args.plot_single_save_path)
    plot_inspect_cluster_request(args.single_model_dir_path, args.plot_cluster_save_path)
