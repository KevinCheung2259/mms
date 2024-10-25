import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import matplotlib.pyplot as plt
from util import smooth_list

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

def plot_inspect_single_model_request(single_model_dir_path, plot_save_path):
    # 获取文件夹中的所有文件
    files = os.listdir(single_model_dir_path)

    for file_name in files:
        if not file_name.endswith('.tsv'):
            continue
        # 模型名称为文件名
        model_name = file_name[:-4]
        model_is_running = []
        model_received_requests = []
        model_returned_requests = []
        model_dropped_requests = []

        # 读取文件每一行的信息到model_is_running, model_received_requests, model_returned_requests, model_dropped_requests
        full_path = os.path.join(single_model_dir_path, file_name)
        print(full_path)
        with open(full_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                model_is_running.append(line.split('\t')[1].strip().lower() == 'true')
                model_received_requests.append(int(line.split('\t')[2]))
                model_returned_requests.append(int(line.split('\t')[3]))
                model_dropped_requests.append(int(line.split('\t')[4]))

        # 将model_is_running, model_received_requests, model_returned_requests, model_dropped_requests转换为整数
        model_is_running = [int(x) for x in model_is_running]
        model_received_requests = [int(x) for x in model_received_requests]
        model_returned_requests = [int(x) for x in model_returned_requests]
        model_dropped_requests = [int(x) for x in model_dropped_requests]

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

def plot_inspect_cluster_request(single_model_dir_path, plot_save_path, interval=1):
    # 获取文件夹中的所有文件
    files = os.listdir(single_model_dir_path)

    # 记录集群在整个过程中的请求数、返回请求数、丢弃请求数、模型是否在运行
    total_received_requests = []
    total_returned_requests = []
    total_dropped_requests = []
    total_is_running = []
    model_num = 0

    for file_name in files:
        if not file_name.endswith('.tsv'):
            continue
        # 模型名称为文件名
        model_num += 1
        model_is_running = []
        model_received_requests = []
        model_returned_requests = []
        model_dropped_requests = []

        # 读取文件每一行的信息到model_is_running, model_received_requests, model_returned_requests, model_dropped_requests
        full_path = os.path.join(single_model_dir_path, file_name)
        print(full_path)
        with open(full_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                model_is_running.append(line.split('\t')[1].strip().lower() == 'true')
                model_received_requests.append(int(line.split('\t')[2]))
                model_returned_requests.append(int(line.split('\t')[3]))
                model_dropped_requests.append(int(line.split('\t')[4]))
        
        model_is_running = [int(x) for x in model_is_running]
        total_is_running.append(model_is_running)
        total_received_requests.append(model_received_requests)
        total_returned_requests.append(model_returned_requests)
        total_dropped_requests.append(model_dropped_requests)

    # 将model_is_running等沿着dim=1的方向求和
    total_is_running = [sum(x) for x in zip(*total_is_running)]
    total_received_requests = [sum(x) for x in zip(*total_received_requests)]
    total_returned_requests = [sum(x) for x in zip(*total_returned_requests)]
    total_dropped_requests = [sum(x) for x in zip(*total_dropped_requests)]

    if interval > 1:
        total_received_requests = smooth_list(total_received_requests, interval=interval)
        total_returned_requests = smooth_list(total_returned_requests, interval=interval)
        total_dropped_requests = smooth_list(total_dropped_requests, interval=interval)
        total_is_running = smooth_list(total_is_running, interval=interval)

    draw_plot(total_received_requests, total_returned_requests, total_dropped_requests, 
            labels=['total_received_requests', 'total_returned_requests', 'total_dropped_requests'],
            colors=['orange', 'green', 'red'], xlabel='Time(s)', ylabel='Requests', title='Cluster Requests', 
            save_path=plot_save_path + '/cluster_requests.png')
    
    total_is_running = [int(x / model_num * 100) for x in total_is_running]
    draw_plot(total_is_running, labels=['total_is_running'], colors=['blue'], xlabel='Time(s)', ylabel='Percentage(%)', 
            ylim=(0, 100), title='Cluster Running Percentage',
            save_path=plot_save_path + '/cluster_running_percentage.png')

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
    plot_inspect_single_model_request(args.single_model_dir_path, args.plot_single_save_path)
    plot_inspect_cluster_request(args.single_model_dir_path, args.plot_cluster_save_path)
