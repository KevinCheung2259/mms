from alpa_serve.trace import Trace, TraceReplay
from typing import List, Dict
import matplotlib.pyplot as plt
import os
import numpy as np

# 对列表进行平均值平滑
def smooth_list(l, interval=5):
    """
    对列表进行平均值平滑，且减少列表元素。
    
    参数:
    l: 输入的列表
    interval: 平滑间隔 (默认值 5)
    
    返回:
    平滑后的列表
    """
    if interval == 1:
        return l
    return [sum(l[i:i+interval]) / interval for i in range(0, len(l), interval)]


def cal_num_requests_per_interval(replays: Dict[str, TraceReplay],
                                  model_name: str,
                                  duration: int,
                                  interval: int) -> List[int]:
    '''
    计算每个时间段内的请求数
    '''
    num_intervals = int(duration // interval)
    num_requests = [0] * num_intervals
    arrival_process = replays[model_name]
    for arrival_time in arrival_process.arrivals:
        interval_index = int((arrival_time - arrival_process.start_seconds) // interval)
        num_requests[interval_index] += 1
    return num_requests


def plot_model_traces(replays: Dict[str, TraceReplay], model_names: List[str], duration: int, interval: int):
    '''
    绘制模型的请求数折线图
    '''
    # 创建子图布局，每行显示 2 个子图
    model_names = np.sort(model_names)
    num_models = len(model_names)
    fig, axs = plt.subplots(nrows=(num_models + 1) // 2, ncols=2, figsize=(14, 10))
    axs = axs.flatten()  # 将子图数组展平，方便迭代访问

    # 生成并绘制每个模型的请求数折线图
    for i, model_name in enumerate(model_names):
        # 假设每个模型的请求数据通过函数 cal_num_requests_per_interval 计算得到
        model_requests = cal_num_requests_per_interval(replays, model_name, duration, interval)
        
        # 绘制到对应子图
        axs[i].plot(model_requests, label=f"{model_name} requests")
        axs[i].set_title(f"{model_name} requests per minute")
        axs[i].set_ylabel("#requests")
        axs[i].set_xlabel("time (minute)")
        axs[i].legend()

    # 移除多余的空白子图
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # 调整布局并显示图表
    plt.tight_layout()

    # 获取当前文件的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # 保存图表到为pdf
    plt.savefig(os.path.join(current_path, "model_requests.pdf"))


def plot_all_models_requests(replays: Dict[str, TraceReplay], model_names: List[str], duration: int, interval: int):
    '''
    汇总并绘制所有模型的请求数折线图
    '''
    # 初始化一个空的列表，用于存储每个模型的请求数
    total_requests = np.zeros(duration // interval)  # 假设每个间隔的请求数初始化为 0

    # 对每个模型进行处理，计算其请求数并累加
    for model_name in model_names:
        # 获取每个模型的请求数
        model_requests = cal_num_requests_per_interval(replays, model_name, duration, interval)
        
        # 将每个模型的请求数累加到 total_requests 中
        total_requests += np.array(model_requests)
    
    arrival_rate = total_requests / interval

    # 以列表形式打印arrival_rate，5个元素为一行
    # for i in range(0, len(arrival_rate), 5):
    #     print(arrival_rate[i:i+5])     

    # 绘制所有模型的总请求数折线图
    plt.figure(figsize=(4, 3))
    plt.plot(arrival_rate, label="Total requests", color="tab:purple")

    # 设置图表标题和标签
    # plt.title("Total Requests for All Models", fontsize=16)
    plt.xlabel("Time (minutes)", fontsize=14)
    plt.ylabel("Arrival Rate", fontsize=14)
    # plt.legend()

    # 调整布局并显示图表
    plt.tight_layout()

    # 获取当前文件的路径
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    # 保存图表为 PDF 文件
    plt.savefig(os.path.join(current_path, "all_models_requests_rate.pdf"))

    # 显示图表
    plt.show()
