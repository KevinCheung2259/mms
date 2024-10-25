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
    return [sum(l[i:i+interval]) / interval for i in range(0, len(l), interval)]