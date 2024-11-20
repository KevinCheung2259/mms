from autots import AutoTS
import pandas as pd
import numpy as np

# 生成伪造数据
date_range = pd.date_range(start='2022-01-01', periods=200, freq='H')  # 生成100小时的时间戳
# requests为1-200
requests = np.arange(1, 201)

data = pd.DataFrame({'timestamp': date_range, 'requests': requests})  # 创建数据框
data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)

# 加载数据
# data = pd.read_csv('data.csv')
# data.columns = ['timestamp', 'requests']
# data['timestamp'] = pd.to_datetime(data['timestamp'])
# data.set_index('timestamp', inplace=True)

# 初始化 AutoTS 模型，仅使用 ARIMA
model = AutoTS(
    forecast_length=5,             # 设置预测的长度为 60 个时间单位
    frequency='infer',              # 自动检测数据频率
    ensemble=None,                  # 不使用集成模型
    model_list=['ARIMA'],           # 指定只使用 ARIMA 模型
    transformer_list=None,        # 不使用数据转换器
    verbose=False                   # 关闭输出
)

# 训练模型
model = model.fit(data)

# 生成预测
prediction = model.predict()
forecast = prediction.forecast

# 输出结果
print("预测结果：")
print(forecast)