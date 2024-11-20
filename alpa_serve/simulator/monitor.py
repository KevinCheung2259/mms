import numpy as np

class Monitor:
    def __init__(self, placement, model_names, prof_ress):
        self.placement = placement
        self.model_names = model_names
        self.prof_ress = prof_ress
        self.high_threshold = 0.3
        self.low_threshold = 0.1

    def compute_capability(self, modeL_prof_ress, parallel_config, max_bs=1):
        slo_scale = 5
        slo = slo_scale * sum(modeL_prof_ress.para_dict[(1, 1, 1)].latency[1])
        latency_mem = modeL_prof_ress.para_dict.get(parallel_config, None)

        if latency_mem is None:
            return 0

        num_stages = parallel_config.pp
        max_cap = 0
        for b, ls in latency_mem.latency.items():
            if b > max_bs:
                continue

            # slo = sum(ls) + (n-1) * max(ls)
            # so, n = ceil((slo - sum(ls)) / max(ls)) + 1
            max_cap = max(max_cap, (slo - sum(ls)) // max(ls) + 1)

        return max_cap * (0.99 ** num_stages)

    def analyse_model_requests_rate_goodput(self, tstamps, num_models, model_ids, good, window_size, i):
        """
        分析在第i个请求时刻前window_size个请求内，每个模型的请求率
        """
        model_requests = [0] * num_models
        model_goodput = [0] * num_models
        duration = tstamps[i] - tstamps[i - window_size]

        for j in range(i - window_size, i):
            if j < 0:
                continue
            model_requests[model_ids[j]] += 1
            model_goodput[model_ids[j]] += good[j]

        model_requests_rate = [0] * num_models
        model_goodput_rate = [0] * num_models

        for j in range(num_models):
            model_requests_rate[j] = model_requests[j] / duration
            model_goodput_rate[j] = model_goodput[j] / (model_requests[j] + 1e-6)

        return model_requests_rate, model_goodput_rate


    def analyse_model_capability(self):
        """
        分析每个模型的吞吐能力
        """
        num_models = len(self.model_names)
        model_capability = [0] * num_models
        model_replica_num = [0] * num_models

        group_configs, group_models = self.placement.group_configs, self.placement.group_models

        for group_config, group_model in zip(group_configs, group_models):
            for model_id in group_model:
                model_capability[model_id] += self.compute_capability(self.prof_ress[model_id], group_config)
                model_replica_num[model_id] += 1

        self.model_capability = model_capability
        self.model_replica_num = model_replica_num


    def decide_whether_to_scale(self, tstamps, num_models, model_ids, good, window_size, i):
        """
        判断是否需要扩容或缩容
        """
        self.model_requests_rate, self.model_goodput_rate = self.analyse_model_requests_rate_goodput(
            tstamps, num_models, model_ids, good, window_size, i)
        
        # 归一化请求率和吞吐能力
        self.model_requests_rate = [rate / max(self.model_requests_rate) for rate in self.model_requests_rate]
        self.model_capability = [cap / max(self.model_capability) for cap in self.model_capability]
        
        # 计算每个模型的负载情况
        self.model_load = [(self.model_requests_rate[i] / (self.model_capability[i] + 1e-6)) * (1 - self.model_goodput_rate[i]) for i in range(num_models)]

        # 计算扩容需求,优先扩容负载高的模型
        self.scale_up_model = sorted([model for model in range(num_models) if self.model_load[model] > self.high_threshold], key=lambda x: self.model_load[x], reverse=True)
        group_num = len(self.placement.group_models)
        # 删除副本数大于等于group_num的模型
        for model in self.scale_up_model:
            if self.model_replica_num[model] >= group_num:
                self.scale_up_model.remove(model)
                print("[Monitor]: Model {} need more GPU to scale up!".format(model))

        # 计算缩容需求,优先缩容副本数最多的模型, 副本数相同的情况下，优先缩容负载最低的模型
        self.scale_down_model = sorted(
            [(model, self.model_replica_num[model], self.model_load[model]) for model in range(num_models) if self.model_load[model] < self.low_threshold],
            key=lambda x: (-x[1], x[2])  # 按副本数降序排列，副本数相同的情况下按负载升序排列
        )
        # 去掉副本数为1的模型
        self.scale_down_model = [model for model, num, _ in self.scale_down_model if num > 1]   
        if len(self.scale_up_model) > 0 and len(self.scale_down_model) > 0:
            return True
        else:
            return False
        

    def find_outliers(self, data, threshold=3):
        """
        使用 IQR 方法找到列表中的异常值索引。
        
        参数:
        data (list): 输入的数字列表。
        
        返回:
        list: 异常值的索引列表。
        """
        # 计算 Q1 和 Q3
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1

        # 找到异常值的边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 找到异常值的索引
        outlier_indices = [i for i in range(len(data)) if data[i] < lower_bound or data[i] > upper_bound]

        return outlier_indices


    def cal_model_memory_usage(self, model_id, model_config):
        """
        计算模型的内存使用量
        """
        return sum(self.prof_ress[model_id].para_dict[model_config].memory[1])

        
        
        


