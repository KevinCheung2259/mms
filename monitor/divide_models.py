from alpa_serve.trace import Trace, TraceReplay
from typing import List, Dict
import multiprocessing
import time
from util import cal_num_requests_per_interval

# 计算两个模型之间请求的相似度
def cal_similarity(model_1: str,
                   model_2: str,
                   duration: int,
                   replays: Dict[str, TraceReplay]) -> float:
    '''
    计算两个模型之间请求的相关性
    '''
    interval_seconds = replays[model_1].interval_seconds
    model_1_requests = cal_num_requests_per_interval(replays, model_1, duration, interval_seconds)
    model_2_requests = cal_num_requests_per_interval(replays, model_2, duration, interval_seconds)
    num_intervals = len(model_1_requests)
    similarity = 0
    for i in range(num_intervals):
        similarity += model_1_requests[i] * model_2_requests[i]
    return similarity / num_intervals

def solve_model_grouping_ILP(model_names: List[str], group_num: int, 
                             similarity_matrix: List[List[float]]) -> List[List[str]]:
    
    import pulp
    from pulp import LpVariable, LpProblem, LpMaximize, lpSum, LpStatus, LpMinimize, value
    tic = time.time()

    model_groups = [[] for _ in range(group_num)]
    num_models = len(model_names)

    # 定义决策变量
    x = LpVariable.matrix("x", (range(num_models), range(group_num)), cat="Binary")
    z = LpVariable.dicts("z", (range(num_models), range(num_models), range(group_num)), cat="Binary")

    # 定义目标函数
    prob = LpProblem("Model_Grouping", LpMinimize)
    # 目标：最小化组中最大相似度
    max_similarity = LpVariable("max_similarity")
    prob += max_similarity

    # 约束条件
    for i in range(num_models):
        prob += lpSum(x[i][j] for j in range(group_num)) == 1  # 每个模型只能分到一个组
    
    for i in range(num_models):
        for k in range(num_models):
            for j in range(group_num):
                prob += z[i][k][j] <= x[i][j]
                prob += z[i][k][j] <= x[k][j]
                prob += z[i][k][j] >= x[i][j] + x[k][j] - 1

    for j in range(group_num):
        # 约束：每组至少有一个模型
        prob += lpSum(x[i][j] for i in range(num_models)) >= 1  

        # 引入中间变量来计算该组的相似度
        group_similarity_sum = LpVariable(f"group_similarity_sum_{j}")

        # 计算该组的相似度总和
        prob += group_similarity_sum == lpSum(similarity_matrix[i][k] * z[i][k][j] 
                                              for i in range(num_models) for k in range(num_models))

        # 约束：该组相似度总和不能超过最大相似度
        prob += group_similarity_sum <= max_similarity

    # 求解
    assert "PULP_CBC_CMD" in pulp.listSolvers(onlyAvailable=True), (
            "Please install ILP solvers by 'sudo apt install coinor-cbc'")
    solver = pulp.PULP_CBC_CMD(mip=True, msg=False, threads=multiprocessing.cpu_count())
    prob.solve(solver)

    status = prob.status
    objective = pulp.value(prob.objective)
    objective = float(objective) if objective is not None else -1.0
    print(f"Model Groups ILP Status: {LpStatus[status]}\tObjective: {objective}")
    print(f"Model Groups ILP Solver Time:  {time.time() - tic:.2f}s")
        
    if prob.status in [pulp.LpStatusInfeasible]:
        raise RuntimeError("Cannot run the function under the given memory budget. Please increase the memory budget.")
    
    for i in range(num_models):
        for j in range(group_num):
            if value(x[i][j]) == 1:
                model_groups[j].append(model_names[i])
    
    return model_groups

def divide_models(model_names: List[str],
                  duration: int,
                  replays: Dict[str, TraceReplay],
                  group_num: int) -> List[List[str]]:
    '''
    根据模型请求到来的模型，将模型分组
    '''
    assert group_num <= len(model_names), 'The number of groups should be less than the number of models'
    num_models = len(model_names)
    similarity_matrix = [[0] * num_models for _ in range(num_models)]
    for i in range(num_models):
        for j in range(num_models):
            similarity_matrix[i][j] = cal_similarity(model_names[i], model_names[j], duration, replays)

    model_groups = solve_model_grouping_ILP(model_names, group_num, similarity_matrix)
    print("Model Groups:")
    for i, group in enumerate(model_groups):
        print(f"Group {i}: {group}")

    # # 打印出每个组的相似度值
    # for i in range(group_num):
    #     similarity_sum = 0
    #     for model_1 in model_groups[i]:
    #         for model_2 in model_groups[i]:
    #             similarity_sum += cal_similarity(model_1, model_2, duration, replays)
    #     print(f"Group {i} similarity sum: {similarity_sum}")

    return model_groups