import re
import numpy as np

def extract_ebitda_prediction(response):
    """
    从模型响应中提取EBITDA预测值
    
    参数:
        response (str): 模型的完整响应文本
        
    返回:
        float: 提取的EBITDA预测值，如果无法提取则返回None
    """
    # 尝试从<answer>标签中提取
    answer_match = re.search(r'<answer>(.*?)</answer>', response)
    if answer_match:
        try:
            return float(answer_match.group(1))
        except (ValueError, TypeError):
            pass
    
    return None

def compute_format_score(response):
    """
    计算格式分数
    
    参数:
        response (str): 模型的响应文本
        
    返回:
        float: 格式分数，范围[0,1]
    """
    # 定义严格的格式模式
    full_pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
    
    # 检查是否完全符合格式要求
    if re.match(full_pattern, response):
        # 检查预测值是否为有效数字
        return 1.0  # 格式完全正确且预测值有效
    
    # 如果格式不完全正确或预测值无效，返回0分
    return 0.0

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    计算EBITDA预测的奖励分数
    
    参数:
        data_source (str): 数据源名称
        solution_str (str): 模型的响应文本
        ground_truth (str): 真实值
        extra_info (dict, optional): 额外信息
        
    返回:
        float: 奖励分数，范围[0,1]
    """
    # if data_source != "ebitda_prediction":
    #     return 0.0
    # 保留最后 300 个字符。请确认这个长度适合你的任务！
    # 如果预测值可能出现在更早的位置而被截断，需要调整或移除此行。
    original_length = len(solution_str)
    solution_str = solution_str[-300:]
     # （可选）可以加日志记录被截断的情况
    # if len(solution_str) < original_length:
    #     print(f"Warning: Truncated solution_str from {original_length} to {len(solution_str)} chars.")
    
    try:

        # 提取预测值
        prediction = extract_ebitda_prediction(solution_str)
        if prediction is None:
            return 0.0  # 如果无法提取预测值，返回0分
        
        # 计算格式分
        format_score = compute_format_score(solution_str)

        # 转换真实值为浮点数
        true_value = float(ground_truth)
        
        # 计算相对误差
        relative_error = abs(prediction - true_value) / (abs(true_value) + 1e-6)
        
        # 根据相对误差计算预测准确度分数
        accuracy_score = 0.0
        if relative_error < 0.01:  # 误差小于1%
            accuracy_score = 1.0
        elif relative_error < 0.05:  # 误差小于5%
            accuracy_score = 0.8
        elif relative_error < 0.1:  # 误差小于10%
            accuracy_score = 0.6
        elif relative_error < 0.2:  # 误差小于20%
            accuracy_score = 0.4
        elif relative_error < 0.5:  # 误差小于50%
            accuracy_score = 0.2
        
        # 综合分数：格式分占10%，准确度分占90%
        final_score = format_score * 0.1 + accuracy_score * 0.9
        return final_score
            
    except (ValueError, TypeError):
        return 0.0  # 如果发生错误，返回0分

# 为了支持多个奖励函数，我们可以添加其他评分函数
def compute_score_strict(data_source, solution_str, ground_truth, extra_info=None):
    """
    更严格的EBITDA预测评分函数
    
    参数:
        data_source (str): 数据源名称
        solution_str (str): 模型的响应文本
        ground_truth (str): 真实值
        extra_info (dict, optional): 额外信息
        
    返回:
        float: 奖励分数，范围[0,1]
    """
    if data_source != "ebitda_prediction":
        return 0.0
    
    try:
        prediction = extract_ebitda_prediction(solution_str)
        if prediction is None:
            return 0.0
        
        true_value = float(ground_truth)
        relative_error = abs(prediction - true_value) / (abs(true_value) + 1e-6)
        
        # 更严格的评分标准
        if relative_error < 0.005:  # 误差小于0.5%
            return 1.0
        elif relative_error < 0.01:  # 误差小于1%
            return 0.8
        elif relative_error < 0.02:  # 误差小于2%
            return 0.6
        elif relative_error < 0.05:  # 误差小于5%
            return 0.4
        elif relative_error < 0.1:  # 误差小于10%
            return 0.2
        else:
            return 0.0
            
    except (ValueError, TypeError):
        return 0.0 