import numpy as np
import torch
from datetime import datetime, timedelta
from typing import List, Tuple

def compute_time_decay(time_diffs: torch.Tensor, 
                      method: str = "exponential",
                      decay_rate: float = 0.1) -> torch.Tensor:
    """
    计算时间衰减权重
    
    Args:
        time_diffs: 时间差张量
        method: 衰减方法 ("exponential", "linear", "inverse")
        decay_rate: 衰减率
        
    Returns:
        衰减权重
    """
    if method == "exponential":
        # 指数衰减: w = exp(-decay_rate * Δt)
        weights = torch.exp(-decay_rate * time_diffs)
    elif method == "linear":
        # 线性衰减: w = max(0, 1 - decay_rate * Δt)
        weights = torch.clamp(1 - decay_rate * time_diffs, min=0)
    elif method == "inverse":
        # 逆衰减: w = 1 / (1 + decay_rate * Δt)
        weights = 1 / (1 + decay_rate * time_diffs)
    else:
        raise ValueError(f"Unknown decay method: {method}")
    
    return weights


def normalize_timestamps(timestamps: torch.Tensor, 
                        range_min: float = 0.0, 
                        range_max: float = 1.0) -> torch.Tensor:
    """
    归一化时间戳到指定范围
    
    Args:
        timestamps: 原始时间戳
        range_min: 目标范围最小值
        range_max: 目标范围最大值
        
    Returns:
        归一化的时间戳
    """
    if len(timestamps) == 0:
        return timestamps
    
    min_val = timestamps.min()
    max_val = timestamps.max()
    
    if max_val == min_val:
        return torch.full_like(timestamps, (range_min + range_max) / 2)
    
    # 线性归一化
    normalized = (timestamps - min_val) / (max_val - min_val)
    # 缩放到目标范围
    normalized = range_min + normalized * (range_max - range_min)
    
    return normalized


def split_by_time_windows(timestamps: torch.Tensor, 
                         window_size: int,
                         stride: int = None) -> List[Tuple[int, int]]:
    """
    将时间戳划分为时间窗口
    
    Args:
        timestamps: 时间戳张量
        window_size: 窗口大小
        stride: 滑动步长（默认等于window_size）
        
    Returns:
        窗口列表，每个窗口为(start_idx, end_idx)
    """
    if stride is None:
        stride = window_size
    
    unique_times = torch.unique(timestamps).sort().values
    windows = []
    
    start_idx = 0
    while start_idx < len(unique_times):
        end_idx = min(start_idx + window_size, len(unique_times))
        windows.append((int(unique_times[start_idx]), int(unique_times[end_idx-1])))
        start_idx += stride
    
    return windows


def compute_time_encoding(timestamps: torch.Tensor, 
                         dimension: int = 64,
                         max_period: float = 10000.0) -> torch.Tensor:
    """
    计算时间的位置编码（类似Transformer的positional encoding）
    
    Args:
        timestamps: 时间戳张量，形状为 [N]
        dimension: 编码维度
        max_period: 最大周期
        
    Returns:
        时间编码，形状为 [N, dimension]
    """
    positions = timestamps.unsqueeze(1)  # [N, 1]
    
    # 计算正弦和余弦的频率
    div_term = torch.exp(
        torch.arange(0, dimension, 2, dtype=torch.float, device=timestamps.device) 
        * -(np.log(max_period) / dimension)
    )
    
    # 计算编码
    encoding = torch.zeros(positions.size(0), dimension, 
                          device=timestamps.device, dtype=torch.float)
    
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)
    
    return encoding