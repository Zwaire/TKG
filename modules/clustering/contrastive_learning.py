import torch
import torch.nn as nn
from typing import Dict, Any

from .event_contrastive import MultiViewEventContrastiveLoss


class MultiViewContrastiveLoss(nn.Module):
    """兼容 trainer 期望接口的多视图对比损失（封装已有实现）"""
    def __init__(self, temperature: float = 0.1, view_weights: Dict[str, float] = None):
        super().__init__()
        self.view_weights = view_weights or {}
        self.impl = MultiViewEventContrastiveLoss(temperature=temperature)

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, timestamps: torch.Tensor):
        # emb1 / emb2 视为结构/语义或两个视图；这里将两者都传入底层实现
        return self.impl(emb1, emb2, torch.zeros((emb1.size(0), 4), dtype=torch.long, device=emb1.device), timestamps)


class MemoryBankContrastiveLoss(nn.Module):
    """简单的记忆银行占位符实现（接口兼容）"""
    def __init__(self, hidden_dim: int = 256, memory_size: int = 8192, temperature: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, timestamps: torch.Tensor = None):
        # 占位：返回零损失以保持训练流程
        return torch.tensor(0.0, device=embeddings.device)


class TemporalContrastiveLoss(MultiViewContrastiveLoss):
    """向后兼容名称，等同于 MultiViewContrastiveLoss"""
    pass
