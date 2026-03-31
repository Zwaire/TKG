import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict

class EventContrastiveLoss(nn.Module):
    """
    事件级对比学习损失
    通过对比学习训练事件嵌入，使相似事件在嵌入空间中接近
    """
    
    def __init__(self, temperature: float = 0.1, 
                 negative_samples: int = 512,
                 temporal_weight: float = 0.5):
        super().__init__()
        
        self.temperature = temperature
        self.negative_samples = negative_samples
        self.temporal_weight = temporal_weight
        
    def forward(self, event_embeddings: torch.Tensor,
                event_triples: torch.Tensor,  # [batch_size, 4] (s, p, o, t)
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        计算事件对比损失
        
        Args:
            event_embeddings: 事件嵌入 [batch_size, hidden_dim]
            event_triples: 事件三元组 [batch_size, 4]
            timestamps: 时间戳 [batch_size]
            
        Returns:
            对比损失
        """
        batch_size = event_embeddings.size(0)
        
        # 归一化嵌入
        embeddings_norm = F.normalize(event_embeddings, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T)  # [batch_size, batch_size]
        
        # 构建时间邻近性掩码
        time_mask = self._build_temporal_mask(timestamps)
        
        # 构建语义相似性掩码（基于共享实体/关系）
        semantic_mask = self._build_semantic_mask(event_triples)
        
        # 正样本掩码：时间邻近且语义相似
        positive_mask = time_mask * semantic_mask
        positive_mask = positive_mask.fill_diagonal_(0)  # 移除自相似
        
        # 负样本掩码：时间远离或语义不相似
        negative_mask = 1 - positive_mask
        negative_mask = negative_mask.fill_diagonal_(0)  # 移除自相似
        
        # 计算对比损失（InfoNCE）
        logits = similarity_matrix / self.temperature
        
        # 正样本相似度
        pos_sim = (logits * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
        
        # 负样本相似度
        neg_sim = (logits * negative_mask).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
        
        # InfoNCE损失
        loss = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)).mean()
        
        return loss
    
    def _build_temporal_mask(self, timestamps: torch.Tensor) -> torch.Tensor:
        """构建时间邻近性掩码"""
        batch_size = timestamps.size(0)
        
        # 计算时间差
        time_diff = torch.abs(timestamps.unsqueeze(0) - timestamps.unsqueeze(1))
        
        # 归一化
        max_diff = time_diff.max()
        if max_diff > 0:
            normalized_diff = time_diff / max_diff
        else:
            normalized_diff = torch.zeros_like(time_diff)
        
        # 时间邻近性：时间差越小，权重越高
        temporal_proximity = 1.0 - normalized_diff
        
        return temporal_proximity
    
    def _build_semantic_mask(self, event_triples: torch.Tensor) -> torch.Tensor:
        """构建语义相似性掩码"""
        batch_size = event_triples.size(0)
        
        # 提取头实体、关系和尾实体
        heads = event_triples[:, 0]
        relations = event_triples[:, 1]
        tails = event_triples[:, 2]
        
        # 计算实体重叠度
        head_sim = (heads.unsqueeze(0) == heads.unsqueeze(1)).float()
        tail_sim = (tails.unsqueeze(0) == tails.unsqueeze(1)).float()
        rel_sim = (relations.unsqueeze(0) == relations.unsqueeze(1)).float()
        
        # 语义相似性：共享实体或关系
        semantic_sim = head_sim + tail_sim + rel_sim
        semantic_sim = semantic_sim / 3.0  # 归一化到[0, 1]
        
        return semantic_sim


class MultiViewEventContrastiveLoss(nn.Module):
    """
    多视图事件对比学习
    结合结构视图和语义视图
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, structural_event_emb: torch.Tensor,
                semantic_event_emb: torch.Tensor,
                event_triples: torch.Tensor,
                timestamps: torch.Tensor) -> torch.Tensor:
        """
        计算多视图事件对比损失
        
        Args:
            structural_event_emb: 结构事件嵌入 [batch_size, hidden_dim]
            semantic_event_emb: 语义事件嵌入 [batch_size, hidden_dim]
            event_triples: 事件三元组 [batch_size, 4]
            timestamps: 时间戳 [batch_size]
            
        Returns:
            多视图对比损失
        """
        # 视图内对比损失
        intra_structural_loss = self._intra_view_loss(
            structural_event_emb, event_triples, timestamps
        )
        
        intra_semantic_loss = self._intra_view_loss(
            semantic_event_emb, event_triples, timestamps
        )
        
        # 跨视图对比损失
        cross_view_loss = self._cross_view_loss(
            structural_event_emb, semantic_event_emb
        )
        
        # 总损失
        total_loss = intra_structural_loss + intra_semantic_loss + cross_view_loss
        
        return total_loss
    
    def _intra_view_loss(self, embeddings: torch.Tensor,
                        event_triples: torch.Tensor,
                        timestamps: torch.Tensor) -> torch.Tensor:
        """视图内对比损失"""
        batch_size = embeddings.size(0)
        
        # 归一化
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        
        # 相似度矩阵
        sim_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # 构建正样本掩码（基于语义和时间相似性）
        semantic_mask = self._build_semantic_mask(event_triples)
        time_mask = self._build_temporal_mask(timestamps)
        positive_mask = semantic_mask * time_mask
        positive_mask = positive_mask.fill_diagonal_(0)
        
        # 计算InfoNCE损失
        exp_sim = torch.exp(sim_matrix)
        
        pos_sim = (exp_sim * positive_mask).sum(dim=1)
        neg_sim = (exp_sim * (1 - positive_mask)).sum(dim=1)
        
        loss = -torch.log(pos_sim / (pos_sim + neg_sim + 1e-8)).mean()
        
        return loss
    
    def _cross_view_loss(self, view1_emb: torch.Tensor,
                        view2_emb: torch.Tensor) -> torch.Tensor:
        """跨视图对比损失"""
        batch_size = view1_emb.size(0)
        
        # 归一化
        view1_norm = F.normalize(view1_emb, p=2, dim=1)
        view2_norm = F.normalize(view2_emb, p=2, dim=1)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(view1_norm, view2_norm.T) / self.temperature
        
        # 正样本是对角线（同一事件的不同视图）
        labels = torch.arange(batch_size, device=view1_emb.device)
        
        # 交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss
    
    def _build_semantic_mask(self, event_triples: torch.Tensor) -> torch.Tensor:
        """构建语义相似性掩码"""
        batch_size = event_triples.size(0)
        
        heads = event_triples[:, 0]
        relations = event_triples[:, 1]
        tails = event_triples[:, 2]
        
        # 计算相似度
        head_eq = (heads.unsqueeze(0) == heads.unsqueeze(1)).float()
        tail_eq = (tails.unsqueeze(0) == tails.unsqueeze(1)).float()
        rel_eq = (relations.unsqueeze(0) == relations.unsqueeze(1)).float()
        
        # 加权组合
        semantic_sim = 0.4 * head_eq + 0.4 * tail_eq + 0.2 * rel_eq
        
        return semantic_sim
    
    def _build_temporal_mask(self, timestamps: torch.Tensor) -> torch.Tensor:
        """构建时间邻近性掩码"""
        batch_size = timestamps.size(0)
        
        time_diff = torch.abs(timestamps.unsqueeze(0) - timestamps.unsqueeze(1))
        max_diff = time_diff.max()
        
        if max_diff > 0:
            normalized = 1.0 - (time_diff / max_diff)
        else:
            normalized = torch.ones_like(time_diff)
        
        return normalized