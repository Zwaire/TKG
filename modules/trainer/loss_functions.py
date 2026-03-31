import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

class MultiTaskLoss(nn.Module):
    """
    多任务损失函数
    整合各种损失用于联合训练
    """
    
    def __init__(self, loss_weights: Dict[str, float] = None):
        super().__init__()
        
        # 默认损失权重
        self.default_weights = {
            'reconstruction': 1.0,    # 链接预测重建损失
            'contrastive': 0.5,       # 对比学习损失
            'temporal': 0.3,          # 时间一致性损失
            'clustering': 0.2,        # 聚类损失
            'regularization': 0.01,   # 正则化损失
        }
        
        # 更新权重
        if loss_weights is not None:
            self.default_weights.update(loss_weights)
        
        self.loss_weights = self.default_weights
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算总损失
        
        Args:
            losses: 各种损失的字典
            
        Returns:
            加权总损失
        """
        total_loss = 0.0
        
        for loss_name, loss_value in losses.items():
            if loss_value is not None:
                weight = self.loss_weights.get(loss_name, 0.0)
                if weight > 0:
                    total_loss += weight * loss_value
        
        return total_loss


class TemporalReconstructionLoss(nn.Module):
    """
    时间感知重建损失（链接预测）
    """
    
    def __init__(self, margin: float = 1.0, norm: int = 2):
        super().__init__()
        self.margin = margin
        self.norm = norm
    
    def forward(self, pos_scores: torch.Tensor,
                neg_scores: torch.Tensor,
                time_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算时间感知重建损失
        
        Args:
            pos_scores: 正样本得分 [batch_size]
            neg_scores: 负样本得分 [batch_size, num_neg]
            time_weights: 时间权重 [batch_size]（可选）
            
        Returns:
            重建损失
        """
        batch_size = pos_scores.size(0)
        num_neg = neg_scores.size(1)
        
        # 扩展正样本得分用于比较
        pos_scores_expanded = pos_scores.unsqueeze(1).expand(-1, num_neg)
        
        # 计算间隔损失
        loss = F.relu(self.margin + neg_scores - pos_scores_expanded)
        
        # 如果有时间权重，应用权重
        if time_weights is not None:
            time_weights = time_weights.unsqueeze(1).expand(-1, num_neg)
            loss = loss * time_weights
        
        # 平均损失
        loss = loss.mean()
        
        return loss


class TemporalConsistencyLoss(nn.Module):
    """
    时间一致性损失
    鼓励相邻时间步的实体嵌入变化平滑
    """
    
    def __init__(self, temporal_smoothing: float = 0.1):
        super().__init__()
        self.temporal_smoothing = temporal_smoothing
    
    def forward(self, current_emb: torch.Tensor,
                previous_emb: torch.Tensor,
                time_gap: torch.Tensor) -> torch.Tensor:
        """
        计算时间一致性损失
        
        Args:
            current_emb: 当前时间步嵌入 [n_entities, hidden_dim]
            previous_emb: 前一时间步嵌入 [n_entities, hidden_dim]
            time_gap: 时间间隔 [n_entities]
            
        Returns:
            时间一致性损失
        """
        # 计算嵌入变化
        embedding_change = torch.norm(current_emb - previous_emb, p=2, dim=1)
        
        # 时间加权：时间间隔越大，允许的变化越大
        allowed_change = self.temporal_smoothing * time_gap
        
        # 惩罚超出允许范围的变化
        loss = F.relu(embedding_change - allowed_change).mean()
        
        return loss


class OrthogonalityConstraint(nn.Module):
    """
    正交性约束损失
    鼓励不同聚类中心的嵌入相互正交
    """
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, cluster_centers: torch.Tensor) -> torch.Tensor:
        """
        计算正交性约束损失
        
        Args:
            cluster_centers: 聚类中心 [n_clusters, hidden_dim]
            
        Returns:
            正交性损失
        """
        n_clusters = cluster_centers.size(0)
        
        # 归一化聚类中心
        centers_norm = F.normalize(cluster_centers, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity = torch.matmul(centers_norm, centers_norm.T)
        
        # 移除对角线
        similarity = similarity - torch.eye(n_clusters, device=cluster_centers.device)
        
        # 惩罚非零的非对角线元素
        loss = torch.norm(similarity, p='fro') ** 2
        
        return self.weight * loss


class AdaptiveLossWeighting(nn.Module):
    """
    自适应损失权重调整
    根据训练进度动态调整损失权重
    """
    
    def __init__(self, initial_weights: Dict[str, float],
                 warmup_epochs: int = 10):
        super().__init__()
        
        self.initial_weights = initial_weights
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
        
        # 可学习的损失权重
        self.log_weights = nn.ParameterDict({
            name: nn.Parameter(torch.log(torch.tensor(weight)))
            for name, weight in initial_weights.items()
        })
    
    def update_epoch(self, epoch: int):
        """更新当前epoch"""
        self.current_epoch = epoch
    
    def get_weights(self) -> Dict[str, float]:
        """获取当前损失权重"""
        weights = {}
        
        for name, log_weight in self.log_weights.items():
            # 应用warmup
            if self.current_epoch < self.warmup_epochs:
                warmup_ratio = self.current_epoch / self.warmup_epochs
                weight = torch.exp(log_weight) * warmup_ratio
            else:
                weight = torch.exp(log_weight)
            
            weights[name] = weight.item()
        
        return weights
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        计算加权损失
        
        Args:
            losses: 各种损失的字典
            
        Returns:
            加权总损失
        """
        total_loss = 0.0
        
        for name, loss_value in losses.items():
            if loss_value is not None and name in self.log_weights:
                log_weight = self.log_weights[name]
                weight = torch.exp(log_weight)
                
                # 添加正则化项防止权重过小
                reg_loss = 0.01 * (log_weight ** 2)
                
                total_loss += weight * loss_value + reg_loss
        
        return total_loss