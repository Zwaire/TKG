import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class TransERelationEncoder(nn.Module):
    """
    TransE关系编码器
    通过实体嵌入计算关系嵌入： h + r ≈ t
    """
    def __init__(self, hidden_dim: int = 256, margin: float = 1.0, 
                 norm: int = 2, scoring_method: str = "distance"):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.margin = margin
        self.norm = norm
        self.scoring_method = scoring_method
        
        # 关系嵌入（可学习的初始嵌入）
        self.relation_emb = None  # 延迟初始化
        
        # 用于计算距离的投影矩阵
        self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 用于计算关系嵌入的神经网络
        self.relation_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def init_relation_embeddings(self, num_relations: int):
        """初始化关系嵌入"""
        self.relation_emb = nn.Embedding(num_relations, self.hidden_dim)
        nn.init.xavier_uniform_(self.relation_emb.weight)
    
    def compute_relation_from_entities(self, head_emb: torch.Tensor, 
                                      tail_emb: torch.Tensor) -> torch.Tensor:
        """
        从头实体和尾实体嵌入计算关系嵌入
        
        Args:
            head_emb: 头实体嵌入 [batch_size, hidden_dim]
            tail_emb: 尾实体嵌入 [batch_size, hidden_dim]
            
        Returns:
            关系嵌入 [batch_size, hidden_dim]
        """
        # 方法1：简单差值 r = t - h
        # relation = tail_emb - head_emb
        
        # 方法2：使用神经网络
        combined = torch.cat([head_emb, tail_emb], dim=-1)
        relation = self.relation_net(combined)
        
        return relation
    
    def score_triple(self, head_emb: torch.Tensor, relation_emb: torch.Tensor,
                    tail_emb: torch.Tensor) -> torch.Tensor:
        """
        计算三元组得分
        
        Args:
            head_emb: 头实体嵌入
            relation_emb: 关系嵌入
            tail_emb: 尾实体嵌入
            
        Returns:
            得分（越高越好）
        """
        if self.scoring_method == "distance":
            # TransE得分： -||h + r - t||
            score = -torch.norm(head_emb + relation_emb - tail_emb, p=self.norm, dim=-1)
        elif self.scoring_method == "cosine":
            # 余弦相似度
            score = F.cosine_similarity(head_emb + relation_emb, tail_emb, dim=-1)
        elif self.scoring_method == "dot":
            # 点积
            score = torch.sum((head_emb + relation_emb) * tail_emb, dim=-1)
        else:
            raise ValueError(f"Unknown scoring method: {self.scoring_method}")
        
        return score
    
    def transE_loss(self, pos_head: torch.Tensor, pos_rel: torch.Tensor, 
                   pos_tail: torch.Tensor, neg_head: torch.Tensor,
                   neg_rel: torch.Tensor, neg_tail: torch.Tensor) -> torch.Tensor:
        """
        TransE间隔损失
        
        Args:
            pos_head, pos_rel, pos_tail: 正样本嵌入
            neg_head, neg_rel, neg_tail: 负样本嵌入
            
        Returns:
            损失值
        """
        # 计算正负样本得分
        pos_score = self.score_triple(pos_head, pos_rel, pos_tail)
        neg_score = self.score_triple(neg_head, neg_rel, neg_tail)
        
        # 间隔损失
        loss = F.relu(self.margin + neg_score - pos_score).mean()
        
        return loss
    
    def forward(self, head_emb: torch.Tensor, tail_emb: torch.Tensor,
                relation_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            head_emb: 头实体嵌入
            tail_emb: 尾实体嵌入
            relation_ids: 关系ID
            
        Returns:
            Tuple[计算得到的关系嵌入, 得分]
        """
        # 计算关系嵌入
        computed_relation = self.compute_relation_from_entities(head_emb, tail_emb)
        
        # 计算得分
        if relation_ids is not None and self.relation_emb is not None:
            # 如果有关系ID，使用可学习的关系嵌入计算得分
            learned_relation = self.relation_emb(relation_ids)
            scores = self.score_triple(head_emb, learned_relation, tail_emb)
        else:
            # 使用计算得到的关系嵌入
            scores = self.score_triple(head_emb, computed_relation, tail_emb)
        
        return computed_relation, scores
    
    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """获取关系嵌入"""
        if self.relation_emb is None:
            raise ValueError("Relation embeddings not initialized. Call init_relation_embeddings first.")
        return self.relation_emb(relation_ids)


class CombinedRelationEncoder(nn.Module):
    """
    组合关系编码器:结合TransE和可学习的关系嵌入
    """
    
    def __init__(self, num_relations: int, hidden_dim: int = 256,
                 use_learned: bool = True, use_transE: bool = True):
        super().__init__()
        
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.use_learned = use_learned
        self.use_transE = use_transE
        
        # 可学习的关系嵌入
        if use_learned:
            self.learned_relation_emb = nn.Embedding(num_relations, hidden_dim)
            nn.init.xavier_uniform_(self.learned_relation_emb.weight)
        
        # TransE编码器
        if use_transE:
            self.transE_encoder = TransERelationEncoder(hidden_dim)
            self.transE_encoder.init_relation_embeddings(num_relations)
        
        # 融合门控机制
        if use_learned and use_transE:
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, head_emb: torch.Tensor, tail_emb: torch.Tensor,
                relation_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        获取关系嵌入
        
        Args:
            head_emb: 头实体嵌入
            tail_emb: 尾实体嵌入
            relation_ids: 关系ID
            
        Returns:
            关系嵌入
        """
        if self.use_learned and relation_ids is not None:
            learned_emb = self.learned_relation_emb(relation_ids)
            
            if not self.use_transE:
                return learned_emb
        
        if self.use_transE:
            transE_emb, _ = self.transE_encoder(head_emb, tail_emb)
            
            if not self.use_learned or relation_ids is None:
                return transE_emb
        
        # 融合两种嵌入
        if self.use_learned and self.use_transE and relation_ids is not None:
            combined = torch.cat([learned_emb, transE_emb], dim=-1)
            gate = self.gate(combined)
            fused = self.fusion(combined)
            relation_emb = gate * learned_emb + (1 - gate) * fused
            return relation_emb
        
        # 默认返回TransE嵌入
        return transE_emb if self.use_transE else learned_emb
    
    def score_triple(self, head_emb: torch.Tensor, relation_emb: torch.Tensor,
                    tail_emb: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        if self.use_transE:
            return self.transE_encoder.score_triple(head_emb, relation_emb, tail_emb)
        return torch.sum((head_emb + relation_emb) * tail_emb, dim=-1)

    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """按关系ID获取关系嵌入。"""
        if self.use_learned:
            return self.learned_relation_emb(relation_ids)

        if self.use_transE:
            return self.transE_encoder.get_relation_embedding(relation_ids)

        raise ValueError("No relation embedding source is enabled.")