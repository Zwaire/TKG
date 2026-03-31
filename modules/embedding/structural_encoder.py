import torch
import torch.nn as nn
from typing import Optional

from models.rgcn_time import TemporalRGCNEncoder
from .relation_encoder import CombinedRelationEncoder

class StructuralEncoder(nn.Module):
    """
    完整结构编码器:RGCN + TransE
    """
    
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 256, rgcn_config: dict = None,
                 relation_config: dict = None):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # 默认配置
        rgcn_config = rgcn_config or {}
        relation_config = relation_config or {}
        
        # RGCN编码器（实体嵌入）
        self.rgcn_encoder = TemporalRGCNEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            **rgcn_config
        )
        
        # 关系编码器
        self.relation_encoder = CombinedRelationEncoder(
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            **relation_config
        )
        
        # 实体投影层（可选）
        self.entity_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, graph_data, edge_time: Optional[torch.Tensor] = None,
                current_time: Optional[torch.Tensor] = None,
                compute_relations: bool = False):
        """
        前向传播
        
        Args:
            graph_data: 图数据
            edge_time: 边时间
            current_time: 当前时间
            compute_relations: 是否计算关系嵌入
            
        Returns:
            实体嵌入，可选的关系嵌入
        """
        # 获取实体嵌入
        entity_embeddings = self.rgcn_encoder(graph_data, edge_time, current_time)
        
        # 应用投影
        entity_embeddings = self.entity_projection(entity_embeddings)
        
        if compute_relations and hasattr(graph_data, 'edge_index') and hasattr(graph_data, 'edge_type'):
            # 计算关系嵌入
            edge_index = graph_data.edge_index
            edge_type = graph_data.edge_type
            
            # 获取头尾实体嵌入
            head_emb = entity_embeddings[edge_index[0]]
            tail_emb = entity_embeddings[edge_index[1]]
            
            # 计算关系嵌入
            relation_embeddings = self.relation_encoder(
                head_emb, tail_emb, edge_type
            )
            
            return entity_embeddings, relation_embeddings
        
        return entity_embeddings
    
    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """获取实体嵌入"""
        return self.rgcn_encoder.get_entity_embedding(entity_ids)
    
    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """获取关系嵌入"""
        return self.relation_encoder.get_relation_embedding(relation_ids)
    
    def score_triple(self, head_emb: torch.Tensor, relation_id: torch.Tensor,
                    tail_emb: torch.Tensor) -> torch.Tensor:
        """计算三元组得分"""
        relation_emb = self.get_relation_embedding(relation_id)
        return self.relation_encoder.score_triple(head_emb, relation_emb, tail_emb)