import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict

from .structural_encoder import StructuralEncoder
from .semantic_encoder import SentenceBERTEncoder
from .relation_encoder import TransERelationEncoder
from .fusion_mlp import FusionMLP

class CompleteEmbeddingModule(nn.Module):
    """
    完整的嵌入模块：整合结构嵌入、语义嵌入和融合
    """
    def __init__(self, num_entities: int, num_relations: int,
                 config: dict):
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.config = config
        
        # 获取配置
        structural_config = config.get('structural', {})
        semantic_config = config.get('semantic', {})
        fusion_config = config.get('fusion', {})
        
        # 结构编码器
        self.structural_encoder = StructuralEncoder(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=structural_config.get('hidden_dim', 256),
            rgcn_config=structural_config,
            relation_config=structural_config.get('relation', {})
        )
        
        # 语义编码器
        self.semantic_encoder = SentenceBERTEncoder(
            model_name=semantic_config.get('text_encoder', 
                                          'sentence-transformers/all-MiniLM-L6-v2'),
            hidden_dim=semantic_config.get('hidden_dim', 256),
            freeze_encoder=semantic_config.get('freeze_encoder', True)
        )
        
        # 融合MLP
        structural_dim = structural_config.get('hidden_dim', 256)
        semantic_dim = semantic_config.get('hidden_dim', 256)
        output_dim = fusion_config.get('output_dim', structural_dim)
        
        self.fusion_mlp = FusionMLP(
            structural_dim=structural_dim,
            semantic_dim=semantic_dim,
            output_dim=output_dim,
            hidden_layers=fusion_config.get('hidden_layers', [512, 256]),
            dropout=fusion_config.get('dropout', 0.2),
            activation=fusion_config.get('activation', 'relu')
        )
        
        self.output_dim = output_dim
    
    def forward(self, graph_data, texts: Optional[List[str]] = None,
                edge_time: Optional[torch.Tensor] = None,
                current_time: Optional[torch.Tensor] = None,
                return_components: bool = False):
        """
        前向传播
        
        Args:
            graph_data: 图数据
            texts: 文本描述列表
            edge_time: 边时间
            current_time: 当前时间
            return_components: 是否返回各组件嵌入
            
        Returns:
            融合嵌入，或（结构嵌入，语义嵌入，融合嵌入）
        """
        # 结构嵌入
        structural_emb = self.structural_encoder(
            graph_data, edge_time, current_time
        )
        
        # 语义嵌入（如果有文本）
        if texts is not None and len(texts) > 0:
            semantic_emb = self.semantic_encoder(texts)
            
            # 确保维度匹配
            if semantic_emb.size(0) != structural_emb.size(0):
                # 截断到最小尺寸
                min_size = min(semantic_emb.size(0), structural_emb.size(0))
                semantic_emb = semantic_emb[:min_size]
                structural_emb = structural_emb[:min_size]
        else:
            # 如果没有文本，使用零向量
            device = structural_emb.device
            semantic_emb = torch.zeros(
                structural_emb.size(0), 
                self.semantic_encoder.hidden_dim,
                device=device
            )
        
        # 融合嵌入
        fused_emb = self.fusion_mlp(structural_emb, semantic_emb)
        
        if return_components:
            return structural_emb, semantic_emb, fused_emb
        
        return fused_emb
    
    def encode_structural(self, graph_data, **kwargs):
        """仅编码结构"""
        return self.structural_encoder(graph_data, **kwargs)
    
    def encode_semantic(self, texts: List[str]):
        """仅编码语义"""
        return self.semantic_encoder(texts)
    
    def get_entity_embedding(self, entity_ids: torch.Tensor):
        """获取实体嵌入"""
        return self.structural_encoder.get_entity_embedding(entity_ids)
    
    def get_relation_embedding(self, relation_ids: torch.Tensor):
        """获取关系嵌入"""
        return self.structural_encoder.get_relation_embedding(relation_ids)