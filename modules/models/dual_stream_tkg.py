import torch
import torch.nn as nn
import dgl
from .rgcn_time import TemporalRGCNEncoder_DGL

class DualStreamTKG(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, 
                 semantic_dim: int = 768, hidden_dim: int = 256,
                 num_rgcn_layers: int = 2, num_bases: int = 10):
        super().__init__()
        
        # --- 结构流 (Structural Stream) ---
        self.struct_encoder = TemporalRGCNEncoder_DGL(
            num_entities=num_entities,
            num_relations=num_relations,
            hidden_dim=hidden_dim,
            num_layers=num_rgcn_layers,
            num_bases=num_bases
        )
        
        # --- 语义流适配器 (Semantic Stream Adapter) ---
        # 将 BERT 的 768 维降维到图的 hidden_dim (256维)，以便对齐
        self.semantic_proj = nn.Sequential(
            nn.Linear(semantic_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- 融合模块 (Fusion Module) ---
        # 使用简单的门控机制 (Gating) 决定信任结构还是语义
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        self.fusion_mlp = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, g: dgl.DGLGraph, current_time: torch.Tensor, 
                event_triples: torch.Tensor, offline_semantic_embs: torch.Tensor):
        """
        g: 历史快照构建的 DGLGraph
        event_triples: 当前批次的事件 [Batch, 4] (src, rel, dst, time)
        offline_semantic_embs: 这批事件对应的 BERT 离线特征 [Batch, 768]
        """
        # 1. 通过 RGCN 获取所有实体的最新结构化嵌入 [N, hidden_dim]
        entity_embs = self.struct_encoder(g, current_time)
        rel_embs = self.struct_encoder.relation_emb.weight
        
        # 2. 提取当前批次事件的【结构级表示】 (Event Structural Repr)
        # e_struct = Entity_s + Relation_r + Entity_o
        src_ids, rel_ids, dst_ids = event_triples[:, 0], event_triples[:, 1], event_triples[:, 2]
        
        event_struct = entity_embs[src_ids] + rel_embs[rel_ids] + entity_embs[dst_ids]
        
        # 3. 提取当前批次事件的【语义级表示】 (Event Semantic Repr)
        # 将 768 维映射到相同的隐藏层空间
        event_semantic = self.semantic_proj(offline_semantic_embs)
        
        # 4. 特征融合 (Gated Fusion)
        concat_feats = torch.cat([event_struct, event_semantic], dim=-1)
        gate = self.fusion_gate(concat_feats)
        
        # 门控机制：让模型自适应选择
        event_fused = gate * event_struct + (1 - gate) * event_semantic
        
        # 返回实体表征(用于主任务链接预测)、两种事件表征(用于对比学习)、融合表征
        return entity_embs, event_struct, event_semantic, event_fused