import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict

class EventEncoder(nn.Module):
    """
    事件编码器：将四元组 (s, p, o, t) 编码为事件嵌入
    """
    
    def __init__(self, entity_encoder, relation_encoder, 
                 time_encoder=None, hidden_dim: int = 256):
        super().__init__()
        
        self.entity_encoder = entity_encoder  # 实体编码器（来自RGCN）
        self.relation_encoder = relation_encoder  # 关系编码器（TransE）
        self.time_encoder = time_encoder  # 时间编码器（可选）
        self.hidden_dim = hidden_dim
        
        # 事件编码MLP
        self.event_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, head_ids: torch.Tensor, relation_ids: torch.Tensor,
                tail_ids: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        编码事件
        
        Args:
            head_ids: 头实体ID [batch_size]
            relation_ids: 关系ID [batch_size]
            tail_ids: 尾实体ID [batch_size]
            timestamps: 时间戳 [batch_size]
            
        Returns:
            事件嵌入 [batch_size, hidden_dim]
        """
        # 获取实体嵌入
        head_emb = self.entity_encoder(head_ids)  # [batch_size, hidden_dim]
        tail_emb = self.entity_encoder(tail_ids)  # [batch_size, hidden_dim]
        
        # 获取关系嵌入
        relation_emb = self.relation_encoder(relation_ids)  # [batch_size, hidden_dim]
        
        # 组合为事件嵌入
        event_emb = torch.cat([head_emb, relation_emb, tail_emb], dim=-1)  # [batch_size, hidden_dim*3]
        
        # 通过MLP
        event_emb = self.event_mlp(event_emb)  # [batch_size, hidden_dim]
        
        # 如果有时间编码器，加入时间信息
        if self.time_encoder is not None:
            time_emb = self.time_encoder(timestamps)  # [batch_size, hidden_dim]
            event_emb = event_emb + 0.1 * time_emb
        
        return event_emb
    
    def encode_batch_events(self, events: List[Tuple[int, int, int, int]]) -> torch.Tensor:
        """批量编码事件"""
        batch_size = len(events)
        
        # 转换为张量
        heads = torch.tensor([e[0] for e in events])
        relations = torch.tensor([e[1] for e in events])
        tails = torch.tensor([e[2] for e in events])
        times = torch.tensor([e[3] for e in events])
        
        # 编码
        return self.forward(heads, relations, tails, times)