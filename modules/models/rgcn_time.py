import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import RelGraphConv

class TimeDecayFunction(nn.Module):
    """时间衰减函数：计算每条边的时间权重"""
    def __init__(self, decay_method: str = 'exponential', decay_rate: float = 0.1):
        super().__init__()
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        
    def forward(self, edge_time: torch.Tensor, current_time: torch.Tensor) -> torch.Tensor:
        # time_diffs: [E]
        time_diffs = current_time - edge_time
        time_diffs = torch.clamp(time_diffs, min=0.0) # 确保时间差非负
        time_diffs = time_diffs / (time_diffs.max() + 1e-8) # 归一化到 [0, 1]
        if self.decay_method == 'exponential':
            # w = exp(-lambda * delta_t)
            weights = torch.exp(-self.decay_rate * time_diffs)
        elif self.decay_method == 'linear':
            weights = torch.clamp(1.0 - self.decay_rate * time_diffs, min=0.0)
        else:
            raise ValueError(f"Unknown decay method: {self.decay_method}")
            
        # 返回形状为 [E, 1] 的权重，适配 DGL 的 norm 参数
        return weights.unsqueeze(1)


class TemporalRGCNEncoder_DGL(nn.Module):
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_bases: int = 10, dropout: float = 0.3,
                 decay_method: str = 'exponential', decay_rate: float = 0.1):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        
        # 【新增1】：加入 LayerNorm 和 Dropout 来稳定数值
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.time_decay = TimeDecayFunction(decay_method, decay_rate)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelGraphConv(
                in_feat=hidden_dim, out_feat=hidden_dim, 
                num_rels=num_relations, regularizer='basis', 
                num_bases=num_bases, self_loop=True,
                dropout=dropout if i < num_layers - 1 else 0.0
            ))
            
        # 【新增2】：使用 Xavier 初始化，严防初始数值过大
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
            
    def forward(self, g: dgl.DGLGraph, current_time: torch.Tensor) -> torch.Tensor:
        h_init = self.entity_emb.weight
        h = self.dropout(h_init) # 初始特征加点 dropout
        
        etypes = g.edata['etype']
        etimes = g.edata['etime']
        edge_weights = self.time_decay(etimes, current_time)
        
        for layer in self.layers:
            h = layer(g, h, etypes, norm=edge_weights)
            h = torch.relu(h)
            
        # 【核心修复】：残差连接后，必须接 LayerNorm 压制数值范围！
        h = h + h_init
        h = self.norm(h) 
        
        return h