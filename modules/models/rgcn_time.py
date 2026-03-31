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
    """基于 DGL 的多层时间衰减 RGCN"""
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_bases: int = 10, dropout: float = 0.3,
                 decay_method: str = 'exponential', decay_rate: float = 0.1):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        
        # 初始节点嵌入与关系嵌入
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        
        # 严谨的时间衰减模块
        self.time_decay = TimeDecayFunction(decay_method, decay_rate)
        
        # DGL 原生关系图卷积层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(RelGraphConv(
                in_feat=hidden_dim, 
                out_feat=hidden_dim, 
                num_rels=num_relations, 
                regularizer='basis', 
                num_bases=num_bases,
                self_loop=True,
                dropout=dropout if i < num_layers - 1 else 0.0
            ))
            
    def forward(self, g: dgl.DGLGraph, current_time: torch.Tensor) -> torch.Tensor:
        """
        g: DGLGraph，必须包含 edge_type ('etype') 和 edge_time ('etime') 特征
        """
        # 1. 初始化节点特征
        h = self.entity_emb.weight
        
        # 2. 提取边类型和边时间
        etypes = g.edata['etype']
        etimes = g.edata['etime']
        
        # 3. 计算时间衰减权重 (Edge Norm)
        edge_weights = self.time_decay(etimes, current_time)
        
        # 4. 逐层前向传播
        for layer in self.layers:
            # DGL 的 RelGraphConv 完美支持传入 norm 作为边权重
            h = layer(g, h, etypes, norm=edge_weights)
            h = torch.relu(h)
            
        return h