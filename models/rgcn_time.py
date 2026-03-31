import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from typing import Optional, Tuple
import math

class TimeDecayFunction(nn.Module):
    """
    时间衰减函数模块
    参考HERLN论文中的时间衰减机制
    """
    def __init__(self, decay_method: str = 'exponential', 
                 decay_rate: float = 0.1,
                 time_dim: int = 64):
        super().__init__()
        self.decay_method = decay_method
        self.decay_rate = decay_rate
        self.time_dim = time_dim
        
        if decay_method == 'learnable':
            # 可学习的时间衰减权重
            self.time_weights = nn.Parameter(torch.randn(time_dim))
        elif decay_method == 'fourier':
            # 傅里叶时间编码
            self.fourier_weights = nn.Parameter(torch.randn(time_dim, time_dim))
    
    def forward(self, time_diffs: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_time: torch.Tensor,
                current_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算时间衰减权重
        
        Args:
            time_diffs: 时间差 [E]
            edge_index: 边索引 [2, E]
            edge_time: 边时间 [E]
            current_time: 当前时间
            
        Returns:
            衰减权重 [E]
        """
        if self.decay_method == 'exponential':
            # 指数衰减: w = exp(-decay_rate * Δt)
            weights = torch.exp(-self.decay_rate * time_diffs)
            
        elif self.decay_method == 'linear':
            # 线性衰减: w = max(0, 1 - decay_rate * Δt)
            weights = torch.clamp(1 - self.decay_rate * time_diffs, min=0)
            
        elif self.decay_method == 'inverse':
            # 逆衰减: w = 1 / (1 + decay_rate * Δt)
            weights = 1 / (1 + self.decay_rate * time_diffs)
            
        elif self.decay_method == 'learnable':
            # 可学习的衰减: w = σ(W * time_encoding(Δt))
            # 使用正弦位置编码表示时间差
            pos_enc = self._positional_encoding(time_diffs.unsqueeze(1))
            weights = torch.sigmoid(torch.matmul(pos_enc, self.time_weights))
            
        elif self.decay_method == 'fourier':
            # 傅里叶时间衰减
            time_enc = self._fourier_time_encoding(time_diffs)
            weights = torch.sigmoid(torch.matmul(time_enc, self.fourier_weights).mean(-1))
            
        else:
            raise ValueError(f"Unknown decay method: {self.decay_method}")
        
        # 归一化权重
        weights = weights / (weights.sum() + 1e-8)
        
        return weights
    
    def _positional_encoding(self, positions: torch.Tensor) -> torch.Tensor:
        """正弦位置编码"""
        d_model = self.time_dim
        pe = torch.zeros(positions.size(0), d_model, device=positions.device)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=positions.device).float() *
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(positions.float() * div_term)
        pe[:, 1::2] = torch.cos(positions.float() * div_term)
        
        return pe
    
    def _fourier_time_encoding(self, time_diffs: torch.Tensor) -> torch.Tensor:
        """傅里叶时间编码"""
        frequencies = torch.linspace(1.0, 10.0, self.time_dim // 2, 
                                    device=time_diffs.device)
        encoding = torch.zeros(len(time_diffs), self.time_dim, 
                              device=time_diffs.device)
        
        for i, freq in enumerate(frequencies):
            encoding[:, 2*i] = torch.sin(time_diffs * freq)
            encoding[:, 2*i+1] = torch.cos(time_diffs * freq)
        
        return encoding


class TemporalRGCNLayer(nn.Module):
    """
    带时间衰减的RGCN层
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 num_relations: int, num_bases: int = 10,
                 use_time_decay: bool = True,
                 decay_method: str = 'exponential',
                 decay_rate: float = 0.1,
                 dropout: float = 0.3,
                 bias: bool = True):
        # in_channels 输入维度, out_channels 输出维度
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.use_time_decay = use_time_decay
        self.dropout = dropout
        
        # RGCN卷积层
        self.rgcn_conv = RGCNConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_relations=num_relations,
            num_bases=num_bases,
            bias=bias
        )
        
        # 时间衰减函数
        if use_time_decay:
            self.time_decay = TimeDecayFunction(
                decay_method=decay_method,
                decay_rate=decay_rate,
                time_dim=out_channels
            )
        
        # 时间感知的参数
        self.time_proj = nn.Linear(out_channels, out_channels)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(out_channels)
        
        # 激活函数
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                edge_time: Optional[torch.Tensor] = None,
                current_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            edge_type: 边类型 [E]
            edge_time: 边时间戳 [E]
            current_time: 当前时间戳
            
        Returns:
            更新后的节点特征 [N, D]
        """
        # 计算RGCN特征
        x_rgcn = self.rgcn_conv(x, edge_index, edge_type)
        
        # 应用时间衰减（如果启用）
        if self.use_time_decay and edge_time is not None:
            # 计算时间差
            if current_time is None:
                current_time = edge_time.max()
            
            time_diffs = current_time - edge_time
            time_diffs = time_diffs / (time_diffs.max() + 1e-8)  # 归一化
            
            # 计算时间衰减权重
            decay_weights = self.time_decay(
                time_diffs, edge_index, edge_time, current_time
            )
            
            # 聚合时间感知的特征
            x_time = self._aggregate_with_time_decay(
                x, edge_index, decay_weights, edge_type
            )
            
            # 融合RGCN特征和时间感知特征
            x_combined = x_rgcn + self.time_proj(x_time)
        else:
            x_combined = x_rgcn
        
        # 应用激活函数和dropout
        x_out = self.activation(x_combined)
        x_out = F.dropout(x_out, p=self.dropout, training=self.training)
        
        # 层归一化
        x_out = self.layer_norm(x_out)
        
        return x_out
    
    def _aggregate_with_time_decay(self, x: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   decay_weights: torch.Tensor,
                                   edge_type: torch.Tensor) -> torch.Tensor:
        """
        使用时间衰减权重进行消息聚合
        
        Args:
            x: 节点特征 [N, D]
            edge_index: 边索引 [2, E]
            decay_weights: 衰减权重 [E]
            edge_type: 边类型 [E]
            
        Returns:
            聚合后的特征 [N, D]
        """
        src, dst = edge_index
        num_nodes = x.size(0)
        
        # 初始化聚合结果
        agg = torch.zeros_like(x)
        
        # 对每个关系类型分别聚合
        for rel in range(self.num_relations):
            mask = (edge_type == rel)
            if mask.any():
                rel_src = src[mask]
                rel_dst = dst[mask]
                rel_weights = decay_weights[mask].unsqueeze(1)  # [E_rel, 1]
                
                # 获取源节点特征
                src_features = x[rel_src]  # [E_rel, D]
                
                # 加权聚合
                weighted_features = src_features * rel_weights
                
                # 使用scatter_add进行聚合
                agg.index_add_(0, rel_dst, weighted_features)
        
        # 计算每个节点的入度（用于归一化）
        degree = torch.zeros(num_nodes, device=x.device)
        degree.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        degree = torch.clamp(degree, min=1.0)
        
        # 归一化
        agg = agg / degree.unsqueeze(1)
        
        return agg


class TemporalRGCNEncoder(nn.Module):
    """
    多层带时间衰减的RGCN编码器
    """
    def __init__(self, num_entities: int, num_relations: int,
                 hidden_dim: int = 256, num_layers: int = 2,
                 num_bases: int = 10, dropout: float = 0.3,
                 use_time_decay: bool = True,
                 decay_method: str = 'exponential',
                 decay_rate: float = 0.1):
        # decay 用于时间衰减的机制的控制，采用的指数型衰减
        super().__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 实体嵌入初始化
        self.entity_emb = nn.Embedding(num_entities, hidden_dim)
        
        # 关系嵌入（用于后续的得分函数）
        self.relation_emb = nn.Embedding(num_relations, hidden_dim)
        
        # 多层RGCN层
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = TemporalRGCNLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases,
                use_time_decay=use_time_decay,
                decay_method=decay_method,
                decay_rate=decay_rate,
                dropout=dropout if i < num_layers - 1 else 0.0,  # 最后一层不用dropout
                bias=True
            )
            self.layers.append(layer)
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 初始化参数
        self._init_parameters()
    
    def _init_parameters(self):
        """初始化模型参数"""
        # Xavier初始化实体嵌入
        nn.init.xavier_uniform_(self.entity_emb.weight)
        
        # 初始化关系嵌入
        nn.init.xavier_uniform_(self.relation_emb.weight)
        
        # 初始化输出投影层
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, graph_data, 
                edge_time: Optional[torch.Tensor] = None,
                current_time: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            graph_data: PyG Data对象,包含edge_index, edge_type, global_node_ids
            edge_time: 边时间戳 [E]
            current_time: 当前时间戳
            
        Returns:
            节点嵌入 [N, D]
        """
        # 获取节点ID和边信息
        node_ids = graph_data.global_node_ids
        edge_index = graph_data.edge_index
        edge_type = graph_data.edge_type

        # 确保所有输入张量在与模型参数相同的设备上，防止 CPU/CUDA 不一致
        device = self.entity_emb.weight.device
        if node_ids is not None:
            node_ids = node_ids.to(device)
        if edge_index is not None:
            edge_index = edge_index.to(device)
        if edge_type is not None:
            edge_type = edge_type.to(device)

        # 如果graph_data中已经有edge_time，使用它并移动到目标设备
        if hasattr(graph_data, 'edge_time') and graph_data.edge_time is not None:
            edge_time = graph_data.edge_time.to(device)
        elif edge_time is not None:
            edge_time = edge_time.to(device)

        # 初始实体嵌入
        x = self.entity_emb(node_ids)
        
        # 通过多层RGCN
        for layer in self.layers:
            x = layer(x, edge_index, edge_type, edge_time, current_time)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x
    
    def get_relation_embedding(self, relation_ids: torch.Tensor) -> torch.Tensor:
        """获取关系嵌入"""
        if relation_ids is None:
            return None
        device = self.relation_emb.weight.device
        return self.relation_emb(relation_ids.to(device))
    
    def get_entity_embedding(self, entity_ids: torch.Tensor) -> torch.Tensor:
        """获取实体嵌入(不通过GNN)"""
        if entity_ids is None:
            return None
        device = self.entity_emb.weight.device
        return self.entity_emb(entity_ids.to(device))