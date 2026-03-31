import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

class FusionMLP(nn.Module):
    """
    融合MLP:将结构嵌入和语义嵌入融合
    """
    
    def __init__(self, structural_dim: int, semantic_dim: int,
                 output_dim: int, hidden_layers: List[int] = None,
                 dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        
        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.output_dim = output_dim
        self.total_input_dim = structural_dim + semantic_dim
        
        # 默认隐藏层
        if hidden_layers is None:
            hidden_layers = [512, 256]
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # 构建MLP层
        layers = []
        input_dim = self.total_input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
        # 门控融合机制（可选）
        self.use_gate = True
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(self.total_input_dim, output_dim),
                nn.Sigmoid()
            )
            self.structural_proj = nn.Linear(structural_dim, output_dim)
            self.semantic_proj = nn.Linear(semantic_dim, output_dim)
    
    def forward(self, structural_emb: torch.Tensor, 
                semantic_emb: torch.Tensor) -> torch.Tensor:
        """
        融合结构嵌入和语义嵌入
        
        Args:
            structural_emb: 结构嵌入 [batch_size, structural_dim]
            semantic_emb: 语义嵌入 [batch_size, semantic_dim]
            
        Returns:
            融合嵌入 [batch_size, output_dim]
        """
        # 确保维度匹配
        if structural_emb.size(0) != semantic_emb.size(0):
            # 广播或截断
            min_size = min(structural_emb.size(0), semantic_emb.size(0))
            structural_emb = structural_emb[:min_size]
            semantic_emb = semantic_emb[:min_size]
        
        # 方法1：简单的MLP融合
        if not self.use_gate:
            combined = torch.cat([structural_emb, semantic_emb], dim=-1)
            fused = self.mlp(combined)
            fused = self.layer_norm(fused)
            return fused
        
        # 方法2：门控融合
        combined = torch.cat([structural_emb, semantic_emb], dim=-1)
        
        # 分别投影
        structural_proj = self.structural_proj(structural_emb)
        semantic_proj = self.semantic_proj(semantic_emb)
        
        # 计算门控权重
        gate_weights = self.gate(combined)
        
        # 门控融合
        fused = gate_weights * structural_proj + (1 - gate_weights) * semantic_proj
        
        # 通过MLP进一步处理
        fused = self.mlp(combined) + fused  # 残差连接
        fused = self.layer_norm(fused)
        
        return fused
    
    def fuse_multiple(self, embeddings_list: List[torch.Tensor]) -> torch.Tensor:
        """
        融合多个嵌入
        
        Args:
            embeddings_list: 嵌入列表
            
        Returns:
            融合嵌入
        """
        if len(embeddings_list) == 0:
            raise ValueError("No embeddings to fuse")
        
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        
        # 拼接所有嵌入
        combined = torch.cat(embeddings_list, dim=-1)
        
        # 通过MLP
        fused = self.mlp(combined)
        fused = self.layer_norm(fused)
        
        return fused


class ResidualFusionMLP(nn.Module):
    """
    带残差连接的融合MLP
    """
    
    def __init__(self, structural_dim: int, semantic_dim: int,
                 output_dim: int, num_blocks: int = 2):
        super().__init__()
        
        self.structural_dim = structural_dim
        self.semantic_dim = semantic_dim
        self.output_dim = output_dim
        
        # 初始融合层
        self.initial_fusion = nn.Sequential(
            nn.Linear(structural_dim + semantic_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
            self.residual_blocks.append(block)
        
        # 最终投影
        self.final_proj = nn.Linear(output_dim, output_dim)
    
    def forward(self, structural_emb: torch.Tensor, 
                semantic_emb: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        # 初始融合
        combined = torch.cat([structural_emb, semantic_emb], dim=-1)
        x = self.initial_fusion(combined)
        
        # 残差连接
        residual = x
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x) + x  # 残差连接
        
        # 最终投影
        x = self.final_proj(x) + residual  # 再次残差连接
        
        return x