import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Optional
import warnings

class SentenceBERTEncoder(nn.Module):
    """
    句子级BERT编码器
    使用预训练的句子Transformer模型
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 hidden_dim: int = 256, freeze_encoder: bool = True):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        
        # 加载tokenizer和模型
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.encoder = AutoModel.from_pretrained(model_name)
            
            # 冻结编码器参数
            if freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        except Exception as e:
            warnings.warn(f"Failed to load model {model_name}: {e}")
            self.encoder = None
            self.tokenizer = None
        
        # 获取编码器输出维度
        if self.encoder is not None:
            encoder_dim = self.encoder.config.hidden_size
        else:
            encoder_dim = hidden_dim
        
        # 投影层（将BERT输出映射到目标维度）
        self.projection = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        编码文本列表
        
        Args:
            texts: 文本字符串列表
            
        Returns:
            文本嵌入 [batch_size, hidden_dim]
        """
        if not texts:
            # 如果没有文本，返回零向量
            device = next(self.parameters()).device
            return torch.zeros(0, self.hidden_dim, device=device)
        
        # 如果没有加载编码器，返回随机嵌入
        if self.encoder is None:
            device = next(self.parameters()).device
            batch_size = len(texts)
            return torch.randn(batch_size, self.hidden_dim, device=device) * 0.01
        
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # 移动到模型设备
        device = next(self.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 编码
        with torch.set_grad_enabled(not self.freeze_encoder):
            outputs = self.encoder(**inputs)
        
        # 使用平均池化（对于句子BERT，通常使用[CLS]或平均池化）
        # 这里使用平均池化
        attention_mask = inputs["attention_mask"]
        token_embeddings = outputs.last_hidden_state
        
        # 创建扩展的attention mask
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # 加权平均
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        sentence_embeddings = sum_embeddings / sum_mask
        
        # 投影到目标维度
        projected = self.projection(sentence_embeddings)
        normalized = self.layer_norm(projected)
        
        return normalized
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """批量编码文本"""
        if not texts:
            device = next(self.parameters()).device
            return torch.zeros(0, self.hidden_dim, device=device)
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.forward(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        return torch.cat(all_embeddings, dim=0)