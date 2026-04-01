import torch
import torch.nn as nn
import torch.nn.functional as F

class TKGLosses(nn.Module):
    def __init__(self, temperature: float = 0.1, lambda_contrastive: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.cross_entropy = nn.CrossEntropyLoss()

    def _calc_distmult_score(self, head_emb: torch.Tensor, rel_emb: torch.Tensor, tail_emb: torch.Tensor) -> torch.Tensor:
        """
        使用 DistMult 知识图谱打分函数。
        公式: Score = (head * rel) dot tail
        :param head_emb: [Batch, hidden_dim]
        :param rel_emb: [Batch, hidden_dim]
        :param tail_emb: [Num_Entities, hidden_dim] (我们要和所有候选尾节点打分)
        :return: [Batch, Num_Entities]
        """
        head_rel = head_emb * rel_emb 
        scores = torch.matmul(head_rel, tail_emb.transpose(0, 1))
        
        scaling_factor = head_emb.size(-1) ** 0.5  # 256的平方根是16
        scores = scores / scaling_factor
        
        return scores

    def compute_link_prediction_loss(self, entity_embs: torch.Tensor, rel_embs: torch.Tensor, 
                                     triples: torch.Tensor) -> torch.Tensor:
        """
        计算主任务损失：预测给定 (主体, 关系) 下的 (客体)
        triples: [Batch, 4] 包含 (src, rel, dst, time)
        """
        src_ids = triples[:, 0]
        rel_ids = triples[:, 1]
        dst_ids = triples[:, 2] # 真实的客体标签

        # 提取当前 Batch 的主体和关系特征
        head_emb = entity_embs[src_ids]
        rel_emb = rel_embs[rel_ids]

        # 计算与所有实体（作为候选客体）的得分
        # 注意：这里 tail_emb 传入的是所有实体的最新特征
        scores = self._calc_distmult_score(head_emb, rel_emb, entity_embs)

        # 这是一个多分类问题，类别数就是图谱中的实体总数
        loss = self.cross_entropy(scores, dst_ids)
        return loss

    def compute_contrastive_loss(self, event_struct: torch.Tensor, event_semantic: torch.Tensor) -> torch.Tensor:
        """
        计算跨模态对比学习损失 (InfoNCE)
        拉近同一个事件的结构特征与语义特征，推远不同事件的特征。
        """
        # L2 归一化，将向量映射到单位超球面上
        z_struct = F.normalize(event_struct, dim=-1)
        z_semantic = F.normalize(event_semantic, dim=-1)

        # 计算相似度矩阵 [Batch, Batch]
        # 对角线元素是正样本对（同一个事件的结构和语义），非对角线是负样本对
        sim_matrix = torch.matmul(z_struct, z_semantic.transpose(0, 1)) / self.temperature

        # 标签：对角线索引 0, 1, 2, ... Batch-1
        batch_size = z_struct.size(0)
        labels = torch.arange(batch_size, device=z_struct.device)

        # 损失可以是对称的：从结构到语义，以及从语义到结构
        loss_s2t = self.cross_entropy(sim_matrix, labels)
        loss_t2s = self.cross_entropy(sim_matrix.transpose(0, 1), labels)
        
        return (loss_s2t + loss_t2s) / 2.0

    def forward(self, entity_embs, rel_embs, event_struct, event_semantic, triples):
        """
        总损失计算
        """
        loss_lp = self.compute_link_prediction_loss(entity_embs, rel_embs, triples)
        loss_cl = self.compute_contrastive_loss(event_struct, event_semantic)
        
        total_loss = loss_lp + self.lambda_contrastive * loss_cl
        
        return total_loss, loss_lp, loss_cl