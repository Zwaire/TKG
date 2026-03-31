import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# 假设我们之前写好的模块都在对应的路径下
from .losses import TKGLosses
from ..data_pipeline.dataset import tkg_collate_fn

class TemporalKGTrainer:
    def __init__(self, model, train_dataset, valid_dataset, config: dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        
        # 实例化我们上一步写的损失函数
        self.loss_fn = TKGLosses(
            temperature=config.get("temperature", 0.1),
            lambda_contrastive=config.get("lambda_contrastive", 0.1)
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get("lr", 0.001), 
            weight_decay=config.get("weight_decay", 1e-5)
        )
        
        # 数据加载器 (注意 batch_size=1，因为我们的 dataset 每次吐出一个时间步的整图)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=tkg_collate_fn)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, collate_fn=tkg_collate_fn)

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total_loss = 0.0
        total_lp_loss = 0.0
        total_cl_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Train]")
        
        for g, target_triples, target_embs, current_time in pbar:
            # 将所有数据搬运到 GPU
            g = g.to(self.device)
            target_triples = target_triples.to(self.device)
            target_embs = target_embs.to(self.device)
            current_time = current_time.to(self.device)
            
            self.optimizer.zero_grad()
            
            # 1. 前向传播：获取实体特征与事件的双模态特征
            entity_embs, event_struct, event_semantic, _ = self.model(
                g, current_time, target_triples, target_embs
            )
            
            # 获取关系嵌入 (从 RGCN 编码器中借用)
            rel_embs = self.model.struct_encoder.relation_emb.weight
            
            # 2. 计算联合损失
            loss, loss_lp, loss_cl = self.loss_fn(
                entity_embs, rel_embs, event_struct, event_semantic, target_triples
            )
            
            # 3. 反向传播与参数更新
            loss.backward()
            
            # 梯度裁剪 (防止图神经网络在长序列中梯度爆炸)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录日志
            total_loss += loss.item()
            total_lp_loss += loss_lp.item()
            total_cl_loss += loss_cl.item()
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}", 
                'LP': f"{loss_lp.item():.4f}", 
                'CL': f"{loss_cl.item():.4f}"
            })
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, epoch_idx: int):
        """简单的验证集评估 (计算 MRR 等指标的骨架)"""
        self.model.eval()
        pbar = tqdm(self.valid_loader, desc=f"Epoch {epoch_idx} [Valid]")
        
        # 在实际打分时，我们通常只看 Link Prediction 的效果
        all_ranks = []
        
        for g, target_triples, target_embs, current_time in pbar:
            g = g.to(self.device)
            target_triples = target_triples.to(self.device)
            current_time = current_time.to(self.device)
            # 验证时可以不需要 target_embs（如果你不做验证集的对比学习）
            
            # 只需要用到 entity_embs 即可预测
            entity_embs = self.model.struct_encoder(g, current_time)
            rel_embs = self.model.struct_encoder.relation_emb.weight
            
            src_ids = target_triples[:, 0]
            rel_ids = target_triples[:, 1]
            dst_ids = target_triples[:, 2] # 真实的 Answer
            
            # 打分: [Num_Events, Num_Entities]
            scores = self.loss_fn._calc_distmult_score(entity_embs[src_ids], rel_embs[rel_ids], entity_embs)
            
            # 计算排名 (这里是一个简化的评测逻辑，实际中可能需要过滤掉训练集中出现过的 true facts，即 filtered-MRR)
            # 找到真实目标在候选者中的排名
            for i in range(len(dst_ids)):
                target_score = scores[i, dst_ids[i]].item()
                # 计算有多少个实体的得分超过了真实实体的得分
                rank = (scores[i] > target_score).sum().item() + 1
                all_ranks.append(rank)
                
        all_ranks = np.array(all_ranks)
        mrr = np.mean(1.0 / all_ranks)
        hits_1 = np.mean(all_ranks <= 1)
        hits_3 = np.mean(all_ranks <= 3)
        hits_10 = np.mean(all_ranks <= 10)
        
        print(f"Validation Results: MRR: {mrr:.4f} | Hits@1: {hits_1:.4f} | Hits@3: {hits_3:.4f} | Hits@10: {hits_10:.4f}")
        return mrr

    def train(self, num_epochs: int):
        best_mrr = 0.0
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            print(f"--- Epoch {epoch} finished. Avg Train Loss: {train_loss:.4f} ---")
            
            val_mrr = self.evaluate(epoch)
            
            # 保存最佳模型
            if val_mrr > best_mrr:
                best_mrr = val_mrr
                torch.save(self.model.state_dict(), "best_tkg_model.pt")
                print("-> Checkpoint Saved!")