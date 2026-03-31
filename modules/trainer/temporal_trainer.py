import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Any
import numpy as np
from tqdm import tqdm
from pathlib import Path

from modules.clustering.contrastive_learning import (
    TemporalContrastiveLoss,
    MultiViewContrastiveLoss,
    MemoryBankContrastiveLoss,
)
from modules.clustering.online_clustering import (
    OnlineClustering,
    ClusterConsistencyLoss
)
from .loss_functions import (
    MultiTaskLoss,
    TemporalReconstructionLoss,
    TemporalConsistencyLoss,
    OrthogonalityConstraint,
    AdaptiveLossWeighting,
)

from modules.data_module import TemporalKGDataset

class TemporalKGTrainer:
    """
    时间感知知识图谱训练器
    支持对比学习和在线聚类的联合训练
    """
    
    def __init__(self, model: nn.Module,
                 clustering_module: OnlineClustering,
                 config: Dict,
                 device: str = "cuda"):
        """
        初始化训练器
        
        Args:
            model: 嵌入模型
            clustering_module: 聚类模块
            config: 训练配置
            device: 训练设备
        """
        # normalize device to torch.device
        self.device = torch.device(device)
        self.model: Any = model.to(self.device)
        self.clustering_module = clustering_module
        self.config = config
        
        # 训练配置
        self.history_window = config['data']['history_window']
        self.num_epochs = config['training']['num_epochs']
        self.batch_size = config['training']['batch_size']
        self.max_samples_per_epoch = config['training'].get('max_samples_per_epoch', None)
        self.global_step = 0
        
        # 损失函数
        self._init_loss_functions()
        self.previous_embeddings: Optional[torch.Tensor] = None
        
        # 优化器
        self._init_optimizer()
        
        # 训练状态
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = {
            'losses': [],
            'metrics': [],
            'cluster_quality': []
        }
        
        # 时间步状态
        self.time_step = 0
        self.previous_graph_repr: Optional[torch.Tensor] = None

        checkpoint_cfg = self.config['training'].get('checkpoint', {})
        self.checkpoint_dir = Path(checkpoint_cfg.get('dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def _init_loss_functions(self):
        """初始化损失函数"""
        loss_config = self.config['training']['loss_weights']
        contrastive_config = self.config['model'].get('clustering', {}).get('contrastive', {})
        
        # 多任务损失
        self.multitask_loss = MultiTaskLoss(loss_config)
        self.contrastive_temperature = float(contrastive_config.get('temperature', 0.1))
        
        # 保留原有组件，便于后续扩展
        self.contrastive_loss = MultiViewContrastiveLoss(
            temperature=contrastive_config.get('temperature', 0.1),
            view_weights={
                'structural_structural': 1.0,
                'semantic_semantic': 1.0,
                'structural_semantic': 0.8,
                'semantic_structural': 0.8
            }
        )
        
        # 记忆银行对比损失
        if contrastive_config.get('use_memory_bank', False):
            hidden_dim = self.config['model']['embedding']['fusion'].get('output_dim', 256)
            self.memory_bank_loss = MemoryBankContrastiveLoss(
                hidden_dim=hidden_dim,
                memory_size=contrastive_config.get('memory_bank_size', 8192),
                temperature=contrastive_config.get('temperature', 0.1)
            )
        
        # 重建损失
        self.reconstruction_loss = TemporalReconstructionLoss(margin=1.0)
        
        # 时间一致性损失
        self.temporal_loss = TemporalConsistencyLoss(
            temporal_smoothing=0.1
        )
        
        # 聚类一致性损失
        self.cluster_consistency_loss = ClusterConsistencyLoss(
            temperature=0.1
        )
        
        # 正交性约束
        self.orthogonality_loss = OrthogonalityConstraint(weight=0.01)
        
        # 自适应损失权重
        if self.config['training'].get('adaptive_weighting', False):
            self.adaptive_weighting = AdaptiveLossWeighting(
                initial_weights=loss_config,
                warmup_epochs=self.config['training']['temporal'].get('warmup_epochs', 10)
            )
    
    def _init_optimizer(self):
        """初始化优化器和学习率调度器"""
        optimizer_config = self.config['training']['optimizer']
        
        # 优化器
        if optimizer_config['type'] == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.0001)
            )
        elif optimizer_config['type'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=optimizer_config.get('weight_decay', 0.01)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_config['type']}")
        
        # 学习率调度器
        if optimizer_config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=1e-6
            )
        elif optimizer_config.get('scheduler') == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=5,
                factor=0.5
            )
        else:
            self.scheduler = None

    def _prepare_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """将 dataset sample 转为训练可用张量。"""
        history_graph = sample['history_graph'].to(self.device)

        triples_np = sample.get('query_triples')
        if triples_np is None or len(triples_np) == 0:
            triples = torch.empty((0, 3), dtype=torch.long, device=self.device)
            timestamps = torch.empty((0,), dtype=torch.float, device=self.device)
        else:
            triples = torch.tensor(triples_np[:, :3], dtype=torch.long, device=self.device)
            timestamps = torch.tensor(triples_np[:, 3], dtype=torch.float, device=self.device)

        texts = sample.get('query_texts', []) or []

        return {
            'history_graph': history_graph,
            'triples': triples,
            'timestamps': timestamps,
            'texts': texts,
        }

    def _compute_event_repr(self, entity_embeddings: torch.Tensor, triples: torch.Tensor):
        """构造事件级结构表示 e = h + r - t。"""
        if triples.numel() == 0:
            empty = torch.empty((0, entity_embeddings.size(1)), device=entity_embeddings.device)
            return empty, empty, empty, empty

        heads = entity_embeddings[triples[:, 0]]
        rels = self.model.get_relation_embedding(triples[:, 1])
        tails = entity_embeddings[triples[:, 2]]
        event_repr = heads + rels - tails
        return event_repr, heads, rels, tails

    def _compute_reconstruction_loss(self, entity_embeddings: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        """TransE 风格链接预测重建损失。"""
        if triples.numel() == 0:
            return torch.tensor(0.0, device=entity_embeddings.device)

        _, heads, rels, tails = self._compute_event_repr(entity_embeddings, triples)
        pos_scores = -torch.norm(heads + rels - tails, p=2, dim=1)

        num_entities = entity_embeddings.size(0)
        neg_tail_ids = torch.randint(0, num_entities, (triples.size(0),), device=entity_embeddings.device)
        neg_tails = entity_embeddings[neg_tail_ids]
        neg_scores = -torch.norm(heads + rels - neg_tails, p=2, dim=1)

        return self.reconstruction_loss(pos_scores, neg_scores.unsqueeze(1))

    def _compute_contrastive_loss(
        self,
        event_repr: torch.Tensor,
        texts: list,
    ) -> torch.Tensor:
        """结构事件表示与文本表示的双向 InfoNCE。"""
        if event_repr.numel() == 0 or not texts:
            return torch.tensor(0.0, device=self.device)

        valid_pairs = [
            (idx, txt)
            for idx, txt in enumerate(texts)
            if isinstance(txt, str) and txt.strip()
        ]
        if len(valid_pairs) < 2:
            return torch.tensor(0.0, device=self.device)

        indices = [i for i, _ in valid_pairs]
        valid_texts = [t for _, t in valid_pairs]
        struct_view = event_repr[indices]
        text_view = self.model.encode_semantic(valid_texts)

        n = min(struct_view.size(0), text_view.size(0))
        if n < 2:
            return torch.tensor(0.0, device=self.device)

        struct_view = F.normalize(struct_view[:n], p=2, dim=1)
        text_view = F.normalize(text_view[:n], p=2, dim=1)

        logits = torch.matmul(struct_view, text_view.T) / self.contrastive_temperature
        labels = torch.arange(n, device=self.device)

        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_s2t + loss_t2s)

    def _compute_temporal_consistency_loss(self, current_embeddings: torch.Tensor) -> torch.Tensor:
        """相邻时间步实体表示平滑约束。"""
        if self.previous_embeddings is None:
            return torch.tensor(0.0, device=self.device)

        if self.previous_embeddings.shape != current_embeddings.shape:
            return torch.tensor(0.0, device=self.device)

        time_gap = torch.ones(current_embeddings.size(0), device=self.device)
        return self.temporal_loss(current_embeddings, self.previous_embeddings, time_gap)

    def _compute_clustering_losses(self, entity_embeddings: torch.Tensor):
        """在线聚类损失和正交约束。"""
        update_freq = max(1, int(self.clustering_module.update_frequency))
        if self.global_step % update_freq != 0:
            z = torch.tensor(0.0, device=self.device)
            return z, z

        if entity_embeddings.size(0) <= self.clustering_module.n_clusters:
            z = torch.tensor(0.0, device=self.device)
            return z, z

        cluster_labels, cluster_centers = self.clustering_module.fit(entity_embeddings.detach())
        cluster_probs = self.clustering_module.predict(entity_embeddings.detach(), soft_assignment=True)
        cluster_loss = self.cluster_consistency_loss(entity_embeddings, cluster_probs)
        reg_loss = self.orthogonality_loss(cluster_centers)

        self._record_cluster_quality(cluster_labels, entity_embeddings)
        return cluster_loss, reg_loss

    def _run_single_step(self, sample: Dict[str, Any], training: bool = True) -> Dict[str, float]:
        batch = self._prepare_sample(sample)
        history_graph = batch['history_graph']
        triples = batch['triples']
        texts = batch['texts']

        entity_embeddings = self.model.encode_structural(history_graph)
        event_repr, _, _, _ = self._compute_event_repr(entity_embeddings, triples)

        losses = {
            'reconstruction': self._compute_reconstruction_loss(entity_embeddings, triples),
            'contrastive': self._compute_contrastive_loss(event_repr, texts),
            'temporal': self._compute_temporal_consistency_loss(entity_embeddings),
        }

        if training:
            cluster_loss, reg_loss = self._compute_clustering_losses(entity_embeddings)
        else:
            cluster_loss = torch.tensor(0.0, device=self.device)
            reg_loss = torch.tensor(0.0, device=self.device)
        losses['clustering'] = cluster_loss
        losses['regularization'] = reg_loss

        total_loss = self.multitask_loss(losses)

        if training:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        self.previous_embeddings = entity_embeddings.detach()
        if training:
            self.global_step += 1

        return {
            'total_loss': float(total_loss.item()),
            'reconstruction_loss': float(losses['reconstruction'].item()),
            'contrastive_loss': float(losses['contrastive'].item()),
            'temporal_loss': float(losses['temporal'].item()),
            'clustering_loss': float(losses['clustering'].item()),
        }
    
    def train_epoch(self, dataset: TemporalKGDataset, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            训练统计信息
        """
        self.model.train()
        epoch_stats = {
            'total_loss': 0.0,
            'reconstruction_loss': 0.0,
            'contrastive_loss': 0.0,
            'temporal_loss': 0.0,
            'clustering_loss': 0.0
        }
        
        self.previous_embeddings = None
        if self.max_samples_per_epoch is None:
            epoch_samples = len(dataset)
        else:
            epoch_samples = min(len(dataset), int(self.max_samples_per_epoch))

        sample_count = max(1, epoch_samples)

        pbar = tqdm(range(epoch_samples), desc=f"Train Epoch {epoch + 1}/{self.num_epochs}")
        for idx in pbar:
            sample = dataset.get(idx)
            step_stats = self._run_single_step(sample, training=True)

            epoch_stats['total_loss'] += step_stats['total_loss']
            epoch_stats['reconstruction_loss'] += step_stats['reconstruction_loss']
            epoch_stats['contrastive_loss'] += step_stats['contrastive_loss']
            epoch_stats['temporal_loss'] += step_stats['temporal_loss']
            epoch_stats['clustering_loss'] += step_stats['clustering_loss']

            pbar.set_postfix({
                'loss': f"{step_stats['total_loss']:.4f}",
                'lr': self.optimizer.param_groups[0]['lr'],
            })

        for key in epoch_stats:
            epoch_stats[key] /= sample_count
        
        # 更新学习率
        if self.scheduler is not None:
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(epoch_stats['total_loss'])
            else:
                self.scheduler.step()
        
        # 更新自适应权重
        if hasattr(self, 'adaptive_weighting'):
            self.adaptive_weighting.update_epoch(epoch)
        
        return epoch_stats
    
    def _record_cluster_quality(self, cluster_labels: np.ndarray,
                              embeddings: torch.Tensor):
        """记录聚类质量指标"""
        # 计算聚类内距离
        intra_cluster_dist = []
        
        for cluster_id in np.unique(cluster_labels):
            mask = cluster_labels == cluster_id
            if mask.sum() > 1:
                cluster_embeddings = embeddings[mask]
                center = cluster_embeddings.mean(dim=0)
                distances = torch.norm(cluster_embeddings - center, p=2, dim=1)
                intra_cluster_dist.append(distances.mean().item())
        
        avg_intra_dist = np.mean(intra_cluster_dist) if intra_cluster_dist else 0
        
        # 记录
        self.train_history['cluster_quality'].append({
            'epoch': self.current_epoch,
            'num_clusters': len(np.unique(cluster_labels)),
            'avg_intra_distance': avg_intra_dist,
            'cluster_sizes': np.bincount(cluster_labels)
        })
    
    def train_temporal_sequence(self, train_dataset : TemporalKGDataset,
                               valid_dataset=None,
                               test_dataset=None):
        """
        时间顺序训练
        
        Args:
            train_dataset: 训练数据集
            valid_dataset: 验证数据集
            test_dataset: 测试数据集
        """
        checkpoint_cfg = self.config['training'].get('checkpoint', {})
        eval_frequency = int(checkpoint_cfg.get('eval_frequency', 1))
        save_frequency = int(checkpoint_cfg.get('save_frequency', 1))

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            train_stats = self.train_epoch(train_dataset, epoch)
            self.train_history['losses'].append({
                'epoch': epoch,
                **train_stats,
            })

            if valid_dataset is not None and (epoch + 1) % max(1, eval_frequency) == 0:
                val_metrics = self.validate(valid_dataset)
                self.train_history['metrics'].append({
                    'epoch': epoch,
                    **val_metrics,
                })

                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    self.save_checkpoint(f"best_model_epoch{epoch + 1}.pt")

            if (epoch + 1) % max(1, save_frequency) == 0:
                self.save_checkpoint(f"checkpoint_epoch{epoch + 1}.pt")

    def validate(self, valid_dataset: TemporalKGDataset) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            valid_dataset: 验证数据集
        Returns:
            验证指标
        """
        self.model.eval()

        self.previous_embeddings = None
        if self.max_samples_per_epoch is None:
            val_samples = len(valid_dataset)
        else:
            val_samples = min(len(valid_dataset), int(self.max_samples_per_epoch))

        total_loss = 0.0
        total_samples = max(1, val_samples)

        with torch.no_grad():
            for idx in tqdm(range(val_samples), desc="Validation"):
                sample = valid_dataset.get(idx)
                stats = self._run_single_step(sample, training=False)
                total_loss += stats['total_loss']

        avg_loss = total_loss / total_samples

        return {'loss': avg_loss}
    
    def save_checkpoint(self, filename: str):
        """保存检查点"""
        path = self.checkpoint_dir / filename
        checkpoint = {
            'epoch': self.current_epoch,
            'time_step': self.time_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }

        torch.save(checkpoint, str(path))
    
    def load_checkpoint(self, filename: str):
        """加载检查点"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.time_step = checkpoint['time_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_history = checkpoint['train_history']