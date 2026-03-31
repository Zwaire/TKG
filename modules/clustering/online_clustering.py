import torch
import numpy as np
from typing import Optional, Tuple
from .event_clustering import EventClustering


class ClusterConsistencyLoss:
    """简单的聚类一致性损失：鼓励软分配更确定（熵最小化）"""
    def __init__(self, temperature: float = 0.1):
        self.temperature = temperature

    def __call__(self, embeddings: torch.Tensor, cluster_probs: torch.Tensor) -> torch.Tensor:
        # cluster_probs: [N, K]
        probs = cluster_probs + 1e-8
        entropy = - (probs * torch.log(probs)).sum(dim=1).mean()
        return entropy


class OnlineClustering:
    """在线聚类模块的轻量封装，使用已有 EventClustering 做拟合/预测。

    提供 `fit(embeddings, timestamps)` 和 `predict(embeddings, soft_assignment)` 接口。
    """
    def __init__(self, n_clusters: int = 20, update_frequency: int = 10,
                 clustering_method: str = 'minibatch_kmeans', use_gpu: bool = False):
        self.n_clusters = n_clusters
        self.update_frequency = update_frequency
        self.clustering_method = clustering_method
        self.use_gpu = use_gpu

        self.clusterer = EventClustering(n_clusters=n_clusters, clustering_method=clustering_method)

    def fit(self, embeddings: torch.Tensor, timestamps: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, torch.Tensor]:
        labels = self.clusterer.fit(embeddings)
        centers = self.clusterer.get_topic_centers()
        return labels, centers

    def predict(self, embeddings: torch.Tensor, soft_assignment: bool = False, temperature: float = 1.0):
        return self.clusterer.predict(embeddings, soft_assignment=soft_assignment, temperature=temperature)
