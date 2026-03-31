import torch
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from typing import List, Tuple, Optional, Dict, Any
import warnings

class EventClustering:
    """
    事件聚类分析（训练后使用）
    对训练好的事件嵌入进行聚类分析
    """
    
    def __init__(self, n_clusters: int = 20,
                 clustering_method: str = "kmeans",
                 random_state: int = 42):
        """
        初始化事件聚类器
        
        Args:
            n_clusters: 聚类数量
            clustering_method: 聚类方法 ("kmeans", "minibatch_kmeans", "spectral")
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.random_state = random_state
        
        # 聚类器
        self.clusterer = None
        self.cluster_centers = None
        self.cluster_labels = None
        
        # 初始化聚类器
        self._init_clusterer()
    
    def _init_clusterer(self):
        """初始化聚类器"""
        if self.clustering_method == "kmeans":
            self.clusterer = KMeans(
                n_clusters=self.n_clusters,
                n_init=10,
                random_state=self.random_state,
                verbose=0
            )
        elif self.clustering_method == "minibatch_kmeans":
            self.clusterer = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=1000,
                n_init=3,
                random_state=self.random_state,
                verbose=0
            )
        else:
            raise ValueError(f"Unsupported clustering method: {self.clustering_method}")
    
    def fit(self, event_embeddings: torch.Tensor) -> np.ndarray:
        """
        拟合事件聚类
        
        Args:
            event_embeddings: 事件嵌入 [n_events, hidden_dim]
            
        Returns:
            聚类标签 [n_events]
        """
        # 转换为numpy数组
        embeddings_np = event_embeddings.detach().cpu().numpy()
        
        # 拟合聚类器
        self.clusterer.fit(embeddings_np)
        
        # 保存结果
        self.cluster_labels = self.clusterer.labels_
        self.cluster_centers = torch.tensor(
            self.clusterer.cluster_centers_,
            device=event_embeddings.device
        )
        
        # 计算聚类质量
        self._compute_cluster_quality(embeddings_np)
        
        return self.cluster_labels
    
    def predict(self, event_embeddings: torch.Tensor, 
                soft_assignment: bool = False,
                temperature: float = 1.0) -> torch.Tensor:
        """
        预测新事件的聚类分配
        
        Args:
            event_embeddings: 事件嵌入 [n_events, hidden_dim]
            soft_assignment: 是否返回软分配
            temperature: 软分配的温度参数
            
        Returns:
            聚类分配（硬分配或软分配）
        """
        if self.cluster_centers is None:
            raise ValueError("Clustering model not fitted. Call fit() first.")
        
        # 计算到聚类中心的距离
        distances = torch.cdist(event_embeddings, self.cluster_centers)  # [n_events, n_clusters]
        
        if soft_assignment:
            # 软分配：使用softmax将距离转换为概率
            probs = torch.softmax(-distances / temperature, dim=1)
            return probs
        else:
            # 硬分配：选择最近的聚类中心
            labels = torch.argmin(distances, dim=1)
            return labels
    
    def get_topic_centers(self) -> torch.Tensor:
        """获取主题中心（聚类中心）"""
        if self.cluster_centers is None:
            raise ValueError("Clustering model not fitted. Call fit() first.")
        return self.cluster_centers
    
    def get_event_topics(self, event_embeddings: torch.Tensor,
                        top_k: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取事件的top-k主题分配
        
        Args:
            event_embeddings: 事件嵌入
            top_k: 返回的主题数量
            
        Returns:
            (主题ID, 主题概率)
        """
        # 计算软分配
        topic_probs = self.predict(event_embeddings, soft_assignment=True)
        
        # 获取top-k主题
        topk_probs, topk_indices = torch.topk(topic_probs, k=top_k, dim=1)
        
        return topk_indices, topk_probs
    
    def _compute_cluster_quality(self, embeddings_np: np.ndarray):
        """计算聚类质量指标"""
        try:
            # 轮廓系数（仅当有多个聚类且样本足够时）
            if len(set(self.cluster_labels)) > 1 and len(embeddings_np) > 10:
                silhouette_avg = silhouette_score(embeddings_np, self.cluster_labels)
                self.silhouette_score = silhouette_avg
            else:
                self.silhouette_score = None
            
            # Calinski-Harabasz指数
            if len(set(self.cluster_labels)) > 1:
                ch_score = calinski_harabasz_score(embeddings_np, self.cluster_labels)
                self.calinski_harabasz_score = ch_score
            else:
                self.calinski_harabasz_score = None
            
            # 聚类大小分布
            unique, counts = np.unique(self.cluster_labels, return_counts=True)
            self.cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
            
        except Exception as e:
            warnings.warn(f"Failed to compute cluster quality metrics: {e}")
            self.silhouette_score = None
            self.calinski_harabasz_score = None
            self.cluster_sizes = {}
    
    def get_cluster_statistics(self) -> Dict[str, Any]:
        """获取聚类统计信息"""
        return {
            'n_clusters': self.n_clusters,
            'method': self.clustering_method,
            'silhouette_score': self.silhouette_score,
            'calinski_harabasz_score': self.calinski_harabasz_score,
            'cluster_sizes': self.cluster_sizes,
            'num_events': sum(self.cluster_sizes.values()) if self.cluster_sizes else 0
        }