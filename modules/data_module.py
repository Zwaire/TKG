import torch
from torch_geometric.data import Data, Dataset
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
import json

class TemporalSnapshot:
    """时间快照类，表示一个时间窗口内的知识图谱"""
    
    def __init__(
        self,
        triples: np.ndarray,
        timestamp: int,
        snapshot_id: int,
        triple_indices: Optional[np.ndarray] = None,
        entity_mapping: Optional[Dict] = None,
    ):
        """
        Args:
            triples: 形状为 [n_triples, 4] 的数组，(s, p, o, t)
            timestamp: 快照的时间戳
            snapshot_id: 快照ID
            entity_mapping: 全局实体ID到局部ID的映射
        """
        self.timestamp = timestamp
        self.snapshot_id = snapshot_id
        self.triples = triples
        self.triple_indices = triple_indices if triple_indices is not None else np.array([], dtype=np.int64)
        
        # 构建边索引和边类型
        if len(triples) > 0:
            self.edge_index = torch.tensor(triples[:, [0, 2]].T, dtype=torch.long)  # [2, E]
            self.edge_type = torch.tensor(triples[:, 1], dtype=torch.long)  # [E]
            self.edge_time = torch.tensor(triples[:, 3], dtype=torch.float)  # [E]
            
            # 获取快照中出现的所有实体
            self.entities = torch.unique(torch.cat([
                self.edge_index[0], self.edge_index[1]
            ]))
            
            # 全局ID到局部ID的映射
            if entity_mapping is not None:
                self.local_to_global = self.entities.tolist()
                self.global_to_local = {
                    global_id: local_id 
                    for local_id, global_id in enumerate(self.local_to_global)
                }
                
                # 转换边索引为局部ID
                self.edge_index_local = torch.stack([
                    torch.tensor([self.global_to_local[g.item()] 
                                for g in self.edge_index[0]]),
                    torch.tensor([self.global_to_local[g.item()] 
                                for g in self.edge_index[1]])
                ])
            else:
                self.edge_index_local = self.edge_index
        else:
            self.edge_index = torch.empty((2, 0), dtype=torch.long)
            self.edge_type = torch.empty((0,), dtype=torch.long)
            self.edge_time = torch.empty((0,), dtype=torch.float)
            self.entities = torch.empty((0,), dtype=torch.long)
            self.edge_index_local = self.edge_index
    
    def to_pyg_data(self) -> Data:
        """转换为PyG Data对象"""
        return Data(
            x=None,  # 节点特征将在嵌入模块中添加
            edge_index=self.edge_index_local,
            edge_type=self.edge_type,
            edge_time=self.edge_time,
            timestamp=torch.tensor([self.timestamp], dtype=torch.float),
            global_node_ids=self.entities,
            num_nodes=len(self.entities)
        )

class TemporalKGDataset(Dataset):
    """时态知识图谱数据集"""
    
    def __init__(self, data_path: str, 
                 history_window: int = 5,
                 time_granularity: str = 'day',
                 split: str = 'train',
                 text_cache_dir: Optional[str] = None,
                 transform=None):
        """
        Args:
            data_path: 数据路径
            history_window: 历史窗口大小
            time_granularity: 时间粒度 ('day', 'week', 'month')
            split: 数据集划分 ('train', 'valid', 'test')
            transform: 可选的数据转换
        """
        super().__init__(transform=transform)
        self.data_path = data_path
        self.history_window = history_window
        self.time_granularity = time_granularity
        self.split = split
        self.text_cache_dir = text_cache_dir
        
        # 加载数据
        self.triples = self._load_triples()
        self.num_entities = self._infer_num_entities()
        self.texts = self._load_texts()
        
        # 构建时间快照
        self.snapshots = self._build_snapshots()
        
        # 构建快照索引
        self.snapshot_indices = self._build_snapshot_indices()
    
    def _load_triples(self) -> np.ndarray:
        """加载三元组数据"""
        file_map = {
            'train': 'train_triples.npy',
            'valid': 'valid_triples.npy', 
            'test': 'test_triples.npy'
        }
        
        file_name = file_map.get(self.split)
        if not file_name:
            raise ValueError(f"Invalid split: {self.split}")
        
        file_path = os.path.join(self.data_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        return np.load(file_path)

    def _infer_num_entities(self) -> int:
        """推断实体总数，用于构建全局实体嵌入索引。"""
        entity_file = os.path.join(self.data_path, "entity2id.json")
        if os.path.exists(entity_file):
            with open(entity_file, "r", encoding="utf-8") as f:
                entity2id = json.load(f)
            return len(entity2id)

        if len(self.triples) == 0:
            return 0

        return int(max(self.triples[:, 0].max(), self.triples[:, 2].max()) + 1)

    def _load_texts(self) -> List[str]:
        """加载当前 split 的文本缓存，不存在时返回空列表。"""
        if self.text_cache_dir is None:
            return []

        text_path = os.path.join(self.text_cache_dir, f"{self.split}_texts.json")
        if not os.path.exists(text_path):
            return []

        with open(text_path, "r", encoding="utf-8") as f:
            texts = json.load(f)

        if not isinstance(texts, list):
            return []

        return texts
    
    def _build_snapshots(self) -> List[TemporalSnapshot]:
        """构建时间快照"""
        if len(self.triples) == 0:
            return []
        
        # 按时间戳稳定排序，保证文本索引对齐
        sorted_idx = np.argsort(self.triples[:, 3], kind='stable')
        sorted_triples = self.triples[sorted_idx]
        
        # 获取唯一时间戳
        unique_timestamps = np.unique(sorted_triples[:, 3])
        
        snapshots = []
        for idx, timestamp in enumerate(unique_timestamps):
            # 获取该时间戳的所有三元组
            mask = sorted_triples[:, 3] == timestamp
            timestamp_triples = sorted_triples[mask]
            timestamp_indices = sorted_idx[mask]
            
            # 创建快照
            snapshot = TemporalSnapshot(
                triples=timestamp_triples,
                timestamp=int(timestamp),
                snapshot_id=idx,
                triple_indices=timestamp_indices,
            )
            snapshots.append(snapshot)
        
        return snapshots
    
    def _build_snapshot_indices(self) -> List[Tuple[int, int]]:
        """构建用于训练的快照索引 (current_idx, history_indices)"""
        indices = []
        
        for current_idx in range(self.history_window, len(self.snapshots)):
            # 获取历史快照的索引
            history_indices = list(range(
                max(0, current_idx - self.history_window),
                current_idx
            ))
            
            # 确保至少有一个历史快照
            if history_indices:
                indices.append((current_idx, history_indices))
        
        return indices
    
    def get_history_subgraph(self, query_time_idx: int) -> Data:
        """
        获取查询时间前的历史子图
        
        Args:
            query_time_idx: 查询时间的快照索引
            
        Returns:
            合并的历史子图
        """
        # 获取历史快照索引
        history_indices = list(range(
            max(0, query_time_idx - self.history_window),
            query_time_idx
        ))
        
        if not history_indices:
            # 如果没有历史快照，返回空图
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                edge_time=torch.empty((0,), dtype=torch.float),
                global_node_ids=torch.arange(self.num_entities, dtype=torch.long),
                num_nodes=self.num_entities
            )
        
        # 合并所有历史快照
        history_snapshots = [self.snapshots[i] for i in history_indices]
        
        # 收集所有的边
        all_edges = []
        all_edge_types = []
        all_edge_times = []
        all_entities = set()
        
        for snapshot in history_snapshots:
            if len(snapshot.edge_index) > 0:
                # 转换为列表形式以便处理
                edges = snapshot.edge_index.T.tolist()
                edge_types = snapshot.edge_type.tolist()
                edge_times = snapshot.edge_time.tolist()
                
                all_edges.extend(edges)
                all_edge_types.extend(edge_types)
                all_edge_times.extend(edge_times)
                
                # 添加实体
                all_entities.update(snapshot.entities.tolist())
        
        if not all_edges:
            return Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_type=torch.empty((0,), dtype=torch.long),
                edge_time=torch.empty((0,), dtype=torch.float),
                global_node_ids=torch.tensor(list(all_entities), dtype=torch.long),
                num_nodes=len(all_entities)
            )
        
        # 转换为张量
        edge_index = torch.tensor(all_edges, dtype=torch.long).T
        edge_type = torch.tensor(all_edge_types, dtype=torch.long)
        edge_time = torch.tensor(all_edge_times, dtype=torch.float)
        global_node_ids = torch.tensor(list(all_entities), dtype=torch.long)
        
        # 历史图直接使用全局实体ID，避免训练阶段再做局部-全局映射
        all_entity_ids = torch.arange(self.num_entities, dtype=torch.long)

        return Data(
            edge_index=edge_index,
            edge_type=edge_type,
            edge_time=edge_time,
            global_node_ids=all_entity_ids,
            num_nodes=self.num_entities
        )
    
    def get_snapshot_by_time(self, timestamp: int) -> Optional[TemporalSnapshot]:
        """根据时间戳获取快照"""
        for snapshot in self.snapshots:
            if snapshot.timestamp == timestamp:
                return snapshot
        return None
    
    def len(self) -> int:
        """返回数据集长度（快照索引对的数量）"""
        return len(self.snapshot_indices)
    
    def get_time_steps(self) -> List[Tuple[int, int]]:
        """返回快照时间步"""
        return self.snapshot_indices
    
    def get(self, idx: int) -> Dict:
        """获取第idx个样本"""
        current_idx, history_indices = self.snapshot_indices[idx]
        
        # 获取当前快照
        current_snapshot = self.snapshots[current_idx]
        
        # 获取历史子图
        history_graph = self.get_history_subgraph(current_idx)
        
        # 获取用于训练的三元组（当前快照的所有边）
        query_triples = current_snapshot.triples

        # 文本与当前快照中的三元组按原始索引对齐
        query_texts = []
        if self.texts:
            for raw_idx in current_snapshot.triple_indices.tolist():
                if 0 <= raw_idx < len(self.texts):
                    query_texts.append(self.texts[raw_idx])
                else:
                    query_texts.append("")
        
        return {
            'history_graph': history_graph,
            'current_snapshot': current_snapshot,
            'query_triples': query_triples,
            'query_texts': query_texts,
            'current_timestamp': current_snapshot.timestamp
        }
    
    def get_all_entities(self) -> torch.Tensor:
        """获取数据集中所有的实体ID"""
        all_entities = set()
        for snapshot in self.snapshots:
            all_entities.update(snapshot.entities.tolist())
        return torch.tensor(list(all_entities), dtype=torch.long)

class TemporalDataLoader:
    """时态知识图谱数据加载器"""
    
    def __init__(self, dataset: TemporalKGDataset, batch_size: int = 32,
                 shuffle: bool = True, num_workers: int = 4):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # 创建索引
        self.indices = list(range(len(dataset)))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_idx >= len(self.indices):
            # 重置迭代器
            if self.shuffle:
                np.random.shuffle(self.indices)
            self.current_idx = 0
            raise StopIteration
        
        # 获取当前批次
        batch_indices = self.indices[self.current_idx:
                                    self.current_idx + self.batch_size]
        batch_data = [self.dataset.get(idx) for idx in batch_indices]
        
        self.current_idx += self.batch_size
        
        # 这里可以添加批次处理逻辑
        return self._collate_fn(batch_data)
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """批次处理函数"""
        # 简单返回批次数据，具体处理逻辑根据模型需求定制
        return {
            'history_graphs': [item['history_graph'] for item in batch],
            'current_snapshots': [item['current_snapshot'] for item in batch],
            'query_triples': [item['query_triples'] for item in batch],
            'timestamps': [item['current_timestamp'] for item in batch]
        }