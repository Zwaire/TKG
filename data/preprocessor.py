import os
import numpy as np
from typing import Dict, List, Tuple, Optional

class ICEWS14Preprocessor:
    """ICEWS14数据集预处理类"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        
    def load_mappings(self):
        """加载实体和关系的映射"""
        entity_file = os.path.join(self.data_path, "entity2id.txt")
        with open(entity_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:]: 
                entity, eid = line.strip().split('\t')
                eid = int(eid)
                self.entity2id[entity] = eid
                self.id2entity[eid] = entity

        relation_file = os.path.join(self.data_path, "relation2id.txt")
        with open(relation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines[:]:
                relation, rid = line.strip().split('\t')
                rid = int(rid)
                self.relation2id[relation] = rid
                self.id2relation[rid] = relation
        
        return len(self.entity2id), len(self.relation2id)
    
    def load_triples(self, file_name: str) -> np.ndarray:
        """加载三元组数据"""
        file_path = os.path.join(self.data_path, file_name)
        triples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    s, p, o, t = line.strip().split('\t')
                    triples.append([
                        int(s), int(p), int(o), int(t)
                    ])
        
        return np.array(triples)

    def split_by_time_windows(self, triples: np.ndarray, 
                             window_size: int = 5) -> List[np.ndarray]:
        """
        按时间窗口划分数据
        Args:
            triples: 形状为 [n, 4] 的数组，每行是 (s, p, o, t)
            window_size: 时间窗口大小（天数）
        Returns:
            划分后的三元组列表
        """
        if len(triples) == 0:
            return []
        
        sorted_idx = np.argsort(triples[:, 3])
        sorted_triples = triples[sorted_idx]
    
        min_time = int(sorted_triples[0, 3])
        max_time = int(sorted_triples[-1, 3])
        
        windows = []
        current_time = min_time
        
        while current_time <= max_time:
            window_end = current_time + window_size
            mask = (sorted_triples[:, 3] >= current_time) & (sorted_triples[:, 3] < window_end)
            window_triples = sorted_triples[mask]
            
            if len(window_triples) > 0:
                windows.append(window_triples)
            
            current_time = window_end
        
        return windows
    
    def get_statistics(self, triples: np.ndarray) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'num_triples': len(triples),
            'num_entities': len(self.entity2id),
            'num_relations': len(self.relation2id),
            'time_range': (triples[:, 3].min(), triples[:, 3].max()),
            'entities_per_relation': {},
            'relations_per_entity': {}
        }
        
        # 计算每个关系对应的实体数量
        for rid in range(stats['num_relations']):
            mask = triples[:, 1] == rid
            if mask.any():
                entities = set(triples[mask][:, 0].tolist() + triples[mask][:, 2].tolist())
                stats['entities_per_relation'][rid] = len(entities)
        
        # 计算每个实体参与的关系数量
        for eid in range(stats['num_entities']):
            mask = (triples[:, 0] == eid) | (triples[:, 2] == eid)
            if mask.any():
                relations = set(triples[mask][:, 1].tolist())
                stats['relations_per_entity'][eid] = len(relations)
        
        return stats
    
    def save_processed_data(self, output_path: str, train: np.ndarray, 
                          valid: np.ndarray, test: np.ndarray, logger):
        """保存处理后的数据"""
        os.makedirs(output_path, exist_ok=True)
        
        # 保存三元组
        np.save(os.path.join(output_path, 'train_triples.npy'), train)
        np.save(os.path.join(output_path, 'valid_triples.npy'), valid)
        np.save(os.path.join(output_path, 'test_triples.npy'), test)
        
        # 保存映射
        import json
        with open(os.path.join(output_path, 'entity2id.json'), 'w') as f:
            json.dump(self.entity2id, f)
        with open(os.path.join(output_path, 'relation2id.json'), 'w') as f:
            json.dump(self.relation2id, f)
        
        logger.info(f"Processed data saved to {output_path}")

class ICEWS18Preprocessor:
    """ICEWS18数据集预处理类"""
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        
    def load_mappings(self):
        """加载实体和关系的映射"""
        entity_file = os.path.join(self.data_path, "entity2id.txt")
        with open(entity_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines: 
                if line.strip():
                    # 兼容不同类型的分隔符
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        parts = line.strip().split()
                    entity, eid = parts[0], int(parts[1])
                    self.entity2id[entity] = eid
                    self.id2entity[eid] = entity

        relation_file = os.path.join(self.data_path, "relation2id.txt")
        with open(relation_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip():
                    parts = line.strip().split('\t')
                    if len(parts) != 2:
                        parts = line.strip().split()
                    relation, rid = parts[0], int(parts[1])
                    self.relation2id[relation] = rid
                    self.id2relation[rid] = relation
        
        return len(self.entity2id), len(self.relation2id)
    
    def load_triples(self, file_name: str) -> np.ndarray:
        """加载五元组数据 (s, p, o, time_step, placeholder)"""
        file_path = os.path.join(self.data_path, file_name)
        triples = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    # 使用无参 split() 自动处理空格或制表符
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        s = int(parts[0])
                        p = int(parts[1])
                        o = int(parts[2])
                        t = int(parts[3])
                        # 处理 ICEWS18 特有的结尾 0 占位符
                        placeholder = int(parts[4]) if len(parts) > 4 else 0
                        
                        triples.append([s, p, o, t, placeholder])
        
        return np.array(triples)

    def split_by_time_windows(self, triples: np.ndarray, 
                             window_size: int = 5) -> List[np.ndarray]:
        """
        按时间窗口划分数据
        Args:
            triples: 形状为 [n, 5] 的数组，每行是 (s, p, o, t, placeholder)
            window_size: 时间窗口大小（天数或时间步数）
        Returns:
            划分后的三元组列表
        """
        if len(triples) == 0:
            return []
        
        # 时间维度仍然在索引 3 的位置
        sorted_idx = np.argsort(triples[:, 3])
        sorted_triples = triples[sorted_idx]
    
        min_time = int(sorted_triples[0, 3])
        max_time = int(sorted_triples[-1, 3])
        
        windows = []
        current_time = min_time
        
        while current_time <= max_time:
            window_end = current_time + window_size
            mask = (sorted_triples[:, 3] >= current_time) & (sorted_triples[:, 3] < window_end)
            window_triples = sorted_triples[mask]
            
            if len(window_triples) > 0:
                windows.append(window_triples)
            
            current_time = window_end
        
        return windows
    
    def get_statistics(self, triples: np.ndarray) -> Dict:
        """获取数据集统计信息"""
        stats = {
            'num_triples': len(triples),
            'num_entities': len(self.entity2id),
            'num_relations': len(self.relation2id),
            'time_range': (triples[:, 3].min(), triples[:, 3].max()),
            'entities_per_relation': {},
            'relations_per_entity': {}
        }
        
        # 计算每个关系对应的实体数量
        for rid in range(stats['num_relations']):
            mask = triples[:, 1] == rid
            if mask.any():
                entities = set(triples[mask][:, 0].tolist() + triples[mask][:, 2].tolist())
                stats['entities_per_relation'][rid] = len(entities)
        
        # 计算每个实体参与的关系数量
        for eid in range(stats['num_entities']):
            mask = (triples[:, 0] == eid) | (triples[:, 2] == eid)
            if mask.any():
                relations = set(triples[mask][:, 1].tolist())
                stats['relations_per_entity'][eid] = len(relations)
        
        return stats
    
    def save_processed_data(self, output_path: str, train: np.ndarray, 
                          valid: np.ndarray, test: np.ndarray):
        """保存处理后的数据"""
        os.makedirs(output_path, exist_ok=True)
        
        # 保存三元组 (npy 格式现在会包含 [n, 5] 的数据)
        np.save(os.path.join(output_path, 'train_triples.npy'), train)
        np.save(os.path.join(output_path, 'valid_triples.npy'), valid)
        np.save(os.path.join(output_path, 'test_triples.npy'), test)
        
        # 保存映射
        import json
        with open(os.path.join(output_path, 'entity2id.json'), 'w', encoding='utf-8') as f:
            json.dump(self.entity2id, f, ensure_ascii=False, indent=2)
        with open(os.path.join(output_path, 'relation2id.json'), 'w', encoding='utf-8') as f:
            json.dump(self.relation2id, f, ensure_ascii=False, indent=2)
        
        print(f"Processed data saved to {output_path}")