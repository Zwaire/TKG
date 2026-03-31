import torch
from torch.utils.data import Dataset
import numpy as np
import dgl
from pathlib import Path
from collections import defaultdict

class TKGSnapshotDataset(Dataset):
    """
    时序知识图谱快照数据集 (滑动窗口机制)
    """
    def __init__(self, data_dir: str, cache_dir: str, split: str, 
                 num_entities: int, history_window: int = 10):
        super().__init__()
        self.history_window = history_window
        self.num_entities = num_entities
        
        # 1. 加载结构化三元组 [N, 4] -> (src, rel, dst, time)
        triples_path = Path(data_dir) / "processed" / f"{split}_triples.npy"
        self.triples = torch.from_numpy(np.load(triples_path)).long()
        
        # 2. 加载离线提取的 BERT 语义特征 [N, 768]
        emb_path = Path(cache_dir) / f"{split}_semantic_emb.pt"
        self.semantic_embs = torch.load(emb_path)
        
        assert len(self.triples) == len(self.semantic_embs), \
            f"数据不匹配: 三元组数量 {len(self.triples)} != 语义特征数量 {len(self.semantic_embs)}"
            
        # 3. 按时间戳排序 (确保时间绝对递增)
        sorted_indices = torch.argsort(self.triples[:, 3])
        self.triples = self.triples[sorted_indices]
        self.semantic_embs = self.semantic_embs[sorted_indices]
        
        # 获取所有唯一的时间戳
        self.unique_times = torch.unique(self.triples[:, 3], sorted=True)
        
        # 4. 构建 时间戳 -> 索引 的映射字典，极大地加速查询
        # 【修复点】：使用 defaultdict，并强制转换为 Python 原生 int
        self.time_to_indices = defaultdict(list)
        # 将 PyTorch 张量转化为标准的 Python list 进行遍历，杜绝底层类型污染
        for i, t in enumerate(self.triples[:, 3].tolist()):
            self.time_to_indices[int(t)].append(i)
            
        print(f"[{split.upper()}] Dataset Loaded: {len(self.unique_times)} unique timestamps.")

    def __len__(self):
        # 必须预留前 history_window 个时间步作为历史背景
        return len(self.unique_times) - self.history_window

    def __getitem__(self, idx):
        # 目标预测时间的索引
        target_time_idx = idx + self.history_window
        # 【修复点】：强制转化为原生 int
        target_time = int(self.unique_times[target_time_idx].item())
        current_time_tensor = torch.tensor([target_time], dtype=torch.float32)
        
        # 1. 提取目标时刻 (Target) 的事件和语义特征
        target_indices = self.time_to_indices[target_time]
        target_triples = self.triples[target_indices]
        target_embs = self.semantic_embs[target_indices]
        
        # 2. 收集历史窗口 (History Window) 内的所有事件索引
        hist_indices = []
        for i in range(idx, target_time_idx):
            # 【修复点】：强制转化为原生 int
            t_val = int(self.unique_times[i].item())
            hist_indices.extend(self.time_to_indices[t_val])
            
        hist_triples = self.triples[hist_indices]
        
        # 3. 构建高效率的 DGL 历史图快照
        src = hist_triples[:, 0]
        dst = hist_triples[:, 2]
        
        g = dgl.graph((src, dst), num_nodes=self.num_entities)
        
        # 将边特征写入图中
        g.edata['etype'] = hist_triples[:, 1]
        g.edata['etime'] = hist_triples[:, 3].float() # 转为 float 以便计算衰减
        
        return g, target_triples, target_embs, current_time_tensor

# --- 为了解决 PyTorch DataLoader 默认的 batch 拼接问题，需要自定义 collate_fn ---
def tkg_collate_fn(batch):
    """
    因为我们按时间步迭代，每个 item 本身就是一个时间步的所有数据（相当于一个自然的 Batch），
    所以直接返回 batch[0] 即可。
    """
    return batch[0]