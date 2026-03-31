import yaml
from pathlib import Path
from modules.models.dual_stream_tkg import DualStreamTKG
from modules.data_pipeline.dataset import TKGSnapshotDataset
from modules.trainer.tkg_trainer import TemporalKGTrainer
import warnings
warnings.filterwarnings("ignore")  # 关闭所有警告
import torch
torch.cuda.empty_cache()

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    # 1. 加载配置 (你可以自己写一个极简的 config.yaml)
    config = {
        "device": "cuda",
        "lr": 0.0001,
        "temperature": 0.1,
        "lambda_contrastive": 0.5, # 调整对比学习的权重
        "num_epochs": 30,
        "history_window": 5, # 用过去5个时间步建图
        "device": "cpu"
    }
    
    data_dir = "./data/ICEWS14s"
    cache_dir = "./data/ICEWS14s/text_cache"
    
    # 请确保你这里的实体/关系数量与真实数据集对齐
    num_entities = 7128  
    num_relations = 230
    
    # 2. 准备数据集
    print("Loading Datasets...")
    train_dataset = TKGSnapshotDataset(data_dir, cache_dir, split="train", 
                                       num_entities=num_entities, history_window=config["history_window"])
    valid_dataset = TKGSnapshotDataset(data_dir, cache_dir, split="valid", 
                                       num_entities=num_entities, history_window=config["history_window"])
    
    # 3. 初始化模型
    print("Initializing Model...")
    model = DualStreamTKG(
        num_entities=num_entities,
        num_relations=num_relations,
        semantic_dim=768,
        hidden_dim=256,
        num_rgcn_layers=2
    )
    
    # 4. 启动训练
    trainer = TemporalKGTrainer(model, train_dataset, valid_dataset, config)
    trainer.train(num_epochs=config["num_epochs"])

if __name__ == "__main__":
    main()