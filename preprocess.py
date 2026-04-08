import yaml
import argparse
from pathlib import Path

# 注意：这里需要改为你针对 ICEWS18 编写的预处理器类
from data.preprocessor import ICEWS18Preprocessor 

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # 参数配置
    parser = argparse.ArgumentParser(description="TKG_ICEWS18_Preprocessor")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--mode", type=str, default="preprocess", help="运行模式，当前默认为预处理")
    parser.add_argument("--experiment", type=str, default="icews18_data_prep", help="实验名称")
    args = parser.parse_args()
    
    config = load_config(args.config)

    # 预处理部分入口
    if args.mode == "preprocess":
        print("Starting ICEWS18 data preprocessing...")
        
        # 1. 实例化针对 ICEWS18 的预处理类
        # 注意：你的 ICEWS18Preprocessor 内部应当已经处理了读取五列（包含末尾0）的逻辑
        preprocessor = ICEWS18Preprocessor(config['data']['data_path'])
        
        # 2. 加载实体和关系映射
        num_entities, num_relations = preprocessor.load_mappings()
        print(f"Loaded {num_entities} entities and {num_relations} relations")
        
        # 3. 同步更新 config 中的实体和关系数量
        if 'model' not in config:
            config['model'] = {}
        config['model']['num_entities'] = num_entities
        config['model']['num_relations'] = num_relations
        
        # 4. 设置输出路径并保存处理后的数据
        output_path = Path(config['data']['data_path']) / "processed"
        
        print("Parsing and saving triples...")
        preprocessor.save_processed_data(
            str(output_path),
            preprocessor.load_triples("train.txt"),
            preprocessor.load_triples("valid.txt"),
            preprocessor.load_triples("test.txt")
        )
        
        print("ICEWS18 data preprocessing successfully completed!")
        
    # exp_logger.finish()

if __name__ == "__main__":
    main()