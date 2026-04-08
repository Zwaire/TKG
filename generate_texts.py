import yaml
import argparse
import json
import numpy as np
from pathlib import Path
import os

# 注意：请确保你的导入路径与你的实际文件名一致
from data.api_text_ICEWS18 import QwenAPIGenerator
from dotenv import load_dotenv
load_dotenv()  # 加载环境变量，确保 API Key 可用

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="TKG_ICEWS18_Text_Generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--split", type=str, default="train", choices=["train", "valid", "test"], help="要生成文本的数据集划分")
    
    # 🌟 新增：用于限制生成数量，方便快速测试
    parser.add_argument("--limit", type=int, default=None, help="仅处理前N条数据用于测试，例如 --limit 100")
    # 🌟 新增：分块保存大小，防范意外中断
    parser.add_argument("--chunk_size", type=int, default=5000, help="每处理多少条数据保存一次到硬盘")
    
    args = parser.parse_args()
    
    config = load_config(args.config)

    # 1. 加载预处理后的数据
    processed_path = Path(config['data']['data_path']) / "processed"
    triples = np.load(processed_path / f"{args.split}_triples.npy")
    
    with open(processed_path / "entity2id.json", 'r', encoding='utf-8') as f:
        id2entity = {int(v): k for k, v in json.load(f).items()}
    with open(processed_path / "relation2id.json", 'r', encoding='utf-8') as f:
        id2relation = {int(v): k for k, v in json.load(f).items()}
        
    triple_list = [(int(s), int(p), int(o), int(t)) for s, p, o, t, *_ in triples]
    
    # 【应用 limit 限制】如果传入了 --limit，则截取前 N 条
    if args.limit is not None:
        triple_list = triple_list[:args.limit]
        print(f"!!! TEST MODE ACTIVATED !!! Limited to generating {args.limit} texts.")
    
    print(f"Loaded {len(triple_list)} triples for {args.split} text generation.")

    # 2. 初始化 API 生成器
    api_config = config['data']['text_generation']['api']
    
    # 【安全读取 API Key】：优先读系统环境变量，其次读 config
    api_key = os.getenv("DASHSCOPE_API_KEY") or api_config.get('api_key')
    if not api_key:
        print("未找到 API Key！请设置环境变量 DASHSCOPE_API_KEY 或将其填入 config.yaml 中。")
        return

    try:
        generator = QwenAPIGenerator(
            api_key=api_key,
            base_url=api_config.get('base_url'),
            cache_dir=config['data']['text_generation']['cache_dir'],
            model=api_config.get('model')
        )
    except Exception as e:
        print(f"大模型 API 初始化失败: {e}")
        return

    # 3. 准备输出路径与断点续传逻辑
    cache_dir = Path(config['data']['text_generation']['cache_dir'])
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_file = cache_dir / f"{args.split}_texts.json"
    
    all_texts = []
    start_idx = 0
    
    # 【断点续传检查】
    if out_file.exists() and args.limit is None: # 测试模式下不触发断点续传
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                all_texts = json.load(f)
            start_idx = len(all_texts)
            if start_idx >= len(triple_list):
                print(f"文件 {out_file} 中已包含 {start_idx} 条数据，生成已完成！无需重复运行。")
                return
            print(f"检测到历史进度！找到 {start_idx} 条已生成数据，将从索引 {start_idx} 处继续生成...")
        except json.JSONDecodeError:
            print(f"历史文件 {out_file} 读取失败，将从头开始生成。")
            all_texts = []

    # 4. 分块调用 API 进行生成
    total_items = len(triple_list)
    requests_per_minute = api_config.get('requests_per_minute', 60)
    concurrency = api_config.get('concurrency', 4)
    triples_per_request = api_config.get('triples_per_request', 10)

    for i in range(start_idx, total_items, args.chunk_size):
        end_idx = min(i + args.chunk_size, total_items)
        chunk = triple_list[i:end_idx]
        
        print(f"Processing chunk: indices {i} to {end_idx} (Total target: {total_items})")
        
        chunk_texts = generator.batch_generate(
            chunk,
            id2entity,
            id2relation,
            requests_per_minute=requests_per_minute,
            concurrency=concurrency,
            triples_per_request=triples_per_request,
            show_progress=True
        )
        
        all_texts.extend(chunk_texts)
        
        # 每处理完一个 chunk 就覆盖保存一次当前所有结果
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(all_texts, f, ensure_ascii=False, indent=2)
            
        print(f"Chunk saved! Total progress: {len(all_texts)} / {total_items}")

    print(f"🎉 任务圆满完成！共成功保存 {len(all_texts)} 条文本到 {out_file}")

if __name__ == "__main__":
    main()