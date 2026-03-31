import torch
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class OfflineSemanticEncoder:
    def __init__(self, cache_dir: str, model_name: str = 'bert-base-uncased', device: str = 'cuda'):
        self.cache_dir = Path(cache_dir)
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading {model_name} onto {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert.eval()

    @torch.no_grad()
    def process_json_file(self, split: str, batch_size: int = 256):
        """直接读取现成的 JSON 文本并提取特征"""
        json_path = self.cache_dir / f"{split}_texts.json"
        if not json_path.exists():
            print(f"File not found: {json_path}")
            return
            
        with open(json_path, 'r', encoding='utf-8') as f:
            texts = json.load(f)
            
        print(f"Processing {split}: loaded {len(texts)} sentences.")
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {split}"):
            batch_texts = texts[i:i+batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                    max_length=64, return_tensors="pt").to(self.device)
            
            outputs = self.bert(**inputs)
            # Mean Pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1) / torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            
            all_embeddings.append(embeddings.cpu())
            
        final_tensor = torch.cat(all_embeddings, dim=0)
        
        # 保存为 .pt 文件，直接放在同目录下即可
        out_path = self.cache_dir / f"{split}_semantic_emb.pt"
        torch.save(final_tensor, out_path)
        print(f"Success! Saved shape {final_tensor.shape} to {out_path}\n")

if __name__ == "__main__":
    # 指向你存放 JSON 文件的目录
    encoder = OfflineSemanticEncoder(cache_dir="C:\\D\\Projects\\TKG_Task\\MyTest\\data\\ICEWS14s\\text_cache")
    encoder.process_json_file("train")
    encoder.process_json_file("valid")
    encoder.process_json_file("test")