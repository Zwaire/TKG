'''
data.api_text_generator 的 Docstring
api调用文件包,主要通过使用Qwen的模型进行文本生成
'''

import os
import json
import time
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from tqdm import tqdm
import warnings
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    warnings.warn("OpenAI package not installed. Please install with: pip install openai")

class QwenAPIGenerator:
    def __init__(self, api_key: str = None, base_url: str = None,
                 cache_dir: str = "./data/text_cache", 
                 model: str = "qwen-plus",
                 max_retries: int = 3,
                 retry_delay: int = 1,
                 timeout: int = 60):
        """
        Args:
            api_key: 百炼API密钥
            base_url: API基础URL
            cache_dir: 缓存目录
            model: 模型名称
            max_retries: 最大重试次数
            retry_delay: 重试延迟（秒）
        """
        # 基础配置
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        if api_key is None:
            api_key = os.getenv("DASHSCOPE_API_KEY")
            if api_key is None:
                raise ValueError("API key not provided and DASHSCOPE_API_KEY environment variable not set")
        if base_url is None:
            base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

        self.logger = None
        self._test_connection()
    
    def set_logger(self, logger):
        self.logger = logger
    
    def _test_connection(self):
        """测试API连接"""
        try:
            # 简单的测试调用
            test_response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say 'OK' if you are ready."},
                ],
                max_tokens=10
            )
            
            if self.logger:
                self.logger.info(f"API connection test successful. Model: {self.model}")
            else:
                print(f"API connection test successful. Model: {self.model}")
                
        except Exception as e:
            error_msg = f"API connection test failed: {e}"
            if self.logger:
                self.logger.error(error_msg)
            else:
                print(error_msg)
            raise
    
    @staticmethod
    def _day_to_date(day_num: int, year: int = 2014) -> str:
        """
        Args:
            day_num: number of the day in the ICEWS14
            year:  2014
        Returns:
            日期字符串，如 "2014年1月1日"
        """
        # 2014年不是闰年，每月天数
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day_num < 1 or day_num > 365:
            raise ValueError(f"Day number must be between 1 and 365 for year {year}, got {day_num}")
        month = 1
        remaining_days = day_num
        for days_in_month in month_days:
            if remaining_days <= days_in_month:
                day = remaining_days
                break
            remaining_days -= days_in_month
            month += 1
        return f"{year}年{month}月{day}日"
    
    def _build_messages(self, subject: str, predicate: str, obj: str, day_num: int) -> List[Dict]:
        """
        构建对话消息列表
        
        Args:
            subject: 主语（实体）
            predicate: 谓语（关系）
            obj: 宾语（实体）
            day_num: 天数（一年中的第几天）
            
        Returns:
            消息列表
        """
        # 将天数转换为日期
        date_str = self._day_to_date(day_num)
        
        # 系统提示
        system_message = {
            "role": "system", 
            "content": """You are a helpful assistant that converts knowledge graph facts into natural English sentences. 
            You are working with the ICEWS14 dataset, which contains events from 2014.
            
            Instructions:
            1. Convert the knowledge graph fact into a clear, concise, and natural English sentence.
            2. Include the exact date from the fact (e.g., '2014年1月1日' becomes 'January 1, 2014').
            3. Use proper English grammar and make the sentence sound natural.
            4. Do not add any additional information, commentary, or explanations.
            5. Keep the sentence simple and factual."""
        }
        
        # 用户提示
        user_content = f"""Convert this knowledge graph fact into a natural English sentence:
                        Subject (who/what): {subject}
                        Predicate (action/relation): {predicate}
                        Object (who/what): {obj}
                        Date: {date_str}

                        Provide only the converted sentence, nothing else."""
        
        user_message = {"role": "user", "content": user_content}
        
        return [system_message, user_message]
    
    def _call_api_with_retry(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """
        调用API，带有重试机制
        
        Args:
            messages: 消息列表
            max_tokens: 最大token数
            
        Returns:
            API响应内容
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9,
                    timeout=self.timeout
                )
                
                content = response.choices[0].message.content.strip()
                
                # 验证响应不为空
                if content and len(content) > 5:
                    return content
                else:
                    raise ValueError("Empty or too short response from API")
                    
            except Exception as e:
                # 在非最后一次尝试时做指数退避 + 随机抖动
                if attempt < self.max_retries - 1:
                    jitter = random.uniform(0, 0.5)
                    wait_time = self.retry_delay * (2 ** attempt) + jitter  # 指数退避 + 抖动
                    if self.logger:
                        self.logger.warning(f"API call failed (attempt {attempt + 1}/{self.max_retries}): {str(e)[:200]}. Retrying in {wait_time:.2f}s...")
                    time.sleep(wait_time)
                else:
                    if self.logger:
                        self.logger.error(f"API call failed after {self.max_retries} attempts: {e}")
                    raise
        
        raise RuntimeError("Max retries exceeded")
    
    def generate_text_for_triple(self, s_id: int, p_id: int, o_id: int, day_num: int,
                                 s_name: str = None, p_name: str = None, 
                                 o_name: str = None) -> str:
        """
        为单个三元组生成文本描述
        
        Args:
            s_id, p_id, o_id: 实体和关系ID
            day_num: 天数（一年中的第几天）
            s_name, p_name, o_name: 实体和关系名称
            
        Returns:
            生成的文本描述
        """

        
        # 获取名称
        s_name = s_name or f"Entity_{s_id}"
        p_name = p_name or f"Relation_{p_id}"
        o_name = o_name or f"Entity_{o_id}"
        
        # 生成文本
        try:
            messages = self._build_messages(s_name, p_name, o_name, day_num)
            text = self._call_api_with_retry(messages)
            
            # 清理文本：移除可能的引号或多余空格
            text = text.strip().strip('"').strip("'").strip()
            
            # 如果生成结果异常，使用简单文本
            if not text or len(text) < 10:
                text = self._generate_simple_text(s_name, p_name, o_name, day_num)
                
        except Exception as e:
            if self.logger:
                self.logger.warning(f"API generation failed: {e}, using simple text")
            # 回退到简单文本
            text = self._generate_simple_text(s_name, p_name, o_name, day_num)
        

        
        return text
    
    def _generate_simple_text(self, s: str, p: str, o: str, day_num: int) -> str:
        """生成简单的文本描述（回退使用）"""
        date_str = self._day_to_date(day_num)
        templates = [
            f"On {date_str}, {s} {p} {o}.",
            f"At {date_str}, {s} performed {p} on {o}.",
            f"The event on {date_str} involved {s} {p} {o}.",
            f"{s} and {o} had interaction {p} on {date_str}."
        ]
        hash_val = int(hashlib.md5(f"{s}_{p}_{o}_{day_num}".encode()).hexdigest(), 16)
        template_idx = hash_val % len(templates)
        
        return templates[template_idx]
    
    def batch_generate(self, triples: List[Tuple[int, int, int, int]],
                       entity_dict: Optional[Dict[int, str]] = None,
                       relation_dict: Optional[Dict[int, str]] = None,
                       batch_size: int = 10,
                       requests_per_minute: int = 60,
                       concurrency: int = 4,
                       triples_per_request: int = 10,
                       show_progress: bool = True) -> List[str]:
        """
        批量生成文本
        Args:
            triples: 三元组列表
            entity_dict: 实体名称映射
            relation_dict: 关系名称映射
            batch_size: 批大小
            requests_per_minute: 每分钟请求数限制
            concurrency: 并发 worker 数量
            triples_per_request: 每次 API 请求中包含的三元组数量
            show_progress: 是否显示进度条
        Returns:
            文本列表（顺序与输入一致）
        """
        class _RateLimiter:
            def __init__(self, rpm: int):
                self.rate_per_sec = rpm / 60.0 if rpm > 0 else float('inf')
                self.capacity = max(1, int(self.rate_per_sec)) if self.rate_per_sec != float('inf') else 1
                self._tokens = self.capacity
                self._last = time.time()
                self._lock = threading.Lock()

            def acquire(self):
                if self.rate_per_sec == float('inf'):
                    return
                while True:
                    with self._lock:
                        now = time.time()
                        # 增加 token
                        self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate_per_sec)
                        self._last = now
                        if self._tokens >= 1:
                            self._tokens -= 1
                            return
                    time.sleep(0.01)
        texts = [None] * len(triples)

        # 所有条目均需生成（移除缓存支持）
        pending_indices = list(range(len(triples)))

        if not pending_indices:
            return texts

        # 按 triples_per_request 分组
        batches = [pending_indices[i:i+triples_per_request] for i in range(0, len(pending_indices), triples_per_request)]

        limiter = _RateLimiter(requests_per_minute)

        # 进度条
        if show_progress:
            pbar = tqdm(total=len(triples), desc="Generating texts via API (batched)", unit="triple")

        completed = 0

        def _generate_batch(batch_indices: List[int]):
            # 构造 batch 中的 triple 信息
            items = []
            for idx in batch_indices:
                s, p, o, day_num = triples[idx]
                s_name = entity_dict.get(int(s)) if entity_dict else None
                p_name = relation_dict.get(int(p)) if relation_dict else None
                o_name = entity_dict.get(int(o)) if entity_dict else None
                items.append((idx, int(s), int(p), int(o), int(day_num), s_name, p_name, o_name))

            # 等待令牌
            limiter.acquire()

            # 发起合并的 prompt 请求
            try:
                responses = self._generate_batch_prompt(items)
            except Exception as e:
                # 如果整批失败，记录并回退到单条生成（以保证鲁棒性）
                if self.logger:
                    self.logger.warning(f"Batch request failed: {e}. Falling back to single-item requests.")
                responses = []
                for t in items:
                    idx, s, p, o, day_num, s_name, p_name, o_name = t
                    try:
                        res = self.generate_text_for_triple(s, p, o, day_num, s_name, p_name, o_name)
                    except Exception:
                        res = self._generate_simple_text(s_name or f"Entity_{s}", p_name or f"Relation_{p}", o_name or f"Entity_{o}", day_num)
                    responses.append((idx, res))

            # responses: either list of (idx, text) or list of texts in same order
            assigned = 0
            for item in responses:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], int):
                    idx, text = item
                else:
                    # assume same order as items
                    idx = items[assigned][0]
                    text = item
                # 清理文本
                text = text.strip().strip('"').strip("'")
                texts[idx] = text
                assigned += 1

            return len(batch_indices)

        # 使用线程池并发执行批次
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            future_to_batch = {ex.submit(_generate_batch, batch): batch for batch in batches}
            for fut in as_completed(future_to_batch):
                try:
                    processed = fut.result()
                except Exception as e:
                    processed = 0
                    if self.logger:
                        self.logger.warning(f"Unhandled error in batch worker: {e}")
                completed += processed
                if show_progress:
                    pbar.update(processed)

        if show_progress:
            pbar.close()

        return texts

    def _generate_batch_prompt(self, items: List[Tuple[int,int,int,int,str,str,str]]) -> List[str]:
        """
        使用单次 API 请求为一组 items 生成多条文本。
        items: list of tuples (idx, s, p, o, day_num, s_name, p_name, o_name)
        返回: list texts in same order
        """
        # 构建 prompt
        lines = []
        for i, (idx, s, p, o, day_num, s_name, p_name, o_name) in enumerate(items, start=1):
            s_n = s_name or f"Entity_{s}"
            p_n = p_name or f"Relation_{p}"
            o_n = o_name or f"Entity_{o}"
            date = self._day_to_date(day_num)
            lines.append(f"{i}. Subject: {s_n} ; Predicate: {p_n} ; Object: {o_n} ; Date: {date}")

        system_message = {
            "role": "system",
            "content": "You are a helpful assistant that converts numbered knowledge graph facts into single natural English sentences. Return each sentence on its own line, in the SAME ORDER as the input. Do NOT add extra commentary or numbering. Use the date format like 'January 1, 2014'. If you cannot create a sentence for an input, return the word 'ERROR' on that line."
        }
        user_message = {"role": "user", "content": "\n".join(lines)}

        messages = [system_message, user_message]

        # 设置合理的 max_tokens：预估每条 40 token
        max_tokens = min(1024, max(128, int(80 * len(items))))

        resp = self._call_api_with_retry(messages, max_tokens=max_tokens)

        # 解析响应：按行分割，最多取 len(items) 行
        result_lines = [l.strip() for l in resp.splitlines() if l.strip()]

        # 如果返回的行数不够，尝试按句号分割或回退到逐条调用
        if len(result_lines) < len(items):
            # 尝试用 sentence split
            splits = [s.strip() for s in resp.replace('\r', '').split('\n') if s.strip()]
            if len(splits) >= len(items):
                result_lines = splits[:len(items)]
            else:
                # 回退：逐条调用保证有结果
                out = []
                for (idx, s, p, o, day_num, s_name, p_name, o_name) in items:
                    try:
                        r = self.generate_text_for_triple(s, p, o, day_num, s_name, p_name, o_name)
                    except Exception:
                        r = self._generate_simple_text(s_name or f"Entity_{s}", p_name or f"Relation_{p}", o_name or f"Entity_{o}", day_num)
                    out.append((idx, r))
                return out

        # 返回按顺序的文本（保证与 items 顺序一致）
        return result_lines[:len(items)]

class SimpleTextGenerator:
    def __init__(self, cache_dir: str = "./data/text_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @staticmethod
    def _day_to_date(day_num: int, year: int = 2014) -> str:
        """将天数转换为日期字符串"""
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        if day_num < 1 or day_num > 365:
            return f"Day {day_num}, 2014"
        month = 1
        remaining_days = day_num
        for days_in_month in month_days:
            if remaining_days <= days_in_month:
                day = remaining_days
                break
            remaining_days -= days_in_month
            month += 1

        month_names = [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ]
        return f"{month_names[month-1]} {day}, {year}"
    
    def generate_text_for_triple(self, s_id: int, p_id: int, o_id: int, day_num: int,
                                 s_name: str = None, p_name: str = None, 
                                 o_name: str = None) -> str:
        """生成简单文本描述"""
        s_name = s_name or f"Entity_{s_id}"
        p_name = p_name or f"Relation_{p_id}"
        o_name = o_name or f"Entity_{o_id}"
        
        date_str = self._day_to_date(day_num)
        
        templates = [
            f"On {date_str}, {s_name} {p_name} {o_name}.",
            f"At {date_str}, {s_name} engaged in {p_name} with {o_name}.",
            f"The event on {date_str} involved {s_name} performing {p_name} on {o_name}.",
            f"{s_name} and {o_name} had interaction {p_name} on {date_str}."
        ]

        import hashlib
        hash_val = int(hashlib.md5(f"{s_id}_{p_id}_{o_id}_{day_num}".encode()).hexdigest(), 16)
        template_idx = hash_val % len(templates)
        
        return templates[template_idx]
    
    def batch_generate(self, triples: List[Tuple[int, int, int, int]],
                       entity_dict: Optional[Dict[int, str]] = None,
                       relation_dict: Optional[Dict[int, str]] = None,
                       **kwargs) -> List[str]:
        """批量生成简单文本"""
        texts = []
        for s, p, o, day_num in triples:
            s_name = entity_dict.get(int(s)) if entity_dict else None
            p_name = relation_dict.get(int(p)) if relation_dict else None
            o_name = entity_dict.get(int(o)) if entity_dict else None
            
            text = self.generate_text_for_triple(s, p, o, day_num, s_name, p_name, o_name)
            texts.append(text)
        
        return texts