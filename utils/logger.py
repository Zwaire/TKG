import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict
import warnings

class SimpleFormatter(logging.Formatter):
    """日志格式化"""
    def __init__(self, fmt=None, datefmt=None):
        if fmt is None:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        if datefmt is None:
            datefmt = '%Y-%m-%d %H:%M:%S'
        super().__init__(fmt, datefmt)

def setup_logger(name: Optional[str] = None,
                 log_dir: Optional[Union[str, Path]] = None,
                 level: Union[int, str] = logging.INFO,
                 console: bool = True,
                 file: bool = True) -> logging.Logger:
    """
    设置并配置日志记录器
    Args:
        name: 记录器名称
        log_dir: 日志文件目录
        level: 日志级别
        console: 是否输出到控制台
        file: 是否输出到文件
    Returns:
        配置的日志记录器
    """
    if name is None:
        name = __name__
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = SimpleFormatter()
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"run_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Log file created: {log_file}")
    logger.propagate = False
    return logger

class ExperimentLogger:
    """实验日志记录器"""
    def __init__(self, experiment_name: str, 
                 log_dir: Union[str, Path],
                 config: Optional[dict] = None):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) / experiment_name
        self.config = config or {}
        
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=self.log_dir,
            level=logging.INFO,
            console=True,
            file=True
        )

        self._save_config()
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.experiment_name}")
        self.logger.info(f"Log directory: {self.log_dir}")
        self.logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)
    
    def _save_config(self):
        # 保存config
        if self.config:
            import json
            config_file = self.log_dir / "config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2)
            self.logger.info(f"Config saved to: {config_file}")
    
    def log_metric(self, metric_name: str, value: float, step: Optional[int] = None):
        step_str = f" (step {step})" if step is not None else ""
        self.logger.info(f"[Metric] {metric_name}{step_str}: {value:.6f}")
    
    def log_message(self, message: str, level: str = "info"):
        level = level.lower()
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
    
    def log_model_info(self, model):
        import torch
        if isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f"Model: {model.__class__.__name__}")
            self.logger.info(f"Total parameters: {total_params:,}")
            self.logger.info(f"Trainable parameters: {trainable_params:,}")
    
    def finish(self):
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment '{self.experiment_name}' completed")
        self.logger.info(f"Results available in: {self.log_dir}")
        self.logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info("=" * 60)