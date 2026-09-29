"""通用工具函数：日志、随机种子、显存统计、参数计数等。"""

import logging
import os
import random
import sys

import numpy as np
import torch


def get_logger(name: str = "opd", level: int = logging.INFO) -> logging.Logger:
    """获取一个统一格式的 logger（重复调用不会重复添加 handler）。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    """设置所有相关库的随机种子，保证 rollout 与训练尽量可复现。"""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def gpu_memory_stats(device: int = 0) -> dict:
    """返回当前 GPU 显存占用（GB），无 GPU 时返回空字典。"""
    if not torch.cuda.is_available():
        return {}
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    return {
        "allocated_gb": round(allocated, 2),
        "reserved_gb": round(reserved, 2),
        "peak_gb": round(peak, 2),
        "total_gb": round(total, 2),
    }


def get_dtype(name: str) -> torch.dtype:
    """把字符串 dtype 转成 torch.dtype。"""
    name = name.lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"不支持的 dtype: {name}，可选 {list(mapping)}")
    return mapping[name]


def ensure_dir(path: str) -> str:
    """目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)
    return path


def safe_decode(tokenizer, token_ids) -> str:
    """安全地把 token id 列表解码为字符串（自动跳过特殊 token）。"""
    return tokenizer.decode(token_ids, skip_special_tokens=True)
