"""数据加载模块。

OPD 只需要 prompt（问题）即可，回复由学生在训练时自己采样生成。
支持 .jsonl / .json / .parquet / .csv 格式，自动兼容常见字段名：
  prompt / instruction / question / query / input
"""

import json
import os
from typing import Optional

from torch.utils.data import Dataset

from .utils import get_logger

logger = get_logger(__name__)

# 常见的 prompt 字段别名
PROMPT_FIELD_ALIASES = ("prompt", "instruction", "question", "query", "input", "problem")


def _read_raw_file(path: str) -> list[dict]:
    """根据扩展名读取文件，返回字典列表。"""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            # 兼容 {"data": [...]} 这种结构
            for key in ("data", "items", "examples"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        return data

    if ext == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return df.to_dict(orient="records")

    if ext == ".csv":
        import pandas as pd

        df = pd.read_csv(path)
        return df.to_dict(orient="records")

    raise ValueError(f"不支持的数据文件格式: {ext}（{path}）")


def _extract_user_text_from_messages(messages) -> Optional[str]:
    """从消息列表（verl/chat 格式）中提取第一条 user 消息文本。"""
    # numpy array / list 都转成 list
    try:
        messages = list(messages)
    except TypeError:
        return None
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str) and content.strip():
                return content.strip()
            if isinstance(content, list):  # 多模态 content 列表
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"].strip()
    return None


def _extract_prompt(example: dict, preferred_field: str) -> Optional[str]:
    """从一条样本中提取 prompt 文本。"""
    # 优先使用用户指定字段
    if preferred_field in example:
        value = example[preferred_field]
        if isinstance(value, str) and value.strip():
            return value.strip()
        # 字段本身可能是消息列表（如 verl 格式的 prompt 字段）
        if isinstance(value, (list, tuple)) or hasattr(value, "tolist"):
            text = _extract_user_text_from_messages(value)
            if text:
                return text

    # 然后尝试常见别名
    for key in PROMPT_FIELD_ALIASES:
        if key in example and isinstance(example[key], str) and example[key].strip():
            return example[key].strip()

    # 兼容 messages 格式（取第一条 user 消息）
    if "messages" in example:
        text = _extract_user_text_from_messages(example["messages"])
        if text:
            return text
    return None


def build_prompt_text(
    tokenizer,
    prompt: str,
    system_prompt: str = "",
) -> str:
    """使用 tokenizer 的 chat template 把原始问题包装成模型输入文本。

    若模型没有 chat template，则退化为简单的 "问题 + 换行" 形式。
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        # add_generation_prompt=True 会在末尾追加 assistant 开头，方便模型续写
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # 没有 chat template 的兜底
    prefix = f"{system_prompt}\n" if system_prompt else ""
    return f"{prefix}{prompt}\n"


class PromptDataset(Dataset):
    """简单的 prompt 数据集，每条返回经过 chat template 包装的文本。"""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        prompt_field: str = "prompt",
        system_prompt: str = "",
        max_samples: int = -1,
        shuffle: bool = True,
        seed: int = 42,
    ):
        raw = _read_raw_file(data_path)
        prompts = []
        for ex in raw:
            p = _extract_prompt(ex, prompt_field)
            if p:
                prompts.append(p)

        if not prompts:
            raise ValueError(f"数据文件 {data_path} 中没有找到任何有效 prompt")

        if shuffle:
            import random

            rng = random.Random(seed)
            rng.shuffle(prompts)

        if max_samples is not None and max_samples > 0:
            prompts = prompts[:max_samples]

        self.prompts = prompts
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        logger.info("从 %s 加载了 %d 条 prompt", data_path, len(prompts))

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> dict:
        raw_prompt = self.prompts[idx]
        prompt_text = build_prompt_text(
            self.tokenizer, raw_prompt, self.system_prompt
        )
        return {"prompt": raw_prompt, "prompt_text": prompt_text}


def collate_prompts(batch: list[dict]) -> list[dict]:
    """自定义 collate：直接返回字典列表，不做 padding（rollout 时逐条处理）。"""
    return batch
