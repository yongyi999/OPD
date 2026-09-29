"""从 ModelScope 下载 OPD 训练数据并转换成统一的 {"prompt": ...} JSONL。

OPD 只需要 prompt（问题），回复由学生在训练时自己采样。

可选数据集：
  - AI-ModelScope/DAPO-Math-17k       数学推理 prompt（约 1.7 万条，推荐）
  - AI-ModelScope/alpaca-gpt4-data-zh 中文通用指令 prompt
  - AI-ModelScope/alpaca-gpt4-data-en 英文通用指令 prompt

用法：
  python scripts/download_data.py --dataset AI-ModelScope/DAPO-Math-17k \
      --out data/train_prompts.jsonl
"""

import argparse
import json
import os
import sys

# 允许从项目根目录导入 src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _extract_prompt  # noqa: E402


def _build_prompt(example: dict) -> str:
    """从一条原始样本中提取 prompt，兼容 alpaca 的 instruction+input 结构。"""
    p = _extract_prompt(example, "prompt")
    if p:
        return p
    # alpaca 格式：instruction + 可选 input
    inst = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    if inp and inp.lower() not in ("no input", "无"):
        return f"{inst}\n{inp}"
    return inst


def main():
    parser = argparse.ArgumentParser(description="下载并转换 OPD 数据")
    parser.add_argument("--dataset", default="AI-ModelScope/DAPO-Math-17k")
    parser.add_argument("--out", default="data/train_prompts.jsonl")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from modelscope.msdatasets import MsDataset

    print(f"加载数据集 {args.dataset} ...")
    ds = MsDataset.load(args.dataset, split=args.split)

    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for example in ds:
            example = dict(example)
            prompt = _build_prompt(example)
            if not prompt:
                continue
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            count += 1
            if args.limit > 0 and count >= args.limit:
                break

    print(f"完成，共写入 {count} 条 prompt -> {args.out}")


if __name__ == "__main__":
    main()
