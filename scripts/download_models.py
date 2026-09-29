"""下载学生与教师模型（默认使用 ModelScope，国内/AutoDL 环境速度快）。

用法：
  python scripts/download_models.py
  python scripts/download_models.py --student Qwen/Qwen3.5-0.8B \
      --teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
      --target_dir models
"""

import argparse
import os


def download_one(model_id: str, target_dir: str) -> str:
    """通过 modelscope 下载单个模型快照。"""
    from modelscope import snapshot_download

    local_dir = os.path.join(target_dir, model_id.split("/")[-1])
    print(f"开始下载 {model_id} -> {local_dir}")
    path = snapshot_download(
        model_id,
        local_dir=local_dir,
        revision="master",
    )
    print(f"完成: {model_id} -> {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="下载 OPD 师生模型")
    parser.add_argument("--student", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--teacher", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    )
    parser.add_argument("--target_dir", default="models")
    parser.add_argument(
        "--only", choices=["student", "teacher", "both"], default="both"
    )
    args = parser.parse_args()

    os.makedirs(args.target_dir, exist_ok=True)
    if args.only in ("student", "both"):
        download_one(args.student, args.target_dir)
    if args.only in ("teacher", "both"):
        download_one(args.teacher, args.target_dir)
    print("全部模型下载完成")


if __name__ == "__main__":
    main()
