"""冒烟测试：用随机初始化的极小模型在 CPU 上跑通完整训练流程。

无需下载大模型（只下载两个 tokenizer，约几十 MB），验证：
  1. 跨 tokenizer 路径（overlap 模式）：Qwen3.5 学生 tokenizer + DeepSeek 教师 tokenizer
  2. 相同 tokenizer 路径（topk 模式）
  3. LoRA、rollout、KL/CE、梯度更新、保存等全部环节

用法：
  python scripts/smoke_test.py
"""

import os
import shutil
import sys
import tempfile

# 让脚本能从项目根导入 src
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
)

TOKENIZER_PATTERNS = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
    "added_tokens.json",
    "generation_config.json",
]


def get_tokenizer_dir(model_id: str) -> str:
    """只下载 tokenizer 相关文件（不下载模型权重）。"""
    from modelscope import snapshot_download

    return snapshot_download(
        model_id, allow_patterns=TOKENIZER_PATTERNS
    )


def build_tiny_model(tok_dir: str, out_dir: str) -> str:
    """用下载的 tokenizer 构建一个随机初始化的极小 Qwen3 模型并保存。"""
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=True,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        torch_dtype="float32",
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return out_dir


def run_one(student_dir: str, teacher_dir: str, tmp: str, tag: str):
    """实例化训练器跑 2 个 optimizer step。"""
    from src.arguments import OPDArguments
    from src.trainer import OnPolicyTrainer

    args = OPDArguments(
        student_model_path=student_dir,
        teacher_model_path=teacher_dir,
        data_path=os.path.join(ROOT, "data", "example_prompts.jsonl"),
        output_dir=os.path.join(tmp, f"out_{tag}"),
        device="cpu",
        model_dtype="fp32",
        use_lora=True,
        gradient_checkpointing=False,
        num_rollouts=2,
        max_prompt_length=128,
        max_response_length=32,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        max_train_samples=4,
        swanlab_mode="disabled",
        debug_steps=2,
        save_steps=-1,
        preview_steps=-1,
        logits_token_chunk=64,
        kl_mode="auto",
    )
    trainer = OnPolicyTrainer(args)
    trainer.train()
    print(f"[{tag}] 冒烟测试通过")


def main():
    tmp = tempfile.mkdtemp(prefix="opd_smoke_")
    try:
        print("下载学生 tokenizer（Qwen3.5-0.8B）...")
        s_tok = get_tokenizer_dir("Qwen/Qwen3.5-0.8B")
        print("下载教师 tokenizer（DeepSeek-R1-Distill-Qwen-7B）...")
        t_tok = get_tokenizer_dir("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

        student_dir = build_tiny_model(s_tok, os.path.join(tmp, "student"))
        teacher_dir = build_tiny_model(t_tok, os.path.join(tmp, "teacher"))

        # 1) 跨 tokenizer：overlap 模式
        print("=" * 60)
        print("测试跨 tokenizer（overlap）路径 ...")
        run_one(student_dir, teacher_dir, tmp, "cross")

        # 2) 相同 tokenizer：topk 模式（教师直接复制学生）
        print("=" * 60)
        print("测试相同 tokenizer（topk）路径 ...")
        same_teacher = os.path.join(tmp, "teacher_same")
        shutil.copytree(student_dir, same_teacher)
        run_one(student_dir, same_teacher, tmp, "same")

        print("=" * 60)
        print("全部冒烟测试通过 ✓")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
