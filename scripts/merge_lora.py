"""把训练得到的 LoRA 适配器合并回基础模型，导出完整模型。

用法：
  python scripts/merge_lora.py --base_model models/Qwen3.5-0.8B \
      --adapter runs/opd/final --out runs/opd/merged
"""

import argparse

import torch
from transformers import AutoConfig


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 适配器")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    from peft import PeftModel

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    is_multimodal = hasattr(config, "vision_config")

    if is_multimodal:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.save_pretrained(args.out)

    # 同时保存 tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.out)
    print(f"合并完成，完整模型已保存到 {args.out}")


if __name__ == "__main__":
    main()
