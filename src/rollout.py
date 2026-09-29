"""On-policy rollout：学生模型针对 prompt 采样自己的回复。

"On-policy" 的核心：训练目标计算在学生自己采样出的序列上，
而不是预先准备好的固定（教师）回复上。
"""

import torch

from .utils import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def _sample_sequences(
    model,
    tokenizer,
    prompt_text: str,
    num_sequences: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对单个 prompt 采样 num_sequences 条回复，返回 (prompt_ids, response_ids)。"""
    enc = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # 直接复制输入实现 n 路采样，比 num_return_sequences 更直观
    input_ids = input_ids.repeat(num_sequences, 1)
    attention_mask = attention_mask.repeat(num_sequences, 1)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # temperature 过低会导致数值问题，给一个下限
    if temperature > 1e-4:
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    output = model.generate(**gen_kwargs)
    prompt_len = input_ids.shape[1]
    response_ids = output[:, prompt_len:]
    return input_ids[:1], response_ids


def generate_student_rollouts(
    student,
    tokenizer,
    prompt_items: list[dict],
    args,
) -> list[dict]:
    """对一批 prompt 采样学生回复。

    返回列表，每个元素对应一个 prompt：
      {
        "prompt": 原始问题,
        "prompt_text": 经 chat template 包装的文本,
        "prompt_ids": 学生 tokenizer 编码的 prompt（1, P）,
        "response_ids": 学生采样回复（n, R_i）,
        "source": "student",
      }
    """
    student.eval()
    rollouts = []
    for item in prompt_items:
        prompt_ids, response_ids = _sample_sequences(
            student,
            tokenizer,
            item["prompt_text"],
            num_sequences=args.num_rollouts,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_response_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        rollouts.append(
            {
                "prompt": item["prompt"],
                "prompt_text": item["prompt_text"],
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "source": "student",
            }
        )
    student.train()
    return rollouts


@torch.no_grad()
def generate_teacher_rollouts(
    teacher,
    teacher_tokenizer,
    prompt_items: list[dict],
    args,
) -> list[dict]:
    """教师混合采样：用教师 tokenizer 的 chat template 采样教师回复。

    用于 teacher_mix_ratio > 0 的场景（冷启动/混合采样）。返回结构与
    学生 rollout 类似，但 prompt_ids 是教师 tokenizer 编码，
    文本层面与学生共享，distillation 模块负责对齐。
    """
    from .data import build_prompt_text

    teacher.eval()
    rollouts = []
    for item in prompt_items:
        teacher_prompt_text = build_prompt_text(
            teacher_tokenizer, item["prompt"], args.system_prompt
        )
        prompt_ids, response_ids = _sample_sequences(
            teacher,
            teacher_tokenizer,
            teacher_prompt_text,
            num_sequences=args.num_rollouts,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_response_length,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
        )
        rollouts.append(
            {
                "prompt": item["prompt"],
                "prompt_text": teacher_prompt_text,
                "prompt_ids": prompt_ids,
                "response_ids": response_ids,
                "source": "teacher",
            }
        )
    return rollouts
