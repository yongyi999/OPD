"""模型加载模块。

- 学生模型：Qwen3.5 系列（多模态混合架构），可选 LoRA 训练
- 教师模型：DeepSeek-R1-Distill-Qwen 系列（纯文本），全程冻结
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .utils import count_parameters, get_dtype, get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------- #
def load_tokenizer(path: str):
    """加载 tokenizer，左/右 padding 统一使用右 padding（训练需要）。"""
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def is_same_tokenizer(tok_a, tok_b) -> bool:
    """通过词表大小 + 若干探针句判断两个 tokenizer 是否等价。"""
    if tok_a.vocab_size != tok_b.vocab_size:
        return False
    probes = [
        "Hello, world!",
        "1 + 1 = 2",
        "请简单解释一下什么是引力。",
        "def f(x):\n    return x + 1",
        "The answer is 42.",
    ]
    for p in probes:
        if tok_a.encode(p, add_special_tokens=False) != tok_b.encode(
            p, add_special_tokens=False
        ):
            return False
    return True


# --------------------------------------------------------------------- #
# 结构辅助函数
# --------------------------------------------------------------------- #
def find_language_module(model: nn.Module):
    """在（可能多模态的）模型中找到文本解码器子模块。

    判据：含有 `layers`（ModuleList）且 block 中带注意力结构的模块。
    多模态模型通常位于 model.model.language_model；纯文本模型为 model.model。
    返回 (模块路径, 模块)；找不到时返回 ("", model)。
    """
    candidates = []
    for name, mod in model.named_modules():
        layers = getattr(mod, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            # 检查前若干个 block 是否为文本解码器（混合架构首个 block
            # 可能是线性注意力层，所以多检查几个）
            is_text = False
            for block in list(layers)[:4]:
                if (
                    hasattr(block, "self_attn")
                    or hasattr(block, "attention")
                    or hasattr(block, "linear_attn")
                    or hasattr(block, "linear_attention")
                ):
                    is_text = True
                    break
            if is_text:
                candidates.append((name, mod, len(layers)))

    if not candidates:
        return "", model
    # 层数最多的候选即真正的文本解码器
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates[0][0], candidates[0][1]


def find_lm_head(model: nn.Module):
    """找到模型的 lm_head（可能在顶层，也可能在 model 下）。"""
    if hasattr(model, "lm_head") and isinstance(model.lm_head, nn.Module):
        return model.lm_head
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "lm_head"):
        return inner.lm_head
    return None


def find_lora_target_suffixes(model: nn.Module) -> list[str]:
    """扫描文本解码器，返回 LoRA 目标模块的后缀名集合。

    默认只挂标准注意力层 q/k/v/o_proj；混合架构中的线性注意力层若命名
    以这些后缀结尾，也会被自动覆盖。
    """
    _, text_mod = find_language_module(model)
    suffixes = set()
    wanted = ("q_proj", "k_proj", "v_proj", "o_proj")
    for name, mod in text_mod.named_modules():
        if isinstance(mod, nn.Linear):
            for w in wanted:
                if name.endswith(w):
                    suffixes.add(w)
    if not suffixes:
        # 兜底：挂所有线性层
        for name, mod in text_mod.named_modules():
            if isinstance(mod, nn.Linear) and "lm_head" not in name:
                suffixes.add(name.split(".")[-1])
    return sorted(suffixes)


# --------------------------------------------------------------------- #
# 模型加载
# --------------------------------------------------------------------- #
def _is_multimodal(config) -> bool:
    """根据 config 判断是否为多模态模型。"""
    if hasattr(config, "vision_config"):
        return True
    architectures = getattr(config, "architectures", None) or []
    return any("ConditionalGeneration" in arch for arch in architectures)


def load_student(args):
    """加载学生模型，必要时套用 LoRA。"""
    dtype = get_dtype(args.model_dtype)
    config = AutoConfig.from_pretrained(
        args.student_model_path, trust_remote_code=True
    )

    common_kwargs = dict(
        pretrained_model_name_or_path=args.student_model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    # "auto" 时不传，让库自己选择默认实现（显式传 "auto" 在某些 CPU/小模型场景会报错）
    if args.attn_implementation and args.attn_implementation != "auto":
        common_kwargs["attn_implementation"] = args.attn_implementation

    if _is_multimodal(config):
        # Qwen3.5 全系为多模态模型，需用 AutoModelForImageTextToText 加载
        from transformers import AutoModelForImageTextToText

        logger.info("学生模型为多模态架构，使用 AutoModelForImageTextToText 加载")
        student = AutoModelForImageTextToText.from_pretrained(**common_kwargs)
    else:
        student = AutoModelForCausalLM.from_pretrained(**common_kwargs)

    if args.gradient_checkpointing and hasattr(student, "gradient_checkpointing_enable"):
        student.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        if args.lora_target_modules == "auto":
            target_modules = find_lora_target_suffixes(student)
        else:
            target_modules = [
                x.strip() for x in args.lora_target_modules.split(",") if x.strip()
            ]
        logger.info("LoRA 目标模块后缀: %s", target_modules)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        student = get_peft_model(student, lora_config)
        # 让梯度可以从输入位置开始回传（LoRA 场景下的保险措施）
        if hasattr(student, "enable_input_require_grads"):
            student.enable_input_require_grads()
        student.print_trainable_parameters()
    else:
        total, trainable = count_parameters(student)
        logger.info("全参数微调：学生模型参数 %d (可训练 %d)", total, trainable)

    student.train()
    return student


def load_teacher(args):
    """加载教师模型并冻结（只做前向，不记录梯度）。"""
    dtype = get_dtype(args.model_dtype)
    teacher_kwargs = dict(
        pretrained_model_name_or_path=args.teacher_model_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    if args.attn_implementation and args.attn_implementation != "auto":
        teacher_kwargs["attn_implementation"] = args.attn_implementation
    teacher = AutoModelForCausalLM.from_pretrained(**teacher_kwargs)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    total, _ = count_parameters(teacher)
    logger.info("教师模型已加载并冻结，参数量 %d (%.2fB)", total, total / 1e9)
    return teacher


# --------------------------------------------------------------------- #
# 教师前向（分块 lm_head，控制显存峰值）
# --------------------------------------------------------------------- #
@torch.no_grad()
def teacher_forward_logits(
    teacher: nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    token_chunk: int = 256,
) -> torch.Tensor:
    """获取教师模型在所有位置上的 logits。

    优先走 base model 隐状态 + 分块 lm_head 的路径（避免一次性生成
    B×T×V 巨大 logits 张量）；不支持时退化为完整前向。
    """
    base = getattr(teacher, "model", None)
    lm_head = find_lm_head(teacher)

    if base is not None and hasattr(base, "layers") and lm_head is not None:
        outputs = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        hidden = outputs.last_hidden_state
        # 分块过 lm_head
        logits_chunks = []
        for s in range(0, hidden.shape[1], token_chunk):
            logits_chunks.append(lm_head(hidden[:, s : s + token_chunk]))
        return torch.cat(logits_chunks, dim=1)

    # 兜底：直接完整前向
    return teacher(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits
