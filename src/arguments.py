"""训练参数定义。

用法：
1. 直接命令行：python train.py --student_model_path models/Qwen3.5-0.8B ...
2. 使用 YAML 配置：python train.py --config configs/default.yaml
3. 命令行参数会覆盖 YAML 中的同名字段。
"""

from dataclasses import dataclass, field, fields
from typing import Optional


@dataclass
class OPDArguments:
    # ------------------------------------------------------------------ #
    # 模型相关
    # ------------------------------------------------------------------ #
    # 学生模型（被训练）。Qwen3.5 全系为多模态混合架构，文本输入可直接使用。
    student_model_path: str = "models/Qwen3.5-0.8B"
    # 教师模型（只提供 token 级分布，不训练）。
    teacher_model_path: str = "models/DeepSeek-R1-Distill-Qwen-7B"
    # 模型精度：bf16 / fp16 / fp32
    model_dtype: str = "bf16"
    # 注意力实现：auto / sdpa / flash_attention_2 / eager
    attn_implementation: str = "auto"
    # 是否启用梯度检查点（省显存，推荐开启）
    gradient_checkpointing: bool = True
    # 是否对学生使用 LoRA（单卡 32GB 推荐开启；关闭则全参数微调）
    use_lora: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    # LoRA 目标模块后缀；"auto" 表示自动从文本模型中扫描 q/k/v/o_proj
    lora_target_modules: str = "auto"
    # 教师是否在 rollout 阶段临时放到 CPU（默认关闭，32GB 足够常驻）
    teacher_cpu_offload: bool = False

    # ------------------------------------------------------------------ #
    # 数据相关
    # ------------------------------------------------------------------ #
    # 数据文件，支持 .jsonl/.json/.parquet/.csv，只需包含 prompt 列
    data_path: str = "data/example_prompts.jsonl"
    # 取数据中哪个字段作为 prompt（兼容 instruction/question/query 等）
    prompt_field: str = "prompt"
    max_prompt_length: int = 1024
    max_response_length: int = 1024
    # 最多使用多少条 prompt，-1 表示全部
    max_train_samples: int = -1
    shuffle_data: bool = True
    # 可选的 system prompt（空字符串表示不使用）
    system_prompt: str = ""

    # ------------------------------------------------------------------ #
    # On-policy rollout（学生自己采样）
    # ------------------------------------------------------------------ #
    # 每个 prompt 采样多少条学生回复（对应 verl 中的 N_RESPONSES）
    num_rollouts: int = 4
    temperature: float = 1.0
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    # 教师混合采样比例（0~1）：0 表示纯 on-policy（学生采样）。
    # 参考 "On-Policy Distillation of Language Models"，冷启动时可设 0.1~0.5。
    teacher_mix_ratio: float = 0.0

    # ------------------------------------------------------------------ #
    # 蒸馏损失
    # ------------------------------------------------------------------ #
    # KL 模式：auto / topk / overlap / sampled / full
    #   auto：师生 tokenizer 相同 -> topk；不同 -> overlap（跨 tokenizer）
    #   topk：只在学生 top-k 集合上计算 KL（Rethinking-OPD 配方）
    #   overlap：跨 tokenizer，只在两个词表的共有 token 上计算 KL（KDFlow 思路）
    #   sampled：只在实际采样到的 token 上计算 log 概率差
    #   full：全词表 KL（仅支持相同 tokenizer，显存占用大）
    kl_mode: str = "auto"
    # topk 模式下的 k 值
    top_k: int = 16
    # 蒸馏温度（同时缩放师生 logits）
    kl_temperature: float = 1.0
    # 教师 logits 温度（Rethinking-OPD 中的 TEACHER_TEMPERATURE）
    teacher_temperature: float = 1.0
    # KD 损失权重（1 - kd_ratio 为 CE 权重）
    kd_ratio: float = 1.0
    # 损失聚合方式：token-mean / seq-mean
    loss_agg_mode: str = "token-mean"
    # 教师 logits 按 token 位置分块大小（控制显存峰值）
    logits_token_chunk: int = 256

    # ------------------------------------------------------------------ #
    # 训练相关
    # ------------------------------------------------------------------ #
    output_dir: str = "runs/opd"
    num_train_epochs: int = 1
    max_steps: int = -1
    # 每个 step 同时处理多少个 prompt（每个 prompt 再展开 num_rollouts 条回复）
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    # 优化器：adamw_torch / paged_adamw_8bit
    optim: str = "adamw_torch"
    seed: int = 42
    save_steps: int = 100
    log_steps: int = 1
    dataloader_num_workers: int = 0
    # 每隔多少 step 往 SwanLab 记录一次学生生成样例（-1 关闭）
    preview_steps: int = 50
    # 训练结束后是否合并 LoRA 并保存完整模型
    merge_after_train: bool = False

    # ------------------------------------------------------------------ #
    # SwanLab 实验记录
    # ------------------------------------------------------------------ #
    # online：云端；local：本地可被 swanlab watch 打开；offline：仅本地；disabled：关闭
    swanlab_mode: str = "online"
    swanlab_project: str = "on-policy-distillation"
    swanlab_entity: Optional[str] = None
    swanlab_experiment_name: str = ""
    swanlab_logdir: str = "runs/swanlab"

    # ------------------------------------------------------------------ #
    # 其他
    # ------------------------------------------------------------------ #
    config: Optional[str] = field(default=None, metadata={"help": "YAML 配置文件路径"})
    device: str = "cuda"
    # 仅跑前 N 个 optimizer step（用于调试，-1 关闭）
    debug_steps: int = -1


def _yaml_to_dict(path: str) -> dict:
    """读取 YAML 文件为字典（需要 pyyaml）。"""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_arguments() -> OPDArguments:
    """解析命令行参数，支持 YAML 配置 + 命令行覆盖。"""
    import argparse

    parser = argparse.ArgumentParser(description="On-Policy Distillation 训练参数")
    # 先注册 --config，便于在解析前读取 YAML
    parser.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    known, _ = parser.parse_known_args()

    yaml_cfg = _yaml_to_dict(known.config) if known.config else {}

    # 根据 dataclass 字段自动注册所有参数
    valid_names = {f.name for f in fields(OPDArguments)}
    type_map = {f.name: f.type for f in fields(OPDArguments)}

    def _py_type(name: str):
        """返回字段声明对应的 Python 基础类型（int/float/str）。"""
        t = type_map.get(name, str)
        # 注解可能是类对象（int）或字符串（"int" / "Optional[int]"）
        if t is int or t in ("int", "Optional[int]"):
            return int
        if t is float or t in ("float", "Optional[float]"):
            return float
        return str

    def _arg_type(name: str):
        return _py_type(name)

    for f in fields(OPDArguments):
        if f.name == "config":
            continue
        ftype = f.type
        # bool 单独处理
        if ftype == "bool" or f.default is True or f.default is False:
            parser.add_argument(
                f"--{f.name}",
                type=lambda x: x.lower() in ("1", "true", "yes", "y"),
                default=None,
                help=f"YAML key: {f.name}",
            )
        else:
            parser.add_argument(f"--{f.name}", type=_arg_type(f.name), default=None)

    args = parser.parse_args()
    cli_dict = {k: v for k, v in vars(args).items() if v is not None and k in valid_names}

    # 优先级：命令行 > YAML > dataclass 默认值
    defaults = {f.name: f.default for f in fields(OPDArguments)}
    merged = {**defaults, **yaml_cfg, **cli_dict}
    # 过滤掉非法字段
    merged = {k: v for k, v in merged.items() if k in valid_names}
    # 对 YAML 来源的值做一次类型兜底（防止 YAML 写成字符串）
    for k, v in list(merged.items()):
        if not isinstance(v, str):
            continue
        py_t = _py_type(k)
        if py_t is int and v.lstrip("-").isdigit():
            merged[k] = int(v)
        elif py_t is float:
            try:
                merged[k] = float(v)
            except ValueError:
                pass
    return OPDArguments(**merged)
