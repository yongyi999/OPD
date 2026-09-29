"""On-Policy Distillation 训练器。

完整流程（每个 optimizer step）：
  1. 取一批 prompt
  2. 学生 on-policy 采样 n 条回复（教师混合采样可选）
  3. 教师对学生自生成的回复提供 token 级分布
  4. 计算 reverse KL（学生采样）/ forward KL（教师采样）+ 可选 CE
  5. 梯度累积、反向传播、参数更新
  6. 指标写入 SwanLab，定期保存与生成样例
"""

import math
import os
import time
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from .data import PromptDataset, collate_prompts
from .distillation import DistillationEngine
from .logger import SwanLabLogger
from .models import (
    load_student,
    load_teacher,
    load_tokenizer,
)
from .rollout import generate_student_rollouts, generate_teacher_rollouts
from .utils import (
    ensure_dir,
    get_logger,
    gpu_memory_stats,
    set_seed,
)

logger = get_logger(__name__)


class OnPolicyTrainer:
    """OPD 训练器。"""

    def __init__(self, args):
        self.args = args
        set_seed(args.seed)
        ensure_dir(args.output_dir)

        # ---------------- Tokenizer ----------------
        logger.info("加载学生 tokenizer: %s", args.student_model_path)
        self.student_tokenizer = load_tokenizer(args.student_model_path)
        logger.info("加载教师 tokenizer: %s", args.teacher_model_path)
        self.teacher_tokenizer = load_tokenizer(args.teacher_model_path)

        # ---------------- 数据 ----------------
        dataset = PromptDataset(
            data_path=args.data_path,
            tokenizer=self.student_tokenizer,
            prompt_field=args.prompt_field,
            system_prompt=args.system_prompt,
            max_samples=args.max_train_samples,
            shuffle=args.shuffle_data,
            seed=args.seed,
        )
        self.dataloader = DataLoader(
            dataset,
            batch_size=args.per_device_train_batch_size,
            shuffle=False,  # dataset 内部已 shuffle
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_prompts,
            drop_last=False,
        )

        # ---------------- 模型 ----------------
        logger.info("加载学生模型 ...")
        self.student = load_student(args)
        logger.info("加载教师模型 ...")
        self.teacher = load_teacher(args)

        # ---------------- 蒸馏引擎 ----------------
        self.engine = DistillationEngine(
            self.student,
            self.teacher,
            self.student_tokenizer,
            self.teacher_tokenizer,
            args,
        )

        # ---------------- 优化器 & 调度器 ----------------
        trainable_params = [p for p in self.student.parameters() if p.requires_grad]
        logger.info("学生可训练参数张量数: %d", len(trainable_params))

        if args.optim == "paged_adamw_8bit":
            import bitsandbytes as bnb

            self.optimizer = bnb.optim.PagedAdamW8bit(
                trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
            )
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay
            )

        # 估算总更新步数
        num_update_steps_per_epoch = math.ceil(
            len(self.dataloader) / args.gradient_accumulation_steps
        )
        if args.max_steps > 0:
            self.max_update_steps = args.max_steps
        else:
            self.max_update_steps = int(
                args.num_train_epochs * num_update_steps_per_epoch
            )

        from transformers import get_scheduler

        self.scheduler = get_scheduler(
            args.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=int(args.warmup_ratio * self.max_update_steps),
            num_training_steps=self.max_update_steps,
        )

        # ---------------- SwanLab ----------------
        self.swanlab = SwanLabLogger(self.args, config_dict=asdict(args))

        # 固定一个预览 prompt，用于观察训练过程中的生成变化
        self.preview_item = dataset[0]

        self.global_step = 0
        logger.info("总 optimizer 步数（预计）: %d", self.max_update_steps)
        if gpu_memory_stats():
            logger.info("初始显存: %s", gpu_memory_stats())

    # ----------------------------------------------------------------- #
    def _move_teacher(self, on_gpu: bool):
        """教师模型在 CPU/GPU 间切换（仅在 teacher_cpu_offload=True 时）。"""
        if not self.args.teacher_cpu_offload:
            return
        target = self.args.device if on_gpu else "cpu"
        self.teacher.to(target)
        if on_gpu:
            torch.cuda.empty_cache()

    # ----------------------------------------------------------------- #
    def _rollout(self, prompt_batch: list[dict]):
        """学生 rollout + 教师混合 rollout。"""
        n_teacher = round(len(prompt_batch) * self.args.teacher_mix_ratio)
        teacher_items = prompt_batch[:n_teacher]
        student_items = prompt_batch[n_teacher:]

        groups = []
        if student_items:
            self._move_teacher(on_gpu=False)  # rollout 时把教师挪走省显存
            groups.extend(
                generate_student_rollouts(
                    self.student,
                    self.student_tokenizer,
                    student_items,
                    self.args,
                )
            )
        if teacher_items:
            self._move_teacher(on_gpu=True)
            groups.extend(
                generate_teacher_rollouts(
                    self.teacher,
                    self.teacher_tokenizer,
                    teacher_items,
                    self.args,
                )
            )
        self._move_teacher(on_gpu=True)  # 计算损失前教师回到 GPU
        return groups

    # ----------------------------------------------------------------- #
    def _log_preview(self):
        """让学生对固定 prompt 生成回复，写入 SwanLab 文本面板。"""
        try:
            from .data import build_prompt_text

            prompt_text = build_prompt_text(
                self.student_tokenizer,
                self.preview_item["prompt"],
                self.args.system_prompt,
            )
            enc = self.student_tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.args.device)
            with torch.no_grad():
                self.student.eval()
                out = self.student.generate(
                    **enc,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    max_new_tokens=min(512, self.args.max_response_length),
                    pad_token_id=self.student_tokenizer.pad_token_id,
                )
                self.student.train()
            text = self.student_tokenizer.decode(
                out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
            self.swanlab.log_text(
                "preview/generation",
                f"Prompt: {self.preview_item['prompt']}\n\nStudent output:\n{text}",
                step=self.global_step,
            )
        except Exception as e:
            logger.warning("预览生成失败: %s", e)

    # ----------------------------------------------------------------- #
    def save_model(self, tag: str = "final"):
        """保存学生模型（LoRA 适配器或完整权重）与 tokenizer。"""
        save_dir = os.path.join(self.args.output_dir, tag)
        ensure_dir(save_dir)
        self.student.save_pretrained(save_dir)
        self.student_tokenizer.save_pretrained(save_dir)
        logger.info("模型已保存到 %s", save_dir)
        return save_dir

    # ----------------------------------------------------------------- #
    def train(self):
        """主训练循环。"""
        logger.info("开始训练 ...")
        accum_loss = 0.0
        accum_metrics: dict = {}
        micro_count = 0
        tokens_seen = 0
        start_time = time.time()
        data_iter = iter(self.dataloader)

        while self.global_step < self.max_update_steps:
            try:
                prompt_batch = next(data_iter)
            except StopIteration:
                break

            # 1) rollout
            groups = self._rollout(prompt_batch)

            # 2) 逐组计算损失并反向
            for group in groups:
                stats = self.engine.compute_group_loss(group)
                if stats is None:
                    continue

                # 梯度累积：损失按累积步数缩放
                scaled_loss = stats["loss"] / self.args.gradient_accumulation_steps
                scaled_loss.backward()

                accum_loss += float(stats["loss"])
                for k, v in stats.items():
                    if k == "loss":
                        continue
                    accum_metrics[k] = accum_metrics.get(k, 0.0) + float(v)
                micro_count += 1
                tokens_seen += int(stats["num_tokens"])

            # 3) 满足累积步数则更新参数（micro_count 可跨 batch 累积）
            if micro_count >= self.args.gradient_accumulation_steps:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.student.parameters() if p.requires_grad],
                    self.args.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                # 记录本轮累积了多少个 micro 组，用于均值归一化
                n_micros = self.args.gradient_accumulation_steps
                micro_count = 0

                # 4) 记录日志
                avg = {k: v / n_micros for k, v in accum_metrics.items()}
                avg["train/loss"] = accum_loss / n_micros
                avg["train/lr"] = float(self.scheduler.get_last_lr()[0])
                avg["train/grad_norm"] = float(grad_norm)
                avg["train/tokens_seen"] = tokens_seen
                avg["train/step_seconds"] = round(
                    (time.time() - start_time) / max(1, self.global_step), 2
                )
                if gpu_memory_stats():
                    avg["gpu/allocated_gb"] = gpu_memory_stats()["allocated_gb"]

                if self.global_step % self.args.log_steps == 0:
                    logger.info(
                        "step %d/%d | loss %.4f | kd %.4f | ce %.4f | "
                        "align %.2f | resp_len %.0f | lr %.2e",
                        self.global_step,
                        self.max_update_steps,
                        avg["train/loss"],
                        avg.get("kd_loss", 0.0),
                        avg.get("ce_loss", 0.0),
                        avg.get("align_ratio", 0.0),
                        avg.get("mean_response_length", 0.0),
                        avg["train/lr"],
                    )
                self.swanlab.log(avg, step=self.global_step)
                accum_loss = 0.0
                accum_metrics = {}

                # 5) 定期保存 & 预览
                if self.args.save_steps > 0 and self.global_step % self.args.save_steps == 0:
                    self.save_model(tag=f"checkpoint-{self.global_step}")
                if (
                    self.args.preview_steps > 0
                    and self.global_step % self.args.preview_steps == 0
                ):
                    self._log_preview()

                if self.args.debug_steps > 0 and self.global_step >= self.args.debug_steps:
                    logger.info("达到 debug_steps=%d，提前结束", self.args.debug_steps)
                    break

        # ---------------- 收尾 ----------------
        final_dir = self.save_model(tag="final")
        if self.args.merge_after_train and self.args.use_lora:
            self._merge_lora(final_dir)
        self.swanlab.finish()
        logger.info("训练结束，总耗时 %.1f 分钟", (time.time() - start_time) / 60)
        logger.info("最终模型目录: %s", final_dir)

    def _merge_lora(self, adapter_dir: str):
        """训练结束后合并 LoRA 适配器并保存完整模型。"""
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoConfig

            cfg = AutoConfig.from_pretrained(
                self.args.student_model_path, trust_remote_code=True
            )
            from .models import _is_multimodal

            cls = AutoModelForImageTextToText if _is_multimodal(cfg) else AutoModelForCausalLM
            base = cls.from_pretrained(
                self.args.student_model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            merged = PeftModel.from_pretrained(base, adapter_dir)
            merged = merged.merge_and_unload()
            merged_dir = os.path.join(self.args.output_dir, "merged")
            merged.save_pretrained(merged_dir)
            self.student_tokenizer.save_pretrained(merged_dir)
            logger.info("LoRA 合并完成，完整模型保存到 %s", merged_dir)
        except Exception as e:
            logger.warning("LoRA 合并失败: %s", e)
