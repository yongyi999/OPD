"""蒸馏损失计算。

支持两种师生搭配：
1. 相同 tokenizer：直接在同一套 token id 上计算
   - topk：学生 top-k 集合上的 KL（Rethinking-OPD 配方，默认）
   - sampled：仅采样 token 上的 log 概率差（verl LOG_PROB_TOP_K=0）
   - full：全词表 KL（分块）
2. 跨 tokenizer（Qwen3.5 学生 + DeepSeek 教师，本项目默认）：
   - overlap：在两个词表的共有 token 子集合上计算 KL（KDFlow 思路），
     并用累积文本匹配对齐师生的 token 位置

学生采样的回复使用 reverse KL（学生概率加权）；
教师混合采样的回复使用 forward KL（教师概率加权）。
"""

import torch
import torch.nn.functional as F

from .data import build_prompt_text
from .models import teacher_forward_logits
from .utils import get_logger

logger = get_logger(__name__)


# ===================================================================== #
# 跨 tokenizer 工具
# ===================================================================== #
def build_overlap_ids(student_tokenizer, teacher_tokenizer, device: str):
    """找出两个 tokenizer 词表中"表面形式相同"的 token 集合。

    返回 (student_overlap_ids, teacher_overlap_ids)，两者一一对应。
    """
    def _norm_vocab(tok):
        # GPT2 风格的 Ġ 与 SentencePiece 风格的 ▁ 都表示词首空格，统一后比较
        return {k.replace("Ġ", "▁"): v for k, v in tok.get_vocab().items()}

    s_vocab = _norm_vocab(student_tokenizer)
    t_vocab = _norm_vocab(teacher_tokenizer)
    common = set(s_vocab.keys()) & set(t_vocab.keys())

    s_ids = [s_vocab[k] for k in common]
    t_ids = [t_vocab[k] for k in common]

    # 保证 eos 也在集合中（两侧 eos 文本可能不同，单独追加）
    s_eos, t_eos = student_tokenizer.eos_token_id, teacher_tokenizer.eos_token_id
    if s_eos not in s_ids:
        s_ids.append(s_eos)
        t_ids.append(t_eos)

    logger.info("师生词表共有 token 数: %d", len(s_ids))
    return (
        torch.tensor(s_ids, dtype=torch.long, device=device),
        torch.tensor(t_ids, dtype=torch.long, device=device),
    )


def _filter_special_ids(tokenizer, ids: list[int]) -> list[int]:
    """过滤掉特殊 token id。"""
    special = set(tokenizer.all_special_ids)
    return [i for i in ids if i not in special]


def align_response_positions(
    student_tokenizer,
    s_response_ids: list[int],
    teacher_tokenizer,
    t_response_ids: list[int],
) -> tuple[list[int], list[int]]:
    """对齐师生对同一段回复文本的 token 位置。

    采用累积文本匹配：各自把 token 逐个解码成文本片段，每当两边累积
    文本相同且当前片段一致时，记录一对对齐位置。类似 KDFlow 的做法。
    返回 (student_indices, teacher_indices)，下标为各自响应 token 序号。
    """
    s_pieces = [student_tokenizer.decode([i]) for i in s_response_ids]
    t_pieces = [teacher_tokenizer.decode([i]) for i in t_response_ids]

    i = j = 0
    s_hist = t_hist = ""
    s_idx, t_idx = [], []

    while i < len(s_pieces) and j < len(t_pieces):
        sp, tp = s_pieces[i], t_pieces[j]
        if s_hist == t_hist and sp == tp:
            s_idx.append(i)
            t_idx.append(j)
            s_hist += sp
            t_hist += tp
            i += 1
            j += 1
        elif len(s_hist) < len(t_hist):
            s_hist += sp
            i += 1
        elif len(s_hist) > len(t_hist):
            t_hist += tp
            j += 1
        else:
            # 累积文本相同但当前片段不同（合并粒度不同），各自前进一步
            s_hist += sp
            t_hist += tp
            i += 1
            j += 1

    return s_idx, t_idx


# ===================================================================== #
# 蒸馏引擎
# ===================================================================== #
class DistillationEngine:
    """对 rollout 结果计算 KD / CE 损失。"""

    def __init__(self, student, teacher, student_tokenizer, teacher_tokenizer, args):
        self.student = student
        self.teacher = teacher
        self.s_tok = student_tokenizer
        self.t_tok = teacher_tokenizer
        self.args = args

        from .models import is_same_tokenizer

        self.same_tokenizer = is_same_tokenizer(student_tokenizer, teacher_tokenizer)
        logger.info("师生 tokenizer 是否相同: %s", self.same_tokenizer)

        # 确定实际使用的 KL 模式
        self.kl_mode = args.kl_mode
        if self.kl_mode == "auto":
            self.kl_mode = "topk" if self.same_tokenizer else "overlap"
        logger.info("使用 KL 模式: %s", self.kl_mode)

        if self.kl_mode == "overlap":
            self.s_overlap, self.t_overlap = build_overlap_ids(
                student_tokenizer, teacher_tokenizer, args.device
            )
        else:
            self.s_overlap = self.t_overlap = None

    # ----------------------------------------------------------------- #
    # 基础：学生前向（训练，带梯度）
    # ----------------------------------------------------------------- #
    def _student_forward(self, full_ids: torch.Tensor):
        """学生模型前向，返回 logits（带梯度）。"""
        full_ids = full_ids.to(self.args.device).reshape(1, -1)  # 保证 2D
        attention_mask = torch.ones_like(full_ids)
        out = self.student(
            input_ids=full_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.logits

    # ----------------------------------------------------------------- #
    # 基础：教师前向（冻结，无梯度）
    # ----------------------------------------------------------------- #
    def _teacher_forward(self, full_ids: torch.Tensor):
        """教师模型前向，返回 logits（无梯度）。"""
        full_ids = full_ids.to(self.args.device).reshape(1, -1)  # 保证 2D
        attention_mask = torch.ones_like(full_ids)
        return teacher_forward_logits(
            self.teacher,
            full_ids,
            attention_mask,
            token_chunk=self.args.logits_token_chunk,
        )

    # ----------------------------------------------------------------- #
    # 相同 tokenizer 的损失
    # ----------------------------------------------------------------- #
    def _same_tokenizer_loss(
        self,
        full_ids: torch.Tensor,
        prompt_len: int,
        from_teacher: bool,
    ) -> dict:
        """对同一条序列计算相同 tokenizer 下的 KD/CE 损失。"""
        s_logits = self._student_forward(full_ids)
        t_logits = self._teacher_forward(full_ids)

        # 响应预测位置：[prompt_len-1, seq_len-2]
        positions = torch.arange(prompt_len - 1, full_ids.shape[0] - 1)
        labels = full_ids[positions + 1].to(self.args.device)

        s_logits = s_logits[0, positions] / self.args.kl_temperature
        t_logits = t_logits[0, positions] / (
            self.args.kl_temperature * self.args.teacher_temperature
        )

        kd_loss = torch.zeros((), device=self.args.device)
        mode = self.kl_mode

        if mode == "topk":
            k = min(self.args.top_k, s_logits.shape[-1])
            if from_teacher:
                # 教师采样的回复：用教师 top-k 集合 + forward KL
                top_vals, top_idx = t_logits.topk(k, dim=-1)
                logp_t = F.log_softmax(top_vals, dim=-1)
                logp_s = F.log_softmax(
                    torch.gather(s_logits, -1, top_idx), dim=-1
                )
                p_t = logp_t.exp()
                kd_loss = (p_t * (logp_t - logp_s)).sum(-1)
            else:
                # 学生采样：学生 top-k 集合 + reverse KL
                top_vals, top_idx = s_logits.topk(k, dim=-1)
                logp_s = F.log_softmax(top_vals, dim=-1)
                logp_t = F.log_softmax(
                    torch.gather(t_logits, -1, top_idx), dim=-1
                )
                p_s = logp_s.exp()
                kd_loss = (p_s * (logp_s - logp_t)).sum(-1)

        elif mode == "sampled":
            logp_s = F.log_softmax(s_logits, dim=-1)
            logp_t = F.log_softmax(t_logits, dim=-1)
            sampled_logp_s = torch.gather(logp_s, -1, labels[:, None]).squeeze(-1)
            sampled_logp_t = torch.gather(logp_t, -1, labels[:, None]).squeeze(-1)
            if from_teacher:
                kd_loss = sampled_logp_t - sampled_logp_s  # forward 方向
            else:
                kd_loss = sampled_logp_s - sampled_logp_t  # reverse 方向

        elif mode == "full":
            # 全词表 KL，按位置分块防止中间张量过大
            kd_chunks = []
            for s in range(0, s_logits.shape[0], self.args.logits_token_chunk):
                sl = s_logits[s : s + self.args.logits_token_chunk]
                tl = t_logits[s : s + self.args.logits_token_chunk]
                logp_s, logp_t = F.log_softmax(sl, -1), F.log_softmax(tl, -1)
                if from_teacher:
                    p_t = logp_t.exp()
                    kd_chunks.append((p_t * (logp_t - logp_s)).sum(-1))
                else:
                    p_s = logp_s.exp()
                    kd_chunks.append((p_s * (logp_s - logp_t)).sum(-1))
            kd_loss = torch.cat(kd_chunks, dim=0)
        else:
            raise ValueError(f"相同 tokenizer 下不支持模式: {mode}")

        # CE：学生在采样 token 上的交叉熵（两种来源都要学）
        ce_loss = F.cross_entropy(
            s_logits, labels, reduction="none"
        )

        # 诊断信息
        with torch.no_grad():
            s_ent = -(F.softmax(s_logits, -1) * F.log_softmax(s_logits, -1)).sum(-1).mean()
            t_ent = -(F.softmax(t_logits, -1) * F.log_softmax(t_logits, -1)).sum(-1).mean()

        return {
            "kd_sum": kd_loss.sum(),
            "ce_sum": ce_loss.sum(),
            "num_tokens": torch.tensor(float(positions.numel()), device=self.args.device),
            "student_entropy": s_ent.detach(),
            "teacher_entropy": t_ent.detach(),
            "align_ratio": torch.tensor(1.0),
        }

    # ----------------------------------------------------------------- #
    # 跨 tokenizer（overlap）的损失
    # ----------------------------------------------------------------- #
    def _cross_tokenizer_loss(
        self,
        raw_prompt: str,
        s_prompt_ids: torch.Tensor,
        s_response_ids: torch.Tensor,
        from_teacher: bool,
    ) -> dict:
        """跨 tokenizer 下，在共有词表子集合上计算 KL。"""
        s_prompt_ids = s_prompt_ids.to(self.args.device).reshape(-1)
        s_response_ids = s_response_ids.to(self.args.device).reshape(-1)
        s_full = torch.cat([s_prompt_ids, s_response_ids], dim=0)
        prompt_len = s_prompt_ids.shape[0]

        # 学生前向
        s_logits = self._student_forward(s_full)  # [1, L, V_s]

        # 学生响应文本（过滤特殊 token），教师据此重新分词
        s_text_ids = _filter_special_ids(
            self.s_tok, s_response_ids.detach().cpu().tolist()
        )
        response_text = self.s_tok.decode(s_text_ids, skip_special_tokens=True)

        # 教师侧文本与编码
        teacher_prompt_text = build_prompt_text(
            self.t_tok, raw_prompt, self.args.system_prompt
        )
        t_prompt_ids = self.t_tok(
            teacher_prompt_text, add_special_tokens=False
        )["input_ids"]
        t_resp_ids = self.t_tok(response_text, add_special_tokens=False)["input_ids"]
        t_full = torch.tensor(
            t_prompt_ids + t_resp_ids, dtype=torch.long, device=self.args.device
        )
        t_prompt_len = len(t_prompt_ids)

        # 教师前向
        t_logits = self._teacher_forward(t_full)  # [1, L_t, V_t]

        # 对齐位置（只在非特殊 token 上对齐）
        s_aligned, t_aligned = align_response_positions(
            self.s_tok, s_text_ids, self.t_tok, t_resp_ids
        )

        kd_sum = torch.zeros((), device=self.args.device)
        if len(s_aligned) > 0:
            s_pos = torch.tensor(
                [prompt_len - 1 + i for i in s_aligned], device=self.args.device
            )
            t_pos = torch.tensor(
                [t_prompt_len - 1 + i for i in t_aligned], device=self.args.device
            )

            # 收集共有词表上的 logits
            s_sub = s_logits[0].index_select(0, s_pos)
            t_sub = t_logits[0].index_select(0, t_pos)
            s_sub = s_sub.index_select(1, self.s_overlap) / self.args.kl_temperature
            t_sub = t_sub.index_select(1, self.t_overlap) / (
                self.args.kl_temperature * self.args.teacher_temperature
            )

            logp_s = F.log_softmax(s_sub, dim=-1)
            logp_t = F.log_softmax(t_sub, dim=-1)
            if from_teacher:
                p_t = logp_t.exp()
                kd = (p_t * (logp_t - logp_s)).sum(-1)
            else:
                p_s = logp_s.exp()
                kd = (p_s * (logp_s - logp_t)).sum(-1)
            kd_sum = kd.sum()

        # CE：学生在自己所有响应 token 上的交叉熵
        ce_positions = torch.arange(prompt_len - 1, s_full.shape[0] - 1)
        ce_labels = s_full[ce_positions + 1]
        ce_loss = F.cross_entropy(
            s_logits[0, ce_positions], ce_labels, reduction="sum"
        )

        with torch.no_grad():
            s_ent = -(F.softmax(s_logits[0, ce_positions], -1)
                      * F.log_softmax(s_logits[0, ce_positions], -1)).sum(-1).mean()
            t_ent = -(F.softmax(t_logits[0, t_prompt_len - 1:], -1)
                      * F.log_softmax(t_logits[0, t_prompt_len - 1:], -1)).sum(-1).mean()
            total_resp = max(len(s_text_ids), 1)
            align_ratio = torch.tensor(len(s_aligned) / total_resp)

        return {
            "kd_sum": kd_sum,
            "ce_sum": ce_loss,
            "num_tokens": torch.tensor(float(ce_positions.numel()), device=self.args.device),
            "student_entropy": s_ent.detach(),
            "teacher_entropy": t_ent.detach(),
            "align_ratio": align_ratio,
        }

    # ----------------------------------------------------------------- #
    # 对外入口：处理一个 prompt 的全部 rollout
    # ----------------------------------------------------------------- #
    def compute_group_loss(self, group: dict) -> dict:
        """计算一个 prompt 对应的 n 条回复的总损失。

        group 来自 rollout.generate_student_rollouts / generate_teacher_rollouts。
        """
        from_teacher = group["source"] == "teacher"
        response_ids = group["response_ids"]  # [n, R]

        # 跨 tokenizer 且为教师采样：先把教师回复转成文本，再用学生 tokenizer 编码
        converted_groups = None
        if from_teacher and not self.same_tokenizer:
            from .data import build_prompt_text as _bpt

            student_prompt_text = _bpt(
                self.s_tok, group["prompt"], self.args.system_prompt
            )
            s_prompt_ids = torch.tensor(
                self.s_tok(student_prompt_text, add_special_tokens=False)["input_ids"],
                device=self.args.device,
            )
            converted = []
            for j in range(response_ids.shape[0]):
                resp = response_ids[j]
                resp = resp[resp != self.t_tok.pad_token_id]
                text_ids = _filter_special_ids(self.t_tok, resp.cpu().tolist())
                text = self.t_tok.decode(text_ids, skip_special_tokens=True)
                s_resp = self.s_tok(text, add_special_tokens=False)["input_ids"]
                converted.append(torch.tensor(s_resp, device=self.args.device))
            # 用学生侧编码重建 group，后续按学生回复流程处理
            response_ids = torch.nn.utils.rnn.pad_sequence(
                converted, batch_first=True, padding_value=self.s_tok.pad_token_id
            )
            group = {
                "prompt": group["prompt"],
                "prompt_ids": s_prompt_ids,
                "source": "teacher",
            }

        kd_total = torch.zeros((), device=self.args.device)
        ce_total = torch.zeros((), device=self.args.device)
        token_total = 0.0
        s_ents, t_ents, align_ratios, resp_lens = [], [], [], []

        for j in range(response_ids.shape[0]):
            resp = response_ids[j]
            # 去掉尾部 padding
            resp = resp[resp != self.s_tok.pad_token_id]
            if resp.numel() == 0:
                continue
            resp_lens.append(resp.numel())

            if self.same_tokenizer:
                s_prompt_ids = group["prompt_ids"].reshape(-1)
                full_ids = torch.cat([s_prompt_ids, resp.to(s_prompt_ids.device)])
                stats = self._same_tokenizer_loss(
                    full_ids, s_prompt_ids.numel(), from_teacher
                )
            else:
                stats = self._cross_tokenizer_loss(
                    group["prompt"],
                    group["prompt_ids"],
                    resp,
                    from_teacher,
                )

            kd_total = kd_total + stats["kd_sum"]
            ce_total = ce_total + stats["ce_sum"]
            token_total += float(stats["num_tokens"])
            s_ents.append(float(stats["student_entropy"]))
            t_ents.append(float(stats["teacher_entropy"]))
            align_ratios.append(float(stats["align_ratio"]))

        if token_total == 0:
            return None

        # token-mean：总损失 / 总 token 数
        kd_mean = kd_total / token_total
        ce_mean = ce_total / token_total
        loss = self.args.kd_ratio * kd_mean + (1 - self.args.kd_ratio) * ce_mean

        return {
            "loss": loss,
            "kd_loss": kd_mean.detach(),
            "ce_loss": ce_mean.detach(),
            "num_tokens": token_total,
            "student_entropy": sum(s_ents) / len(s_ents),
            "teacher_entropy": sum(t_ents) / len(t_ents),
            "align_ratio": sum(align_ratios) / len(align_ratios),
            "mean_response_length": sum(resp_lens) / len(resp_lens),
        }
