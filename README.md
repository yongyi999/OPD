# On-Policy Distillation：Qwen3.5 学生 × DeepSeek 教师

基于 **On-Policy Distillation（在线策略蒸馏，OPD）** 的单卡训练项目：学生模型针对问题**自己采样**回复，教师模型在学生自己生成的序列上提供 **token 级分布**，学生通过最小化 KL 散度向教师学习。

- **学生模型（Student）**：[`Qwen/Qwen3.5-0.8B`](https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B)（Qwen3.5 系列最小模型，混合线性注意力架构，LoRA 训练）
- **教师模型（Teacher）**：[`deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`](https://www.modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)（全程冻结，只提供 token 级分布）
- **实验记录**：[SwanLab](https://swanlab.cn) 云端看板，实时观察 loss / KL / 熵 / 对齐率 / 生成样例
- **目标硬件**：单张 **RTX 5090（32GB）**，Python 3.12 + CUDA 12.8 + PyTorch 2.8.0

---

## 1. 方法简介

### 什么是 On-Policy Distillation

传统（off-policy）蒸馏：先准备好教师生成的固定回复，学生在这些**固定文本**上做监督学习（SFT）。

**On-Policy Distillation**：每个训练步让学生**自己采样**回复，然后让教师在这些回复上逐 token 给出分布，学生在"自己会犯的错误"上学习，缓解暴露偏差（exposure bias）。

```
每个训练步：
  1. 取一个 prompt
  2. 学生用当前策略采样 n 条回复（on-policy rollout）
  3. 教师在学生自生成的序列上做前向，给出 token 级分布
  4. 计算 KL 损失，只更新学生
```

本项目默认搭配是**跨 tokenizer**（Qwen3.5 词表 248k，DeepSeek/Qwen2 词表 152k），处理方式：
- 学生用自己的 tokenizer 采样，教师用自己的 tokenizer 重新编码；
- 用**累积文本匹配**对齐两边的 token 位置；
- 只在两个词表的**共有 token 子集合**（约 13 万 token）上计算 KL（思路来自 [KDFlow](https://github.com/songmzhang/KDFlow)）。

### 损失公式

- 学生采样的回复：**reverse KL**（学生概率加权，[MiniLLM](https://github.com/microsoft/LMOps/tree/main/minillm) 思路）

  $$L = \sum_{v \in \mathcal{V}_{ov}} p_s(v)\,(\log p_s(v) - \log p_t(v))$$

- 教师混合采样的回复（可选，`teacher_mix_ratio>0`）：**forward KL**（教师概率加权）
- 可选混入 CE：`kd_ratio` 控制 KD 权重（`1-kd_ratio` 为 CE 权重）

---

## 2. 项目结构

```
OPD/
├── train.py                  # 训练入口
├── requirements.txt          # 依赖清单
├── configs/
│   └── default.yaml          # 默认配置（RTX 5090 单卡）
├── src/
│   ├── arguments.py          # 训练参数（YAML + 命令行）
│   ├── data.py               # prompt 数据加载、chat template
│   ├── models.py             # 师生模型加载、LoRA、教师前向
│   ├── rollout.py            # 学生 on-policy 采样、教师混合采样
│   ├── distillation.py       # KL 损失、跨 tokenizer 对齐
│   ├── trainer.py            # 训练主循环
│   ├── logger.py             # SwanLab 封装
│   └── utils.py              # 工具函数
├── scripts/
│   ├── download_models.py    # 下载师生模型
│   ├── download_data.py      # 下载并转换训练数据
│   ├── merge_lora.py         # 合并 LoRA 适配器
│   └── smoke_test.py         # 冒烟测试（CPU，极小模型）
├── data/
│   └── example_prompts.jsonl # 内置示例 prompt（中英混合，约 85 条）
└── runs/                     # 训练产物（gitignore）
```

---

## 3. AutoDL 环境搭建（重点）

### 3.1 租用实例

1. 在 [AutoDL](https://www.autodl.com) 选择 **RTX 5090（32GB）** 单卡；
2. 镜像选择（或手动创建）：
   - **Ubuntu 22.04 / 24.04**
   - **Python 3.12**
   - **CUDA 12.8**（PyTorch 2.8.0 自带运行时，AutoDL 镜像只要驱动支持即可）
3. 开机后进入终端（JupyterLab 终端或 SSH）。

> RTX 5090 是 Blackwell 架构（sm_120），**必须使用 CUDA 12.8 版本的 PyTorch**，低版本 CUDA 的 torch 无法调用显卡。

### 3.2 拉取代码

```bash
git clone https://github.com/yongyi999/OPD.git
cd OPD
```

### 3.3 安装 PyTorch（cu128）

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 应输出: 2.8.0 True NVIDIA GeForce RTX 5090...
```

### 3.4 安装其余依赖

```bash
pip install -r requirements.txt
```

> 其中 `bitsandbytes` 用于 8-bit 优化器；如安装失败可忽略，默认使用普通 AdamW。

**（可选）安装 Qwen3.5 混合架构加速包**，能显著加快线性注意力层（rollout 与前向）：

```bash
pip install causal-conv1d flash-linear-attention
```

不装也能跑，日志出现 `The fast path is not available ... Falling back to torch implementation` 属正常，只是速度较慢。

### 3.5 登录 SwanLab

```bash
pip install swanlab
swanlab login
```

按提示在 [swanlab.cn](https://swanlab.cn) 注册并复制 API Key。也可以用环境变量：

```bash
export SWANLAB_API_KEY="你的key"
```

> 不想用云端时，训练时加 `--swanlab_mode local`，数据写本地，之后用 `swanlab watch runs/swanlab` 打开。

### 3.6 下载模型

```bash
python scripts/download_models.py
```

默认下载到：
- `models/Qwen3.5-0.8B`（学生，约 1.8GB）
- `models/DeepSeek-R1-Distill-Qwen-7B`（教师，约 15GB）

### 3.7 准备训练数据

**方式 A（快速试用）**：直接使用内置的 `data/example_prompts.jsonl`，无需下载。

**方式 B（推荐，正式训练）**：下载数学推理数据集（DAPO-Math-17k，该镜像为 179 万行版本）：

```bash
python scripts/download_data.py --dataset AI-ModelScope/DAPO-Math-17k \
    --out data/train_prompts.jsonl
```

其他可选数据：

```bash
# 中文通用指令
python scripts/download_data.py --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --out data/train_prompts.jsonl
# 英文通用指令
python scripts/download_data.py --dataset AI-ModelScope/alpaca-gpt4-data-en \
    --out data/train_prompts.jsonl
```

OPD 只需要 prompt（问题），回复由学生在训练时自己生成。

---

## 4. 启动训练

### 先用内置数据跑通（建议第一次先跑）

```bash
python train.py --config configs/default.yaml --debug_steps 2 --swanlab_mode local
```

### 正式训练（使用下载的数据）

```bash
python train.py --config configs/default.yaml \
    --data_path data/train_prompts.jsonl \
    --max_train_samples 10000
```

或直接一键脚本：

```bash
bash run.sh
```

### 常用自定义参数

```bash
python train.py --config configs/default.yaml \
    --num_rollouts 4 \              # 每个 prompt 采样几条回复
    --max_response_length 1024 \    # 回复最大长度
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --save_steps 100 \
    --preview_steps 50 \
    --swanlab_project on-policy-distillation
```

全部参数见 `src/arguments.py` 与 `configs/default.yaml`。

---

## 5. 在 SwanLab 查看训练

启动后终端会打印 SwanLab 实验链接，云端面板可看到：

| 指标 | 含义 |
|------|------|
| `train/loss` | 总损失（KD + CE） |
| `kd_loss` | 平均每个 token 的 KL 散度 |
| `ce_loss` | 学生在采样 token 上的交叉熵 |
| `student_entropy` / `teacher_entropy` | 学生 / 教师分布熵 |
| `align_ratio` | 跨 tokenizer 位置对齐率（越高越好） |
| `mean_response_length` | 学生回复平均长度 |
| `train/lr` / `train/grad_norm` | 学习率 / 梯度范数 |
| `gpu/allocated_gb` | 显存占用 |
| `preview/generation` | 学生对固定 prompt 的生成样例（可看进步） |

---

## 6. 训练产物与合并

模型默认保存在 `runs/opd/`：
- `final/`：最终 LoRA 适配器（默认）
- `checkpoint-<step>/`：定期检查点

**合并 LoRA 为完整模型**（便于推理/部署）：

```bash
python scripts/merge_lora.py \
    --base_model models/Qwen3.5-0.8B \
    --adapter runs/opd/final \
    --out runs/opd/merged
```

合并后用 `AutoModelForCausalLM.from_pretrained("runs/opd/merged")` 直接加载。

---

## 7. 冒烟测试（可选，验证代码）

无需大模型，用随机极小模型在 CPU 上跑通完整流程（跨 tokenizer + 相同 tokenizer 两条路径）：

```bash
python scripts/smoke_test.py
```

---

## 8. 调参建议

- **LoRA**：默认 `r=64, alpha=128`，单卡 32GB 足够；显存紧张可降到 `r=32`。
- **全参数微调**：加 `--use_lora false`（0.8B 学生可行；建议同时 `--optim paged_adamw_8bit`）。
- **rollout 数量**：`num_rollouts=4` 是性价比甜点；更多（8）更稳但更慢。
- **温度**：采样温度 1.0；学生回复太发散可降到 0.8。
- **冷启动**：如果学生太弱、KL 学不动，可设 `--teacher_mix_ratio 0.2`（混入教师采样，使用 forward KL）。
- **学习率**：LoRA 用 1e-5；全参用 2e-6 ~ 5e-6。
- **跨 tokenizer 对齐率低**：真实模型通常 0.7~0.95；若异常低，检查 prompt 中是否有特殊字符。

---

## 9. 常见问题

1. **`torch.cuda.is_available()` 为 False**：torch 装错了 CUDA 版本，必须用 `--index-url .../cu128`。
2. **`CUDA error: no kernel image is available`**：同上，Blackwell 必须 CUDA 12.8 + torch 2.8。
3. **显存不足（OOM）**：减小 `--per_device_train_batch_size 1`、`--max_response_length 512`、`--lora_r 32`，或开启 `--teacher_cpu_offload`（rollout 时教师临时放 CPU）。
4. **SwanLab 初始化失败**：训练不受影响，可加 `--swanlab_mode offline` 或检查网络/API Key。
5. **`The fast path is not available`**：未装 fla/conv1d，仅是速度提示，可忽略。

---

## 10. 参考与致谢

本项目在方法与代码思路上参考了以下工作：

- **MiniLLM**: Knowledge Distillation of Large Language Models（reverse KL）— https://github.com/microsoft/LMOps/tree/main/minillm
- **Rethinking On-Policy Distillation**（ICML 2026 FoGen Workshop，top-k 配方）— https://github.com/Thinking-Space/Rethinking-OPD
- **KDFlow**（跨 tokenizer 蒸馏、overlap 子词表）— https://github.com/songmzhang/KDFlow
- **On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes**
- **verl**（rollout/token-reward 设计）— https://github.com/verl-project/verl
- **ModelScope**（模型与数据下载）— https://www.modelscope.cn
- **SwanLab**（实验管理）— https://swanlab.cn

---

## 11. License

[Apache License 2.0](LICENSE)

模型与数据的使用请分别遵循 Qwen、DeepSeek 及对应数据集的许可协议。
