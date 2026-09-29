#!/bin/bash
# 一键启动 OPD 训练（AutoDL / Ubuntu）
# 前置：已按 README 安装依赖、下载模型与数据

set -e

# 1) 下载模型（已下载可跳过）
# python scripts/download_models.py

# 2) 下载并转换数据（使用 example_prompts 可跳过）
# python scripts/download_data.py --dataset AI-ModelScope/DAPO-Math-17k \
#     --out data/train_prompts.jsonl

# 3) SwanLab 登录（云端模式；也可设置环境变量 SWANLAB_API_KEY）
# swanlab login

# 4) 启动训练
python train.py --config configs/default.yaml "$@"
