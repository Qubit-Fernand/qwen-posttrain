#!/usr/bin/env bash
# setup_env.sh - 在新机器上复现 llm-posttrain 环境 (Qwen 后训练)
set -euo pipefail

echo '== 1. 检查 conda =='
if ! command -v conda >/dev/null 2>&1; then
  echo '请先安装 miniconda: https://docs.conda.io/en/latest/miniconda.html'
  exit 1
fi

echo '== 2. 配置清华镜像 + 接受 ToS =='
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

echo '== 3. 创建 conda 环境 llm-posttrain (python 3.11) =='
conda create -y -n llm-posttrain python=3.11
conda activate llm-posttrain

echo '== 4. 安装 torch (cu124, 兼容 driver 550) =='
pip install torch --index-url https://download.pytorch.org/whl/cu124

echo '== 5. 安装 HF 全家桶 + swanlab (清华镜像) =='
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple   transformers trl huggingface_hub datasets accelerate peft swanlab

echo '== 6. 配置国内模型下载镜像 =='
grep -q HF_ENDPOINT ~/.zshrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
grep -q HF_ENDPOINT ~/.bashrc 2>/dev/null || echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc

echo '== 7. swanlab 登录 (交互式) =='
swanlab login

echo '== 完成! 之后用: conda activate llm-posttrain =='
