# Qwen-PostTrain

Qwen3-1.7B 在 GSM8K 上的最小后训练 demo：SFT（监督微调）+ GRPO（RL）。

## 目录结构

| 路径 | 内容 |
|---|---|
| Qwen3-1.7B/ | 模型权重（bf16，来自 HuggingFace，国内走 hf-mirror 下载） |
| data/ | GSM8K 数据集（openai/gsm8k, config=main）：train 7473 条 / test 1319 条；字段 question + answer（answer 结尾是 井号井号 数字 的标准答案）。HF datasets 缓存格式 |
| sft_gsm8k.py | SFT demo：前 500 条 GSM8K，监督微调学标准答案 |
| grpo_gsm8k.py | GRPO demo：200 条 GSM8K，reward = 生成答案数字是否等于标准答案 |
| test_swanlab.py | swanlab 监控台连通性测试脚本（官方示例） |
| swanlog/ | swanlab 本地运行日志（test_swanlab.py 产生） |
| checkpoints/ | 训练输出目录（跑训练后自动生成） |

## 环境

- conda 环境：llm-posttrain（python 3.11, torch 2.6.0+cu124）
- 激活：source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-posttrain
- 监控：swanlab（已登录 Qubit 账号）

## 运行

    cd ~/Qwen-PostTrain
    conda activate llm-posttrain

    # 1) 监控台连通性测试（已跑通）
    python test_swanlab.py

    # 2) SFT（前 500 条，约 10-20 分钟）
    HF_ENDPOINT=https://hf-mirror.com python sft_gsm8k.py

    # 3) GRPO（200 条，reward 从 ~0 往上涨）
    HF_ENDPOINT=https://hf-mirror.com python grpo_gsm8k.py

监控面板：https://swanlab.cn/@Fernand （项目 qwen3-1.7b-sft / qwen3-1.7b-grpo）
