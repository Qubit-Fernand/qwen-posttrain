# Qwen-PostTrain

Qwen3-1.7B 在 GSM8K 上的最小后训练 demo：SFT（监督微调）+ GRPO（RL）+ baseline eval。

## 目录结构

| 路径 | 内容 |
|---|---|
| scripts/ | 所有 Python 脚本 |
| scripts/sft_gsm8k.py | SFT demo：前 500 条 GSM8K，监督微调学标准答案 |
| scripts/grpo_gsm8k.py | GRPO demo：200 条 GSM8K，reward = 生成答案数字是否等于标准答案 |
| scripts/eval_gsm8k.py | 在 GSM8K test 上测 pass@1（baseline / SFT 后 / GRPO 后都用它） |
| scripts/test_swanlab.py | swanlab 监控台连通性测试（官方示例） |
| Qwen3-1.7B/ | 模型权重（bf16，.gitkeep 占位；本地/新机器自行下载） |
| data/ | GSM8K 数据集（openai/gsm8k, config=main）：train 7473 / test 1319；answer 结尾是 井号井号 数字 的标准答案。HF datasets arrow 缓存格式 |
| setup_env.sh | 新机器一键复现环境（conda + torch + HF 全家桶 + swanlab） |
| swanlog/ | swanlab 本地运行日志 |
| checkpoints/ | 训练输出目录（跑训练后自动生成） |

## 环境

- conda 环境：llm-posttrain（python 3.11, torch 2.6.0+cu124）
- 激活：source ~/miniconda3/etc/profile.d/conda.sh && conda activate llm-posttrain
- 监控：swanlab（已登录 Qubit 账号）

## 运行（都从项目根目录执行，不要 cd 进 scripts）

    cd ~/Qwen-PostTrain
    conda activate llm-posttrain

    # 1) 监控台连通性测试
    python scripts/test_swanlab.py

    # 2) 原始模型 baseline（先 200 条快速看）
    python scripts/eval_gsm8k.py ./Qwen3-1.7B 200

    # 3) SFT（前 500 条，约 10-20 分钟）
    HF_ENDPOINT=https://hf-mirror.com python scripts/sft_gsm8k.py

    # 4) GRPO（200 条，reward 从 ~0 往上涨）
    HF_ENDPOINT=https://hf-mirror.com python scripts/grpo_gsm8k.py

监控面板：https://swanlab.cn/@Fernand
- qwen3-1.7b-eval：baseline / SFT 后 / GRPO 后的 pass@1 对比
- qwen3-1.7b-sft：SFT loss 曲线
- qwen3-1.7b-grpo：GRPO reward 曲线

## 当前运行状态与已知问题

现有脚本已完成初步流程运行，但 GRPO 日志持续显示 `grad_norm=0`，生成几乎全部触及长度上限，且未保存 GRPO 模型。原因与参数是否更新仍待验证，不能据此声称强化学习已有效。详细证据、评估限制和排查顺序见 [Codex Memory](codex_memory.md) 与 [诊断指标](diagnostics/2026-09-22-run-review.json)。
