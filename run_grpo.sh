#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-posttrain
cd ~/Qwen-PostTrain
python scripts/grpo_gsm8k.py 2>&1 | tee grpo_train.log
