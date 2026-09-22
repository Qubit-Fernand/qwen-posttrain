#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-posttrain
cd ~/Qwen-PostTrain
python scripts/eval_gsm8k.py ./checkpoints/sft-gsm8k/checkpoint-64 200 2>&1 | tee eval_sft.log
