#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm-posttrain
cd ~/Qwen-PostTrain
python scripts/sft_then_eval.py 2>&1 | tee sft_eval_inline.log
