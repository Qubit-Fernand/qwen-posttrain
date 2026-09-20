# grpo_gsm8k.py — Qwen3-1.7B 在 GSM8K 上的最小 GRPO demo
# 运行（激活 llm-posttrain 环境后）:
#   HF_ENDPOINT=https://hf-mirror.com python grpo_gsm8k.py
import re
import swanlab
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

# ---- TODO: 模型权重下完后, 改成本地路径 ----
MODEL_PATH = 'Qwen/Qwen3-1.7B'
MAX_SAMPLES = 200           # GRPO 更贵(每步要生成), demo 先取 200 条

# 1) 数据: 只要 question 当 prompt; answer 不是训练目标, 留作 reward 的标准答案
ds = load_dataset('openai/gsm8k', 'main', split=f'train[:{MAX_SAMPLES}]', cache_dir='./data')

def to_prompt(ex):
    return {'prompt': [{'role': 'user', 'content': ex['question']}],
            'answer': ex['answer']}

ds = ds.map(to_prompt, remove_columns=['question'])

# 2) reward: 从模型生成里提最终数字, 和 GSM8K 标准答案(#### 后的数字)比对
def extract_num(text):
    nums = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None

def reward_accuracy(completions, answer, **kwargs):
    rewards = []
    for comp, gt in zip(completions, answer):
        gen = comp[-1]['content']                 # 模型生成的回答
        gt_num = extract_num(gt.split('####')[-1])  # 标准答案数字
        pred_num = extract_num(gen)                 # 模型猜的数字
        rewards.append(1.0 if gt_num and pred_num and gt_num == pred_num else 0.0)
    return rewards

# 3) 模型
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype='bfloat16')
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

swanlab.init(project='qwen3-1.7b-grpo',
             config={'model': MODEL_PATH, 'max_samples': MAX_SAMPLES})

# 4) GRPO 配置 (单卡 A100 40G)
args = GRPOConfig(
    output_dir='./checkpoints/grpo-gsm8k',
    num_generations=4,               # K: 每道题采样 4 个回答, 组内按 reward 排名
    max_completion_length=512,       # 每答最多生成多少 token
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=1e-6,              # RL 阶段用小 lr
    num_train_epochs=1,
    bf16=True,
    logging_steps=5,
    save_strategy='no',
    report_to='swanlab',
    gradient_checkpointing=True,
)

# 5) 开跑
trainer = GRPOTrainer(model=model, args=args, train_dataset=ds,
                      processing_class=tokenizer,
                      reward_funcs=[reward_accuracy])
trainer.train()
swanlab.finish()
