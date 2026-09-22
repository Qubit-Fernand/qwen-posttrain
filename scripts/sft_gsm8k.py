# sft_gsm8k.py — Qwen3-1.7B 在 GSM8K 上的最小 SFT demo
# 运行（激活 llm-posttrain 环境后）:
#   HF_ENDPOINT=https://hf-mirror.com python sft_gsm8k.py
import swanlab
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

# ---- TODO: 模型权重下完后，把这里改成本地路径 ----
MODEL_PATH = "./Qwen3-1.7B"
MAX_SAMPLES = 500          # demo 先取 500 条，跑通后再加大

# 1) 数据: question -> user, answer -> assistant
ds = load_dataset("openai/gsm8k", "main", split=f"train[:{MAX_SAMPLES}]", cache_dir="./data")

def to_messages(ex):
    return {"messages": [
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]},
    ]}

ds = ds.map(to_messages, remove_columns=ds.column_names)

# 2) 模型 + tokenizer (bf16)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype="bfloat16")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# LoRA: 只训练低秩适配器, 不存完整权重
lora_config = LoraConfig(
    r=8, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM",
)

# 3) swanlab 监控
swanlab.init(project="qwen3-1.7b-sft",
             config={"model": MODEL_PATH, "max_samples": MAX_SAMPLES})

# 4) 训练配置 (单卡 A100 40G)
args = SFTConfig(
    output_dir="./checkpoints/sft-gsm8k",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,      # 等效 batch 16
    learning_rate=1e-5,
    num_train_epochs=2,
    bf16=True,
    max_length=1024,
    gradient_checkpointing=True,
    logging_steps=5,
    save_strategy="epoch",                # demo 不存 checkpoint，省空间
    report_to="swanlab",
)

# 5) 开跑
trainer = SFTTrainer(model=model, args=args, train_dataset=ds, processing_class=tokenizer, peft_config=lora_config)
trainer.train()
swanlab.finish()
