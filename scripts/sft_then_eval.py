#!/usr/bin/env python3
"""SFT 训练完直接在同一个进程里 eval, 不存 checkpoint"""
import re, torch, swanlab
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

MODEL_PATH = './Qwen3-1.7B'
N_TRAIN = 500
N_EVAL = 200

# 1) 数据
ds = load_dataset('openai/gsm8k', 'main', split=f'train[:{N_TRAIN}]')
def fmt(ex):
    q = ex['question']
    a = ex['answer'].split('####')[-1].strip()
    return {'messages': [
        {'role':'user','content':q},
        {'role':'assistant','content':f'The answer is {a}.'}
    ]}
ds = ds.map(fmt, remove_columns=ds.column_names)

# 2) 模型 + LoRA
tok = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16).cuda()
lora = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'],
                  lora_dropout=0.05, bias='none', task_type='CAUSAL_LM')

swanlab.init(project='qwen3-1.7b-sft', experiment_name='sft-then-eval')
args = SFTConfig(
    output_dir='./checkpoints/sft-tmp', per_device_train_batch_size=8,
    num_train_epochs=2, learning_rate=1e-4, logging_steps=10,
    save_strategy='no', report_to='swanlab', max_length=512,
)
trainer = SFTTrainer(model=model, args=args, train_dataset=ds,
                     processing_class=tok, peft_config=lora)
trainer.train()
swanlab.finish()

# 3) 同一个进程里直接 eval
print('\n=== SFT 完成, 开始 eval ===')
model.config.use_cache = True
model.eval()
eval_ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{N_EVAL}]')
correct = 0
total = 0
swanlab.init(project='qwen3-1.7b-eval', experiment_name='sft-eval-inline')
for i, ex in enumerate(eval_ds):
    q = ex['question']
    ref = ex['answer'].split('####')[-1].strip()
    messages = [{'role':'user','content':q}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    nums = re.findall(r'-?[\d,]+\.?\d*', text)
    pred = nums[-1].replace(',','') if nums else ''
    if pred and abs(float(pred) - float(ref)) < 1e-4:
        correct += 1
    total += 1
    if total % 20 == 0:
        acc = correct / total
        print(f'{total}/{N_EVAL}  acc={acc:.4f}')
        swanlab.log({'acc_running': acc, 'samples': total})

final = correct / total
print(f'\n=== SFT eval: {correct}/{total} = {final:.4f} ===')
swanlab.log({'final_acc': final})
swanlab.finish()
