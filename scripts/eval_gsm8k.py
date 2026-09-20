# eval_gsm8k.py - 在 GSM8K test 上测模型 pass@1 准确率
# 输出: 累计准确率(对了多少题/总共多少题), 不是单个 batch 的即时准确率
# 用法: python eval_gsm8k.py [模型路径] [样本数]
import re, sys
import swanlab
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else './Qwen3-1.7B'
MAX_SAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BATCH_SIZE = 8

def extract_num(text):
    nums = re.findall(r'-?[\d,]+(?:\.\d+)?', text)
    return nums[-1].replace(',', '') if nums else None

# test split, 不走训练缓存
ds = load_dataset('openai/gsm8k', 'main', split=f'test[:{MAX_SAMPLES}]', cache_dir='./data')

print(f'loading model: {MODEL_PATH}')
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype='bfloat16').cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = 'left'
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 根据模型来源自动分配曲线颜色:
#   baseline (./Qwen3-1.7B)  -> 灰
#   SFT 后 (checkpoints/sft*) -> 蓝
#   GRPO 后 (checkpoints/grpo*) -> 橙
if 'sft' in MODEL_PATH.lower():
    run_color = 'blue'
elif 'grpo' in MODEL_PATH.lower():
    run_color = 'orange'
else:
    run_color = 'gray'

swanlab.init(project='qwen3-1.7b-eval',
             color=run_color,
             config={'model': MODEL_PATH, 'samples': MAX_SAMPLES, 'split': 'test'})

correct = 0
total = 0
for i in range(0, MAX_SAMPLES, BATCH_SIZE):
    batch = ds.select(range(i, min(i + BATCH_SIZE, MAX_SAMPLES)))
    texts = [tokenizer.apply_chat_template(
                [{'role': 'user', 'content': q}],
                tokenize=False, add_generation_prompt=True)
             for q in batch['question']]
    inputs = tokenizer(texts, return_tensors='pt', padding=True,
                       truncation=True, max_length=512).to('cuda')
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    for j, o in enumerate(out):
        gen = tokenizer.decode(o[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        gt = extract_num(batch['answer'][j].split('####')[-1])
        pred = extract_num(gen)
        ok = gt is not None and pred is not None and gt == pred
        correct += int(ok)
        total += 1
    # acc 是【累计】准确率, 不是当前 batch(8 题)的即时准确率
    # total 是累计做过的题目数; 1 个 batch = BATCH_SIZE 道题
    acc = correct / total
    swanlab.log({
        'Total Accuracy (GSM8K pass@1)': acc,
        'Samples Done': total,
    })
    print(f'  {total}/{MAX_SAMPLES}  acc={acc:.4f}')

print(f'== final: {correct}/{total} = {correct/total:.4f} ==')
swanlab.finish()
