# Codex Memory

## 项目定位与访问

本项目是 Qwen3-1.7B 在 GSM8K 上的单卡后训练 demo，包括 LoRA SFT、GRPO 和评估；尚不能当作已经验证有效的强化学习训练流程。

- A100 仓库：`/data/users/zhoushuo/Qwen-PostTrain`。
- 从 Mac 访问优先使用 `ssh-vpn A100`；运行脚本从仓库根目录执行。
- Conda 环境：`llm-posttrain`。
- 远端：`git@github.com:Qubit-Fernand/qwen-posttrain.git`，分支 `main`。
- 模型权重、checkpoints、运行日志与派生 Arrow 缓存不提交；已跟踪的原始 GSM8K 数据保留。
- `run_sft.sh`、`run_grpo.sh`、`run_eval_sft.sh`、`run_sft_eval.sh` 是现有运行入口；执行会启动训练或评估，检查仓库时不要自动运行。

## 2026-09-22：历史运行问题记录

本次重新读取了 2026-09-20 的日志与当前脚本，未启动新的训练，也未修复以下训练/评估问题。结构化证据和日志 SHA-256 见 `diagnostics/2026-09-22-run-review.json`。原始日志保留在 A100 仓库根目录，但被 Git 忽略。

### 1. GRPO 梯度范数持续为 0：高优先级待排查

证据：`grpo_train.log` 中 epoch 0.1 至 1.0 的 10 次记录全部为 `grad_norm=0`。运行约 647.4 秒，完成约 50 个 optimizer step 的进度；流程完成不等于参数有效更新。

注意：不是所有指标都为 0。记录中的 reward 为 0.05–0.125，`reward_std` 非零；多数 loss 接近 0，但有一次约 0.003294。GRPO 的组内优势归一化可能使平均 loss 接近 0，不能仅凭 loss 小就断言训练失败。`clip_ratio/*=0` 本身也不能作为故障证据。

当前代码在 `scripts/grpo_gsm8k.py` 中以 `PeftModel.from_pretrained(base_model, MODEL_PATH)` 加载 SFT adapter，再 `merge_and_unload()`。加载时没有明确设置可训练性；合并后的参数是否可训练、trainer 是否改变该状态，尚未在这次检查中实测。参数冻结是待验证假设，不是已确认根因。注释“GRPO 需要完整模型”不宜作为普遍结论，GRPO 也可以使用可训练的 LoRA adapter。

后续验证：

1. 在最终模型构造后及 trainer 初始化后统计 `requires_grad=True` 的参数数量，并核对 optimizer 参数组。
2. 用一个小批次验证 backward 后有效梯度及 optimizer step 前后的参数差值。
3. 核对日志中梯度范数的计算与记录方式；若参数确实更新，继续查为何显示为 0。
4. 明确选择“继续训练 LoRA”或“合并后全参数训练”，再配置可训练参数和显存预算。

### 2. 生成几乎全部触及 512 token 上限

`max_completion_length=512`。10 个记录窗口里有 9 个 `completions/clipped_ratio=1`，另 1 个为 0.9875；多数窗口生成平均长度为 512。

这意味着很多样本可能在输出最终答案前被截断。当前正则提取“最后一个数字”，有可能把推理过程中的中间数字当成答案，奖励与评估需要检查。不要把 `completions/clipped_ratio`（生成截断）与 PPO/GRPO 的 `clip_ratio/*`（策略比率裁剪）混淆。

后续抽查 prompt、生成文本、EOS/结束原因、答案提取结果，核对思考模式与 token 预算，再决定是否增加长度、调整模板或改用简短回答任务。不要只扩大 GPU 规模。

### 3. GRPO 没有保存训练结果

脚本设置 `save_strategy='no'`，训练结束也未调用最终保存；检查时 `checkpoints/grpo-gsm8k` 没有模型文件。当前没有可用于独立加载验证的 GRPO 产物，不能声称已评估 GRPO 后的模型。

修复时应同时保存模型/adapter、tokenizer、训练配置和评估所需元数据，并做独立加载后的测试。

### 4. 现有 baseline / SFT 评估没有展示提升

同一主评估脚本在 GSM8K test 前 200 条上的日志为：

| 模型 | 正确数 | 准确率 |
| --- | --- | --- |
| Qwen3-1.7B baseline | 18/200 | 9% |
| SFT checkpoint-64 | 16/200 | 8% |

这是当前生成长度、模板与数字提取规则下的观测，不足以据此判断 SFT 普遍无效。需要先排查截断与答案解析，再用统一评估配置比较。

### 5. 另一条 SFT＋eval 路径中途崩溃

`scripts/sft_then_eval.py` 对预测字符串移除了逗号，但对参考答案直接调用 `float(ref)`。`sft_eval_inline.log` 报错：

```text
ValueError: could not convert string to float: '2,125'
```

最后一个进度输出是 `140/200 acc=0.1857`，不是完整 200 条结果，也不表示恰好在第 141 条失败。这条脚本的监督目标、学习率和生成长度与主流程不同，不能把中途指标直接当成主 SFT 的提升结论。

后续统一预测与参考答案的数字归一化，并补充逗号、负数、小数、缺失最终答案等解析验证。

### 6. SFT 标签与复现条件需核对

- 主 SFT 脚本使用 `messages`，未显式设置 `assistant_only_loss=True`。按已安装 TRL 的实现，没有适用掩码时整个序列参与 loss；若目标只学习 assistant 输出，应核对模板支持及实际 labels，不能只加参数便认定正确。
- SFT 已改为 `save_strategy="epoch"`，但旁边注释仍说“不存 checkpoint”，注释与行为不一致。
- `setup_env.sh` 的多项依赖未固定版本。此处记录的是检查当天的版本，不保证完全等同于历史运行时的依赖状态。

## 扩大训练前的优先顺序

1. 验证 GRPO 可训练参数、有效梯度和参数更新。
2. 检查生成截断，统一答案解析与评估配置。
3. 保存并独立加载 GRPO 产物。
4. 确认 baseline → SFT → GRPO 的可复现结果，再迁移到太极扩大训练。

本次提交保留现有脚本行为，以上问题均作为未解决事项记录；没有把假设写成已验证修复。
