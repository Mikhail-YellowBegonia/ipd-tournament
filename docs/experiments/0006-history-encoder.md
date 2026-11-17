# 历史编码器与辅助损失

- 日期：2026-11-17
- 提交（git commit）：PR #13（待合入）
- 关联决策：ADR 0002（分层历史编码）、ADR 0003（训练/评测分离）
- 对应 issue：#5（路线图任务 7）

## 假设

1. 在纯 Python 评测端之前，训练端可以用 PyTorch 实现较完整的分层编码器：
   场内 n-gram 计数嵌入 + 场间时间衰减注意力 + 本场 GRU。
2. 20 场历史、25 回合、嵌入维度 32 的模型在 CPU/batch=1 下可以在 1 ms 内完成
   一次前向；这给 #7 的纯 Python 重写留下明确的时间预算。
3. 辅助损失（预测对手下一手 + 对比学习）能提供有效梯度，且只使用模拟器标签。

## 设置

### 编码器

实现：`ipd/agents/encoder.py`，`HistoryEncoder`。

- 词表 27 =（主体动作 3）×（对方动作 3）×（收益阶段 3）；`m=0, c=1, b=2`。
- 场内：每场跳过含 `m` 的步，对有效步做 n-gram 计数并除以有效步数，
  再乘 `Embedding(27, 32)`，得到 20 个场向量。
- 场间：可学习 query + 4 头 `MultiheadAttention`；无效场用 `-1e9` 掩码，
  时间衰减 `0.5 ** (age / 10)` 以对数形式加到注意力 mask 上；全空历史返回零向量。
- 本场：GRU（1 层，隐藏 32）读本场动作对的 n-gram 嵌入；初始隐状态由历史嵌入
  经线性层 + 轮次嵌入（8 轮）得到；输出与本场嵌入残差相加后 LayerNorm。
- 参数：12,832（d=32）；d=24 时 7,512，d=16 时 3,600。
- 接口：`forward(own_history, opp_history, my_decisions, opp_decisions, rs, length)`，
  输入形状 `(batch,20,25)` / `(batch,length)`，输出 `(batch,32)`。
  **注意**：考试接口只给对手历史，因此 `own_history` 表示历史主体（对手）自己的
  动作，`opp_history` 表示其对手的动作；`observation_tensors()` 负责把九参数接口
  的一次观测转成张量。`encode_history()` 只输出历史池化嵌入，供对比学习使用。

### 辅助损失

实现：`ipd/training/auxiliary.py`。

- `NextMovePredictor`：状态嵌入 -> 隐层 -> 2 类 logits；`next_move_loss()` 交叉熵。
- `contrastive_loss()`：对称 InfoNCE；同一策略两场历史作为正样本，batch 内其余为负。
- `history_contrastive_loss()`：复用 `HistoryEncoder.encode_history()` 生成两个视图。
- `combined_auxiliary_loss()`：按权重合并两项损失，供 #6 策略网络直接调用。

### 基准环境

- `.venv` Python 3.12.13，torch 2.15.0，CPU；`torch.set_num_threads(1)`。
- batch=1，历史 20 × 25 完整可见，`length=24`（最坏的本场长度）。
- 预热 100 次后测 5 组 × 100 次；关闭 GC；`torch.inference_mode()`。
- batch=32 另测一次，仅作参考。

## 结果

### 单次前向耗时（batch=1，完整历史）

| 组 | 最坏 (ms) | 中位 (ms) | 平均 (ms) |
|---|---:|---:|---:|
| 1 | 0.519 | 0.213 | 0.228 |
| 2 | 0.507 | 0.211 | 0.226 |
| 3 | 0.433 | 0.210 | 0.218 |
| 4 | 0.520 | 0.209 | 0.222 |
| 5 | 0.484 | 0.212 | 0.224 |

- 全部 5 组的最坏值 0.433–0.520 ms，低于 1 ms 目标；中位数约 0.21 ms，
  距目标约 4.7 倍余量。
- batch=32 平均约 0.79 ms/次，批处理没有线性放大。
- 参数 12,832；模型很小，适合后续蒸馏/进化策略（d=24 时 7,512 参数）。

### 单元测试

`tests/test_encoder.py` 共 21 项，覆盖：

- 空历史（全 `m`）、完整历史、10%/50%/90% 逐步隐藏；
- batch=1 与 batch=32、批内逐样本一致性、eval 确定性；
- n-gram 索引正确性（单步 c/c 与 c/b 的嵌入对齐）、隐藏步跳过；
- 注意力掩码（无效场权重 ~0）、时间衰减（越近权重越大）、全空池化输出为零；
- `length=0`、轮次条件影响输出、梯度回传、输入形状/取值校验；
- 性能：100 次前向最坏 ≤ 1 ms；
- 辅助损失：下一手预测头形状/损失/可学习、对比学习相似视图损失更低、
  batch=1 与非法形状、历史对比损失梯度、组合损失梯度。

本分支全量测试（venv，含 axelrod）102 项通过（#2 的评估测试在另一个 PR）；
系统 Python 无 torch 时 21 项编码器测试自动跳过，其余 75 项通过
（axelrod 6 项也跳过）。

## 结论与下一步

- 1 ms 性能目标已达成，且有余量；d=32/12.8k 参数可以作为 #6 的默认配置。
- 编码器只使用考试接口实际可见的对手历史；#7 的纯 Python 实现需要复刻
  n-gram 嵌入、时间衰减注意力、GRU 和轮次嵌入，权重导出脚本要包含
  `phase_of_turn`/`time_decay` 常量。
- #6 可以直接实例化 `HistoryEncoder` + `NextMovePredictor`，用
  `encode_history()`/`forward()` 产出状态嵌入；辅助损失有独立单测。
- 如果后续进化策略需要把参数量压到 5k 以下，优先把 `embed_dim` 降到 24
  （7,512 参数）；降到 16 则 3,600 参数，但表达能力会下降。

## 复现

```bash
.venv/bin/python -m unittest tests.test_encoder
.venv/bin/python -m unittest discover -s tests -t .
.venv/bin/python -m ipd.agents.bench_encoder --runs 3 --reps 100
```
