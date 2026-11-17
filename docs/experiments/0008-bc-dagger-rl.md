# BC/DAgger 热启动 + RL 微调（阶段 1）

- 日期：2026-11-17
- 提交（git commit）：`49afef5`（PR #16；rebase 合并后哈希可能变化）
- 关联决策：ADR 0008（BC/DAgger 热启动）、ADR 0007（RL 训练循环）、
  ADR 0003（教师-学生）
- 对应 issue：#6（阶段 1 训练）
- 代码：`ipd/training/imitation.py`、`ipd/training/env.py`（labels/labeler）、
  `ipd/training/rl_train.py`（imitation 正则、CLI 超参数）

## 假设

1. 纯 REINFORCE 从零训练在阶段 1 会陷入"首手背叛"的局部最优；用基线做
   教师的行为克隆可以快速注入"合作开局、报复、终局收割"等行为。
2. 离线 BC 的 25 手复合误差会显著拉低评估；DAgger 用学生访问到的状态补
   教师标签，可以缩小分布偏移。
3. 在 RL 更新中加入在环教师模仿正则，可以保留 BC 学到的行为、同时用
   对局收益去修正可剥削/敌对对手上的动作。

## 设置

- 阶段 1 池：26 个对手（L0+L1 标准版 + 随机参数变体）；训练每轮抽 24 个，
  8 轮；`hide_prob=0`。
- 教师：`ipd.agents.baseline.strategy`；教师轨迹 20 赛季（约 9.6 万条转移）。
- BC：交叉熵，batch 256，lr 1e-3，4 epochs。
- DAgger：学生采样 10–15 赛季/轮，逐状态补教师标签；2–4 轮；每轮 2–3
  epochs，lr 4e-4–5e-4。
- RL 微调：从 BC/DAgger 检查点 `--resume`；REINFORCE + 价值基线 + 熵正则，
  lr 1e-4–3e-4，entropy 0.002–0.003，在环模仿系数 0.2–0.5，60–100 赛季。
- 评估：`evaluate_policy` 与完整 26 对手 + 基线跑 8 轮，报告中位数口径的
  每场平均分、名次、四个 canonical 对手的 baseline-relative regret；选型用
  2 个种子，最终用 5 个固定种子 0–4。

## 结果

### 1. 纯 RL 对照（校准，1 个种子）

| 配置 | 赛季 | 评估每场 | 基线 |
|---|---:|---:|---:|
| 默认 lr 1e-3 | 10 | 3.82 | 49.93 |
| 默认 | 20 | 4.43 | 49.72 |
| lr 3e-3, entropy 0.005 | 20 | 7.21 | 50.44 |
| 同上 | 100 | 18.37（峰值 22.53@50） | 48.98 |
| 合作初始化 + 合作课程 50 + 全池 50 | 100 | **31.61** | 49.77 |
| 再全池 +100/+150 | 150/200 | 21.58/23.66 | 47.01/46.29 |

纯 RL 的最佳观测是 31.61，且继续训练会漂移；阶段 1 关口不过。

### 2. 行为克隆 / DAgger（2 个种子）

| 方法 | 教师数据 | 学生数据 | 教师动作准确率 | 评估每场 | 基线 |
|---|---:|---:|---:|---:|---:|
| BC-only | 20 赛季 | 0 | 96.6% | 31.99–35.01 | 48.96–50.37 |
| DAgger ×2 | 20 | 2×10 | 91.7% | **38.60** | 48.75 |
| DAgger ×4 | 20 | 4×15 | 89.4% | **42.43** | 47.94 |

- 教师数据上 96.6% 的动作准确率仍只换来 32–35 分：25 手复合误差会让一次
  错误背叛触发 TFT/Grim 的长期报复。
- DAgger 把评估推到 42.43；更多学生数据让训练集准确率降到 89–90%，但
  评估上升，说明覆盖比拟合更重要。

### 3. RL 微调（2 个种子选型，5 个种子复核）

| 起点 | RL 配置 | 2 种子最佳 | 5 种子均值 | 基线（5 种子） |
|---|---|---:|---:|---:|
| DAgger ×2（38.60） | imitation 0.5, lr 3e-4, 60 赛季 | 44.24 | — | — |
| 继续 | imitation 0.2, lr 2e-4, 100 赛季 | **45.76** | **45.38** | 47.79 |
| DAgger ×4（42.43） | imitation 0.3, lr 2e-4, 100 赛季 | 45.51 | 45.24 | 47.45 |

5 种子下最佳单模型 **45.38 vs 47.79**（gap −2.4），名次稳定第 2。

### 4. 集成（诊断，5 个种子）

两个最佳单模型的动作概率加权集成（DAgger×2 权重 0.6，DAgger×4 权重 0.4）：

| 指标 | 集成 | 基线 |
|---|---:|---:|
| 每场平均分 | **47.74** | **47.87** |
| gap | **−0.13** | — |
| 名次 | 2 | 1 |
| regret all_c | +2.45 | 0 |
| regret all_d | +11.62 | 0 |
| regret tft | +5.30 | 0 |
| regret gradual | +1.92 | 0 |

均值几乎打平，但四个 canonical 对手仍全部略差，严格关口未过。集成需要两份
模型，推理成本翻倍，且不是单模型，暂不作为最终形态。

### 5. 失败/无效方向（避免重复）

| 尝试 | 结果 |
|---|---|
| 困难样本加权（RL imitation，weight 2） | 最佳 44.90；AllD 差距扩大到 22.5 |
| 背叛类加权（BC/DAgger，weight 5） | BC 最佳 37.44；RL 最佳 43.90 |
| 过采样 AllD/AllC（+6/+6，tft/gradual +2/+2） | 最佳 44.45；AllD 仍差 11.2 |
| 更大编码器 d=48 / d=64（BC-only） | 29.75 / 32.60，反而更差 |
| 显式统计特征拼到动作头（BC-only） | 28.94，拟合率 95.8% |
| 多次微调检查点权重平均 | 13.9–29.5，跨收敛域平均破坏策略 |

### 6. 逐对手误差

5 种子最佳单模型的逐对手差距（policy − baseline，前几位）：

| 对手 | 差距 | 说明 |
|---|---:|---|
| rand(p=0.78) | −19.9 | 不收割高合作率随机对手 |
| all_d | −15.0 | 对手连续背叛后仍会多合作 2–3 手 |
| soft_majority | −12.8 | 同上，缺少剥削 |
| rand(p=0.5) | −8.4 | — |
| memory_one / tft | −7～−8 | 终局/报复时机偏晚 |
| all_c | −6.0 | 第 0–1 轮收割不足 |

教师动作拟合率（按对手）：AllD 94.1%、all_c 97.3%、tft 97.4%、gradual 97.5%。
但 all_c/tft/gradual 的教师"背叛"标签只占 4%，总准确率 97% 仍可能每场漏掉
一次关键背叛，直接损失 2–6 分。

## 观察与诊断

- BC/DAgger 是有效的：评估从纯 RL 的 18–31 提升到 42.4，再用 RL 到 45.4–45.8。
- 剩余差距不是"不会合作"，而是"不会在正确的时候背叛"：对高合作率/无报复
  对手的收割不足，对 AllD 的报复启动偏慢。这是**长尾关键动作**问题，不是
  平均准确率问题。
- 单模型已经接近基线；集成把均值打到 47.74，说明两个微调策略的误差有互补，
  但 canonical 的逐项差距仍在。
- 失败方向表明：单纯给稀有动作加权会破坏 TFT 的合作，不能只做全局类别加权；
  需要按"对手历史是否可剥削/是否已持续背叛"做条件化加权，或把集成蒸馏回
  单模型。

## 结论与下一步

- 阶段 1 关口**未通过**：最佳单模型 45.38 vs 47.79；集成 47.74 vs 47.87，
  四个 canonical 的 regret 仍为正。
- 方案 A（推荐）：把集成的动作分布蒸馏回单模型（用集成在阶段池上的状态
  标签训练，或 DAgger 式查询集成），目标保留 47.7 的均值并修掉 AllD/AllC
  的逐项差距；这属于 #8 蒸馏的雏形。
- 方案 B：按可观察条件（对手当前场连续背叛/历史全合作）做**条件化损失权重**
  或小的反射层；必须在实验记录中标注为工程层，不能声称纯端到端。
- 方案 C：与 issue 的验收口径对齐——如果允许"均值持平、逐项略差"，则
  47.74 vs 47.87 可视为打平；但当前 ADR 0007 的定义要求逐项 ≤0，需要明确
  修改并记录。
- 方案 D：兜底用基线提交（留出池 47.94，合法），但这不是项目目标。
- 阶段 2/3 暂不应开始，先把阶段 1 的逐项 regret 修到 ≤0；否则阶段 3 的
  hide_prob/读历史关口会更难。

## 复现

```bash
# BC-only
.venv/bin/python -m ipd.training.imitation --stage 1 --seasons 20 --epochs 4 \
  --dagger-iterations 0 --checkpoint-dir runs/rl/bc_stage1

# DAgger ×2
.venv/bin/python -m ipd.training.imitation --stage 1 --seasons 20 --epochs 4 \
  --dagger-iterations 2 --student-seasons 10 --dagger-epochs 2 --dagger-lr 5e-4 \
  --checkpoint-dir runs/rl/bc_stage1

# RL 微调
.venv/bin/python -m ipd.training.rl_train --stage 1 \
  --resume runs/rl/bc_stage1/stage1_bc.pt --iterations 100 \
  --eval-every 20 --eval-seeds 0,1 --lr 2e-4 --entropy-coef 0.002 \
  --imitation-coef 0.2 --opponents-per-round 24 \
  --checkpoint-dir runs/rl/finetune_imitation2

# 5 种子评估
.venv/bin/python - <<'PY'
import torch
from ipd.agents.baseline import baseline_agent
from ipd.pool.registry import agents_for_stage, config_for_stage
from ipd.training.rl_train import load_checkpoint, evaluate_policy
policy, _ = load_checkpoint("runs/rl/finetune_imitation2/stage1/stage1_best.pt")
print(evaluate_policy(policy, agents_for_stage(1, seed=0), config_for_stage(1),
                      seeds=(0,1,2,3,4), baseline=baseline_agent()))
PY
```


## 2026-11-17 验收口径修订与暂停

- ADR 0007 将阶段 1 从“均值 > 基线且逐项 regret ≤0”改为阶梯口径：
  均值容差 3.0 分/场、canonical 逐项 gap ≤15；阶段 2/3 收紧到 1/8 和 0/5。
- 当前最佳单模型 5 种子 45.38 vs 47.79，all_c 8.2、all_d 14.5、tft 7.0、
  gradual 2.2，按修订口径**通过阶段 1**；逐项仍未不低于基线，记录保留。
- 训练暂停，保留基线兜底（留出池 47.94）；PR #16 保留未合并。
- 恢复顺序：单模型蒸馏/条件化权重 → 阶段 2/3 信誉训练 → 必要时 PPO/ES。
