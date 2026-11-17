# 全池诊断 harness 验收（T2）

- 日期：2026-11-17
- 提交（git commit）：`2542f57`
- 关联决策：ADR 0009（尾部优先、规则先于模型）
- 实施计划：[plan 0001](../plans/0001-tail-first.md) T2
- 代码：`ipd/eval/diagnose.py`、`tests/test_eval_diagnose.py`

## 假设

1. 把实验 0009 的手工全池流程固化成一条命令，应能**复现**已有数字
   （全池 solo 48.36、官方 30/种子协议、guard +1.30）。
2. 逐对手 + tag + 尾部 + 克隆噪声的输出，应能直接指出下一批要修的
   对手，而不是只看一个均值。

## 设置

- 全池：130 个留出策略，完整 8 轮，`hide_prob=0.1`，`timeout=0.5`。
- 席位：candidate、baseline、baseline#clone（噪声对照）；`--no-clone`
  时去掉克隆。
- 默认 seeds 0–4；变体只跑 seed 0（plan 0001 的"四道门"之一）。
- 30/种子对照：`--size 30 --seeds 0,1,2,3,4 --no-clone`，与
  `ipd.eval.holdout` 完全同 roster。
- 测试：`python -m unittest discover -s tests -t .`，**218 项通过**
  （新增 `tests/test_eval_diagnose.py` 16 项）。

## 结果

### 1. 全池 solo：复现实验 0009

```
python3 -m ipd.eval.diagnose --agent ipd.agents.baseline:baseline_agent \
    --no-baseline --no-clone --seeds 0
```

- candidate **48.36** 分/场，名次 1/131，0 故障，25.9 s；
- 尾部与实验 0009 完全一致：prober -138.75、history_classifier -55.50、
  endgame_preempt -51.25、all_d -45.00、…；
- 结论：harness 与手工诊断同口径。

### 2. 30/种子协议：与 `ipd.eval.holdout` 逐字段一致

| 字段 | holdout | diagnose |
|---|---:|---:|
| candidate_per_match | 48.919 | 48.919 |
| baseline_per_match | 48.773 | 48.773 |
| gap | +0.147 | +0.147 |
| faults | 0 | 0 |
| n_agents | 32 | 32 |
| size | 30 | 30 |

逐种子 candidate 一致（41.80, 45.76, 49.01, 54.67, 53.35）。结论：需要
与旧协议对照时，加 `--no-clone` 即可。

### 3. 全池 5 seeds：guard 复现 +1.30，gate 通过

```
python3 -m ipd.eval.diagnose --agent ipd.agents.baseline_guard:baseline_guard_agent --gate
```

| 指标 | 数值 |
|---|---:|
| candidate | 50.85 |
| baseline | 49.55 |
| **gap** | **+1.30** |
| worst seed gap | -0.55 |
| clone | 49.37 |
| **clone noise** | **-0.18** |
| worst seed noise | -1.29 |
| 耗时 | 151.3 s |
| faults | 0 |
| GATE | **PASS** |

逐种子与实验 0009 一致：+2.91、+0.63、+1.00、+2.54、-0.55。结论：harness
的主用例（全池 + 克隆）与手工脚本可互相复现。

### 4. 标准池 config 变体（seed 0，`--no-clone`）

| 变体 | gap |
|---|---:|
| `history_update="round"` | +7.26 |
| `hide_scope="step"` | +7.30 |
| `shuffle_schedule=False` | +10.10 |
| `hide_prob=0.0` | +7.93 |

标准池含 prober，所以 gap 远大于留出池；四种配置均为正，无崩溃。

### 5. 新输出：下一批尾部目标

全池 5 seeds 的 `worst by delta vs baseline`（candidate − baseline）：

| 对手 | delta | 标签 |
|---|---:|---|
| `ho_basicb_hard_majority` | -15.75 | basic, conditional |
| `ho_histb_headtail_split` | -14.80 | conditional, endgame, history |
| `ho_hista_hidden_discount` | -12.17 | conditional, history |
| `ho_mixb_reputation_threshold` | -9.00 | history, noisy |
| `ho_basica_rand_p10` | -5.75 | basic, hostile, noisy |
| `ho_condb_graded_revenge` | -5.50 | conditional, noisy |
| `ho_mixb_mode_selector` | -5.30 | history, mixed |
| `ho_mixa_rate_zone_retaliate` | -5.08 | conditional, mixed, noisy |
| `ho_basicb_soft_majority` | -4.83 | basic, conditional, history |
| `ho_mixa_exploit_cycle_history` | -4.35 | history, mixed |

按 tag 的 delta：hostile **+20.65**、mixed **+7.57**、basic **+5.50**、
noisy +0.57、history -0.11、cooperative -0.26、conditional -0.36、
endgame -0.40。守卫主要帮 hostile/basic/mixed，对 endgame/conditional
略负但都在 ~0.4 分内（与噪声同量级）。

## 观察

- Harness 复现了三套已知数字（全池 solo、官方 30/种子、guard 全池），
  可以作为后续规则/分类器的唯一决策入口。
- `--gate` 按 plan 0001 的 T1 门槛工作：全池均值 >= 基线 +1.0、最差种子
  >= 基线 -1.0；当前 guard PASS。
- 新的尾部清单把 T4 的目标具体化：`hard_majority`、`headtail_split`、
  `hidden_discount`、`reputation_threshold` 是下一批规则要打的点；
  `all_d` 仍是 -45 的理论下限，不修。
- by-tag 的负值（endgame/conditional ~-0.4）不要过度反应，先看逐对手
  的 delta；只有能稳定超过克隆噪声（~±1.3/种子）的才值得改。
- 每次全池 5 seeds 约 2.5 分钟；T2 之后所有候选统一用这个口径。

## 结论与下一步

- T2 完成：`ipd/eval/diagnose.py` 可作为全池/抽样/变体/尾部的统一评测入口；
  218 项测试通过。
- 下一步：
  1. T3 dev pool（60 个）：把 130 冻结为最终测试；
  2. T4 按上面的尾部清单逐条修规则（先 hard_majority / headtail_split）；
  3. T5 分类器/q 只学规则修不动的残差。

## 复现

```bash
# 全池 solo（应得 48.36）
python3 -m ipd.eval.diagnose --agent ipd.agents.baseline:baseline_agent \
    --no-baseline --no-clone --seeds 0

# 官方 30/种子对照（应与 ipd.eval.holdout 一致）
python3 -m ipd.eval.diagnose --agent ipd.agents.baseline:baseline_agent \
    --size 30 --seeds 0,1,2,3,4 --no-clone

# 全池 5 seeds + 克隆 + gate
python3 -m ipd.eval.diagnose \
    --agent ipd.agents.baseline_guard:baseline_guard_agent --gate

# 标准池 config 变体
python3 -m ipd.eval.diagnose \
    --agent ipd.agents.baseline_guard:baseline_guard_agent \
    --pool canonical --seeds 0 --variants --no-clone

# 测试
python3 -m unittest tests.test_eval_diagnose
```
