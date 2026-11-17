# 尾部优先：反试探守卫 `baseline_guard` 与全池诊断

- 日期：2026-11-17
- 提交（git commit）：`6f5c2fa`（`baseline-guard` 分支）
- 关联决策：ADR 0009（尾部优先、规则先于模型）、ADR 0007（阶段 1 关口）
- 实施计划：[plan 0001](../plans/0001-tail-first.md)
- 代码：`ipd/agents/baseline_guard.py`、`tests/test_baseline_guard.py`

## 假设

1. 阶段 1 的端到端失败不是"策略容量不够"，而是少数对手上的**灾难性错误**；
   逐对手诊断应能定位到具体机制，而不是"处处差一点"。
2. 如果最大的负分来自可解释的对手模型假设错误，一条纯 Python 规则就能
   补掉大部分，且不需要训练、没有运行时风险。
3. 官方"30 个/种子 × 5 种子"协议的噪声可能大于小规则的真实收益；决策
   必须用全池 + 克隆对照。

## 设置

- 全池：130 个留出策略（`ipd.pool.holdout`），完整 8 轮，`hide_prob=0.1`，
  `timeout=0.5`，`history_update="match"`，`shuffle_schedule=True`。
- 诊断：baseline + 130 留出策略，seed 0；用 `pair_scores` 拆逐对手得分。
- 候选：`baseline_guard` = baseline 的 DP + "对手首手 B → 我第 2 手 B"。
- 验证：候选 + baseline + baseline 克隆（同函数、不同席位） + 130 留出策略，
  seeds 0–4；效应 = 候选 − baseline，噪声 = 克隆 − baseline。
- 标准池：候选 + baseline + `canonical()` 23 个，seed 0。
- 配置变体：`history_update="round"`、`hide_scope="step"`、
  `shuffle_schedule=False`、`hide_prob=0.0`，各 seed 0。
- 测试：`python -m unittest discover -s tests -t .`，202 项。

## 结果

### 1. 全池诊断（baseline，seed 0）

- 每场平均 **48.36**，池内第 1，0 故障；130 个对手、8 轮，约 30.5 s。
- **16 个对手为负**，其中 3 个低于 -50。

| 对手 | 每场得分 | 标签 |
|---|---:|---|
| `ho_basicb_prober` | **-138.75** | basic, hostile, mixed |
| `ho_condc_history_classifier` | -55.50 | conditional, history |
| `ho_conda_history_endgame_preempt` | -51.25 | endgame, history |
| `ho_basica_all_d` | -45.00 | basic, hostile |
| `ho_histb_headtail_split` | -43.25 | conditional, endgame, history |
| `ho_histb_fame_executioner` | -43.12 | endgame, history, hostile |
| `ho_basica_rand_p10` | -32.38 | basic, hostile, noisy |
| `ho_conda_warmup3_defect` | -32.00 | basic, hostile |

按标签（每场均值）：cooperative 69.05、conditional 59.42、history 48.15、
noisy 44.54、basic 40.53、mixed 40.39、endgame 31.57、hostile -21.31。

### 2. prober 的机制：跨场历史污染

prober 的规则（`ipd/pool/holdout/basic_b.py`）：先手 B-C-C 试探；若对手
第 2 手仍合作（`opp_decisions[1] == "c"`），永久背叛；否则退回 TFT。

全池对局中，prober 对其他对手合作，baseline 的跨场对手模型把它当成
"高合作先验"，于是本场全程合作。逐手序列（rounds 1–7）：

```
baseline  b c c c c c c c c c c c c c c c c c c c c c c c b
prober    b c c b b b b b b b b b b b b b b b b b b b b b b
```

单场 -139～-144，接近"全合作对 AllD"的 -170 地板。2 人直接对局时
（无跨场历史）baseline 只输 2 分，说明问题不是单挑打不过，而是历史污染。

### 3. `baseline_guard` 全池 5 seeds

| seed | guard | baseline | clone | 效应 | 噪声 |
|---:|---:|---:|---:|---:|---:|
| 0 | 51.14 | 48.24 | 48.51 | **+2.91** | +0.27 |
| 1 | 51.21 | 50.58 | 51.03 | +0.63 | +0.46 |
| 2 | 50.89 | 49.89 | 48.60 | +1.00 | -1.29 |
| 3 | 51.87 | 49.33 | 49.68 | **+2.54** | +0.35 |
| 4 | 49.16 | 49.71 | 49.00 | -0.55 | -0.70 |
| **均值** | — | — | — | **+1.30** | **-0.18** |

- 4/5 种子为正，最差 -0.55；克隆噪声均值 -0.18、单个种子可达 ±1.3。
- prober 单场从 -138.75 翻到 +71.00。
- 负分对手从 16 降到 14。

### 4. 标准池与配置变体

- 标准池（canonical 23）：guard **51.80** vs baseline **43.19**，效应 **+8.61**
  （主要来自 prober；同 roster 比较）。
- 变体（全池，seed 0）：

| 变体 | guard | baseline | clone | 效应 | 噪声 |
|---|---:|---:|---:|---:|---:|
| `history_update="round"` | 49.70 | 48.33 | 48.54 | +1.37 | +0.21 |
| `hide_scope="step"` | 50.12 | 48.97 | 49.43 | +1.14 | +0.46 |
| `shuffle_schedule=False` | 51.53 | 51.12 | 51.13 | +0.40 | +0.01 |
| `hide_prob=0.0` | 50.82 | 48.58 | 49.74 | +2.24 | +1.16 |

四种配置下效应均为正，未出现崩溃；`shuffle_schedule=False` 时效应最小
（+0.40），说明 prober 的历史污染依赖"轮内对局顺序/历史更新"这些尚未
确认的赛程细节（roadmap B1 问题 1）。

### 5. 测试与运行时间

- 全量测试 **202 项通过**（新增 `tests/test_baseline_guard.py` 15 项）。
- 守卫只在 `length == 1` 做一次比较，运行时间与 baseline 同量级
  （baseline 0.045 ms 平均 / 0.205 ms 最坏，实验 0004）。

## 观察

- 差距是**尾部**的：前 3 个负分对手贡献了大部分 gap；不是"处处差 5%"。
- 最大的洞来自**对手模型假设错误**（对手对所有人一视同仁），不是策略
  容量；一条规则就能补，验证了 ADR 0009 的方向。
- 官方 30/种子协议上，同一策略对自身的 gap 可达 ±3（5 seeds 控制组
  [-2.88, +1.92, -1.90, +3.15, +0.44]），所以该协议只能筛查、不能决策。
- 本规则是在 130 上发现的，130 是测试集；+1.30 是"有希望"而非"最终"。
  必须用 dev pool 复核（plan T3），并在合并提交版前重跑四道门。
- `shuffle_schedule=False` 下收益缩小，提示赛程未确认前不要把这条规则的
  收益当成确定值。

## 结论与下一步

- T1 `baseline_guard` 达到门槛：全池均值 +1.30（>= +1.0）、worst -0.55
  （>= -1.0）、标准池不降、四种配置不崩；测试 202 项通过。
- 但**不直接合入 main**：等 dev pool（T3）复核 + T6 独立提交版（零 import、
  parity）完成后再合并；main 的兜底仍是原基线。
- 下一步：
  1. T2 `ipd/eval/diagnose.py`：把本实验的全池/克隆/逐对手流程固化；
  2. T3 dev pool（60 个）：复核 guard，并作为后续规则/分类器的迭代集；
  3. T4 尾部规则：`history_classifier`、`endgame_preempt`、
     `fame_executioner/headtail_split`、`warmup3`、`rand_p10/hard_majority`；
  4. T5 分类器/q 只学规则修不动的残差。

## 复现

```bash
# 单测
python3 -m unittest tests.test_baseline_guard

# 官方协议（注意：30/种子噪声大，仅筛查）
python3 -m ipd.eval.holdout \
    --agent ipd.agents.baseline_guard:baseline_guard_agent \
    --size 30 --seeds 0,1,2,3,4

# 全池 5 seeds + 克隆对照（决策用；约 2.5 分钟）
python3 - <<'PY'
from ipd.agents import baseline, baseline_guard
from ipd.pool.holdout import HOLDOUT_AGENTS
from ipd.sim import Agent, TournamentConfig, run_tournament

hold = list(HOLDOUT_AGENTS)
for seed in range(5):
    cfg = TournamentConfig()
    agents = [
        Agent(name="guard", fn=baseline_guard.strategy, family="baseline_guard", layer="BASE"),
        Agent(name="base", fn=baseline.strategy, family="baseline", layer="BASE"),
        Agent(name="clone", fn=baseline.strategy, family="baseline", layer="BASE"),
    ] + hold
    res = run_tournament(agents, seed=seed, cfg=cfg)
    div = cfg.rounds * (len(agents) - 1)
    g, b, k = (s / div for s in res.scores[:3])
    print(f"seed {seed}: guard {g:.2f} base {b:.2f} effect {g-b:+.2f} noise {k-b:+.2f}")
PY
```
