# Dev 池（60 个）与 guard 硬补丁检验

- 日期：2026-11-17
- 提交（git commit）：`4d2e4a1`（dev 池代码）
- 关联决策：ADR 0009（尾部优先）、**ADR 0010（对手建模优先于对手补丁）**
- 实施计划：[plan 0001](../plans/0001-tail-first.md) T3
- 代码：`ipd/pool/dev/`、`tests/test_dev_pool.py`、
  `ipd/eval/diagnose.py`（`--pool dev`）
- 指南：[docs/eval/dev_pool_guidelines.md](../eval/dev_pool_guidelines.md)

## 假设

1. 130 留出池已被 `guard_open` 用来发现规则；继续在它上面调参会把测试集
   变成验证集。需要一个独立的 60 个策略的 dev 池，作为规则/模型的迭代集。
2. 如果 `guard_open` 是通用的"反试探"机制，它应在 dev 池上复现 130 的
   +1.30；如果复现不了，说明它只是 prober 的硬补丁，不能提交。

## 设置

### 生成

- 6 个隔离 subagent，各自只读 `.dev_staging/CONTRACT.md` 与自己的主题，
  不读 `ipd/`、`tests/`、`docs/`、`.holdout_staging/`、其他作者文件。
- 每人 10 个策略、6 个模块，共 **60**；名称 `dv_` 前缀。
- 主题：经典基础、周期/随机、本场条件/终局、读历史、混合/噪声、
  混合/历史/终局/噪声。
- 机器检查器 `.dev_staging/check_dev_module.py`：6 个模块全部
  **0 error / 0 warning**。

### 标签覆盖（60 个）

| 标签 | 数量 | 下限 |
|---|---:|---:|
| basic | 18 | — |
| conditional | 38 | 18（30%） |
| cooperative | 5 | — |
| endgame | 15 | 6（10%） |
| history | 18 | 12（20%） |
| hostile | 2 | — |
| mixed | 16 | — |
| noisy | 15 | 6（10%） |

### 测试与集成

- `tests/test_dev_pool.py` **16 项通过**；全量 **234 项通过**。
- 独立性：AST 白名单、无 `global`/`nonlocal`、无 `random.seed`；
- 行为探针：history 真读历史、conditional 响应本场、endgame 末手背叛、
  noisy 重复调用出现两种动作；指纹多样性 >= 54/60；
- 隔离：`dv_` 名字与 `agents_for_stage()` 四个阶段无交集；
- 全池 + 标准池 0 故障；`--pool dev` 入口可用。

### 评估

```
python3 -m ipd.eval.diagnose \
    --agent ipd.agents.baseline_guard:baseline_guard_agent --pool dev --gate
```

60 个对手、seeds 0–4、候选/基线/克隆三席位，38.2 s。

## 结果

### 1. guard 在 dev 池上没有复现 130 的收益

| 指标 | dev 池 | 130 留出池（实验 0009） |
|---|---:|---:|
| candidate | 50.45 | 50.85 |
| baseline | 51.06 | 49.55 |
| **gap** | **-0.62** | **+1.30** |
| worst seed gap | -2.61 | -0.55 |
| clone noise | -0.45 | -0.18 |
| rank | 2.2/63 | 1.2/133 |

逐种子 gap：+0.08、-2.61、-0.78、+1.80、-1.56。**GATE FAIL**
（均值 -0.62 < +1.0）。结论：+1.30 不是可迁移的通用收益。

### 2. 为什么 guard 是硬补丁

| 对手 | 首手 | 试探规则 | 对第 2 手合作 | 对第 2 手背叛 | 正确应对 |
|---|---|---|---|---|---|
| 130 `ho_basicb_prober` | B | B-C-C | 永久背叛（惩罚合作） | 退回 TFT | 第 2 手背叛 |
| dev `dv_basicb_opening_probe` | B | B-B | 转为合作（奖励忍让） | 永久背叛（惩罚报复） | 第 2 手合作 |

两者在 `length == 1` 的可观测信息完全一样（"首手 B"），正确动作却相反。
`guard_open` 把"首手 B"这一个低信息量线索直接硬编码成"第 2 手背叛"，
必然在其中一边出错：

- 对 prober：第 2 手背叛 → 不触发永久背叛，修好（-138.75 → +71.00）；
- 对 dev opening_probe：第 2 手背叛 → 触发永久背叛，反而变坏（-101.12）。

这不是"规则不够多"，而是单一线索 + 硬编码动作的必然结果。继续按测试集
尾部逐个加 `if`，会在 dev / 真实考场触发另一个对手，且规则不可组合。

### 3. dev 池的尾部（用于选特征，不是补丁清单）

`worst by delta vs baseline`：

| 对手 | delta | 标签 |
|---|---:|---|
| `dv_hista_good_games` | -11.85 | conditional, history |
| `dv_conda_noisy_leak_end1` | -9.28 | conditional, endgame, noisy |
| `dv_hista_exploit_soft` | -8.75 | history, noisy |
| `dv_mixb_rep_switch` | -7.00 | history, mixed |
| `dv_hista_type_classify` | -6.40 | conditional, history |
| `dv_mixb_endgame_harvest` | -5.45 | endgame, mixed |
| `dv_mixb_hist_forecast` | -5.42 | history, mixed |
| `dv_basicb_random_p30` | -4.88 | noisy |
| `dv_hista_endgame_telegraph` | -4.50 | conditional, endgame, history |
| `dv_mixb_mixed_clock` | -3.60 | history, mixed |

按 tag 的 delta：cooperative +2.02、basic +0.09、conditional -0.15、
mixed -1.11、noisy -1.29、endgame -1.33、history -1.48。

候选最差绝对分：`dv_basicb_opening_probe` -101.12、`dv_basica_alld` -45.00、
`dv_hista_exploit_soft` -27.02、`dv_mixb_endgame_harvest` -8.72、
`dv_mixa_score_driven_mix` -4.70。

这些数字告诉我们**缺哪些特征**（终局倾向、历史-本场不一致、类型切换、
随机性、声誉），而不是"给这十个对手各写一个分支"。

## 观察

- Dev 池达到了目的：在合并 guard 之前抓到了"holdout 特例"。这正是 T3
  存在的理由；130 从此只做最终 go/no-go。
- guard 是一个有价值的**消融**：它证明"首手模式"必须被利用，但响应方向
  取决于对手类型；把它做成无条件规则是错误层级。
- 130 与 dev 的尾部清单不同（130：hard_majority/headtail_split/
  hidden_discount；dev：good_games/noisy_leak_end1/exploit_soft/
  rep_switch）。只修一边的规则会在另一边复现 guard 的问题。
- dev 池 60 个、5 seeds 约 38 s，适合日常迭代；全池 130 约 150 s，
  只做最终确认。

## 结论与下一步

- T3 完成：60 个独立策略、0 故障、16 项测试、`--pool dev` 入口可用。
- T1 guard 结论：**holdout +1.30，dev -0.62；拒绝作为提交候选**。
  保留为消融/机制证据；其洞察转入通用 q 模型的特征集（首手模式、
  历史-本场不一致、类型切换），不再作为独立规则。
- 下一步（ADR 0010 / T4）：
  1. 通用对手模型 v0：`detectors` 9 类统计 + 本场合作率/连续背叛/上一手 +
     历史-本场不一致 + 首手/终局/阶段特征 → logistic/浅 GBDT → q → 原 DP；
  2. 训练用 pool L0–L3 + Axelrod + 合成，不用 dev/130 训练；dev 做模型
     选择，130 只做最终；
  3. 双边门槛：dev 与 130 都不降、效应 > 克隆噪声、0 故障、纯标准库；
  4. 如果 v0 在双边都拿不到正收益，提交基线（兜底）。
- T5（类型后验/决策聚焦 q）是 v1 的可选进阶，不再是"修残差"的补丁层。

## 复现

```bash
# dev 池测试
python3 -m unittest tests.test_dev_pool

# dev 池 guard 评估
python3 -m ipd.eval.diagnose \
    --agent ipd.agents.baseline_guard:baseline_guard_agent --pool dev --gate

# dev 池单个种子快速筛查
python3 -m ipd.eval.diagnose \
    --agent ipd.agents.baseline_guard:baseline_guard_agent --pool dev --seeds 0
```
