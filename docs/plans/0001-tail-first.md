# 0001 尾部优先实施计划

- 日期：2026-11-17
- 状态：进行中
- 关联决策：ADR 0009（尾部优先、规则先于模型）
- 上游：ADR 0001（RL 主路线）、ADR 0007（阶段 1 关口）、ADR 0004（信誉成本）

## 目标

在 12 月 15 日截止前，把**可提交**的每场平均分从基线水平稳定提高 >=1.0，
同时保持：纯标准库、单次决策 p99 < 5 ms、无跨调用可变状态、0 故障、
单文件可交付。学习模型只作为挑战者，不阻塞兜底提交。

## 当前基线

| 指标 | 数值 | 来源 |
|---|---:|---|
| 留出池（130 全池，seed 0） | 48.36 分/场，第 1 | 2026-11-17 诊断 |
| 留出池（30/种子 × 5 种子） | 47.94 分/场，名次 1.2/31 | 实验 0007 |
| 标准池（canonical 23） | 41.98 分/场 | 2026-11-17 诊断 |
| 单次决策 | 0.045 ms 平均 / 0.205 ms 最坏 | 实验 0004 |
| 负分对手 | 16 / 130 | 2026-11-17 诊断 |
| 最差对手 | prober -138.75 分/场 | 2026-11-17 诊断 |

关键诊断：prober 用 B-C-C 试探，基线把它的跨场"好人历史"当真，本场全程
合作；`guard_open`（对手首手 B → 我第 2 手 B）在 130 上 +1.30，但 dev 上
-0.62——它是硬补丁，不是通用机制（ADR 0010）。

## 评测协议（决策用）

1. **全池**：130 个留出策略 + 候选 + 基线 + 基线克隆（噪声对照），完整 8 轮，
   `hide_prob=0.1`，`timeout=0.5`，seeds 0–4。
2. **指标**：候选与基线的每场平均分、gap、worst seed、负分对手数、逐对手
   得分、按 tag 聚合；克隆-基线的 gap 作为噪声带。
3. **快速筛查**：30 个/种子 × 5 种子，仅用于淘汰明显坏的候选，不用于决策；
   该协议同一策略对自身 gap 可达 ±3。
4. **稳健性**：至少 4 种 `TournamentConfig` 变体（`history_update="round"`、
   `hide_scope="step"`、`shuffle_schedule=False`、`hide_prob=0.0`）各跑一次，
   不允许崩或大幅回退。
5. **数据纪律**：130 冻结；迭代用 dev pool（60 个，独立编写）与标准池。
6. **双边门槛**：任何改动必须在 dev + 130 上都不降，且效应 > 克隆噪声；
   只在一边有效视为池特例，回退。
7. **合规**：零 import（提交版）、无模块级可变状态、不改输入、异常返回
   `"c"`、Python 3.10–3.12、确定性、0 故障。

## 工作项

### T1 反试探守卫 `baseline_guard`（holdout 通过；dev 未通过；不作为提交候选）

- `ipd/agents/baseline_guard.py`：baseline + "对手首手 B → 我第 2 手 B"。
- 单测：触发/不触发、九参数透传、异常→合作、不改输入、无状态。
- 验证：全池 5 seeds + 标准池 + 4 种 config 变体。
- holdout：全池 +1.30（worst -0.55），标准池 +8.61，gate PASS（实验 0009）。
- **dev：gap -0.62（worst -2.61），未复现（实验 0011）。**
- 结论：单一线索硬补丁，拒绝作为提交候选；保留为消融/机制证据；
  洞察（首手模式、历史-本场不一致）转入 T4 特征。见 ADR 0010。

### T2 全池诊断/评测 harness（✅ 2026-11-17）

- `ipd/eval/diagnose.py`：候选 vs 基线 vs 克隆的逐对手、尾部、tag 聚合、
  config 变体、`--gate`；输出 JSON + 表格。
- 验收（实验 0010）：
  - 全池 solo 48.36，与实验 0009 一致；
  - 30/种子协议与 `ipd.eval.holdout` 的 candidate/baseline/gap/faults 完全一致；
  - 全池 5 seeds guard 复现 +1.30（worst -0.55，noise -0.18），`--gate` PASS
    （仅验证 harness 复现；guard 本身已被 ADR 0010 降级）。

### T3 dev pool（60 个，✅ 2026-11-17）

- 6 个隔离作者 × 10 个策略 = 60；`dv_` 前缀；标签覆盖 history 18、
  conditional 38、endgame 15、noisy 15（下限 12/18/6/6）。
- `tests/test_dev_pool.py` 16 项、全量 234 项通过；0 故障；与
  `agents_for_stage()` 无交集；`--pool dev` 已接入。
- 首个用途：guard 在 dev 上 -0.62，未通过泛化（实验 0011）——这正是
  dev 池要抓的 holdout 特例。

### T4 通用对手模型 v0：特征 q + DP（本周）

**不做针对具体对手的硬补丁**（ADR 0010）。尾部清单只作为"缺什么特征"
的诊断，不逐条加 `if`。

- 特征：
  - `ipd/detectors/profile.py` 的 9 类统计（coop/opening/last/endgame/
    unprovoked/retaliate/copy/exploit/grim/endgame_k）；
  - 本场：合作率、连续背叛长度、上一手、`length`/阶段、比分差；
  - 历史-本场不一致：历史合作率 − 本场合作率；对手对他人 vs 对我的差异；
  - 首手模式（用于区分 prober B-C-C / probe B-B，但不直接决定动作）。
- 模型：多项 logistic 或浅 GBDT → q（或类型后验 → q）；接回 baseline 的 DP。
- 训练：pool L0–L3 + Axelrod + 合成；不用 dev/130 训练；dev 做模型选择，
  130 只做最终。
- 指标：DP 的 payoff loss / 锦标赛分数，不是分类准确率；置信度门控回退
  Beta q。
- 双边门槛：dev 与 130 都不降、效应 > 克隆噪声、0 故障、纯标准库。
- 失败回退：v0 在双边都拿不到正收益 → 提交基线。

### T5 类型后验 / 决策聚焦 q（v1，可选）

- 在 T4 之上，把 q 升级为"类型后验 + 条件转移"或"决策聚焦损失"
  （predict-then-optimize），而不是按对手名写分支。
- 训练/验证协议同 T4；只有 v0 在双边稳定后才做。
- 这是上限探针，不是主线；不达标就停在 v0 或基线。

### T6 独立提交版（下周）

- 把最终候选合并成零 import 的单文件 `submission/strategy.py` 候选。
- parity：与仓库版 10k 随机状态逐动作一致；跨 Python 3.10–3.12。
- 调试包到手后：`validate_submission.py`、`run_local.py --league`。

## 反模式（明确不做）

- 针对具体对手名/单一线索的硬补丁（guard 是教训：130 +1.30，dev -0.62）。
- 按测试集尾部逐个加 `if`；尾部清单只用来选特征。
- 只在 130 上有效、dev 上无效的改动。
- 用 130 调参（130 只做最终 go/no-go）。

## 时间表（27 天）

| 天 | 工作 |
|---|---|
| D1 | T1 guard（holdout 通过/dev 未通过，降级）+ T2 harness + 文档/ADR |
| D2–4 | T3 dev pool（✅）+ T4 特征/训练数据 |
| D5–7 | T4 logistic v0 + dev 验证 |
| D8–14 | T4 GBDT/类型后验 v1；dev + 130 双边验证；T6 提交版 |
| D15–21 | 收敛单文件、parity、跨版本、合规；必要时回到基线 |
| D22–27 | 缓冲、调试包验证、最终提交与 300 字说明 |

## 风险与回退

- **测试集过拟合**：130 只用于最终；dev pool + config 变体缓解。
- **硬补丁回归**：只在一边有效的改动视为池特例，回退（guard 教训）。
- **规则交互**：每条规则单独提交；合并后重跑四道门。
- **赛程假设**：`history_update`/`hide_scope`/`shuffle_schedule` 未确认；
  config 变体是代理。调试包到位后重跑。
- **调参噪声**：全池 5 seeds + 克隆对照；小的（<1.0）收益不决策。
- **回退**：任何候选不过门槛，提交基线；基线始终可提交（0.074 ms、0 故障）。

## 关联文档

- [ADR 0009：尾部优先](../decisions/0009-tail-first-rule-layer.md)
- [ADR 0010：对手建模优先于对手补丁](../decisions/0010-opponent-model-over-patches.md)
- [实验 0009：反试探守卫](../experiments/0009-tail-first-guard.md)
- [实验 0011：dev 池与 guard 硬补丁检验](../experiments/0011-dev-pool-guard-generalization.md)
- [路线图](../roadmap.md)
