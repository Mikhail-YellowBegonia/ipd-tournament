# 留出测试池：130 个独立编写策略与基线参照

- 日期：2026-11-17
- 提交（git commit）：待填（PR #15）
- 关联决策：ADR 0001（基线是参照尺）、ADR 0003（评测对齐）、ADR 0006（分阶段）
- 对应 issue：#10（路线图任务 11）
- 代码：`ipd/pool/holdout/`、`ipd/agents/holdout_template.py`、
  `ipd/eval/holdout.py`、`tests/test_holdout_pool.py`
- 指南：`docs/eval/holdout_pool_guidelines.md`

## 假设

1. 用"没有看过我们代码"的独立编写者批量生成 100–200 个策略，可以近似真实
   同学池的多样性，并暴露训练模型对未知对手的过拟合。
2. 行为覆盖比"名字不同"更重要：读历史、条件响应、终局背叛、加噪声四类必须
   占足够比例，且标签要能被行为探针验证。
3. 手工基线在留出池上仍然是很强的参照；训练模型最终必须在**同一抽样**上
   超过它，且要报告均值与最差种子，而不是只看一次运气好的抽样。

## 设置

### 生成与审查

- 按 issue #10 的选项 B：10 个隔离的 subagent，各自只拿到课程规则、接口契约
  （`CONTRACT.md`）与主题，不读取 `ipd/agents`、`ipd/pool`、`ipd/training`；
  每个写 1 个模块、13 个策略，共 **10 × 13 = 130** 个。
- 主题：基础/周期/随机（basic_a、basic_b）、条件响应（conditional_a/b/c）、
  读历史（history_a/b/c）、混合与加噪声（mixed_a、mixed_b）。
- 第一轮机器检查：导入白名单、九参数签名、`STRATEGIES` 契约、异常/非法返回、
  标签合法性；第二轮人工审查 + 仓库内 AST 与行为测试。
- 独立性：策略模块只允许 `random`/`math`/`collections`/`itertools`/`functools`；
  AST 检查禁止 `ipd` 导入、`global`/`nonlocal`、`random.seed`、`print`、`open`。
- 不进入训练：`agents_for_stage()` 的任何阶段都不包含留出策略（单测覆盖）。

### 标签覆盖

| 标签 | 数量 | 占比 | issue 下限 |
|---|---:|---:|---:|
| basic | 29 | 22.3% | — |
| cooperative | 22 | 16.9% | — |
| hostile | 9 | 6.9% | — |
| conditional | 89 | 68.5% | 30% |
| history | 54 | 41.5% | 20% |
| endgame | 35 | 26.9% | 10% |
| noisy | 39 | 30.0% | 10% |
| mixed | 27 | 20.8% | — |

行为探针（`tests/test_holdout_pool.py`）：每个 `history` 策略在至少一组历史
变化下改变决策；每个 `conditional` 策略在至少一组本场对手行为变化下改变决策；
每个 `endgame` 策略末手背叛；每个 `noisy` 策略重复调用出现两种动作。

### 评估协议

- 留出池 130 个策略；每个种子无放回抽 30 个（真实考场 n 通常 20–40），
  与候选、基线一起跑完整 8 轮、`hide_prob=0.1`、超时 0.5 s。
- 5 个固定种子 0–4；报告每场平均分、池内名次、逐种子分数与标签覆盖。
- 命令：

```bash
python -m ipd.eval.holdout --agent ipd.agents.baseline:baseline_agent \
    --no-baseline --size 30 --seeds 0,1,2,3,4
python -m ipd.eval.holdout --agent tft --size 30 --seeds 0,1,2,3,4
```

## 结果

### 全池集成（1 个种子，8 轮）

153 个玩家（130 留出 + 23 个标准对手），`hide_prob=0.1`，种子 0：

- 耗时 38.7 s；**0 故障**，单个玩家最大故障数 0；
- 留出策略名次范围 1–153，中位 77；
- 前 5 名：`ho_hista_shadow_forecast`、`ho_conda_warmup5_rate_gate`、
  `ho_hista_provocation_budget`、`ho_condb_graded_revenge`、
  `reputation_gate(threshold=0.25)[tft]`（第 4 并列）。
  说明留出池里确实有比标准对手池更强的策略，不是"弱鸡池"。

### 单次决策耗时

最坏平均 0.043 ms/次（`ho_histb_unprovoked_milker`，完整 20×25 历史、20 次
调用平均），距 0.5 s 限制约 1 万倍余量。

### 基线参照（抽样 30/种子，5 种子）

| 策略 | 每场平均分 | 池内名次（均值） | 逐种子总分 |
|---|---:|---:|---|
| baseline（不加对手基线） | **47.94** | **1.2 / 31** | 10426, 10678, 11498, 12973, 11948 |
| tft（与基线同池） | 43.98 | 9.4 / 32 | 9745, 9175, 11894, 12579, 11140 |
| baseline（与 tft 同池） | 49.01 | — | 10581, 11060, 12472, 13218, 13439 |

- 基线在 5 个抽样中 4 次第 1、1 次第 2；在 30 个留出对手里仍是最强参照。
- tft 比同池基线低 **5.03 分/场**，名次均值 9.4/32；留出池对它并不友好。
- 两种基线数字（47.94 vs 49.01）来自不同 roster（是否把 tft 计入对手），
  不能直接比较；最终验收要用"候选与基线在同一 roster"的 gap。

### 行为多样性

- 行为指纹（7 探针 × 4 历史情境 × 25 手）：130 个策略 → **125 个不同指纹**；
  少量近似策略（最多 3 个同指纹）来自不同模块的参数化变体，属于可接受的重复。
- 人工抽查：约 9 类脚本对手下的行为签名两两不同；读历史策略确实对历史变化
  有反应，不是只把历史参数"读而未用"。

## 观察

- 留出池的读历史比例 41.5% 明显高于训练阶段 3 的 L3 占比（约 25%），
  加上终局/噪声策略更多，整体难度高于训练池；这正是最终验收想要的分布。
- 130 个策略中既有全合作/全背叛这类简单对手，也有基于历史信誉、终局检测、
  同行比较的强策略；全池前 5 名压过标准对手池，说明覆盖没有退化成"简单规则
  的重复"。
- 抽样 30 个/种子会带来名次波动（基线也从 1 到 2）；验收必须看 5 个种子的
  均值与最差，而不是单次排名。
- 5 个指纹重复说明"独立编写"仍可能收敛到相近规则；后续可补充更怪的策略
  （例如对 `rs`/`opp_id` 敏感、故意自毁、周期性诈唬），但当前覆盖已满足
  issue 下限。

## 结论与下一步

- 留出池已满足 issue #10 的产出与验收条件：130 个、0 故障、覆盖达标、
  无代码依赖、有评估入口与基线参照。
- 下一步（#6 训练完成后）：
  1. 把训练检查点包装成 `Agent`（`ipd.training.policy.policy_agent` 或导出的
     纯 Python 运行时），用 `ipd.eval.holdout` 在同一 30/种子的抽样上与基线比；
  2. 报告 5 个种子的均值、与基线的 gap、最差种子与标签覆盖；
  3. 如果基线仍占优，按 issue #6 的关键关口处理诊断（学习率、数据量、辅助
     损失权重、塑形项），而不是换用留出池调参（留出池不能用于训练）。
- 验收门槛建议：候选 5 个种子的平均每场分 ≥ 基线 + 0（即 gap ≥ 0），且至少
  3/5 个种子不落后；同时报告最差种子，避免"均值赢、最差崩"。

## 复现

```bash
.venv/bin/python -m unittest tests.test_holdout_pool
python -m ipd.eval.holdout --check
python -m ipd.eval.holdout --agent ipd.agents.baseline:baseline_agent \
    --no-baseline --size 30 --seeds 0,1,2,3,4
python -m ipd.eval.holdout --agent tft --size 30 --seeds 0,1,2,3,4
```
