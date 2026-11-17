# 留出测试池编写指南（issue #10）

- 日期：2026-11-17
- 对应 issue：#10（路线图任务 11）
- 实现：`ipd/pool/holdout/`（130 个策略）、`ipd/agents/holdout_template.py`、
  `ipd/eval/holdout.py`、`tests/test_holdout_pool.py`
- 关联：ADR 0003（评测对齐）、ADR 0006（分阶段对手复杂度）、实验 0007

## 目的

留出测试池模拟"没有看过我们代码的真实同学"：最终训练出的策略必须在**没有参与
训练**的 100–200 个对手上超过手工基线（ADR 0001），才能判断泛化是否成立。
因此：

- **只用于最终验收与泛化检查，不进入训练**（`ipd.pool.agents_for_stage()` 不含它们）；
- 编写者不查看 `ipd/agents/`、`ipd/pool/`（除接口说明）、`ipd/training/` 的实现；
- 每个策略独立实现，只依赖标准库。

## 接口

函数签名固定九个参数，参数名必须一致：

```python
def strategy(rs, length, my_id, my_decisions, opp_id, opp_decisions,
             repay, opp_history_decisions, opp_opp_history_decisions):
    ...
    return "c"  # 或 "b"
```

| 参数 | 含义 |
|---|---|
| `rs` | 当前轮次，0–7 |
| `length` | 本场已完成回合数，0–24 |
| `my_id` / `opp_id` | 双方组号 |
| `my_decisions` | 本场自己此前动作，长度 `length` 的元组 |
| `opp_decisions` | 本场对手此前动作，长度 `length` 的元组 |
| `repay` | 25×4 元组，`repay[length] = (T, S, R, P)` |
| `opp_history_decisions` | 对手最近 20 场（20×25）动作 |
| `opp_opp_history_decisions` | 上述各场中对手的对手（我方）的动作 |

历史中的 `"m"` 表示补位或该步被隐藏；最新一场在最后一行。收益表：

| 回合 | T | S | R | P |
|---:|---:|---:|---:|---:|
| 0–9 | 4 | −6 | 2 | −1 |
| 10–19 | 5 | −7 | 3 | −2 |
| 20–24 | 6 | −8 | 4 | −3 |

单回合得分：我 c 对方 c→R；我 c 对方 b→S；我 b 对方 c→T；我 b 对方 b→P。

## 硬性要求

1. 只用标准库，且只允许 `random`、`math`、`collections`、`itertools`、`functools`。
2. 不得保存跨调用可变状态：不写 `global` / `nonlocal`，不用模块级缓存或类实例状态。
3. 不修改传入参数；不读写文件、不联网、不 `print`；不调用 `random.seed()`。
4. 每次调用返回 `"c"` 或 `"b"`；函数体整体包在 `try/except Exception: return "c"`。
5. 每个策略独立实现；同一文件内可共享纯函数工具，但不要从其他留出文件 import。

## 模块格式与标签

```python
"""<主题> 留出策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


@register("ho_<prefix>_<slug>", ("conditional", "history"))
def my_strategy(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """一句话说明。"""
    try:
        ...
        return "c"
    except Exception:
        return "c"
```

- 名称全局唯一，以 `ho_` 开头并带模块前缀（如 `ho_hista_`）。
- 标签从 `basic`、`cooperative`、`hostile`、`conditional`、`history`、
  `endgame`、`noisy`、`mixed` 中选 1–3 个，**必须如实反映行为**：
  - `history`：实际读取并使用了 `opp_history_decisions` 或
    `opp_opp_history_decisions`；
  - `conditional`：根据本场对手此前动作/合作率/比分改变行为；
  - `endgame`：最后 1–5 手有可观察的背叛；
  - `noisy`：合作/背叛概率都在 (0,1) 内，或按概率翻转；
  - `mixed`：在多个基础策略/模式间切换。
- 每个模块 13 个彼此不同的策略（行为不同，不只是换参数）。

## 行为覆盖（issue #10 验收）

| 特征 | 当前数量（130） | 占比 | 下限 |
|---|---:|---:|---:|
| 读历史（history） | 54 | 41.5% | 20% |
| 条件响应（conditional） | 89 | 68.5% | 30% |
| 终局背叛（endgame） | 35 | 26.9% | 10% |
| 加噪声（noisy） | 39 | 30.0% | 10% |

标签总计数：basic 29、cooperative 22、hostile 9、mixed 27。
覆盖检查：`python -m ipd.eval.holdout --check`（同时检查 100–200 规模）。

## 编写与审查流程

1. **独立编写**：编写者只拿到课程规则、本指南与
   `ipd/agents/holdout_template.py`；不查看实现代码。当前 130 个策略即由 10 个
   隔离的 subagent 按此约束生成（每个 13 个，只给契约与主题），人工审查后合入。
2. **契约检查**：每个模块用独立检查器验证导入、签名、标签、非法返回值与异常；
   仓库内测试再做 AST 检查（无 `ipd` 导入、无 `global/nonlocal`、无 `random.seed`、
   无 `print/open`）。
3. **行为探针**：`tests/test_holdout_pool.py` 对标签做行为验证：
   - 标 `history` 的策略必须在至少一组历史变化下改变决策；
   - 标 `conditional` 的策略必须在至少一组本场对手行为变化下改变决策；
   - 标 `endgame` 的策略末手必须背叛；
   - 标 `noisy` 的策略重复调用必须出现两种动作。
4. **模拟器集成**：全部留出策略 + 标准对手跑一轮锦标赛，要求 0 故障；
   单次决策耗时远低于 0.5 s（当前最坏约 0.04 ms）。
5. **多样性**：行为指纹 130 → 125 个不同簇，接近全异；允许少量近似策略。

## 评估用法

```bash
# 池子规模 + 行为覆盖
python -m ipd.eval.holdout --check

# 基线在留出池抽样上的参照成绩（30 个/种子，5 个固定种子）
python -m ipd.eval.holdout --agent ipd.agents.baseline:baseline_agent \
    --no-baseline --size 30 --seeds 0,1,2,3,4

# 候选 vs 基线（同一抽样）
python -m ipd.eval.holdout --agent tft --size 30 --seeds 0,1,2,3,4
```

默认每个种子从 130 个里抽 30 个（真实考场 n 通常 20–40），跑完整 8 轮；
`--size 130` 用整池但耗时约为 8 倍。最终验收建议报告 5 个种子的均值、
与基线的分差，以及各次抽取的标签覆盖。

## 新增对手

- 在 `ipd/pool/holdout/` 新建模块（或按主题追加到现有模块），使用上面的
  `register` / `STRATEGIES` 格式；模块名加入 `ipd/pool/holdout/__init__.py`
  的 `MODULE_NAMES`，总数保持在 100–200。
- 更新本指南的覆盖表与实验 0007，并重跑：

```bash
.venv/bin/python -m unittest tests.test_holdout_pool
.venv/bin/python -m ipd.eval.holdout --check
```

## 验收清单（对应 issue #10）

- [x] 编写指南（本文件）
- [x] 示例模板 `ipd/agents/holdout_template.py`
- [x] 至少 100 个独立编写的对手（130 个）
- [x] 模拟器中 0 故障（全池 + 标准对手，1 轮；单测覆盖）
- [x] 行为覆盖：history 41.5%、conditional 68.5%、endgame 26.9%、noisy 30.0%
- [x] 不与我方代码有依赖（AST 检查 + 人工审查）
- [x] 评估入口与基线参照成绩（实验 0007）
