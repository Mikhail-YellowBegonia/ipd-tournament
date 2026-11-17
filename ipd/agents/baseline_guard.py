"""带"首手防守"的基线候选：在 baseline 的 DP 之上加一条反试探规则。

背景（ADR 0009 / 实验 0009）
--------------------------
留出池里的 ``ho_basicb_prober`` 用 B-C-C 试探：对手第 2 手若仍合作，它就
永久背叛；它对其他对手合作，因此 baseline 的跨场对手模型会把它的历史当成
"好人"先验，本场全程合作，单场被打到 -139 左右（接近"全合作对 AllD"的
-170 地板）。逐手序列是 baseline ``bcccc…c`` 对 prober ``bccbbbb…b``。

规则
----
若对手首手背叛（``length == 1`` 且 ``opp_decisions[0] == "b"``），第 2 手
直接背叛。这一手正好是 prober 的触发条件：它看到 ``opp_decisions[1] == "b"``
就不再进入永久背叛模式，退回 TFT。规则不依赖历史，也不假设对手是谁。

全池 130 × 5 seeds（同场克隆对照）上，该规则把 prober 从 -138.75 翻到
+71.00，整体效应 +1.30（4/5 种子为正）；标准池 41.98 -> 52.03。

约束
----
与 baseline 相同：只用标准库；不保存跨调用可变状态；不修改传入数据；
异常时返回合作。提交版应把 baseline 与本规则合并成零 import 的单文件
（见 plan 0001 的 T6）。
"""

from ipd.agents.baseline import strategy as _baseline_strategy
from ipd.sim.agent import Agent

COOPERATE = "c"
DEFECT = "b"

# 对手首手背叛后，我方在第 2 手（length == 1）防守。
GUARD_LENGTH = 1

__all__ = ["strategy", "baseline_guard_agent", "COOPERATE", "DEFECT"]


def _apply_guard(action, length, opp_decisions):
    """在 baseline 的动作上叠加首手防守；纯函数，便于单测。"""
    if (
        length == GUARD_LENGTH
        and action == COOPERATE
        and opp_decisions
        and opp_decisions[0] == DEFECT
    ):
        return DEFECT
    return action


def strategy(
    rs,
    length,
    my_id,
    my_decisions,
    opp_id,
    opp_decisions,
    repay,
    opp_history_decisions,
    opp_opp_history_decisions,
):
    """考试九参数接口：baseline 决策 + 反试探守卫；异常时返回 ``"c"``。"""
    try:
        action = _baseline_strategy(
            rs,
            length,
            my_id,
            my_decisions,
            opp_id,
            opp_decisions,
            repay,
            opp_history_decisions,
            opp_opp_history_decisions,
        )
        return _apply_guard(action, length, opp_decisions)
    except Exception:
        return COOPERATE


def baseline_guard_agent(name="baseline_guard"):
    """把候选包装成模拟器/评测可用的 ``Agent``。"""
    return Agent(name=name, fn=strategy, family="baseline_guard", layer="BASE")
