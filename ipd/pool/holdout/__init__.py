"""留出测试池（issue #10）：由独立编写者提交的 130 个策略。

编写者只依据课程规则与 `docs/eval/holdout_pool_guidelines.md`，不查看
`ipd/agents`、`ipd/pool`、`ipd/training` 的实现；每个策略模块只导入标准库
（`random`/`math`/`collections`/`itertools`/`functools`），不导入 `ipd`。

本文件只负责加载与元数据整理，不把留出池放进训练对手池：
`ipd.pool.agents_for_stage()` 不会返回它们。留出池只用于最终验收与泛化检查。

用法::

    from ipd.pool.holdout import HOLDOUT_AGENTS, holdout_agents, tag_counts

    agents = holdout_agents()            # 新的列表，元素为 ipd.sim.Agent
    print(tag_counts())                  # {"history": 54, ...}
"""

import importlib
import inspect

from ipd.sim.agent import Agent

# 每个模块 13 个策略；加载顺序固定，便于复现。
MODULE_NAMES = (
    "basic_a",
    "basic_b",
    "conditional_a",
    "conditional_b",
    "conditional_c",
    "history_a",
    "history_b",
    "history_c",
    "mixed_a",
    "mixed_b",
)

VALID_TAGS = frozenset(
    {
        "basic",
        "cooperative",
        "hostile",
        "conditional",
        "history",
        "endgame",
        "noisy",
        "mixed",
    }
)

TAG_DESCRIPTIONS = {
    "basic": "固定规则/经典策略",
    "cooperative": "以合作为主",
    "hostile": "以背叛为主",
    "conditional": "响应本场对手行为",
    "history": "读取并使用对手历史",
    "endgame": "终局切换为背叛",
    "noisy": "带随机性",
    "mixed": "多策略/多模式切换",
}

PARAMETERS = (
    "rs",
    "length",
    "my_id",
    "my_decisions",
    "opp_id",
    "opp_decisions",
    "repay",
    "opp_history_decisions",
    "opp_opp_history_decisions",
)


def _load():
    """导入各模块并校验契约；返回 `(name, tags, fn, module_name)` 元组。"""
    entries = []
    seen = set()
    for module_name in MODULE_NAMES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        strategies = getattr(module, "STRATEGIES", None)
        if not isinstance(strategies, (tuple, list)):
            raise ValueError(f"{module_name}: STRATEGIES must be a list")
        for item in strategies:
            if not (isinstance(item, (tuple, list)) and len(item) == 3):
                raise ValueError(f"{module_name}: bad strategy entry {item!r}")
            name, tags, fn = item
            if not isinstance(name, str) or not name.startswith("ho_"):
                raise ValueError(f"{module_name}: bad strategy name {name!r}")
            if name in seen:
                raise ValueError(f"duplicate holdout strategy name: {name}")
            if not callable(fn):
                raise ValueError(f"{name}: strategy is not callable")
            if not isinstance(tags, (tuple, list)) or not tags:
                raise ValueError(f"{name}: tags must be a non-empty tuple")
            unknown = set(tags) - VALID_TAGS
            if unknown:
                raise ValueError(f"{name}: unknown tags {sorted(unknown)}")
            signature = inspect.signature(fn)
            if list(signature.parameters) != list(PARAMETERS):
                raise ValueError(
                    f"{name}: expected parameters {PARAMETERS}, "
                    f"got {list(signature.parameters)}"
                )
            seen.add(name)
            entries.append((name, tuple(tags), fn, module_name))
    if not 100 <= len(entries) <= 200:
        raise ValueError(
            f"holdout pool should have 100-200 strategies, got {len(entries)}"
        )
    return tuple(entries)


STRATEGIES = _load()

HOLDOUT_AGENTS = tuple(
    Agent(
        name=name,
        fn=fn,
        family="holdout",
        layer="HOLDOUT",
        params={"tags": tags, "module": module_name},
    )
    for name, tags, fn, module_name in STRATEGIES
)


def holdout_agents():
    """返回留出池 Agent 列表（浅拷贝，避免调用方修改模块级元组）。"""
    return list(HOLDOUT_AGENTS)


def tag_counts():
    """返回每个标签覆盖的策略数。"""
    counts = {}
    for _name, tags, _fn, _module in STRATEGIES:
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items()))


def agents_with_tag(tag):
    """返回带指定标签的 Agent 列表。"""
    if tag not in VALID_TAGS:
        raise ValueError(f"unknown tag: {tag}")
    return [
        agent
        for agent in HOLDOUT_AGENTS
        if tag in agent.params.get("tags", ())
    ]


def module_counts():
    """按源模块统计策略数。"""
    counts = {}
    for _name, _tags, _fn, module_name in STRATEGIES:
        counts[module_name] = counts.get(module_name, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "HOLDOUT_AGENTS",
    "MODULE_NAMES",
    "PARAMETERS",
    "STRATEGIES",
    "TAG_DESCRIPTIONS",
    "VALID_TAGS",
    "agents_with_tag",
    "holdout_agents",
    "module_counts",
    "tag_counts",
]
