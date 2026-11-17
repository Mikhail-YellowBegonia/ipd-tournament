"""Dev 测试池（plan 0001 / T3）：60 个独立编写的策略，用于迭代。

与留出池（`ipd/pool/holdout`，130 个）分离：

- 130 留出池**冻结**为最终测试，只用于提交前的 go/no-go；
- 本池用于规则/分类器的开发期评估（`ipd.eval.diagnose --pool dev`）；
- `ipd.pool.agents_for_stage()` 不包含本池，避免进入训练。

编写流程与留出池相同：6 个隔离作者、每人 10 个策略、只依赖标准库、
不导入 `ipd`。生成阶段的契约与检查器在 `.dev_staging/`（本地）。
"""

import importlib
import inspect

from ipd.sim.agent import Agent

# 每个模块 10 个策略；加载顺序固定，便于复现。
MODULE_NAMES = (
    "dev_basic_a",
    "dev_basic_b",
    "dev_conditional_a",
    "dev_history_a",
    "dev_mixed_a",
    "dev_mixed_b",
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

EXPECTED_COUNT = 60


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
            if not isinstance(name, str) or not name.startswith("dv_"):
                raise ValueError(f"{module_name}: bad strategy name {name!r}")
            if name in seen:
                raise ValueError(f"duplicate dev strategy name: {name}")
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
    if len(entries) != EXPECTED_COUNT:
        raise ValueError(f"dev pool should have {EXPECTED_COUNT} strategies, got {len(entries)}")
    return tuple(entries)


STRATEGIES = _load()

DEV_AGENTS = tuple(
    Agent(
        name=name,
        fn=fn,
        family="dev",
        layer="DEV",
        params={"tags": tags, "module": module_name},
    )
    for name, tags, fn, module_name in STRATEGIES
)


def dev_agents():
    """返回 dev 池 Agent 列表（浅拷贝，避免调用方修改模块级元组）。"""
    return list(DEV_AGENTS)


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
        for agent in DEV_AGENTS
        if tag in agent.params.get("tags", ())
    ]


def module_counts():
    """按源模块统计策略数。"""
    counts = {}
    for _name, _tags, _fn, module_name in STRATEGIES:
        counts[module_name] = counts.get(module_name, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "DEV_AGENTS",
    "EXPECTED_COUNT",
    "MODULE_NAMES",
    "PARAMETERS",
    "STRATEGIES",
    "TAG_DESCRIPTIONS",
    "VALID_TAGS",
    "agents_with_tag",
    "dev_agents",
    "module_counts",
    "tag_counts",
]
