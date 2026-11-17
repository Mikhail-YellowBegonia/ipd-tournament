"""解析评估命令的 `--agent` 参数。

支持三类写法：

- 族名：`tft`、`all_c`、`gradual`；
- 带参数/基础策略的族：`end_defect(k=1)[tft]`、
  `by_phase[tft|tf2t|grim]`；
- 模块或文件：`ipd.agents.baseline:baseline_agent`、
  `ipd.agents.baseline:strategy`、`./my_agent.py`。

模块属性如果是 `Agent` 直接使用；如果是工厂函数（无需必填参数）就调用；
否则当作九参数策略函数包装成 `Agent`。
"""

import ast
import importlib
import importlib.util
import inspect
from pathlib import Path
import re
import sys

from ipd.pool.registry import FAMILY_BY_NAME, make_agent
from ipd.sim import Agent

_FAMILY_RE = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"(?:\((?P<params>[^()]*)\))?"
    r"(?:\[(?P<bases>.+)\])?$"
)


def _parse_value(text):
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        # 允许 pattern=cb 这类不加引号的字符串
        return text


def _parse_params(text):
    if not text or not text.strip():
        return {}
    params = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"bad parameter (expected key=value): {item!r}")
        params[key.strip()] = _parse_value(value)
    return params


def _parse_bases(text):
    if not text:
        return ()
    return tuple(load_agent(part.strip()) for part in text.split("|") if part.strip())


def load_agent(spec):
    """把命令行写法解析成 `Agent`。

    Raises:
        ValueError: 族名未知且不是可导入的模块/文件。
        TypeError: 模块属性既不是 Agent、工厂函数，也不是策略函数。
    """
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("agent spec must be a non-empty string")
    spec = spec.strip()

    match = _FAMILY_RE.match(spec)
    if match and match.group("name") in FAMILY_BY_NAME:
        name = match.group("name")
        params = _parse_params(match.group("params"))
        bases = _parse_bases(match.group("bases"))
        return make_agent(name, params, bases)

    # 没有参数/基础策略的未知名字，按模块处理；带括号的族写错则直接报错。
    if match and not match.group("params") and not match.group("bases"):
        return _load_module_agent(spec)

    if match:
        name = match.group("name")
        raise ValueError(
            f"unknown family: {name!r}; "
            f"known families: {', '.join(sorted(FAMILY_BY_NAME))}"
        )
    return _load_module_agent(spec)


def _import_file(path):
    path = Path(path).resolve()
    if not path.exists():
        raise ValueError(f"agent file not found: {path}")
    module_name = f"_ipd_eval_agent_{abs(hash(path))}"
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot import agent file: {path}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module
    module_spec.loader.exec_module(module)
    return module


def _load_module_agent(spec):
    module_name, sep, attr_name = spec.partition(":")
    attr_name = attr_name or "strategy"
    if module_name.endswith(".py") or Path(module_name).exists():
        module = _import_file(module_name)
    else:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise ValueError(f"cannot import agent module {module_name!r}") from exc
    try:
        attr = getattr(module, attr_name)
    except AttributeError as exc:
        raise ValueError(f"module {module_name!r} has no attribute {attr_name!r}") from exc
    return _as_agent(attr, spec)


def _as_agent(attr, name):
    if isinstance(attr, Agent):
        return attr
    if not callable(attr):
        raise TypeError(f"agent object {name!r} is not an Agent or callable")

    # 九参数策略函数有必填参数，不能直接调用；工厂函数没有必填参数。
    try:
        signature = inspect.signature(attr)
        required = [
            p
            for p in signature.parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
    except (TypeError, ValueError):
        required = []
    if required:
        return Agent(name=name, fn=attr, family=name, layer="CUSTOM")

    result = attr()
    if isinstance(result, Agent):
        return result
    if callable(result):
        return Agent(name=name, fn=result, family=name, layer="CUSTOM")
    raise TypeError(f"factory {name!r} did not return an Agent or callable")


def load_agents(specs):
    """批量解析；重复的族写法不会去重，由调用方决定。"""
    return [load_agent(spec) for spec in specs]


__all__ = ["load_agent", "load_agents"]
