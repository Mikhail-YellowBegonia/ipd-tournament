from dataclasses import dataclass, field
from typing import Callable

# 与考试接口一致的九参数策略函数，返回 "c" 或 "b"
StrategyFn = Callable[..., str]


@dataclass(frozen=True)
class Agent:
    """一个参赛者。name 需在同一场锦标赛中唯一。"""

    name: str
    fn: StrategyFn
    family: str = ""
    layer: str = ""
    params: dict = field(default_factory=dict, compare=False, hash=False)
