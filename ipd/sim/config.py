from dataclasses import dataclass
from typing import Optional

from ipd.sim import rules


@dataclass(frozen=True)
class TournamentConfig:
    """锦标赛配置。标注"待确认"的字段需在拿到调试包后对齐（见 docs/roadmap.md）。"""

    rounds: int = rules.ROUNDS
    history_matches: int = rules.HISTORY_MATCHES

    # 待确认：历史隐藏的概率与粒度。
    # "match"：整场记录隐藏；"step"：逐步隐藏，两份历史同步。
    hide_prob: float = 0.1
    hide_scope: str = "match"

    # 待确认：历史何时更新。
    # "match"：每场结束立即写入；"round"：整轮结束后统一写入。
    history_update: str = "match"

    # 待确认：每轮内对局顺序是否随机。
    shuffle_schedule: bool = True

    # 单次决策超时（秒）；None 表示不计时，供训练使用。
    timeout: Optional[float] = 0.5
    # 同一种子内累计故障次数达到该值后，后续一律按合作处理。
    fault_limit: int = 3

    def __post_init__(self):
        if self.hide_scope not in ("match", "step"):
            raise ValueError(f"hide_scope must be 'match' or 'step': {self.hide_scope}")
        if self.history_update not in ("match", "round"):
            raise ValueError(f"history_update must be 'match' or 'round': {self.history_update}")
        if not 0.0 <= self.hide_prob <= 1.0:
            raise ValueError(f"hide_prob out of range: {self.hide_prob}")
