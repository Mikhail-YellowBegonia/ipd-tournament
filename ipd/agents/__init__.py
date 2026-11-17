"""手工基线（参照尺，issue #3）与后续学习型智能体。"""

from ipd.agents.baseline import baseline_agent, strategy
from ipd.agents.baseline_guard import baseline_guard_agent

__all__ = ["baseline_agent", "baseline_guard_agent", "strategy"]
