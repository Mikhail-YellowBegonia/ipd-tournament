"""对手池。分层说明见 docs/design/architecture.md#对手池。"""

from ipd.pool.registry import FAMILIES, canonical, make_agent, sample_agent, sample_population, uniquify

__all__ = ["FAMILIES", "canonical", "make_agent", "sample_agent", "sample_population", "uniquify"]
