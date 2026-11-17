"""对手池。分层说明与分阶段配置见 docs/design/architecture.md#对手池、ADR 0006。"""

from ipd.pool.registry import (
    DEFAULT_MIX,
    FAMILIES,
    STAGE_LAYERS,
    agents_by_layer,
    agents_for_stage,
    canonical,
    config_for_stage,
    make_agent,
    sample_agent,
    sample_population,
    standard_agent,
    uniquify,
)

__all__ = [
    "DEFAULT_MIX",
    "FAMILIES",
    "STAGE_LAYERS",
    "agents_by_layer",
    "agents_for_stage",
    "canonical",
    "config_for_stage",
    "make_agent",
    "sample_agent",
    "sample_population",
    "standard_agent",
    "uniquify",
]
