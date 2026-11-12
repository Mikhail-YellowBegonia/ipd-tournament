"""检测器库：识别历史中的行为模式，供 L3 对手与信誉塑形共用（ADR 0004）。"""

from ipd.detectors.profile import EVENTS, Profile, match_profile, profile

__all__ = ["EVENTS", "Profile", "match_profile", "profile"]
