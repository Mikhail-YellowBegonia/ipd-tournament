"""L0：无条件策略，不看本场对手动作，也不读历史。"""

import random

C, D = "c", "b"


def all_c():
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return C
    return s


def all_d():
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return D
    return s


def rand(p=0.5):
    """以概率 p 合作。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return C if random.random() < p else D
    return s


def periodic(pattern="cb"):
    """按固定模式循环出手，如 "cb" 为交替、"ccb" 为两合作一背叛。"""
    if not pattern or set(pattern) - {C, D}:
        raise ValueError(f"bad pattern: {pattern!r}")

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        return pattern[length % len(pattern)]
    return s
