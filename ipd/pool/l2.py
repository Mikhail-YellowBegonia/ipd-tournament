"""L2：感知回合数的包装器，以及模拟同学代码特性的包装器。

每个包装器接收一个基础策略函数，返回新的策略函数。
"""

import random

from ipd.sim import rules

C, D = "c", "b"


def end_defect(base, k=1):
    """最后 k 手无条件背叛。"""
    start = rules.TURNS - k

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if length >= start:
            return D
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def opening(base, pattern="b"):
    """固定起手式，之后交给基础策略。"""
    n = len(pattern)

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if length < n:
            return pattern[length]
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def by_phase(bases):
    """三个收益阶段各用一个基础策略。"""
    if len(bases) != len(rules.PHASES):
        raise ValueError("need one base strategy per phase")

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        base = bases[rules.phase_of(length)]
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def noisy(base, eps=0.05):
    """以概率 eps 翻转基础策略的动作，模拟失误。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        move = base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
        if random.random() < eps:
            return C if move == D else D
        return move
    return s


def faulty(base, p=0.02):
    """以概率 p 抛异常，模拟有 bug 的提交（累计 3 次后被判为永久合作）。"""
    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if random.random() < p:
            raise RuntimeError("simulated bug")
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s
