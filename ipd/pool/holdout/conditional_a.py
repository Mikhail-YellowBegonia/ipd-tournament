"""阶段/阈值/终局条件类留出策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _defects(decisions):
    return sum(1 for d in decisions if d == "b")


def _coop_rate(decisions):
    c = 0
    n = 0
    for d in decisions:
        if d == "c":
            c += 1
            n += 1
        elif d == "b":
            n += 1
    return (c / n) if n else 0.5


def _last_move(decisions, default="c"):
    if decisions:
        return decisions[-1]
    return default


def _score_diff(length, my_decisions, opp_decisions, repay):
    total = 0
    n = min(length, len(my_decisions), len(opp_decisions), len(repay))
    for i in range(n):
        t, s, r, p = repay[i]
        mine = my_decisions[i]
        theirs = opp_decisions[i]
        if mine == "c":
            total += r if theirs == "c" else s
        else:
            total += t if theirs == "c" else p
    return total


def _history_coop_rate(history):
    c = 0
    n = 0
    for row in history:
        for d in row:
            if d == "c":
                c += 1
                n += 1
            elif d == "b":
                n += 1
    return (c / n) if n else 0.5


def _history_endgame_defect_rate(history):
    b = 0
    n = 0
    for row in history:
        for d in row[-5:]:
            if d == "b":
                b += 1
                n += 1
            elif d == "c":
                n += 1
    return (b / n) if n else 0.0


@register("ho_conda_warmup3_defect", ("basic", "hostile"))
def warmup3_defect(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """前 3 手合作，随后无条件永久背叛。"""
    try:
        if length < 3:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_conda_warmup5_rate_gate", ("cooperative", "conditional"))
def warmup5_rate_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """前 5 手合作，之后对手本场合作率不低于 3/4 才继续合作，否则背叛。"""
    try:
        if length < 5:
            return "c"
        if _coop_rate(opp_decisions) >= 0.75:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_conda_two_strikes_forgive", ("conditional", "endgame"))
def two_strikes_forgive(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """对手本场背叛满 2 次后转背叛，但对手连续 3 次合作则原谅；最后 3 手背叛。"""
    try:
        if length >= 22:
            return "b"
        if _defects(opp_decisions) >= 2:
            recent = opp_decisions[-3:]
            if len(recent) >= 3 and all(d == "c" for d in recent):
                return "c"
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_conda_history_rate_modes", ("history", "mixed"))
def history_rate_modes(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手近 20 场合作率在合作 / 以牙还牙 / 背叛三种模式间切换。"""
    try:
        rate = _history_coop_rate(opp_history_decisions)
        if rate < 0.35:
            return "b"
        if rate > 0.8:
            return "c"
        return _last_move(opp_decisions, "c")
    except Exception:
        return "c"


@register("ho_conda_history_endgame_preempt", ("history", "endgame"))
def history_endgame_preempt(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                            repay, opp_history_decisions, opp_opp_history_decisions):
    """若对手历史终局爱背叛就提前强硬，否则合作到最后一并背叛最后 2 手。"""
    try:
        if _history_endgame_defect_rate(opp_history_decisions) > 0.4:
            return "b"
        if length >= 23:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_conda_phase_scoped_grim", ("conditional", "cooperative"))
def phase_scoped_grim(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """按 10 手一阶段：阶段内对手背叛过就背叛到底，进入新阶段重新合作。"""
    try:
        start = (length // 10) * 10
        if "b" in opp_decisions[start:]:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_conda_score_swing_end1", ("conditional", "endgame"))
def score_swing_end1(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """前半段以牙还牙；10 手后落后 8 分以上转攻、领先 8 分以上转守；最后 1 手背叛。"""
    try:
        if length >= 24:
            return "b"
        if length >= 10:
            diff = _score_diff(length, my_decisions, opp_decisions, repay)
            if diff <= -8:
                return "b"
            if diff >= 8:
                return "c"
        return _last_move(opp_decisions, "c")
    except Exception:
        return "c"


@register("ho_conda_tft_end1", ("conditional", "endgame"))
def tft_end1(rs, length, my_id, my_decisions, opp_id, opp_decisions,
             repay, opp_history_decisions, opp_opp_history_decisions):
    """以牙还牙，但最后 1 手抢先背叛。"""
    try:
        if length >= 24:
            return "b"
        return _last_move(opp_decisions, "c")
    except Exception:
        return "c"


@register("ho_conda_last5_defect", ("cooperative", "endgame"))
def last5_defect(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """前 20 手始终合作，最后 5 手背叛。"""
    try:
        if length >= 20:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_conda_grim_then_end2", ("conditional", "endgame"))
def grim_then_end2(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """对手首次背叛后永久背叛；若对手一直合作，则在最后 2 手背叛。"""
    try:
        if length >= 23:
            return "b"
        if "b" in opp_decisions:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_conda_mixed_mode_switch", ("conditional", "endgame", "mixed"))
def mixed_mode_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """开局 4 手合作；对手合作率高走宽容以牙还牙，低则一直背叛；最后 2 手背叛。"""
    try:
        if length >= 23:
            return "b"
        if length < 4:
            return "c"
        if _coop_rate(opp_decisions) >= 0.6:
            if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "c":
                return "c"
            return _last_move(opp_decisions, "c")
        return "b"
    except Exception:
        return "c"


@register("ho_conda_noisy_phases", ("endgame", "mixed", "noisy"))
def noisy_phases(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """前 10 手高概率合作，10-19 手按概率，最后 5 手背叛。"""
    try:
        if length >= 20:
            return "b"
        if length < 10:
            return "c" if random.random() < 0.72 else "b"
        return "c" if random.random() < 0.34 else "b"
    except Exception:
        return "c"


@register("ho_conda_pavlov_end3", ("basic", "conditional", "endgame"))
def pavlov_end3(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """赢则重复、输则切换（Pavlov），最后 3 手背叛。"""
    try:
        if length >= 22:
            return "b"
        if length == 0:
            return "c"
        my_last = my_decisions[-1]
        opp_last = opp_decisions[-1]
        won = (my_last == "c" and opp_last == "c") or (
            my_last == "b" and opp_last == "c")
        if won:
            return my_last
        return "b" if my_last == "c" else "c"
    except Exception:
        return "c"
