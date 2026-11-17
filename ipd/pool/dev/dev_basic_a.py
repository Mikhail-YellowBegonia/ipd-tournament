"""经典基础与极端合作/敌对 dev 策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _last(moves, default="c"):
    """取最近一步的合法决策（"c"/"b"），没有就返回默认值。"""
    if not moves:
        return default
    for move in reversed(moves):
        if move == "c" or move == "b":
            return move
    return default


def _tally(moves):
    """统计合作与背叛的步数，忽略 "m" 补位。"""
    coop = 0
    defect = 0
    for move in moves:
        if move == "c":
            coop += 1
        elif move == "b":
            defect += 1
    return coop, defect


def _score(shape, mine, theirs):
    """按 (T, S, R, P) 收益表计算单步得分。"""
    T, S, R, P = shape
    if mine == "c":
        return R if theirs == "c" else S
    return T if theirs == "c" else P


@register("dv_basica_allc", ("basic", "cooperative"))
def always_cooperate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """永远合作，不看任何历史。"""
    try:
        return "c"
    except Exception:
        return "c"


@register("dv_basica_alld", ("basic", "hostile"))
def always_defect(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """永远背叛，不看任何历史。"""
    try:
        return "b"
    except Exception:
        return "b"


@register("dv_basica_tft", ("basic", "conditional"))
def tit_for_tat(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """首手合作，之后原样复制对手上一手。"""
    try:
        if length <= 0:
            return "c"
        return _last(opp_decisions, "c")
    except Exception:
        return "c"


@register("dv_basica_tf2t", ("basic", "conditional"))
def tit_for_two_tats(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """宽容两报还一报：对手连续两手背叛才回以背叛。"""
    try:
        if length < 2:
            return "c"
        if opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basica_grim", ("basic", "conditional"))
def grim_trigger(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """冷酷触发：对手只要背叛过一次，此后永远背叛。"""
    try:
        for move in opp_decisions:
            if move == "b":
                return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basica_gtft", ("basic", "conditional", "noisy"))
def generous_tit_for_tat(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                         repay, opp_history_decisions, opp_opp_history_decisions):
    """宽容一报还一报：对手上一手背叛时以 0.2 概率原谅。"""
    try:
        if length <= 0:
            return "c"
        if _last(opp_decisions, "c") == "c":
            return "c"
        if random.random() < 0.2:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basica_wsls", ("basic", "conditional"))
def win_stay_lose_shift(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """赢则重复上一手，输则切换到相反动作。"""
    try:
        if length <= 0:
            return "c"
        mine = _last(my_decisions, "c")
        theirs = _last(opp_decisions, "c")
        index = min(length - 1, len(repay) - 1)
        T, S, R, P = repay[index]
        gained = _score((T, S, R, P), mine, theirs)
        if gained >= R:
            return mine
        return "b" if mine == "c" else "c"
    except Exception:
        return "c"


@register("dv_basica_soft_majority", ("basic", "conditional", "cooperative"))
def soft_majority(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """软多数：对手合作次数不少于背叛次数就继续合作。"""
    try:
        if length <= 0:
            return "c"
        coop, defect = _tally(opp_decisions)
        return "c" if coop >= defect else "b"
    except Exception:
        return "c"


@register("dv_basica_hard_majority", ("basic", "conditional"))
def hard_majority(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """硬多数：只有对手合作率严格超过三分之二才合作。"""
    try:
        if length <= 0:
            return "c"
        coop, defect = _tally(opp_decisions)
        total = coop + defect
        if total <= 0:
            return "c"
        return "c" if coop * 3 > total * 2 else "b"
    except Exception:
        return "c"


@register("dv_basica_period_cbb", ("basic", "hostile"))
def periodic_coop_betray_betray(rs, length, my_id, my_decisions, opp_id,
                                opp_decisions, repay, opp_history_decisions,
                                opp_opp_history_decisions):
    """三步周期：合作-背叛-背叛，循环往复。"""
    try:
        return "c" if length % 3 == 0 else "b"
    except Exception:
        return "c"
