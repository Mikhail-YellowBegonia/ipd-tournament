"""周期/随机/概率/阈值变体 dev 策略，独立编写，只用标准库。

主题：周期模式、随机 p 合作、按概率翻转、以牙还牙变体、开场试探。
"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _coop_count(decisions):
    """统计一段决策序列里合作的次数。"""
    count = 0
    for move in decisions:
        if move == "c":
            count += 1
    return count


def _coop_rate(decisions):
    """一段决策序列的合作率；序列为空时按全合作处理。"""
    if not decisions:
        return 1.0
    return _coop_count(decisions) / float(len(decisions))


@register("dv_basicb_period_cb", ("basic",))
def period_cb(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """两拍周期：轮次为偶数时合作，为奇数时背叛。"""
    try:
        if rs % 2 == 0:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basicb_period_ccdd", ("basic",))
def period_ccdd(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """四拍周期 c,c,b,b：按本场已进行手数取相位。"""
    try:
        phase = length % 4
        if phase < 2:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basicb_period_defect_first", ("basic",))
def period_defect_first(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """三拍周期 b,c,c：先背叛一次示强，随后两手合作，循环往复。"""
    try:
        phase = rs % 3
        if phase == 0:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basicb_random_p30", ("noisy",))
def random_p30(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """每手以 0.3 的概率合作，其余背叛（偏背叛的随机策略）。"""
    try:
        if random.random() < 0.3:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basicb_random_p70", ("noisy",))
def random_p70(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """每手以 0.7 的概率合作，其余背叛（偏合作的随机策略）。"""
    try:
        if random.random() < 0.7:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basicb_stoch_rate_threshold", ("conditional", "noisy"))
def stoch_rate_threshold(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                         repay, opp_history_decisions, opp_opp_history_decisions):
    """随机阈值对赌：抽一个均匀阈值，本场对手合作率高于阈值就合作，否则背叛。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if random.random() < rate:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_basicb_tft_copy", ("basic", "conditional"))
def tft_copy(rs, length, my_id, my_decisions, opp_id, opp_decisions,
             repay, opp_history_decisions, opp_opp_history_decisions):
    """经典以牙还牙：首手合作，之后完全照抄对手上一手。"""
    try:
        if length == 0:
            return "c"
        if opp_decisions[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basicb_tft_suspicious", ("basic", "conditional"))
def tft_suspicious(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """疑心版以牙还牙：首手背叛试探，之后仍照抄对手上一手。"""
    try:
        if length == 0:
            return "b"
        if opp_decisions[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basicb_opening_probe", ("basic", "conditional"))
def opening_probe(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """开场两手背叛做试探：对手若两手都忍让则转为合作，被对手连续两手背叛再惩罚。"""
    try:
        if length < 2:
            return "b"
        if opp_decisions[0] != "c" or opp_decisions[1] != "c":
            return "b"
        if opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_basicb_tft_two_strikes", ("basic", "conditional"))
def tft_two_strikes(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """两报还一报：只有对手连续两手背叛才还击，单次背叛予以容忍。"""
    try:
        if length < 2:
            return "c"
        if opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"
