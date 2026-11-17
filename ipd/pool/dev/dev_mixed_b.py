"""混合/历史/终局/噪声主题 dev 策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _rate(moves, default=0.0):
    good = 0
    total = 0
    for move in moves:
        if move == "c":
            good += 1
            total += 1
        elif move == "b":
            total += 1
    if not total:
        return default
    return good / total


def _cells(hist):
    out = []
    if not hist:
        return out
    for row in hist:
        for move in row:
            if move == "c" or move == "b":
                out.append(move)
    return out


def _profile(hist, opp_hist):
    """返回 (对手历史合作率, 对手的对手历史合作率, 可见样本数)。"""
    own = _cells(hist)
    rival = _cells(opp_hist)
    if not own:
        return 0.0, 0.0, 0
    return _rate(own), _rate(rival, _rate(own)), len(own)


def _visible_share(hist):
    total = 0
    seen = 0
    if not hist:
        return 0.0
    for row in hist:
        for move in row:
            total += 1
            if move == "c" or move == "b":
                seen += 1
    if not total:
        return 0.0
    return seen / total


def _recent_rate(hist, k):
    if not hist:
        return 0.0, 0
    out = []
    for row in hist[-k:]:
        for move in row:
            if move == "c" or move == "b":
                out.append(move)
    return _rate(out), len(out)


def _early_rate(hist, k):
    if not hist:
        return 0.0, 0
    out = []
    for row in hist[:k]:
        for move in row:
            if move == "c" or move == "b":
                out.append(move)
    return _rate(out), len(out)


def _trail(moves, target):
    count = 0
    for move in reversed(moves):
        if move == target:
            count += 1
        else:
            break
    return count


def _totals(my_decisions, opp_decisions, repay, length):
    mine = 0
    theirs = 0
    limit = length
    for bound in (len(my_decisions), len(opp_decisions), len(repay)):
        if bound < limit:
            limit = bound
    for index in range(limit):
        t_val, s_val, r_val, p_val = repay[index]
        my_move = my_decisions[index]
        opp_move = opp_decisions[index]
        if my_move == "c":
            mine += r_val if opp_move == "c" else s_val
        else:
            mine += t_val if opp_move == "c" else p_val
        if opp_move == "c":
            theirs += r_val if my_move == "c" else s_val
        else:
            theirs += t_val if my_move == "c" else p_val
    return mine, theirs


@register("dv_mixb_rep_switch", ("history", "mixed"))
def rep_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """按历史信誉（合作率＋被剥削程度）在温和、强硬、对冲模式间切换。"""
    try:
        if length == 0 or not opp_decisions:
            return "c"
        trust, rival, seen = _profile(
            opp_history_decisions, opp_opp_history_decisions)
        current = _rate(opp_decisions)
        last = opp_decisions[-1]
        if seen < 5:
            return last
        exploit = rival - trust
        if trust >= 0.62 and exploit <= 0.12:
            mode = "warm"
        elif trust <= 0.30:
            mode = "cold"
        else:
            mode = "hedge"
        if mode == "warm":
            if _trail(opp_decisions, "b") >= 2:
                return "b"
            if current < 0.35:
                return "b"
            if current >= 0.90 and length >= 9 and length % 3 == 0:
                return "b"
            return "c"
        if mode == "cold":
            if current >= 0.80 and _trail(my_decisions, "c") >= 2:
                return "c"
            return "b"
        if exploit > 0.25:
            return "b" if length % 2 == 0 else "c"
        if current >= 0.55:
            return "c"
        return "b" if last == "b" else "c"
    except Exception:
        return "c"


@register("dv_mixb_hist_forgiver", ("cooperative", "history"))
def hist_forgiver(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """宽恕型针锋相对：历史信誉好就多让一步，并周期性单方面重启合作。"""
    try:
        if length == 0 or not opp_decisions:
            return "c"
        trust, rival, seen = _profile(
            opp_history_decisions, opp_opp_history_decisions)
        current = _rate(opp_decisions)
        last = opp_decisions[-1]
        if last == "c":
            return "c"
        if trust >= 0.55 and length % 5 == 0:
            return "c"
        if _trail(opp_decisions, "b") >= 2:
            return "b"
        limit = 0.35 if trust >= 0.55 else 0.72
        if current >= limit:
            return "c"
        if seen < 10:
            return "c" if length % 3 == 1 else "b"
        return "b"
    except Exception:
        return "c"


@register("dv_mixb_hist_endgame", ("endgame", "history", "mixed"))
def hist_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """历史信誉决定常规模式（互惠/压制/榨取），最后两手固定收割。"""
    try:
        if length >= 23:
            return "b"
        if length == 0 or not opp_decisions:
            return "c"
        current = _rate(opp_decisions)
        last = opp_decisions[-1]
        if _visible_share(opp_history_decisions) < 0.80:
            return "b" if current < 0.60 else "c"
        trust, rival, seen = _profile(
            opp_history_decisions, opp_opp_history_decisions)
        if seen < 10:
            return "c" if length % 3 else "b"
        if trust >= 0.62 and rival - trust <= 0.12:
            mode = "reciprocate"
        elif trust <= 0.35:
            mode = "contain"
        else:
            mode = "leverage"
        if mode == "reciprocate":
            return "c" if (current >= 0.42 or last == "c") else "b"
        if mode == "contain":
            return "c" if _trail(opp_decisions, "c") >= 2 else "b"
        if length <= 2:
            return "b"
        return "c" if last == "c" else "b"
    except Exception:
        return "c"


@register("dv_mixb_hist_forecast", ("history", "mixed"))
def hist_forecast(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """用跨场历史近端/远端合作率做趋势外推，再按预测在跟注、抢跑、匀速间切换。"""
    try:
        if length == 0 or not opp_decisions:
            return "c"
        if _visible_share(opp_history_decisions) < 0.80:
            return opp_decisions[-1]
        recent, recent_n = _recent_rate(opp_history_decisions, 3)
        early, early_n = _early_rate(opp_history_decisions, 6)
        current = _rate(opp_decisions)
        if recent_n < 4 or early_n < 4:
            return opp_decisions[-1]
        trend = recent - early
        forecast = 0.55 * recent + 0.30 * current + 0.15 * (0.5 + trend)
        if forecast >= 0.62:
            return "c" if (current >= 0.25 or opp_decisions[-1] == "c") else "b"
        if forecast <= 0.38:
            return "b"
        return "c" if length % 2 == 1 else ("b" if opp_decisions[-1] == "b" else "c")
    except Exception:
        return "c"


@register("dv_mixb_noisy_meter", ("mixed", "noisy"))
def noisy_meter(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """按本场比分位置在压制、安抚、计量三种随机模式间切换。"""
    try:
        if length == 0 or not opp_decisions:
            return "c"
        mine, theirs = _totals(my_decisions, opp_decisions, repay, length)
        current = _rate(opp_decisions)
        gap = mine - theirs
        if gap >= 5:
            p = 0.30 * current
        elif gap <= -5:
            p = 0.80 + 0.15 * current
        else:
            p = 0.50 + 0.40 * (current - 0.5)
        if p < 0.08:
            p = 0.08
        elif p > 0.92:
            p = 0.92
        return "c" if random.random() < p else "b"
    except Exception:
        return "c"


@register("dv_mixb_noisy_wsls", ("basic", "noisy"))
def noisy_wsls(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """经典赢留输变，每次决策附加固定概率抖动以跳出互相背叛的循环。"""
    try:
        if length == 0 or not my_decisions:
            return "c"
        t_val, s_val, r_val, p_val = repay[length - 1]
        last_mine = my_decisions[-1]
        last_opp = opp_decisions[-1] if opp_decisions else "c"
        if last_mine == "c":
            gain = r_val if last_opp == "c" else s_val
        else:
            gain = t_val if last_opp == "c" else p_val
        if gain > 0:
            plan = last_mine
        else:
            plan = "b" if last_mine == "c" else "c"
        if random.random() < 0.18:
            plan = "c" if plan == "b" else "b"
        return plan
    except Exception:
        return "c"


@register("dv_mixb_noisy_endgame", ("endgame", "noisy"))
def noisy_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """随机化比例回应，领先/落后微调概率，最后两轮强制收割。"""
    try:
        if length >= 23:
            return "b"
        if length == 0 or not opp_decisions:
            return "c"
        current = _rate(opp_decisions)
        mine, theirs = _totals(my_decisions, opp_decisions, repay, length)
        p = 0.30 + 0.55 * current
        if mine < theirs - 4:
            p += 0.18
        if length >= 20:
            p -= 0.22
        if p < 0.08:
            p = 0.08
        elif p > 0.92:
            p = 0.92
        return "c" if random.random() < p else "b"
    except Exception:
        return "c"


@register("dv_mixb_endgame_harvest", ("endgame", "mixed"))
def endgame_harvest(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """三段式：先示好测试，再按对手合作率榨取或互惠，最后三手收割。"""
    try:
        if length >= 22:
            return "b"
        if length == 0 or not opp_decisions:
            return "c"
        current = _rate(opp_decisions)
        if length < 4:
            return "c"
        if length >= 16:
            if current >= 0.50:
                return "b" if length % 2 == 0 else "c"
            return "c" if current >= 0.25 else "b"
        if current >= 0.80:
            return "b" if length % 2 == 0 else "c"
        if current <= 0.20:
            return "b"
        return opp_decisions[-1]
    except Exception:
        return "c"


@register("dv_mixb_endgame_counter", ("conditional", "endgame"))
def endgame_counter(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """盯住本场对手的连续动作：连合则合、连叛则叛，末两轮背叛。"""
    try:
        if length >= 23:
            return "b"
        if length == 0 or not opp_decisions:
            return "c"
        if _trail(opp_decisions, "b") >= 2:
            return "b"
        if _trail(opp_decisions, "c") >= 2:
            return "c"
        if _rate(opp_decisions) >= 0.50:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_mixb_mixed_clock", ("history", "mixed"))
def mixed_clock(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """四段人格（合作/宽厚/严格针锋/背叛）按历史信誉重排后轮转。"""
    try:
        if length == 0 or not opp_decisions:
            return "c"
        trust, rival, seen = _profile(
            opp_history_decisions, opp_opp_history_decisions)
        current = _rate(opp_decisions)
        last = opp_decisions[-1]
        if seen < 5:
            order = ("soft", "tft", "soft", "tft")
        elif trust >= 0.60:
            order = ("coop", "soft", "tft", "defect")
        else:
            order = ("coop", "strict", "defect", "tft")
        persona = order[(length // 5) % 4]
        if persona == "coop":
            return "c"
        if persona == "defect":
            return "b"
        if persona == "tft":
            return last
        if persona == "strict":
            return "b" if current < 0.50 else last
        if _trail(opp_decisions, "b") >= 3:
            return "b"
        if current < 0.30 and last == "b":
            return "b"
        return "c"
    except Exception:
        return "c"
