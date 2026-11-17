"""经典反应式 IPD 策略留出池（basic / conditional 主题，独立编写）。

13 个彼此不同的经典或半经典反应式规则：tit-for-tat、suspicious TFT、TF2T、
stochastic TFT、generous TFT（有限宽恕额度）、Grim、Pavlov / WSLS、Joss、
Prober、soft majority（带跨场声誉先验）、hard majority、memory-one
（firm-but-fair），以及"有限惩罚 + 终局背叛"的宽恕 TFT。
只用标准库；不保存任何跨调用状态。
"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _coop_rate(moves):
    """本场对手合作率；若还没有真实决策则返回 None。"""
    total = 0
    coop = 0
    for move in moves:
        if move == "c":
            coop += 1
            total += 1
        elif move == "b":
            total += 1
    if total == 0:
        return None
    return coop / total


def _history_coop_rate(rows):
    """跨场历史合作率，忽略 "m"（补位 / 隐藏）；无有效步时返回 None。"""
    total = 0
    coop = 0
    for row in rows:
        for move in row:
            if move == "c":
                coop += 1
                total += 1
            elif move == "b":
                total += 1
    if total == 0:
        return None
    return coop / total


def _payoffs(repay, index, mine, theirs):
    """按收益表返回 (我的得分, 对方得分)。"""
    t, s, r, p = repay[index]
    if mine == "c" and theirs == "c":
        return r, r
    if mine == "c":
        return s, t
    if theirs == "c":
        return t, s
    return p, p


@register("ho_basicb_tft", ("basic", "conditional"))
def tft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
        repay, opp_history_decisions, opp_opp_history_decisions):
    """经典 TFT：首手合作，之后逐手复制对手上一手。"""
    try:
        if not opp_decisions:
            return "c"
        return "c" if opp_decisions[-1] == "c" else "b"
    except Exception:
        return "c"


@register("ho_basicb_suspicious_tft", ("basic", "conditional"))
def suspicious_tft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """怀疑版 TFT：先背叛一次试探，此后完全复制对手上一手。"""
    try:
        if not opp_decisions:
            return "b"
        return "c" if opp_decisions[-1] == "c" else "b"
    except Exception:
        return "c"


@register("ho_basicb_tf2t", ("basic", "cooperative", "conditional"))
def tf2t(rs, length, my_id, my_decisions, opp_id, opp_decisions,
         repay, opp_history_decisions, opp_opp_history_decisions):
    """TF2T：只有在对手连续两手背叛时才背叛，其余情况一律合作。"""
    try:
        if not opp_decisions:
            return "c"
        if len(opp_decisions) >= 2 and opp_decisions[-1] == "b" \
                and opp_decisions[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_basicb_stft", ("basic", "conditional", "noisy"))
def stft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
         repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 TFT：对手上一手背叛时以概率 0.72 报复，连续背叛则必然报复。"""
    try:
        if not opp_decisions:
            return "c"
        if opp_decisions[-1] == "c":
            return "c"
        if len(opp_decisions) >= 2 and opp_decisions[-2] == "b":
            return "b"
        return "b" if random.random() < 0.72 else "c"
    except Exception:
        return "c"


@register("ho_basicb_gtft", ("basic", "cooperative", "conditional"))
def gtft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
         repay, opp_history_decisions, opp_opp_history_decisions):
    """宽恕型 TFT：整场只宽恕前两次背叛（额度 g=2），额度用尽后永久报复。"""
    try:
        if not opp_decisions:
            return "c"
        if opp_decisions[-1] == "c":
            return "c"
        defections = sum(1 for move in opp_decisions if move == "b")
        if defections <= 2:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basicb_grim", ("basic", "conditional", "hostile"))
def grim(rs, length, my_id, my_decisions, opp_id, opp_decisions,
         repay, opp_history_decisions, opp_opp_history_decisions):
    """Grim（冷酷触发）：对手一旦背叛过就一直背叛，否则一直合作。"""
    try:
        if not opp_decisions:
            return "c"
        for move in opp_decisions:
            if move == "b":
                return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_basicb_pavlov", ("basic", "conditional"))
def pavlov(rs, length, my_id, my_decisions, opp_id, opp_decisions,
           repay, opp_history_decisions, opp_opp_history_decisions):
    """Pavlov / WSLS：上一手得分不低于 R（赢）就重复，低于 R（输）就换手。"""
    try:
        if not my_decisions or not opp_decisions:
            return "c"
        index = min(len(my_decisions), len(opp_decisions)) - 1
        t, s, r, p = repay[index]
        mine = my_decisions[index]
        theirs = opp_decisions[index]
        my_payoff, _ = _payoffs(repay, index, mine, theirs)
        if my_payoff >= r:
            return mine
        return "b" if mine == "c" else "c"
    except Exception:
        return "c"


@register("ho_basicb_joss", ("basic", "conditional", "noisy"))
def joss(rs, length, my_id, my_decisions, opp_id, opp_decisions,
         repay, opp_history_decisions, opp_opp_history_decisions):
    """Joss：TFT 加偷袭——对已背叛过的对手以 0.1 概率趁机背叛，报复时 0.9 还手。"""
    try:
        if not opp_decisions:
            return "c"
        rate = _coop_rate(opp_decisions)
        if opp_decisions[-1] == "b":
            if rate is None or rate < 0.35:
                return "b"
            return "b" if random.random() < 0.9 else "c"
        if rate is not None and 0.25 <= rate <= 0.9:
            for move in opp_decisions:
                if move == "b":
                    return "b" if random.random() < 0.1 else "c"
        return "c"
    except Exception:
        return "c"


@register("ho_basicb_prober", ("basic", "hostile", "mixed"))
def prober(rs, length, my_id, my_decisions, opp_id, opp_decisions,
           repay, opp_history_decisions, opp_opp_history_decisions):
    """Prober：先手 B-C-C 试探；若对手仍合作则长期剥削，否则退回 TFT 模式。"""
    try:
        if len(opp_decisions) < 3:
            return ("b", "c", "c")[len(opp_decisions)]
        if opp_decisions[1] == "c":
            return "b"
        return "c" if opp_decisions[-1] == "c" else "b"
    except Exception:
        return "c"


@register("ho_basicb_soft_majority", ("basic", "conditional", "history"))
def soft_majority(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """软多数派：本场对手合作率占七成、其跨场历史合作率占三成的加权多数。"""
    try:
        prior = _history_coop_rate(opp_history_decisions)
        rate = _coop_rate(opp_decisions)
        if rate is None:
            return "c" if (prior is None or prior >= 0.5) else "b"
        if prior is None:
            prior = 0.5
        return "c" if 0.7 * rate + 0.3 * prior >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_basicb_hard_majority", ("basic", "conditional"))
def hard_majority(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """硬多数派：先背叛；只有对手近 6 手几乎全合作且总合作率达标才转为合作。"""
    try:
        if not opp_decisions:
            return "b"
        window = opp_decisions[-6:]
        if len(window) < 4:
            return "b"
        coops = sum(1 for move in window if move == "c")
        rate = _coop_rate(opp_decisions)
        if coops >= len(window) - 1 and rate is not None and rate >= 0.7:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basicb_memory_one", ("basic", "conditional"))
def memory_one(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """记忆一手（firm-but-fair）：只在"我刚合作而对手背叛"后背叛，其余都合作。"""
    try:
        if not my_decisions or not opp_decisions:
            return "c"
        if my_decisions[-1] == "c" and opp_decisions[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_basicb_forgiving_tft", ("basic", "conditional", "endgame"))
def forgiving_tft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """宽恕 TFT：惩罚两手后主动恢复合作；最后三手（22–24）无条件背叛。"""
    try:
        if length >= 22:
            return "b"
        if not opp_decisions:
            return "c"
        if opp_decisions[-1] == "c":
            return "c"
        if len(my_decisions) >= 2 and my_decisions[-1] == "b" \
                and my_decisions[-2] == "b":
            return "c"
        return "b"
    except Exception:
        return "c"
