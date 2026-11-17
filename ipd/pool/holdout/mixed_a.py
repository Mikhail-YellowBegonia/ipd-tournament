"""随机混合 / 多阶段切换主题的留出策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _rate(seq):
    """返回非 'm' 决策中合作的比例；没有有效样本时返回 None。"""
    coop = 0
    total = 0
    for x in seq:
        if x == "c":
            coop += 1
            total += 1
        elif x == "b":
            total += 1
    if total == 0:
        return None
    return coop / total


def _mean_rate(rows):
    """把二维历史摊平后算合作率；全为 'm' 时返回 None。"""
    coop = 0
    total = 0
    for row in rows:
        for x in row:
            if x == "c":
                coop += 1
                total += 1
            elif x == "b":
                total += 1
    if total == 0:
        return None
    return coop / total


def _last(seq):
    return seq[-1] if seq else None


def _payoff(mine, theirs, table):
    """table = (T, S, R, P)，返回 mine 这一步的得分。"""
    if mine == "c":
        return table[2] if theirs == "c" else table[1]
    return table[0] if theirs == "c" else table[3]


def _my_total(my_decisions, opp_decisions, repay):
    total = 0
    n = min(len(my_decisions), len(opp_decisions), 25)
    for i in range(n):
        total += _payoff(my_decisions[i], opp_decisions[i], repay[i])
    return total


def _opp_total(my_decisions, opp_decisions, repay):
    total = 0
    n = min(len(my_decisions), len(opp_decisions), 25)
    for i in range(n):
        total += _payoff(opp_decisions[i], my_decisions[i], repay[i])
    return total


@register("ho_mixa_round_cycle", ("conditional", "mixed"))
def round_cycle(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """按 rs%3 在三套宽严不同的报复规则之间轮换。"""
    try:
        if rs % 3 == 0:
            if len(opp_decisions) >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
                return "b"
            return "c"
        if rs % 3 == 1:
            return "b" if _last(opp_decisions) == "b" else "c"
        if len(opp_decisions) >= 2 and "b" in opp_decisions[-2:]:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_mixa_rand_open_tft", ("mixed", "noisy"))
def rand_open_tft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """前两手随机出招，之后切换到带噪声的以牙还牙。"""
    try:
        if length < 2:
            return "c" if random.random() < 0.5 else "b"
        base = "b" if _last(opp_decisions) == "b" else "c"
        if random.random() < 0.2:
            base = "b" if base == "c" else "c"
        return base
    except Exception:
        return "c"


@register("ho_mixa_weighted_pick", ("mixed", "noisy"))
def weighted_pick(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """每步按随机权重在永远合作/永远背叛/TFT/冷酷触发四个基础规则中抽一个执行。"""
    try:
        draw = random.random()
        if draw < 0.2:
            return "c"
        if draw < 0.45:
            return "b"
        if draw < 0.8:
            return "b" if _last(opp_decisions) == "b" else "c"
        return "b" if "b" in opp_decisions else "c"
    except Exception:
        return "c"


@register("ho_mixa_history_score_gate", ("conditional", "history"))
def history_score_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """把对手长程历史合作率与本场合作率加权成一分，过阈值才合作。"""
    try:
        hist_rate = _mean_rate(opp_history_decisions)
        game_rate = _rate(opp_decisions)
        if hist_rate is None:
            hist_rate = 0.5
        if game_rate is None:
            game_rate = 0.5
        score = 0.55 * hist_rate + 0.45 * game_rate
        return "c" if score >= 0.6 else "b"
    except Exception:
        return "c"


@register("ho_mixa_score_gap_switch", ("conditional", "mixed"))
def score_gap_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """按当前比分差切换宽厚、平衡、加码报复三种模式。"""
    try:
        if length == 0:
            return "c"
        gap = _my_total(my_decisions, opp_decisions, repay) - \
            _opp_total(my_decisions, opp_decisions, repay)
        if gap >= 6:
            if len(opp_decisions) >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
                return "b"
            return "c"
        if gap <= -6:
            if _last(opp_decisions) == "b":
                return "b"
            return "b" if length % 2 == 0 else "c"
        return "b" if _last(opp_decisions) == "b" else "c"
    except Exception:
        return "c"


@register("ho_mixa_tft_endgame3", ("conditional", "endgame"))
def tft_endgame3(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """以牙还牙为主，但最后三手一定转为背叛。"""
    try:
        if length >= 22:
            return "b"
        return "b" if _last(opp_decisions) == "b" else "c"
    except Exception:
        return "c"


@register("ho_mixa_block_mode_endgame", ("mixed", "noisy", "endgame"))
def block_mode_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """按 opp_id/length 块/rs 定一个模式，模式内加噪声翻牌，最后两手背叛。"""
    try:
        if length >= 23:
            return "b"
        mode = (opp_id * 3 + length // 5 + rs) % 3
        if mode == 0:
            base = "b" if _last(opp_decisions) == "b" else "c"
        elif mode == 1:
            if len(opp_decisions) >= 2 and "b" in opp_decisions[-2:]:
                base = "b"
            else:
                base = "c"
        else:
            rate = _rate(opp_decisions)
            base = "c" if rate is None or rate >= 0.5 else "b"
        if random.random() < 0.25:
            base = "b" if base == "c" else "c"
        return base
    except Exception:
        return "c"


@register("ho_mixa_rate_zone_retaliate", ("conditional", "mixed", "noisy"))
def rate_zone_retaliate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手本场合作率分区：高合作回报、低合作强硬、中间区随机报复长度。"""
    try:
        rate = _rate(opp_decisions)
        if rate is None:
            return "c" if random.random() < 0.6 else "b"
        if rate >= 0.85:
            return "c"
        if rate <= 0.25:
            return "b"
        window = 1 + random.randrange(3)
        recent = opp_decisions[-window:]
        if "b" in recent and random.random() > 0.3:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_mixa_countertype_history", ("mixed", "history"))
def countertype_history(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """先把对手分成纯合作/纯背叛/类TFT/混合四型，再选对应的反制规则。"""
    try:
        coop = 0
        defect = 0
        mirror_hits = 0
        mirror_total = 0
        for row_idx in range(len(opp_history_decisions)):
            row = opp_history_decisions[row_idx]
            other = opp_opp_history_decisions[row_idx]
            for i in range(len(row)):
                if row[i] == "c":
                    coop += 1
                elif row[i] == "b":
                    defect += 1
                if i > 0:
                    a = row[i]
                    b = other[i - 1]
                    if (a == "c" or a == "b") and (b == "c" or b == "b"):
                        mirror_total += 1
                        if a == b:
                            mirror_hits += 1
        known = coop + defect
        if known == 0:
            return "c"
        if defect == 0:
            return "c"
        if coop == 0:
            return "b"
        mirror_ratio = mirror_hits / mirror_total if mirror_total else 0.5
        if mirror_ratio >= 0.7:
            return "b" if _last(opp_decisions) == "b" else "c"
        if coop / known >= 0.6:
            return "b" if length % 4 == 3 else "c"
        return "b" if _last(opp_decisions) == "b" or length % 2 == 1 else "c"
    except Exception:
        return "c"


@register("ho_mixa_block_phases", ("conditional", "mixed"))
def block_phases(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """按 length//5 在三段之间整段轮换：无条件合作、零容忍惩罚、条件合作加周期试探。"""
    try:
        block = length // 5
        phase = block % 3
        start = block * 5
        block_opp = opp_decisions[start:] if length > start else ()
        if phase == 0:
            return "c"
        if phase == 1:
            return "b" if "b" in block_opp else "c"
        if length % 4 == 3:
            return "b"
        rate = _rate(block_opp)
        if rate is None:
            rate = _rate(opp_decisions)
        if rate is None:
            return "c"
        if rate >= 0.6 and _last(opp_decisions) != "b":
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixa_rand_threshold", ("conditional", "noisy"))
def rand_threshold(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """每步随机抽一个合作阈值，再与对手近期合作率比较。"""
    try:
        window = 3 + (rs % 3)
        recent = opp_decisions[-window:] if length > 0 else ()
        rate = _rate(recent)
        if rate is None:
            return "c" if random.random() < 0.7 else "b"
        threshold = random.uniform(0.25, 0.75)
        return "c" if rate >= threshold else "b"
    except Exception:
        return "c"


@register("ho_mixa_exploit_cycle_history", ("mixed", "history"))
def exploit_cycle_history(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                          repay, opp_history_decisions, opp_opp_history_decisions):
    """用对手及其对手的历史估计可剥削性，在周期剥削与防御性镜像间切换。"""
    try:
        opp_rate = _mean_rate(opp_history_decisions)
        meta_rate = _mean_rate(opp_opp_history_decisions)
        if opp_rate is None:
            opp_rate = 0.5
        if meta_rate is None:
            meta_rate = 0.5
        if opp_rate >= 0.7 and meta_rate >= 0.6:
            return "b" if length % 3 == 2 else "c"
        if meta_rate < 0.4:
            if _last(opp_decisions) == "b":
                return "b"
            return "b" if length % 4 == 0 else "c"
        game_rate = _rate(opp_decisions)
        if game_rate is None:
            return "c"
        if game_rate >= 0.5 and _last(opp_decisions) != "b":
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixa_coop_probe_noise", ("cooperative", "noisy"))
def coop_probe_noise(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """以合作为主，随机插入试探性背叛；遇到连续两次背叛则大概率报复。"""
    try:
        if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b" if random.random() < 0.8 else "c"
        return "b" if random.random() < 0.2 else "c"
    except Exception:
        return "c"
