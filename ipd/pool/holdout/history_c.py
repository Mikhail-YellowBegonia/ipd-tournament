"""读历史的鲁棒/加权/分阶段留出策略，独立编写，只用标准库。"""

import math
import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# --------------------------------------------------------------- 纯工具函数

def _count_valid(moves):
    n = 0
    for move in moves:
        if move != "m":
            n += 1
    return n


def _rate(moves, default=0.5):
    """合作率，隐藏补位的 m 直接跳过（视为无信息）。"""
    total = 0
    coop = 0
    for move in moves:
        if move == "m":
            continue
        total += 1
        if move == "c":
            coop += 1
    if total == 0:
        return default
    return coop / float(total)


def _games_rate(games, default=0.5):
    """多场历史的总合作率，按有效步统计。"""
    total = 0
    coop = 0
    for row in games:
        for move in row:
            if move == "m":
                continue
            total += 1
            if move == "c":
                coop += 1
    if total == 0:
        return default
    return coop / float(total)


def _exp_decay_rate(games, lam, default=0.5):
    """越新的场次权重越大（指数衰减）的合作率。"""
    rows = list(games)
    n = len(rows)
    total = 0.0
    weight = 0.0
    for idx, row in enumerate(rows):
        age = n - 1 - idx
        w = math.exp(-lam * age)
        for move in row:
            if move == "m":
                continue
            weight += w
            if move == "c":
                total += w
    if weight <= 0.0:
        return default
    return total / weight


def _phase_map(repay):
    """把收益结构相同的连续步归为同一阶段。"""
    phases = []
    phase = 0
    prev = None
    for row in repay:
        key = tuple(row)
        if prev is not None and key != prev:
            phase += 1
        phases.append(phase)
        prev = key
    return phases


def _phase_games_rate(games, phases, phase, default=0.5):
    """只在指定阶段的步上统计历史合作率。"""
    total = 0
    coop = 0
    for row in games:
        for idx, move in enumerate(row):
            if idx >= len(phases) or phases[idx] != phase or move == "m":
                continue
            total += 1
            if move == "c":
                coop += 1
    if total == 0:
        return default
    return coop / float(total)


def _conditional_rate(own_games, other_games, condition, default=0.5):
    """配对历史：P(对手这手合作 | 对手的对手这手给出 condition)。"""
    hit = 0
    total = 0
    for own_row, other_row in zip(own_games, other_games):
        for own_move, other_move in zip(own_row, other_row):
            if own_move == "m" or other_move == "m":
                continue
            if other_move != condition:
                continue
            total += 1
            if own_move == "c":
                hit += 1
    if total == 0:
        return default
    return hit / float(total)


def _repeat_defect_rate(games):
    """历史中“上一手背叛后又背叛”的比例，度量背叛的持续性；无样本返回 None。"""
    after_b = 0
    stayed_b = 0
    for row in games:
        prev = None
        for move in row:
            if move == "m":
                continue
            if prev == "b":
                after_b += 1
                if move == "b":
                    stayed_b += 1
            prev = move
    if after_b == 0:
        return None
    return stayed_b / float(after_b)


def _tail_streak(moves, target="b"):
    """末尾连续 target 的个数。"""
    streak = 0
    for move in reversed(tuple(moves)):
        if move == target:
            streak += 1
        else:
            break
    return streak


def _pair_scores(mine, theirs, row):
    t_score, s_score, r_score, p_score = row
    if mine == "c":
        my_score = r_score if theirs == "c" else s_score
    else:
        my_score = t_score if theirs == "c" else p_score
    if theirs == "c":
        opp_score = r_score if mine == "c" else s_score
    else:
        opp_score = t_score if mine == "c" else p_score
    return my_score, opp_score


def _score_gap(my_decisions, opp_decisions, repay):
    """本场我领先对手的累计分差（按每一步的收益表结算）。"""
    gap = 0
    n = min(len(my_decisions), len(opp_decisions), len(repay))
    for i in range(n):
        my_score, opp_score = _pair_scores(my_decisions[i], opp_decisions[i], repay[i])
        gap += my_score - opp_score
    return gap


# --------------------------------------------------------------- 13 个策略

@register("ho_histc_phase_gate", ("history", "conditional"))
def phase_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """只看当前收益阶段的历史合作率，与本场合作率双门限放行。"""
    try:
        if length == 0:
            return "c"
        phases = _phase_map(repay)
        phase = phases[min(length, len(phases) - 1)]
        h_rate = _phase_games_rate(opp_history_decisions, phases, phase)
        m_rate = _rate(opp_decisions)
        if opp_decisions[-1] == "b" and m_rate < 0.5:
            return "b"
        if h_rate >= 0.55 and m_rate >= 0.5:
            return "c"
        if h_rate <= 0.3 and m_rate <= 0.5:
            return "b"
        return "c" if (h_rate + m_rate) / 2.0 > 0.55 else "b"
    except Exception:
        return "c"


@register("ho_histc_exp_decay", ("history", "conditional", "noisy"))
def exp_decay_window(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """近场指数加权 + 本场最近四手混合打分，中间区间按概率合作。"""
    try:
        if length == 0:
            return "c"
        h_rate = _exp_decay_rate(opp_history_decisions, 0.8)
        peer = _exp_decay_rate(opp_opp_history_decisions, 0.8)
        m_rate = _rate(opp_decisions[-4:])
        score = 0.45 * h_rate + 0.25 * peer + 0.30 * m_rate
        if score >= 0.79:
            return "c"
        if score <= 0.58:
            return "b"
        p_coop = 0.15 + 0.70 * (score - 0.58) / 0.21
        return "c" if random.random() < p_coop else "b"
    except Exception:
        return "c"


@register("ho_histc_long_trust", ("history", "conditional"))
def long_memory_trust(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """长记忆：全局合作率定信任档，落后或本场连续被坑时偏向宽恕。"""
    try:
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        if _tail_streak(opp_decisions) >= 2:
            return "b"
        if m_rate < 0.25:
            return "b"
        trust = 0.7 * h_rate + 0.3 * peer
        gap = _score_gap(my_decisions, opp_decisions, repay)
        if trust >= 0.6:
            return "c"
        if trust >= 0.45 and (gap < -3 or m_rate >= 0.5):
            return "c"
        if trust >= 0.35 and m_rate >= 0.6:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_histc_recent_clean", ("history", "noisy"))
def recent_clean_window(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """把 m 当无信息，只信最近 3 场的有效步；样本太少则退回本场判断。"""
    try:
        if length == 0:
            return "c"
        recent = opp_history_decisions[-3:]
        valid = 0
        for row in recent:
            valid += _count_valid(row)
        m_rate = _rate(opp_decisions)
        if valid < 10:
            return "c" if m_rate >= 0.5 else "b"
        h_rate = _games_rate(recent)
        peer = _games_rate(opp_opp_history_decisions[-3:])
        score = 0.50 * h_rate + 0.30 * peer + 0.20 * m_rate
        if score >= 0.85:
            return "c"
        if score <= 0.62:
            return "b"
        p_coop = 0.15 + 0.70 * (score - 0.62) / 0.23
        return "c" if random.random() < p_coop else "b"
    except Exception:
        return "c"


@register("ho_histc_recip_profile", ("history", "conditional"))
def reciprocal_profile(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """配对两份历史，估计对手被合作时回报、被背叛时报复的倾向。"""
    try:
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        m_rate = _rate(opp_decisions)
        if h_rate < 0.3:
            return "b"
        if m_rate < 0.35:
            return "b"
        reward = _conditional_rate(opp_history_decisions,
                                   opp_opp_history_decisions, "c")
        punish = _conditional_rate(opp_history_decisions,
                                   opp_opp_history_decisions, "b")
        if reward >= 0.55 and punish < 0.65:
            return "c" if m_rate >= 0.5 else "b"
        if reward < 0.4:
            return "b"
        return "c" if m_rate >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_histc_peer_norm", ("history", "cooperative", "conditional"))
def peer_norm(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """用对手的对手的合作率估计环境规范，再决定对上一手背叛的宽恕度。"""
    try:
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        trust = 0.5 * h_rate + 0.5 * peer
        if opp_decisions[-1] == "b":
            return "c" if trust >= 0.8 else "b"
        return "c" if (trust >= 0.25 or m_rate >= 0.5) else "b"
    except Exception:
        return "c"


@register("ho_histc_trend_windows", ("history", "conditional"))
def trend_windows(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """短窗对比长窗：近期合作率明显下滑则收紧，明显回暖则放宽。"""
    try:
        if length == 0:
            return "c"
        long_rate = _games_rate(opp_history_decisions)
        short_rate = _games_rate(opp_history_decisions[-4:])
        m_rate = _rate(opp_decisions)
        trend = short_rate - long_rate
        if trend <= -0.25:
            return "b"
        if trend >= 0.25:
            return "c"
        base = (long_rate + m_rate) / 2.0
        if base >= 0.5:
            return "b" if m_rate < 0.34 else "c"
        return "c" if m_rate >= 0.76 else "b"
    except Exception:
        return "c"


@register("ho_histc_stage_modes", ("history", "conditional", "mixed"))
def stage_modes(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """三阶段换模式：前期试探门限、中期互惠宽恕、后期按比分施压。"""
    try:
        if length == 0:
            return "c"
        phases = _phase_map(repay)
        phase = phases[min(length, len(phases) - 1)]
        m_rate = _rate(opp_decisions)
        h_phase = _phase_games_rate(opp_history_decisions, phases, phase)
        if phase == 0:
            peer = _games_rate(opp_opp_history_decisions)
            score = 0.5 * h_phase + 0.2 * peer + 0.3 * m_rate
            return "c" if score >= 0.7 else "b"
        if phase == 1:
            if opp_decisions[-1] == "c":
                return "c" if m_rate >= 0.4 else "b"
            if h_phase >= 0.8 and m_rate >= 0.6:
                return "c"
            return "b"
        gap = _score_gap(my_decisions, opp_decisions, repay)
        if m_rate >= 0.6 and h_phase >= 0.5 and gap >= -4:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_histc_endgame_gate", ("history", "endgame"))
def endgame_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """好历史才合作，但最后 4 手无条件背叛。"""
    try:
        if length >= 21:
            return "b"
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        if h_rate >= 0.6 and m_rate >= 0.5 and peer >= 0.4:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_histc_endgame_noise", ("history", "endgame", "noisy"))
def endgame_noise(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """前中期按历史加权概率合作，最后 3 手固定背叛。"""
    try:
        if length >= 22:
            return "b"
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        score = 0.50 * h_rate + 0.25 * peer + 0.25 * m_rate
        if score >= 0.86:
            return "c"
        if score <= 0.45:
            return "b"
        p_coop = 0.15 + 0.70 * (score - 0.45) / 0.41
        return "c" if random.random() < p_coop else "b"
    except Exception:
        return "c"


@register("ho_histc_noise_forgive", ("history", "noisy"))
def noise_forgive(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """区分偶发与持续的背叛：孤立背叛加权宽恕，其余按概率合作。"""
    try:
        if length == 0:
            return "c"
        recent = opp_history_decisions[-6:]
        h_rate = _games_rate(recent)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        repeat = _repeat_defect_rate(recent)
        bonus = 0.12 if (repeat is not None and repeat < 0.25) else 0.0
        score = 0.40 * h_rate + 0.15 * peer + 0.45 * m_rate + bonus
        if score >= 0.78:
            return "c"
        if score <= 0.55:
            return "b"
        p_coop = 0.15 + 0.70 * (score - 0.55) / 0.23
        return "c" if random.random() < p_coop else "b"
    except Exception:
        return "c"


@register("ho_histc_trust_ladder", ("history", "conditional"))
def trust_ladder(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """把长期合作率量化成 5 级信任阶梯，每级对应不同规则。"""
    try:
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        trust = 0.7 * h_rate + 0.3 * peer
        if trust < 0.2:
            level = 0
        elif trust < 0.4:
            level = 1
        elif trust < 0.6:
            level = 2
        elif trust < 0.8:
            level = 3
        else:
            level = 4
        if level == 0:
            return "b"
        if level == 1:
            return "c" if (m_rate >= 0.7 and peer >= 0.6) else "b"
        if level == 2:
            return "c" if (m_rate >= 0.5 and opp_decisions[-1] == "c") else "b"
        if level == 3:
            return "b" if (opp_decisions[-1] == "b" and m_rate < 0.5) else "c"
        if len(opp_decisions) >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_histc_mode_switch", ("history", "conditional", "mixed"))
def mode_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """按长期历史在宽容 / 一报还一报 / 冷酷三种基础玩法之间切换。"""
    try:
        if length == 0:
            return "c"
        h_rate = _games_rate(opp_history_decisions)
        peer = _games_rate(opp_opp_history_decisions)
        m_rate = _rate(opp_decisions)
        if h_rate >= 0.85 and peer >= 0.6:
            mode = "forgiving"
        elif h_rate >= 0.4:
            mode = "tit_for_tat"
        else:
            mode = "grim"
        if mode == "forgiving":
            return "b" if _tail_streak(opp_decisions) >= 2 else "c"
        if mode == "tit_for_tat":
            if opp_decisions[-1] == "b":
                return "c" if m_rate >= 0.55 else "b"
            return "c" if m_rate >= 0.3 else "b"
        return "b" if "b" in opp_decisions else "c"
    except Exception:
        return "c"
