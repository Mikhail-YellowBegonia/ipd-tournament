"""响应本场信号的条件策略留出池（独立编写，只用标准库）。

每条策略都读取本场 ``opp_decisions``（合作率 / 上一手 / 连续背叛 / 比分差 /
背叛次数）或对手历史场次，做不同的结构反应；其中若干条叠加随机化与终局背叛。
"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------------------
# 纯工具函数（无状态、不修改入参）
# ---------------------------------------------------------------------------

def _coop_rate(seq):
    """序列里的合作比例，忽略补位/隐藏的 'm'。"""
    total = 0
    coop = 0
    for move in seq:
        if move == "c":
            coop += 1
            total += 1
        elif move == "b":
            total += 1
    if total == 0:
        return 0.5
    return coop / float(total)


def _betrayals(seq):
    """序列里背叛的次数。"""
    count = 0
    for move in seq:
        if move == "b":
            count += 1
    return count


def _streak(seq, target):
    """序列末尾连续等于 target 的长度。"""
    run = 0
    for move in reversed(seq):
        if move == target:
            run += 1
        else:
            break
    return run


def _hist_rows(hist):
    """历史中至少含一个真实动作（c/b）的行，保持原顺序。"""
    rows = []
    for row in hist:
        real = 0
        for move in row:
            if move == "c" or move == "b":
                real += 1
        if real:
            rows.append(row)
    return rows


def _hist_coop_rate(hist, recent):
    """历史场次（可只取最近 recent 行）的合作比例，忽略 'm'。"""
    rows = _hist_rows(hist)
    if recent is not None:
        rows = rows[-recent:]
    coop = 0
    total = 0
    for row in rows:
        for move in row:
            if move == "c":
                coop += 1
                total += 1
            elif move == "b":
                total += 1
    if total == 0:
        return 0.5
    return coop / float(total)


def _match_score(my_decisions, opp_decisions, repay):
    """用收益表把本场已发生的步数折算成 (我的分, 对手的分)。"""
    mine = 0
    theirs = 0
    n = len(my_decisions)
    if len(opp_decisions) < n:
        n = len(opp_decisions)
    for i in range(n):
        if i >= len(repay):
            break
        t, s, r, p = repay[i]
        my_move = my_decisions[i]
        opp_move = opp_decisions[i]
        if my_move == "c" and opp_move == "c":
            mine += r
            theirs += r
        elif my_move == "c" and opp_move == "b":
            mine += s
            theirs += t
        elif my_move == "b" and opp_move == "c":
            mine += t
            theirs += s
        elif my_move == "b" and opp_move == "b":
            mine += p
            theirs += p
    return mine, theirs


# ---------------------------------------------------------------------------
# 13 条策略
# ---------------------------------------------------------------------------

@register("ho_condb_thresh50", ("conditional", "cooperative"))
def thresh50(rs, length, my_id, my_decisions, opp_id, opp_decisions,
             repay, opp_history_decisions, opp_opp_history_decisions):
    """对手本场合作率不低于 0.5 就合作，否则背叛。"""
    try:
        if length == 0:
            return "c"
        if _coop_rate(opp_decisions) >= 0.5:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_thresh70_grace", ("conditional", "cooperative"))
def thresh70_grace(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """前两手无条件合作，之后合作率需达 0.7；对手近两手回合作则恢复合作。"""
    try:
        if length < 2:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 0.7:
            return "c"
        if rate >= 0.55 and _streak(opp_decisions, "c") >= 2:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_thresh80_end", ("conditional", "endgame"))
def thresh80_end(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """对手合作率需达 0.8 才合作，最后两手一律背叛。"""
    try:
        if length >= 23:
            return "b"
        if length == 0:
            return "c"
        if _coop_rate(opp_decisions) >= 0.8:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_mirror_forgive", ("conditional", "noisy"))
def mirror_forgive(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """镜像对手上一手；对手背叛时按概率宽恕，合作率过低则不宽恕。"""
    try:
        if length == 0:
            return "c"
        if opp_decisions[-1] == "c":
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate < 0.3:
            return "b"
        forgive = 0.12 if _streak(opp_decisions, "b") <= 1 else 0.05
        if random.random() < forgive:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_streak2", ("conditional",))
def streak2(rs, length, my_id, my_decisions, opp_id, opp_decisions,
            repay, opp_history_decisions, opp_opp_history_decisions):
    """对手连续背叛两手才报复，对手一恢复合作就立刻原谅。"""
    try:
        if length == 0:
            return "c"
        run = _streak(opp_decisions, "b")
        if run >= 2:
            return "b"
        if run == 0:
            return "c"
        if my_decisions and my_decisions[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_condb_score_gap", ("conditional",))
def score_gap(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """按本场比分差与场均收益行动：场均跌破互背叛保底线就背叛，领先且对手合作就守住合作。"""
    try:
        if length == 0:
            return "c"
        mine, theirs = _match_score(my_decisions, opp_decisions, repay)
        gap = mine - theirs
        p_now = repay[length - 1][3]
        avg = mine / float(max(1, length))
        if avg <= p_now:
            return "b"
        if gap <= -3:
            return "b"
        if gap >= 3 and opp_decisions[-1] == "b":
            return "b"
        if opp_decisions[-1] == "c":
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_phase_revenge", ("conditional", "endgame"))
def phase_revenge(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """报复长度随 rs 阶段变长（早期宽恕快、后期记仇久），最后两手背叛。"""
    try:
        if length >= 23:
            return "b"
        if length == 0:
            return "c"
        last_betray = -1
        for i in range(length):
            if opp_decisions[i] == "b":
                last_betray = i
        if last_betray < 0:
            return "c"
        punish = 1 + rs // 3
        if length - last_betray <= punish:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_condb_rs_ladder", ("conditional",))
def rs_ladder(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """合作率门槛随本场 rs 抬高（0.4+0.06*rs），rs 后段额外收紧。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        threshold = 0.4 + 0.06 * rs
        if rs >= 6:
            threshold = max(threshold, 0.85)
        if rate >= threshold:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_graded_revenge", ("conditional", "noisy"))
def graded_revenge(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手本场背叛总次数分级报复：背叛越多，报复概率越高。"""
    try:
        if length == 0:
            return "c"
        betrayals = _betrayals(opp_decisions)
        if betrayals <= 1:
            return "c"
        if betrayals >= 6:
            return "b"
        if _coop_rate(opp_decisions) >= 0.75 and _streak(opp_decisions, "b") == 0:
            return "c"
        if random.random() < min(0.85, 0.15 * betrayals):
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_condb_hist_prior", ("conditional", "history"))
def hist_prior(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """把对手最近 6 场历史的合作率当先验，与本场合作率加权后取阈值。"""
    try:
        if length == 0:
            return "c"
        in_match = _coop_rate(opp_decisions)
        prior = _hist_coop_rate(opp_history_decisions, 6)
        blended = 0.55 * in_match + 0.45 * prior
        if blended >= 0.62:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_trust_from_others", ("conditional", "history"))
def trust_from_others(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """参考别人如何对待该对手：历史里别人多合作且对手回报就合作，别人都背叛就背叛。"""
    try:
        if length == 0:
            return "c"
        others_rate = _hist_coop_rate(opp_opp_history_decisions, 8)
        opp_rate = _hist_coop_rate(opp_history_decisions, 8)
        in_match = _coop_rate(opp_decisions)
        if others_rate >= 0.6 and opp_rate >= 0.5 and in_match >= 0.4:
            return "c"
        if others_rate <= 0.35:
            return "b"
        if in_match >= 0.6 and opp_rate >= 0.5:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_mixed_switch", ("mixed", "conditional", "noisy"))
def mixed_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """按合作率在全合作 / 以牙还牙 / 全背叛间切换，模糊区随机挑模式。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 0.85:
            return "c"
        if rate <= 0.35:
            return "b"
        if random.random() < 0.5:
            return "c"
        if opp_decisions[-1] == "c":
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_condb_noisy_band", ("conditional", "noisy", "endgame"))
def noisy_band(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """带抖动的合作率阈值（极端值不抖动），中段按概率翻转，最后两手背叛。"""
    try:
        if length >= 23:
            return "b"
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        threshold = 0.6 + random.uniform(-0.08, 0.08)
        if rate >= 0.9:
            return "c"
        if rate <= 0.2:
            return "b"
        if rate >= threshold:
            return "c"
        if random.random() < 0.15:
            return "c"
        return "b"
    except Exception:
        return "c"
