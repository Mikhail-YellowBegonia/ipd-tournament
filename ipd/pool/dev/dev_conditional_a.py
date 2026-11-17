"""本场条件响应 + 终局 dev 策略，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


def _coop_rate(moves):
    """本场动作序列中合作的比例；空序列返回 0.5。"""
    if not moves:
        return 0.5
    good = 0
    for move in moves:
        if move == "c":
            good += 1
    return good / float(len(moves))


def _prior_coop_rate(hist):
    """对手最近 20 场历史中的合作比例，忽略补位/隐藏的 m。"""
    good = 0
    seen = 0
    for row in hist:
        for move in row:
            if move == "m":
                continue
            seen += 1
            if move == "c":
                good += 1
    if seen == 0:
        return 0.5
    return good / float(seen)


def _match_score(mine, theirs, repay):
    """按 repay 表计算前 length 步的累计得分。"""
    score = 0
    for i in range(len(mine)):
        t, s, r, p = repay[i]
        if mine[i] == "c":
            score += r if theirs[i] == "c" else s
        else:
            score += t if theirs[i] == "c" else p
    return score


@register("dv_conda_score_catchup", ("conditional",))
def score_catchup(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """本场比分落后于对手时转为背叛抢分，领先或持平则继续合作。"""
    try:
        if length <= 0:
            return "c"
        mine = _match_score(my_decisions, opp_decisions, repay)
        theirs = _match_score(opp_decisions, my_decisions, repay)
        if mine < theirs:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_score_press", ("conditional",))
def score_press(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """本场领先超过 6 分就压榨性背叛，否则（含落后）保持合作。"""
    try:
        if length <= 0:
            return "c"
        mine = _match_score(my_decisions, opp_decisions, repay)
        theirs = _match_score(opp_decisions, my_decisions, repay)
        if mine - theirs > 6:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_rate_floor", ("conditional", "cooperative"))
def rate_floor(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """以合作为默认，本场对手合作率跌破 0.5 且样本足够时才背叛。"""
    try:
        if len(opp_decisions) < 4:
            return "c"
        if _coop_rate(opp_decisions) < 0.5:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_proportional_end5", ("conditional", "endgame"))
def proportional_end5(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手本场合作率做比例式回应，最后 5 手固定背叛。"""
    try:
        if 25 - length <= 5:
            return "b"
        if length <= 0:
            return "c"
        if _coop_rate(opp_decisions) >= 0.55:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_conda_soft_forgive_end2", ("conditional", "endgame"))
def soft_forgive_end2(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """软报复：对手上一手背叛且自己上一手合作时回敬一次，随后宽容；最后 2 手背叛。"""
    try:
        if 25 - length <= 2:
            return "b"
        if not opp_decisions:
            return "c"
        if opp_decisions[-1] == "b" and (not my_decisions or my_decisions[-1] == "c"):
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_warm_grim_end3", ("conditional", "endgame"))
def warm_grim_end3(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """硬报复 + warmup：前 3 手无条件合作，之后对手只要背叛过一次就永久背叛；最后 3 手背叛。"""
    try:
        if 25 - length <= 3:
            return "b"
        if length < 3:
            return "c"
        for move in opp_decisions:
            if move == "b":
                return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_noisy_leak_end1", ("conditional", "endgame", "noisy"))
def noisy_leak_end1(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手本场合作率定合作倾向，并有 15% 概率泄露（翻转）；最后一手背叛。"""
    try:
        if 25 - length <= 1:
            return "b"
        if _coop_rate(opp_decisions) >= 0.5:
            base = "c"
        else:
            base = "b"
        if random.random() < 0.15:
            return "b" if base == "c" else "c"
        return base
    except Exception:
        return "c"


@register("dv_conda_patience3_end1", ("conditional", "endgame"))
def patience3_end1(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """耐心线：只有对手连续 3 手都背叛才报复，否则合作；最后一手背叛。"""
    try:
        if 25 - length <= 1:
            return "b"
        if len(opp_decisions) < 3:
            return "c"
        recent = opp_decisions[-3:]
        if recent[0] == "b" and recent[1] == "b" and recent[2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_conda_blend_reputation_end2", ("conditional", "endgame", "history"))
def blend_reputation_end2(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                          repay, opp_history_decisions, opp_opp_history_decisions):
    """把对手近 20 场合作声誉与本场合作率各半混合，低混合率才背叛；最后 2 手背叛。"""
    try:
        if 25 - length <= 2:
            return "b"
        if length <= 0:
            return "c"
        prior = _prior_coop_rate(opp_history_decisions)
        current = _coop_rate(opp_decisions)
        blended = 0.5 * prior + 0.5 * current
        if blended >= 0.6:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_conda_leader_lock_end5", ("conditional", "endgame"))
def leader_lock_end5(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """中局宽容型以牙还牙；最后 5 手按本场比分：领先就锁分背叛，落后就求和。"""
    try:
        if 25 - length <= 1:
            return "b"
        if 25 - length <= 5:
            mine = _match_score(my_decisions, opp_decisions, repay)
            theirs = _match_score(opp_decisions, my_decisions, repay)
            if mine - theirs >= 0:
                return "b"
            return "c"
        if not opp_decisions:
            return "c"
        if opp_decisions[-1] == "b" and _coop_rate(opp_decisions) < 0.5:
            return "b"
        return "c"
    except Exception:
        return "c"
