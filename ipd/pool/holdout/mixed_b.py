"""噪声包装留出策略集（mixed_b）：独立编写，仅用标准库。

13 个策略围绕 TFT / Grim / 条件阈值 / 历史信誉 / 阶段切换等骨架，
分别叠加不同机制的随机性：翻转概率、随机宽恕、随机阈值、带噪终局背叛、
偶发随机背叛、噪声 + 历史信誉门槛、噪声 + 阶段切换等。
所有函数均为无跨调用状态的纯决策函数。
"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn

    return deco


def _coop_rate(seq):
    """序列中合作占有效步（c/b）的比例；全为 m 或空序列返回 0.5。"""
    c = 0
    d = 0
    for x in seq:
        if x == "c":
            c += 1
        elif x == "b":
            d += 1
    total = c + d
    if total == 0:
        return 0.5
    return c / total


def _hist_counts(hist):
    c = 0
    d = 0
    for row in hist:
        for x in row:
            if x == "c":
                c += 1
            elif x == "b":
                d += 1
    return c, d


def _hist_rate(hist):
    """近 20 场历史里的合作比例；无有效步时返回 0.5。"""
    c, d = _hist_counts(hist)
    if c + d == 0:
        return 0.5
    return c / (c + d)


def _hist_uniform(hist):
    """历史全 c 返回 "c"，全 b 返回 "b"，其余（含全 m）返回 None。"""
    c, d = _hist_counts(hist)
    if c and not d:
        return "c"
    if d and not c:
        return "b"
    return None


def _flip(move, p):
    """以概率 p 把动作翻转（合作<->背叛）。"""
    if random.random() < p:
        if move == "c":
            return "b"
        return "c"
    return move


@register("ho_mixb_tft_tremble", ("noisy", "conditional"))
def tft_tremble(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """带噪以牙还牙：跟对手上一手，中段以 9% 概率翻转；完美合作/全背叛时无噪声。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        base = opp_decisions[-1]
        return _flip(base, 0.09)
    except Exception:
        return "c"


@register("ho_mixb_grim_random_forgive", ("noisy", "conditional"))
def grim_random_forgive(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions,
                        opp_opp_history_decisions):
    """带随机宽恕的冷酷策略：一旦被背叛就持续惩罚，每回合有 9% 概率宽恕一次。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        betrayed = False
        for x in opp_decisions:
            if x == "b":
                betrayed = True
                break
        if not betrayed:
            return "c"
        if random.random() < 0.09:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_reputation_threshold", ("noisy", "history"))
def reputation_threshold(rs, length, my_id, my_decisions, opp_id,
                         opp_decisions, repay, opp_history_decisions,
                         opp_opp_history_decisions):
    """历史信誉门槛：混合两层历史合作率得信誉分，与随机的门槛比较后决定动作。"""
    try:
        uni = _hist_uniform(opp_history_decisions)
        uni2 = _hist_uniform(opp_opp_history_decisions)
        if uni == "c" and uni2 == "c":
            return "c"
        if uni == "b" and uni2 == "b":
            return "b"
        rep = _hist_rate(opp_history_decisions)
        rep2 = _hist_rate(opp_opp_history_decisions)
        here = _coop_rate(opp_decisions) if length else 0.5
        trust = 0.55 * rep + 0.25 * rep2 + 0.20 * here
        cut = random.uniform(0.20, 0.95)
        if trust >= cut:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_random_cutoff", ("noisy", "conditional"))
def random_cutoff(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """随机阈值：本场对手合作率与每回合随机抽出的门槛比较，高于门槛才合作。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        cut = random.uniform(0.15, 0.85)
        if rate >= cut:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_noisy_endgame", ("noisy", "endgame", "conditional"))
def noisy_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """带噪 TFT，但最后三手/终局轮固定背叛（末手段不参与随机翻转）。"""
    try:
        if length >= 22 or rs >= 7:
            return "b"
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        return _flip(opp_decisions[-1], 0.09)
    except Exception:
        return "c"


@register("ho_mixb_occasional_betray", ("noisy", "cooperative", "conditional"))
def occasional_betray(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions,
                      opp_opp_history_decisions):
    """以合作为主：约 9% 概率偶发随机背叛；对手合作率低于 0.4 时转为背叛。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        if rate < 0.4:
            return "b"
        if random.random() < 0.09:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_mixb_phase_switch", ("noisy", "mixed", "endgame"))
def phase_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """分阶段切换模式：早期带噪 TFT、中期冷酷+随机宽恕、后期剥削，末尾固定背叛。"""
    try:
        if length >= 21 or rs >= 7:
            return "b"
        if length == 0:
            return "c"
        if length < 8:
            return _flip(opp_decisions[-1], 0.10)
        if length < 16:
            clean = True
            for x in opp_decisions:
                if x == "b":
                    clean = False
                    break
            if clean:
                return "c"
            if random.random() < 0.12:
                return "c"
            return "b"
        if random.random() < 0.25:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_history_blend", ("noisy", "history", "conditional"))
def history_blend(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """本场合作率与两层历史信誉加权成信任分，叠加均匀噪声后按 0.5 判定。"""
    try:
        uni = _hist_uniform(opp_history_decisions)
        uni2 = _hist_uniform(opp_opp_history_decisions)
        if uni == "c" and uni2 == "c":
            return "c"
        if uni == "b" and uni2 == "b":
            return "b"
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        rep = _hist_rate(opp_history_decisions)
        rep2 = _hist_rate(opp_opp_history_decisions)
        score = 0.45 * rate + 0.35 * rep + 0.20 * rep2
        score += random.uniform(-0.35, 0.35)
        if score >= 0.5:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_recent_window", ("noisy", "conditional"))
def recent_window(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """只看最近三手的条件惩罚：窗口内有背叛则以 9% 概率宽恕，否则背叛。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        dirty = False
        for x in opp_decisions[-3:]:
            if x == "b":
                dirty = True
                break
        if dirty:
            if random.random() < 0.09:
                return "c"
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_mixb_pavlov_noise", ("noisy", "conditional"))
def pavlov_noise(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """带噪赢留输变：按上一手收益判断好坏，好则重复、坏则反转，再叠 9% 翻转。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        t, s, r, p = repay[length - 1]
        mine = my_decisions[-1]
        theirs = opp_decisions[-1]
        if mine == "c" and theirs == "c":
            score = r
        elif mine == "c":
            score = s
        elif theirs == "c":
            score = t
        else:
            score = p
        if score >= r:
            base = mine
        elif mine == "c":
            base = "b"
        else:
            base = "c"
        return _flip(base, 0.09)
    except Exception:
        return "c"


@register("ho_mixb_reputation_endgame", ("noisy", "history", "endgame"))
def reputation_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions,
                       opp_opp_history_decisions):
    """历史信誉定基调，随机阈值 + 双向小噪声推进，最后三手/终局轮固定背叛。"""
    try:
        if length >= 22 or rs >= 7:
            return "b"
        uni = _hist_uniform(opp_history_decisions)
        uni2 = _hist_uniform(opp_opp_history_decisions)
        if uni == "c" and uni2 == "c":
            return "c"
        if uni == "b" and uni2 == "b":
            return "b"
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        rep = _hist_rate(opp_history_decisions)
        rep2 = _hist_rate(opp_opp_history_decisions)
        trust = 0.5 * rate + 0.3 * rep + 0.2 * rep2
        cut = random.uniform(0.35, 0.75)
        if trust < cut:
            if random.random() < 0.10:
                return "c"
            return "b"
        if random.random() < 0.08:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_mixb_mode_selector", ("mixed", "history"))
def mode_selector(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """按历史信誉与进行阶段，在 TFT / 全合作 / 冷酷 / 全背叛四种模式间切换。"""
    try:
        uni = _hist_uniform(opp_history_decisions)
        uni2 = _hist_uniform(opp_opp_history_decisions)
        if uni == "c" and uni2 == "c":
            return "c"
        if uni == "b" and uni2 == "b":
            return "b"
        rep = _hist_rate(opp_history_decisions)
        rep2 = _hist_rate(opp_opp_history_decisions)
        trust = 0.6 * rep + 0.4 * rep2
        if length == 0:
            if trust >= 0.5:
                return "c"
            return "b"
        if trust >= 0.7:
            if length < 5:
                return opp_decisions[-1]
            return "c"
        if trust >= 0.45:
            for x in opp_decisions:
                if x == "b":
                    return "b"
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_mixb_tit_for_two_tats", ("noisy", "conditional"))
def tit_for_two_tats(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions,
                     opp_opp_history_decisions):
    """两报还一报：连续两次被背叛才报复，其余时候以 9% 概率偶发背叛。"""
    try:
        if length == 0:
            return "c"
        rate = _coop_rate(opp_decisions)
        if rate >= 1.0:
            return "c"
        if rate <= 0.0:
            return "b"
        if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        if length == 1 and opp_decisions[-1] == "b":
            return "b"
        if random.random() < 0.09:
            return "b"
        return "c"
    except Exception:
        return "c"
