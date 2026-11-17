"""混合条件策略留出池：试探后决定、最近两手组合、合作率×阶段联合判断、
Gradual 式报复、Pavlov 变体、终局背刺、首领-跟随、二阶段切换、随机混合等。

独立编写，仅依赖标准库。
"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn

    return deco


C = "c"
D = "b"
M = "m"


# ---------------------------------------------------------------- 纯工具函数

def _rate(seq):
    """合作率（忽略补位标记）；无有效样本返回 None。"""
    seen = 0
    coop = 0
    for d in seq:
        if d == M:
            continue
        seen += 1
        if d == C:
            coop += 1
    if not seen:
        return None
    return coop / float(seen)


def _hist_rate(hist):
    """全部历史步（20 场 × 25 步）的合作率；忽略 "m"。"""
    seen = 0
    coop = 0
    for row in hist:
        for d in row:
            if d == M:
                continue
            seen += 1
            if d == C:
                coop += 1
    if not seen:
        return None
    return coop / float(seen)


def _hist_tail_rate(hist, k=5):
    """每场最后 k 步的合作率，用于观察对手的终局习性。"""
    seen = 0
    coop = 0
    for row in hist:
        vals = [d for d in row if d != M]
        for d in vals[-k:]:
            seen += 1
            if d == C:
                coop += 1
    if not seen:
        return None
    return coop / float(seen)


def _streak(seq, val):
    """结尾处连续 val 的长度。"""
    n = 0
    for d in reversed(seq):
        if d != val:
            break
        n += 1
    return n


def _scores(my_decisions, opp_decisions, repay):
    """按收益表回放本场比分，返回 (我的累计分, 对手累计分)。"""
    me = 0
    them = 0
    n = min(len(my_decisions), len(opp_decisions))
    for i in range(n):
        mine = my_decisions[i]
        theirs = opp_decisions[i]
        if mine not in (C, D) or theirs not in (C, D):
            continue
        t, s, r, p = repay[i]
        if mine == C and theirs == C:
            me += r
            them += r
        elif mine == C:
            me += s
            them += t
        elif theirs == C:
            me += t
            them += s
        else:
            me += p
            them += p
    return me, them


# ---------------------------------------------------------------- 13 个策略

@register("ho_condc_probe_two_strike", ("conditional", "mixed", "cooperative"))
def probe_two_strike(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """首手合作做试探：对手开局若背叛，则整场转入“连续两次合作才放行”的惩罚模式；
       否则按一报还一报走。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        if opp[0] == D:
            if length == 1:
                return D
            if opp[-1] == D or opp[-2] == D:
                return D
            return C
        return C if opp[-1] == C else D
    except Exception:
        return C


@register("ho_condc_combo_matrix", ("conditional", "noisy"))
def combo_matrix(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手最近两手组合查表：连合合作、连背背叛、刚回头大概率原谅、刚背刺大概率回击，
       两个模糊组合上带随机。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        if length == 1:
            return C if opp[0] == C else D
        last, prev = opp[-1], opp[-2]
        if last == C and prev == C:
            return C
        if last == D and prev == D:
            return D
        if last == C:
            return C if random.random() < 0.8 else D
        if my_decisions[-1] == C:
            return D if random.random() < 0.75 else C
        return D if random.random() < 0.55 else C
    except Exception:
        return C


@register("ho_condc_gradual_echo", ("conditional",))
def gradual_echo(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """Gradual 式报复：对手第 n 次背叛就回击 min(n, 4) 手，回击期内再次被刺则顺延并加长。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        defects = 0
        last_defect = -1
        for i, d in enumerate(opp):
            if d == D:
                defects += 1
                last_defect = i
        if last_defect < 0:
            return C
        needed = min(defects, 4)
        since = length - 1 - last_defect
        if since < needed:
            return D
        return C
    except Exception:
        return C


@register("ho_condc_rate_phase_gate", ("conditional", "history"))
def rate_phase_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """合作率×阶段联合判断：把 20 场历史合作率当先验与本场合作率融合，收益阶段越高门槛越低；
       若历史显示对手爱在末段背刺，则提前抬高合作门槛。"""
    try:
        prior = _hist_rate(opp_history_decisions)
        if length == 0:
            if prior is None:
                return C
            return D if prior < 0.25 else C
        cur = _rate(opp_decisions)
        if cur is None:
            cur = 0.5
        if prior is None:
            blended = cur
        else:
            blended = (cur * length + prior * 4.0) / float(length + 4.0)
        if length < 10:
            thr = 0.62
        elif length < 20:
            thr = 0.55
        else:
            thr = 0.45
        tail = _hist_tail_rate(opp_history_decisions, 5)
        if length >= 18 and tail is not None and tail < 0.5:
            thr = 0.85
        if blended >= thr:
            return C
        if opp_decisions[-1] == C and _streak(opp_decisions, C) >= 2:
            return C
        return D
    except Exception:
        return C


@register("ho_condc_pavlov_wsls", ("conditional", "basic"))
def pavlov_wsls(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """Pavlov 赢留输变：上一手得分不低于 R 就重复，否则翻转；被单次背刺时当噪声原谅一手。"""
    try:
        if length == 0:
            return C
        t, s, r, p = repay[length - 1]
        mine = my_decisions[-1]
        theirs = opp_decisions[-1]
        if mine == C and theirs == C:
            gain = r
        elif mine == C:
            gain = s
        elif theirs == C:
            gain = t
        else:
            gain = p
        if gain >= r:
            return C if mine == C else D
        if theirs == D and _streak(opp_decisions, D) == 1:
            return C
        return D if mine == C else C
    except Exception:
        return C


@register("ho_condc_pavlov_endgame", ("conditional", "endgame", "mixed"))
def pavlov_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """WSLS 主体 + 中段收割：对手很软且我落后就提前背叛拿 T；对手已崩则收手；
       最后 5 手一律背叛。"""
    try:
        if length == 0:
            return C
        if length >= 20:
            return D
        opp = opp_decisions
        rate = _rate(opp)
        if rate is None:
            rate = 0.5
        me, them = _scores(my_decisions, opp, repay)
        if length >= 15 and rate >= 0.85 and me <= them:
            return D
        if length >= 16 and rate <= 0.4:
            return D
        t, s, r, p = repay[length - 1]
        mine = my_decisions[-1]
        theirs = opp[-1]
        if mine == C and theirs == C:
            gain = r
        elif mine == C:
            gain = s
        elif theirs == C:
            gain = t
        else:
            gain = p
        if gain >= r:
            return C if mine == C else D
        return D if mine == C else C
    except Exception:
        return C


@register("ho_condc_leader_follower", ("conditional", "mixed"))
def leader_follower(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """首领-跟随式：比分领先当“首领”，只做防卫性合作（被刺才回击）；落后当“跟随者”，
       抢 T 追分，只在对手连续两手合作时才给合作。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        me, them = _scores(my_decisions, opp, repay)
        if me >= them:
            return D if opp[-1] == D else C
        if length >= 2 and opp[-1] == C and opp[-2] == C:
            return C
        return D
    except Exception:
        return C


@register("ho_condc_two_phase_trust", ("conditional", "mixed", "cooperative"))
def two_phase_trust(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """二阶段切换：前 12 手不计前嫌地合作（仅连续两次被背叛才回击），之后按第一阶段
       对手合作率切到宽和 / 严格 / 断交三种模式。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        if length < 12:
            return D if _streak(opp, D) >= 2 else C
        first = _rate(opp[:12])
        if first is None:
            first = 0.5
        if first >= 0.8:
            return D if _streak(opp, D) >= 2 else C
        if first >= 0.5:
            return C if opp[-1] == C else D
        return D
    except Exception:
        return C


@register("ho_condc_reputation_gate", ("conditional", "history"))
def reputation_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """信誉闸门：用对手 20 场历史合作率分流——劣迹斑斑直接背叛，信誉好则宽容
       （只惩罚连续两次背刺），中性按一报还一报。"""
    try:
        prior = _hist_rate(opp_history_decisions)
        opp = opp_decisions
        if length == 0:
            return D if (prior is not None and prior < 0.3) else C
        if prior is not None and prior < 0.3:
            return D
        if prior is not None and prior >= 0.75:
            if length >= 2 and opp[-1] == D and opp[-2] == D:
                return D
            return C
        return C if opp[-1] == C else D
    except Exception:
        return C


@register("ho_condc_history_classifier", ("conditional", "history"))
def history_classifier(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """历史分类：统计对手过往面对“背叛它的对手”时是否仍合作——惯犯就一起烂；
       老好人则落后时榨取、领先时维持其合作意愿；只有互惠者才长期给合作。"""
    try:
        prior = _hist_rate(opp_history_decisions)
        opp = opp_decisions
        me, them = _scores(my_decisions, opp, repay)
        if length == 0:
            return D if (prior is not None and prior < 0.3) else C
        if prior is not None and prior < 0.3:
            return D
        n_def = 0
        n_def_c = 0
        for i, row in enumerate(opp_history_decisions):
            if i < len(opp_opp_history_decisions):
                other = opp_opp_history_decisions[i]
            else:
                other = ()
            for j, d in enumerate(row):
                od = other[j] if j < len(other) else M
                if d != C and d != D:
                    continue
                if od == D:
                    n_def += 1
                    if d == C:
                        n_def_c += 1
        pushover = n_def >= 4 and n_def_c / float(n_def) >= 0.6
        if pushover:
            if me < them:
                return D                     # 落后：榨取软手的 T
            if opp[-1] == D and (length < 2 or opp[-2] == D):
                return D                     # 领先：仍惩罚双背刺
            return C                         # 领先：保持对方合作意愿
        if prior is not None and prior >= 0.75:
            cur = _rate(opp)
            if cur is not None and cur >= 0.5:
                if not (length >= 2 and opp[-1] == D and opp[-2] == D):
                    return C
        return D
    except Exception:
        return C


@register("ho_condc_endgame_pressure", ("conditional", "endgame"))
def endgame_pressure(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """终局压力：最后 5 手必定背叛；对手本场越软，收割起点越提前（软手从第 15 手开始）。"""
    try:
        if length == 0:
            return C
        if length >= 20:
            return D
        opp = opp_decisions
        rate = _rate(opp)
        if rate is None:
            rate = 0.5
        if length >= 15 and rate >= 0.8:
            return D
        return C if opp[-1] == C else D
    except Exception:
        return C


@register("ho_condc_noisy_mix", ("conditional", "noisy", "mixed"))
def noisy_mix(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """随机混合：按对手合作率过滤两个极端，中间地带在 TFT / 宽恕型 / 掠夺型三种模式间
       随机切换，并叠 10% 执行噪声。"""
    try:
        if length == 0:
            return D if random.random() < 0.2 else C
        opp = opp_decisions
        rate = _rate(opp)
        if rate is None:
            rate = 0.5
        if rate <= 0.2:
            return D
        if rate >= 0.9:
            return C
        roll = random.random()
        if roll < 0.4:
            base = C if opp[-1] == C else D
        elif roll < 0.65:
            if opp[-1] == C or (length >= 2 and opp[-2] == C):
                base = C
            else:
                base = D
        else:
            base = D if random.random() < 0.6 else C
        if random.random() < 0.1:
            return D if base == C else C
        return base
    except Exception:
        return C


@register("ho_condc_scoreboard_endgame", ("conditional", "endgame"))
def scoreboard_endgame(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """比分驱动终局：领先越多收割越早（第 17 手起），势均力敌第 20 手起，落后则拖到第 22 手
       并先试图合作；被连续两次背刺立刻回击。"""
    try:
        if length == 0:
            return C
        opp = opp_decisions
        me, them = _scores(my_decisions, opp, repay)
        margin = me - them
        if margin >= 10:
            start = 17
        elif margin >= 0:
            start = 20
        else:
            start = 22
        if length >= start:
            return D
        if _streak(opp, D) >= 2:
            return D
        if margin < -6 and length >= 12:
            return C
        return C if opp[-1] == C else D
    except Exception:
        return C
