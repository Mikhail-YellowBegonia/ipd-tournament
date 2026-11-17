"""历史读牌 / 终局识别主题的留出策略，独立编写，只用标准库。

覆盖的规则族：
- 逐场统计对手最后 1 / 2 / 3 手的背叛比例，并区分"只在末段翻脸"与"从头烂到尾"；
- 按对手典型终局提前量（onset）提前动手；
- 用对齐历史（对手 vs 对手的对手）算末段劫掠率、剥削率、无端背叛率；
- 信誉门槛（整体合作率、近期加权合作率）分档响应；
- 坏名声 -> 全程背叛 / 只剥削；
- 末段收益抬升趋势检测；
- 带随机扰动的信誉/模式策略。
"""

import collections
import random

STRATEGIES = []


def register(name, tags):
    """把 (name, tags, fn) 收进模块级 STRATEGIES。"""
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------- 纯函数小工具

def _valid(moves):
    """剔除补位/隐藏标记 m，只留真实决策。"""
    return [m for m in moves if m == "c" or m == "b"]


def _clean_rows(hist):
    """把历史按场拆成非空 c/b 序列；整行隐藏的场直接跳过。"""
    rows = []
    for row in hist:
        moves = _valid(row)
        if moves:
            rows.append(moves)
    return rows


def _coop_rate(rows):
    total = 0
    good = 0
    for moves in rows:
        total += len(moves)
        good += moves.count("c")
    if not total:
        return None
    return good / total


def _defect_rate(rows):
    rate = _coop_rate(rows)
    if rate is None:
        return None
    return 1.0 - rate


def _tail_defect_rate(rows, k):
    """各场最后 k 手里 b 的占比。"""
    total = 0
    bad = 0
    for moves in rows:
        tail = moves[-k:]
        total += len(tail)
        bad += tail.count("b")
    if not total:
        return None
    return bad / total


def _tail_any_rate(rows, k):
    """最后 k 手里出现过 b 的场次占比。"""
    if not rows:
        return None
    hits = 0
    for moves in rows:
        if "b" in moves[-k:]:
            hits += 1
    return hits / len(rows)


def _onset_distances(rows, span=5):
    """每场从末尾往前数，最后一次背叛离终局的距离（0 = 末手才翻脸）。"""
    out = []
    for moves in rows:
        tail = moves[-span:]
        distance = -1
        for idx in range(len(tail)):
            if tail[idx] == "b":
                distance = len(tail) - 1 - idx
        if distance >= 0:
            out.append(distance)
    return out


def _paired_rows(hist, other):
    """对齐两份历史，得到每场 [(对手动作, 对手的对手动作)] 序列。"""
    out = []
    for row, other_row in zip(hist, other):
        pairs = []
        for mine, theirs in zip(row, other_row):
            if (mine == "c" or mine == "b") and (theirs == "c" or theirs == "b"):
                pairs.append((mine, theirs))
        if pairs:
            out.append(pairs)
    return out


def _pair_exploit_rate(pairs):
    """对方 c 而它 b 的回合占比：吃白食程度。"""
    total = 0
    hits = 0
    for row in pairs:
        total += len(row)
        for mine, theirs in row:
            if mine == "b" and theirs == "c":
                hits += 1
    if not total:
        return None
    return hits / total


def _pair_tail_harvest(pairs, k):
    """末 k 手里"对方 c、它 b"的回合占比：终局劫掠习惯。"""
    total = 0
    hits = 0
    for row in pairs:
        for mine, theirs in row[-k:]:
            total += 1
            if mine == "b" and theirs == "c":
                hits += 1
    if not total:
        return None
    return hits / total


def _pair_unprovoked_rate(pairs):
    """无端背叛：它 b 而对方上一手（或开场）是 c 的回合占比。"""
    total = 0
    hits = 0
    for row in pairs:
        for idx in range(len(row)):
            total += 1
            mine, theirs = row[idx]
            if mine != "b":
                continue
            if idx == 0 or row[idx - 1][1] == "c":
                hits += 1
    if not total:
        return None
    return hits / total


# ---------------------------------------------------------------- 1. 末手背叛比例

@register("ho_histb_tail1_preempt", ("conditional", "endgame", "history"))
def tail1_preempt(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手历史"末手背叛"的场次比例分档：惯犯提前五手设伏，清白者宽容但末两手收网。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        last_bad = _tail_any_rate(rows, 1)
        if last_bad is not None and last_bad >= 0.34:
            if length >= 20:
                return "b"
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        if length >= 23:
            return "b"
        if opp and opp.count("c") == 0:
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 2. 末两手背叛占比

@register("ho_histb_tail2_bait", ("conditional", "endgame", "history"))
def tail2_bait(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """先看整体合作率：名声烂的直接全程压；只在末两手收网的掠夺者先喂熟再连偷，其余宽容到末手。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        rep = _coop_rate(rows)
        tail = _tail_defect_rate(rows, 2)
        if rep is not None and rep < 0.5:
            return "b"
        if tail is not None and tail > 0.5:
            if length >= 23:
                return "b"
            return "c"
        if length >= 24:
            return "b"
        if len(opp) >= 2 and opp[-1] == "b" and opp[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 3. 开局/末段对比

@register("ho_histb_headtail_split", ("conditional", "endgame", "history"))
def headtail_split(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """比较历史开局三手与末三手的背叛率：只末段翻脸的掠夺者提前四手偷，开局就烂的全程压，清白者末三手收网。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        tail3 = _tail_defect_rate(rows, 3)
        head_total = 0
        head_bad = 0
        for moves in rows:
            head = moves[:3]
            head_total += len(head)
            head_bad += head.count("b")
        head3 = head_bad / head_total if head_total else 0.0
        if tail3 is not None and tail3 > 0.5:
            if head3 >= 0.2:
                return "b"
            if length >= 21:
                return "b"
            return "c"
        if length >= 22:
            return "b"
        if len(opp) >= 2 and opp[-1] == "b" and opp[-2] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 4. 典型终局提前量

@register("ho_histb_onset_median", ("conditional", "endgame", "history"))
def onset_median(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """估计对手每场"第几手开始收网"的提前量，比它早两手下手；无此习惯者末四手收割、只对连续背叛报复。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        onsets = _onset_distances(rows, 5)
        if onsets:
            ordered = sorted(onsets)
            onset = ordered[len(ordered) // 2]
            frontload = onset + 2
            if frontload > 5:
                frontload = 5
        else:
            frontload = 4
        if length >= 25 - frontload:
            return "b"
        if onsets:
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        if len(opp) >= 3 and opp[-3:].count("b") == 3:
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 5. 对齐历史末段劫掠

@register("ho_histb_paired_harvest", ("cooperative", "endgame", "history"))
def paired_harvest(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """对齐两份历史看对手是否在别人输诚时于末段下刀：有这习惯提前四手截胡，否则宽厚为主、迟一手报复。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        pairs = _paired_rows(opp_history_decisions, opp_opp_history_decisions)
        opp = _valid(opp_decisions)
        own_tail = _tail_defect_rate(rows, 2)
        harvest = _pair_tail_harvest(pairs, 2)
        if own_tail is None:
            own_tail = 0.0
        if harvest is None:
            harvest = 0.0
        score = 0.5 * own_tail + 0.5 * harvest
        if score > 0.4:
            if length >= 21:
                return "b"
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        if length >= 24:
            return "b"
        if len(opp) >= 2 and opp[-2] == "b":
            return "b"
        if len(opp) == 1 and opp[0] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 6. 末两手形状

@register("ho_histb_finalpair_read", ("endgame", "history"))
def finalpair_read(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """读对手每场最后两手的形状：惯用 (c,b) 是最后一手收网型，惯用 (b,b) 提前三手设防；其余末两手收割。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        counts = collections.Counter()
        for moves in rows:
            if len(moves) >= 2:
                counts[moves[-2] + moves[-1]] += 1
        total = sum(counts.values())
        cb = counts.get("cb", 0) / total if total else 0.0
        bb = counts.get("bb", 0) / total if total else 0.0
        if cb >= 0.4:
            if length >= 23:
                return "b"
            return "c"
        if bb >= 0.5:
            if length >= 22:
                return "b"
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        if length >= 23:
            return "b"
        if length >= 10:
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 7. 信誉三档门槛

@register("ho_histb_rep_gate3", ("endgame", "history", "mixed"))
def rep_gate3(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """按历史合作率分三档：烂名声全程背叛；名声模糊严格对等、末手收网；好名声末五手收割、第 15 手后见血即压。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        rep = _coop_rate(rows)
        if rep is None:
            rep = 0.6
        opp = _valid(opp_decisions)
        if rep < 0.45:
            return "b"
        if rep >= 0.75:
            if length >= 20:
                return "b"
            if length >= 15 and opp and opp.count("b") >= 1:
                return "b"
            if len(opp) >= 2 and opp[-1] == "b" and opp[-2] == "b":
                return "b"
            return "c"
        if length >= 24:
            return "b"
        if opp and opp[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 8. 近期加权信誉

@register("ho_histb_rep_recency", ("endgame", "history"))
def rep_recency(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """近期对局权重更高的信誉分：越新越算数，名声差提前五手动手，中等提前三手，好名声宽容。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        weight_sum = 0.0
        coop_sum = 0.0
        for idx, moves in enumerate(rows):
            weight = 1.0 + idx
            weight_sum += weight * len(moves)
            coop_sum += weight * moves.count("c")
        rep = coop_sum / weight_sum if weight_sum else 0.6
        if rep < 0.5:
            if length >= 20:
                return "b"
            if opp and opp[-1] == "b":
                return "b"
            return "c"
        if rep < 0.8:
            if length >= 22:
                return "b"
            if len(opp) >= 2 and opp[-1] == "b" and opp[-2] == "b":
                return "b"
            return "c"
        if length >= 24:
            return "b"
        if opp and opp.count("c") > 0:
            return "c"
        if opp:
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 9. 无端背叛 -> 只剥削

@register("ho_histb_unprovoked_milker", ("conditional", "history", "mixed"))
def unprovoked_milker(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """用"无端背叛"与"剥削"两个指数的组合（自身背叛率、对方刚给 c 后立刻下刀、对方 c 而它 b）判档：惯犯只吸奶不硬碰。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        pairs = _paired_rows(opp_history_decisions, opp_opp_history_decisions)
        opp = _valid(opp_decisions)
        rep = _defect_rate(rows)
        unprovoked = _pair_unprovoked_rate(pairs)
        exploit = _pair_exploit_rate(pairs)
        if rep is None:
            rep = 0.0
        if unprovoked is None:
            unprovoked = 0.0
        if exploit is None:
            exploit = 0.0
        index = 0.45 * rep + 0.30 * unprovoked + 0.25 * exploit
        if index > 0.4:
            if opp:
                hard = opp.count("b") / len(opp)
                if hard > 0.5:
                    return "b"
                if opp[-1] == "c":
                    return "b"
            return "c"
        if opp and opp[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 10. 坏名声全压

@register("ho_histb_fame_executioner", ("endgame", "history", "hostile"))
def fame_executioner(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """逐场统计对手高背叛对局：恶名昭著整场全压；略有污点从第 4 手起压；清白者养到末五手再收割。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        tainted = 0
        clean = True
        for moves in rows:
            if len(moves) >= 5 and moves.count("b") / len(moves) >= 0.6:
                tainted += 1
            if moves.count("b"):
                clean = False
        if tainted >= 2:
            return "b"
        if not clean:
            if length >= 4:
                return "b"
            return "c"
        if length >= 20:
            return "b"
        if opp and opp.count("b") * 2 > len(opp):
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 11. 末段变坏趋势

@register("ho_histb_trend_escalate", ("endgame", "history"))
def trend_escalate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """比较开局五手与收官五手的背叛率差：末段才变坏就提前设伏，开局也烂就全程压，未见恶化则中期起严格对等。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        opp = _valid(opp_decisions)
        early_total = 0
        early_bad = 0
        late_total = 0
        late_bad = 0
        for moves in rows:
            head = moves[:5]
            tail = moves[-5:]
            early_total += len(head)
            early_bad += head.count("b")
            late_total += len(tail)
            late_bad += tail.count("b")
        early = early_bad / early_total if early_total else 0.0
        late = late_bad / late_total if late_total else 0.0
        gap = late - early
        if gap > 0.25 or late > 0.6:
            if early >= 0.4:
                return "b"
            frontload = 4 if gap > 0.45 else 2
            if length >= 25 - frontload:
                return "b"
            return "c"
        if length >= 23:
            return "b"
        if length >= 16 and opp and opp[-1] == "b":
            return "b"
        return "c"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 12. 噪声信誉

@register("ho_histb_noisy_reputation", ("history", "noisy"))
def noisy_reputation(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """对手合作率与"对手的对手"合作率合成信任分：极端分值确定动作，中间地带按分数随机试探。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        others = _clean_rows(opp_opp_history_decisions)
        opp = _valid(opp_decisions)
        rep = _coop_rate(rows)
        climate = _coop_rate(others)
        if rep is None:
            rep = 0.5
        if climate is None:
            climate = 0.5
        if not opp:
            return "c"
        trust = 0.6 * rep + 0.4 * climate
        if trust >= 0.9:
            return "c"
        if trust <= 0.15:
            return "b"
        rate = opp.count("b") / len(opp)
        if rate > 0.5:
            return "b"
        if rate == 0.0 and len(opp) >= 5:
            return "c"
        p_coop = 0.25 + 0.5 * trust
        if random.random() < p_coop:
            return "c"
        return "b"
    except Exception:
        return "c"


# ---------------------------------------------------------------- 13. 噪声模式轮换

@register("ho_histb_noisy_modes", ("history", "mixed", "noisy"))
def noisy_modes(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """用对手自身合作率与其对手合作率合成信任分挑模式（宽容/全压/末段收割/对等），模式内部再随机扰动。"""
    try:
        rows = _clean_rows(opp_history_decisions)
        others = _clean_rows(opp_opp_history_decisions)
        opp = _valid(opp_decisions)
        rep = _coop_rate(rows)
        climate = _coop_rate(others)
        if rep is None:
            rep = 0.5
        if climate is None:
            climate = 0.5
        if not opp:
            return "c"
        trust = 0.5 * rep + 0.5 * climate
        if trust >= 0.95:
            if length >= 24:
                return "b"
            if len(opp) >= 2 and opp[-1] == "b" and opp[-2] == "b":
                return "b"
            return "c"
        if trust <= 0.25:
            return "b"
        if rep <= 0.25:
            return "b"
        tail = _tail_defect_rate(rows, 2)
        if tail is not None and tail >= 0.6:
            if length >= 21:
                return "b" if random.random() < 0.9 else "c"
            return "c" if random.random() < 0.75 else "b"
        if length >= 23:
            return "b" if random.random() < 0.85 else "c"
        if opp[-1] == "b":
            return "b" if random.random() < 0.85 else "c"
        return "c" if random.random() < 0.8 else "b"
    except Exception:
        return "c"
