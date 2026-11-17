"""读历史主题 dev 策略 A：历史合作率、被背叛反应、终局倾向、类型分类与软手剥削。"""

import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------- 纯工具函数


def _coop_count(decisions):
    """本场对手序列里 "c" 的个数。"""
    count = 0
    for move in decisions:
        if move == "c":
            count += 1
    return count


def _stats(rows):
    """(合作率, 有效步数)：把 20×25 历史里所有 c/b 摊平统计，m 跳过。"""
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
        return 0.5, 0
    return coop / total, total


def _tail_stats(rows, tail):
    """只看每场历史最后 tail 步的合作率与有效步数。"""
    coop = 0
    total = 0
    for row in rows:
        for move in row[-tail:]:
            if move == "c":
                coop += 1
                total += 1
            elif move == "b":
                total += 1
    if total == 0:
        return 0.5, 0
    return coop / total, total


def _recent_rows(rows, keep):
    """从最新一场往前取 keep 场非空历史（不足则全给）。"""
    picked = []
    for row in reversed(rows):
        has_move = False
        for move in row:
            if move == "c" or move == "b":
                has_move = True
                break
        if has_move:
            picked.append(row)
            if len(picked) >= keep:
                break
    return picked


def _betrayals(my_decisions, opp_decisions):
    """本场我被坑的次数：我合作、对手背叛。"""
    span = min(len(my_decisions), len(opp_decisions))
    hurt = 0
    for i in range(span):
        if my_decisions[i] == "c" and opp_decisions[i] == "b":
            hurt += 1
    return hurt


def _mutual_coops(my_decisions, opp_decisions):
    """本场双方都合作的次数。"""
    span = min(len(my_decisions), len(opp_decisions))
    shared = 0
    for i in range(span):
        if my_decisions[i] == "c" and opp_decisions[i] == "c":
            shared += 1
    return shared


def _streak(decisions, move):
    """序列末尾连续等于 move 的长度。"""
    run = 0
    for past in reversed(decisions):
        if past != move:
            break
        run += 1
    return run


def _exploit_pairs(opp_history_decisions, opp_opp_history_decisions):
    """(剥削率, 有效步数)：对手在对手合作的那一步自己背叛。"""
    hits = 0
    total = 0
    rows = min(len(opp_history_decisions), len(opp_opp_history_decisions))
    for r in range(rows):
        own_row = opp_history_decisions[r]
        foe_row = opp_opp_history_decisions[r]
        steps = min(len(own_row), len(foe_row))
        for s in range(steps):
            own_move = own_row[s]
            foe_move = foe_row[s]
            if own_move in ("c", "b") and foe_move in ("c", "b"):
                total += 1
                if own_move == "b" and foe_move == "c":
                    hits += 1
    if total == 0:
        return 0.5, 0
    return hits / total, total


def _reaction_stats(opp_history_decisions, opp_opp_history_decisions):
    """(后手合作率, 样本数) 两组：对手的对手上一步合作/背叛之后。"""
    coop_after_c = 0
    seen_after_c = 0
    coop_after_b = 0
    seen_after_b = 0
    rows = min(len(opp_history_decisions), len(opp_opp_history_decisions))
    for r in range(rows):
        own_row = opp_history_decisions[r]
        foe_row = opp_opp_history_decisions[r]
        steps = min(len(own_row), len(foe_row))
        for s in range(1, steps):
            reply = own_row[s]
            previous = foe_row[s - 1]
            if reply != "c" and reply != "b":
                continue
            if previous == "c":
                seen_after_c += 1
                if reply == "c":
                    coop_after_c += 1
            elif previous == "b":
                seen_after_b += 1
                if reply == "c":
                    coop_after_b += 1
    rate_after_c = 0.5 if seen_after_c == 0 else coop_after_c / seen_after_c
    rate_after_b = 0.5 if seen_after_b == 0 else coop_after_b / seen_after_b
    return rate_after_c, seen_after_c, rate_after_b, seen_after_b


def _good_games(opp_history_decisions, floor=0.8):
    """历史里合作率达到 floor 的「好局」数量。"""
    good = 0
    for row in opp_history_decisions:
        coop = 0
        total = 0
        for move in row:
            if move == "c":
                coop += 1
                total += 1
            elif move == "b":
                total += 1
        if total >= 10 and coop / total >= floor:
            good += 1
    return good


# ---------------------------------------------------------------- 十个策略


@register("dv_hista_rep_gate", ("conditional", "history"))
def rep_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
             repay, opp_history_decisions, opp_opp_history_decisions):
    """用 20 场历史合作率设信誉门槛，本场露出敌意时一票否决。"""
    try:
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            return "c"
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if played >= 3 and live < 0.25:
            return "b"
        if reputation >= 0.6:
            return "c"
        if reputation >= 0.4 and played >= 5 and live >= 0.65:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("dv_hista_recent_window", ("conditional", "history"))
def recent_window(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """只信最近 5 场历史，记录中性时用本场对位镜像决定。"""
    try:
        window, seen = _stats(_recent_rows(opp_history_decisions, 5))
        played = len(opp_decisions)
        if seen == 0 and played == 0:
            return "c"
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if played >= 3 and live < 0.34:
            return "b"
        if seen == 0:
            return "c"
        if window >= 0.6:
            return "c"
        if window >= 0.4:
            return opp_decisions[-1] if opp_decisions else "c"
        return "b"
    except Exception:
        return "c"


@register("dv_hista_betrayal_tolerance", ("conditional", "cooperative", "history"))
def betrayal_tolerance(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """被背叛时按对手历史信誉折算可忍次数，超支才报复。"""
    try:
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            budget = 1
        else:
            budget = int(round(3.0 * reputation))
        if _betrayals(my_decisions, opp_decisions) > budget:
            return "b"
        if reputation < 0.7 and _streak(opp_decisions, "b") >= 2:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("dv_hista_endgame_scanner", ("endgame", "history"))
def endgame_scanner(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """扫历史每场末段 5 步，判断对手终局是否会偷袭并提前收手。"""
    try:
        if length >= 23:
            return "b"
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            return "c"
        end_reputation, end_seen = _tail_stats(opp_history_decisions, 5)
        if reputation >= 0.6:
            if end_seen > 0 and end_reputation <= 0.4 and length >= 18:
                return "b"
            return "c"
        if reputation <= 0.3:
            return "b"
        if end_reputation < 0.5 and length >= 20:
            return "b"
        return "c" if end_reputation >= 0.5 else "b"
    except Exception:
        return "c"


@register("dv_hista_type_classify", ("conditional", "history"))
def type_classify(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手历史合作率与其对手的合作率把对手分成四类，再分别取打法。"""
    try:
        own, own_seen = _stats(opp_history_decisions)
        foe, foe_seen = _stats(opp_opp_history_decisions)
        if own_seen == 0 and foe_seen == 0:
            return "c"
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if played >= 3 and live < 0.3:
            return "b"
        if own >= 0.7 and foe >= 0.6:
            return "c"
        if own >= 0.7:
            return "c" if live >= 0.5 else "b"
        if own <= 0.35 and foe >= 0.5:
            return "b"
        if own <= 0.35:
            return "b"
        return "c" if live >= 0.55 else "b"
    except Exception:
        return "c"


@register("dv_hista_exploit_soft", ("history", "noisy"))
def exploit_soft(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """对历史上一直合作、其对手也软的软手，按概率下黑手收割。"""
    try:
        own, own_seen = _stats(opp_history_decisions)
        foe, foe_seen = _stats(opp_opp_history_decisions)
        if own_seen == 0:
            return "c"
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if played >= 3 and live < 0.3:
            return "b"
        if own >= 0.75 and foe_seen > 0 and foe >= 0.9:
            return "c" if length < 12 else "b"
        if own >= 0.75 and foe_seen > 0 and foe >= 0.35:
            chance = 0.35 + 0.30 * foe
            return "b" if random.random() < chance else "c"
        if own >= 0.75:
            return "c" if live >= 0.5 else "b"
        if own <= 0.4:
            return "b"
        return "c" if live >= 0.5 else "b"
    except Exception:
        return "c"


@register("dv_hista_fairness_index", ("conditional", "history"))
def fairness_index(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """看对手对「对手上一步」的反应，区分条件合作者与老好人再定宽容度。"""
    try:
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            return "c"
        after_c, n_c, after_b, n_b = _reaction_stats(
            opp_history_decisions, opp_opp_history_decisions)
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if played >= 3 and live < 0.35:
            return "b"
        if reputation <= 0.35:
            return "b"
        retaliates = (n_c >= 10 and n_b >= 10 and after_c - after_b >= 0.25)
        if retaliates:
            return "b" if _streak(opp_decisions, "b") >= 2 else "c"
        if after_c >= 0.8 and after_b >= 0.8:
            return "c"
        return "c" if live >= 0.5 else "b"
    except Exception:
        return "c"


@register("dv_hista_grudge_score", ("conditional", "history"))
def grudge_score(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """把历史剥削率、本场被坑次数与互惠次数合成怨恨分，越线翻脸。"""
    try:
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            return "c"
        exploit, exploit_seen = _exploit_pairs(opp_history_decisions,
                                               opp_opp_history_decisions)
        if exploit_seen == 0:
            exploit = 0.5
        hurt = _betrayals(my_decisions, opp_decisions)
        shared = _mutual_coops(my_decisions, opp_decisions)
        score = (3.0 * (1.0 - reputation) + 2.0 * exploit
                 + 1.2 * hurt - 1.5 * shared)
        if _streak(opp_decisions, "c") >= 3:
            score -= 1.2
        if _streak(opp_decisions, "b") >= 2:
            score += 1.0
        return "b" if score >= 2.0 else "c"
    except Exception:
        return "c"


@register("dv_hista_good_games", ("conditional", "history"))
def good_games(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """数历史上「合作率 ≥ 0.8 的好局」定信任档位，再按本场表现微调。"""
    try:
        good = _good_games(opp_history_decisions)
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if good >= 12:
            if played >= 3 and live < 0.4:
                return "b"
            return "c"
        if good <= 3:
            if played >= 8 and live >= 0.85:
                return "c"
            return "b"
        if played == 0:
            return "c"
        if live >= 0.55:
            return "c"
        if live <= 0.35:
            return "b"
        return opp_decisions[-1] if opp_decisions else "c"
    except Exception:
        return "c"


@register("dv_hista_endgame_telegraph", ("conditional", "endgame", "history"))
def endgame_telegraph(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """看对手在历史末段如何对待其对手，判断它会不会终局偷袭再定收割时点。"""
    try:
        if length >= 24:
            return "b"
        reputation, seen = _stats(opp_history_decisions)
        if seen == 0:
            return "c"
        end_own, end_seen = _tail_stats(opp_history_decisions, 4)
        end_foe, foe_seen = _tail_stats(opp_opp_history_decisions, 4)
        squeezes = (end_seen > 0 and foe_seen > 0
                    and end_own <= 0.45 and end_foe >= 0.55)
        punished = (end_seen > 0 and foe_seen > 0
                    and end_own >= 0.6 and end_foe <= 0.45)
        if squeezes and length >= 17:
            return "b"
        strike_at = 23 if punished else 21
        if length >= strike_at:
            return "b"
        played = len(opp_decisions)
        live = 0.0 if played == 0 else _coop_count(opp_decisions) / played
        if reputation < 0.4:
            return "b"
        if played >= 4 and live < 0.35:
            return "b"
        if reputation >= 0.6 and live >= 0.5:
            return "c"
        return opp_decisions[-1] if opp_decisions else "c"
    except Exception:
        return "c"
