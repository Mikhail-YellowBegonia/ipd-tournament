"""读历史做类型判断的留出策略池，独立编写，只用标准库。"""

import random

STRATEGIES = []


def register(name, tags):
    """把策略登记进 STRATEGIES。"""
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------- 纯工具函数

def _rate(seq):
    """返回序列的 (合作率, 有效步数)；空序列给中性 0.5。"""
    count = 0
    coops = 0
    for x in seq:
        if x != "m":
            count += 1
            if x == "c":
                coops += 1
    if count == 0:
        return 0.5, 0
    return coops / count, count


def _hist_rate(hist):
    """20×25 历史的整体 (合作率, 有效步数)。"""
    count = 0
    coops = 0
    for row in hist:
        for x in row:
            if x != "m":
                count += 1
                if x == "c":
                    coops += 1
    if count == 0:
        return 0.5, 0
    return coops / count, count


def _recent_rate(seq, k):
    """序列末尾 k 步的 (合作率, 有效步数)。"""
    if not seq or k <= 0:
        return 0.5, 0
    return _rate(seq[-k:])


def _hist_tail_rate(hist, k):
    """历史按时间展开后末尾 k 个有效步的 (合作率, 有效步数)。"""
    flat = _flat_valid(hist)
    return _recent_rate(flat, k)


def _last_match_rate(hist):
    """历史最后一整场的 (合作率, 有效步数)。"""
    if not hist:
        return 0.5, 0
    return _rate(hist[-1])


def _flat_valid(hist):
    """把 20×25 历史按时间压成一维有效步列表。"""
    flat = []
    for row in hist:
        for x in row:
            if x != "m":
                flat.append(x)
    return flat


def _trailing_run(seq, symbol):
    """序列末尾连续 symbol 的长度。"""
    run = 0
    for x in reversed(seq):
        if x == symbol:
            run += 1
        else:
            break
    return run


def _longest_run(seq, symbol):
    """序列里最长 symbol 连击长度。"""
    best = 0
    run = 0
    for x in seq:
        if x == symbol:
            run += 1
            if run > best:
                best = run
        else:
            run = 0
    return best


def _exploited_count(mine, theirs):
    """本场里我合作、对手背叛的次数。"""
    count = 0
    limit = min(len(mine), len(theirs))
    for i in range(limit):
        if mine[i] == "c" and theirs[i] == "b":
            count += 1
    return count


def _total_scores(mine, theirs, repay):
    """按已打回合结算双方总收益 (我, 对手)。"""
    my_total = 0
    their_total = 0
    limit = min(len(mine), len(theirs), len(repay))
    for i in range(limit):
        T, S, R, P = repay[i]
        a = mine[i]
        b = theirs[i]
        if a == "c":
            my_total += R if b == "c" else S
        else:
            my_total += T if b == "c" else P
        if b == "c":
            their_total += R if a == "c" else S
        else:
            their_total += T if a == "c" else P
    return my_total, their_total


def _first_move_betrayal(hist):
    """历史各场首个可见动作的背叛率。"""
    matches = 0
    betrays = 0
    for row in hist:
        for x in row:
            if x != "m":
                matches += 1
                if x == "b":
                    betrays += 1
                break
    if matches == 0:
        return 0.0, 0
    return betrays / matches, matches


def _reciprocity(hist, oohist):
    """对手在其对手先合作之后的镜像回应率。"""
    eligible = 0
    hits = 0
    for row, orow in zip(hist, oohist):
        for i in range(1, len(row)):
            if orow[i - 1] == "c" and row[i] in ("c", "b"):
                eligible += 1
                if row[i] == "c":
                    hits += 1
    if eligible == 0:
        return 0.0, 0
    return hits / eligible, eligible


def _exploit_share(hist, oohist):
    """对手背叛中“紧接对方合作”的比例。"""
    defections = 0
    exploited = 0
    for row, orow in zip(hist, oohist):
        for i in range(len(row)):
            if row[i] == "b":
                defections += 1
                if i >= 1 and orow[i - 1] == "c":
                    exploited += 1
    if defections == 0:
        return 0.0, 0
    return exploited / defections, defections


def _hidden_share(hist):
    """历史里被隐藏/补位 (m) 的格子占比。"""
    cells = 0
    hidden = 0
    for row in hist:
        for x in row:
            cells += 1
            if x == "m":
                hidden += 1
    if cells == 0:
        return 0.0, 0
    return hidden / cells, cells


def _endgame_betrayal(hist, k):
    """各场最后 k 个有效步的平均背叛率。"""
    rates = []
    for row in hist:
        tail = []
        for x in row:
            if x != "m":
                tail.append(x)
        if not tail:
            continue
        tail = tail[-k:]
        betrays = 0
        for x in tail:
            if x == "b":
                betrays += 1
        rates.append(betrays / len(tail))
    if not rates:
        return 0.5
    return sum(rates) / len(rates)


# ---------------------------------------------------------------- 策略本体

@register("ho_hista_bayes_type", ("history", "conditional"))
def bayes_type(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """以历史合作率为先验、本场近 8 手为证据，算对手是合作型的后验概率。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        last, last_n = _last_match_rate(opp_history_decisions)
        prior = 0.5 * h + 0.5 * (last if last_n else h)
        prior = min(0.98, max(0.02, prior))
        window = opp_decisions[-8:]
        coops = 0
        for x in window:
            if x == "c":
                coops += 1
        defects = len(window) - coops
        like_c = (0.85 ** coops) * (0.15 ** defects)
        like_d = (0.15 ** coops) * (0.85 ** defects)
        post = (prior * like_c) / (prior * like_c + (1.0 - prior) * like_d + 1e-12)
        return "c" if post >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_hista_ema_trust", ("history", "conditional"))
def ema_trust(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """历史做 0.9 衰减平滑得底色，与本场近 6 手合成信任分，并带迟滞带。"""
    try:
        ema = 0.5
        seen = 0
        for row in opp_history_decisions:
            for x in row:
                if x == "m":
                    continue
                ema = 0.9 * ema + (0.1 if x == "c" else 0.0)
                seen += 1
        if seen == 0:
            ema = 0.5
        local, _ = _recent_rate(opp_decisions, 6)
        trust = 0.4 * ema + 0.6 * local
        if trust >= 0.58:
            return "c"
        if trust <= 0.45:
            return "b"
        if length >= 1 and opp_decisions[-1] == "c":
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_hista_firstmove_audit", ("history", "conditional"))
def firstmove_audit(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """用历史首手背叛率做背景审查：审查不过就默认敌对，通过则走宽容互惠。"""
    try:
        fm, fm_n = _first_move_betrayal(opp_history_decisions)
        recent5, _ = _recent_rate(opp_decisions, 5)
        if fm >= 0.4:
            if length >= 5 and recent5 >= 0.999:
                return "c"
            return "b"
        if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        if _exploited_count(my_decisions, opp_decisions) >= 3:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_hista_hidden_discount", ("history", "conditional"))
def hidden_discount(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """隐藏比例给历史合作率打折，最近一场大量被隐藏则额外降信任。"""
    try:
        hidden, _ = _hidden_share(opp_history_decisions)
        visible, visible_n = _hist_rate(opp_history_decisions)
        trust_hist = visible * (1.0 - 0.5 * hidden) if visible_n else 0.5
        last_hidden, _ = _hidden_share(opp_history_decisions[-1:])
        local, _ = _recent_rate(opp_decisions, 6)
        score = 0.35 * trust_hist + 0.55 * local + 0.10 * (1.0 - hidden)
        if hidden >= 0.35 and local < 0.8:
            return "b"
        if last_hidden >= 0.6 and local < 0.9:
            return "b"
        return "c" if score >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_hista_exploit_index", ("history", "conditional"))
def exploit_index(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """统计对手趁其对手合作时背叛的比例，再叠加本场我被剥削的比率决定惩罚。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        share, defections = _exploit_share(
            opp_history_decisions, opp_opp_history_decisions)
        local, _ = _recent_rate(opp_decisions, 6)
        exploited_now = _exploited_count(my_decisions, opp_decisions)
        my_rate = exploited_now / length if length else 0.0
        hostility = 0.45 * (1.0 - h) + 0.3 * share + 0.35 * my_rate
        if defections == 0 and h >= 0.9 and exploited_now <= 1:
            return "c"
        if hostility >= 0.5:
            return "b"
        if length >= 3 and local <= 0.34:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_hista_endgame_window", ("history", "conditional", "endgame"))
def endgame_window(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """按对手历史终局背叛率决定提前收网，最后 5 手一律背叛。"""
    try:
        if length >= 20:
            return "b"
        h, _ = _hist_rate(opp_history_decisions)
        eg = _endgame_betrayal(opp_history_decisions, 3)
        local, _ = _recent_rate(opp_decisions, 6)
        trust = 0.45 * h + 0.55 * local
        if eg >= 0.5 and rs >= 5:
            return "b"
        if eg >= 0.75 and trust < 0.8:
            return "b"
        return "c" if trust >= 0.5 else "b"
    except Exception:
        return "c"


@register("ho_hista_noisy_band", ("history", "conditional", "noisy"))
def noisy_band(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """信任分两端给确定动作，灰区按信任分概率随机化合作。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        peer, _ = _hist_rate(opp_opp_history_decisions)
        local, _ = _rate(opp_decisions)
        if length >= 1 and local >= 0.999:
            return "c"
        if length >= 1 and local <= 0.001:
            return "b"
        trust = 0.4 * h + 0.3 * peer + 0.3 * local
        if trust >= 0.8:
            return "c"
        if trust <= 0.3:
            return "b"
        return "c" if random.random() < trust else "b"
    except Exception:
        return "c"


@register("ho_hista_mode_switch", ("history", "conditional", "mixed"))
def mode_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """按历史类型在宽容互惠、冷酷报复、比分追赶三种模式之间切换。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        if h >= 0.65:
            exploited = _exploited_count(my_decisions, opp_decisions)
            if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
                return "b"
            if exploited >= 3:
                return "b"
            return "c"
        if h <= 0.3:
            if "b" in opp_decisions:
                return "b"
            return "c"
        mine, theirs = _total_scores(my_decisions, opp_decisions, repay)
        if length >= 4 and mine + 2 < theirs:
            return "b"
        if length >= 2 and opp_decisions[-1] == "b" and opp_decisions[-2] == "b":
            return "b"
        return "c" if mine >= theirs else "b"
    except Exception:
        return "c"


@register("ho_hista_peer_rank", ("history", "conditional"))
def peer_rank(rs, length, my_id, my_decisions, opp_id, opp_decisions,
              repay, opp_history_decisions, opp_opp_history_decisions):
    """把对手合作率与“别人对它的合作率”对比做同行排名，落后同行则加重怀疑。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        peer, _ = _hist_rate(opp_opp_history_decisions)
        gap = h - peer
        local, _ = _recent_rate(opp_decisions, 6)
        suspicion = (0.5 * (1.0 - h) + 0.5 * (1.0 - local)
                     + 0.3 * max(0.0, -gap))
        if gap <= -0.3:
            return "b"
        if suspicion >= 0.5:
            return "b"
        if length >= 6 and local >= 0.9 and h >= 0.4:
            return "c"
        return "c" if suspicion < 0.42 else "b"
    except Exception:
        return "c"


@register("ho_hista_streak_policy", ("history", "conditional", "cooperative"))
def streak_policy(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """看历史与本场的背叛连击：连击短就宽恕，连击长且历史差就报复。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        flat = _flat_valid(opp_history_decisions)
        longest = _longest_run(flat, "b")
        hist_run = _trailing_run(flat, "b")
        run = _trailing_run(opp_decisions, "b")
        if h >= 0.6:
            if run >= 2:
                return "b"
            if hist_run >= 8 and h < 0.75:
                return "b"
            return "c"
        if run >= 1:
            return "b"
        if longest >= 10 and h < 0.35:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_hista_provocation_budget", ("history", "conditional", "cooperative"))
def provocation_budget(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                       repay, opp_history_decisions, opp_opp_history_decisions):
    """用历史信任度换算“忍耐额度”，本场被剥削次数超过额度才转入惩罚。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        budget = int(3.0 * h + 0.5)
        used = _exploited_count(my_decisions, opp_decisions)
        local, _ = _recent_rate(opp_decisions, 4)
        if length >= 1 and opp_decisions[-1] == "b" and h < 0.45:
            return "b"
        if used > budget:
            return "b"
        if local <= 0.25 and h < 0.6:
            return "b"
        return "c"
    except Exception:
        return "c"


@register("ho_hista_phase_switch", ("history", "conditional", "endgame"))
def phase_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """按 rs 阶段换权重：开局重历史、中局看同行、收官重本场，最后 3 手收网背叛。"""
    try:
        if length >= 22:
            return "b"
        h, _ = _hist_rate(opp_history_decisions)
        peer, _ = _hist_rate(opp_opp_history_decisions)
        local, _ = _recent_rate(opp_decisions, 5)
        if rs <= 1:
            score = 0.7 * h + 0.3 * local
            bar = 0.45
        elif rs <= 5:
            score = 0.4 * h + 0.45 * local + 0.15 * peer
            bar = 0.6
        else:
            score = 0.25 * h + 0.75 * local
            bar = 0.6
        if length >= 18 and local < 0.5:
            return "b"
        return "c" if score >= bar else "b"
    except Exception:
        return "c"


@register("ho_hista_shadow_forecast", ("history", "conditional"))
def shadow_forecast(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """估计对手镜像回应率，比较现在背叛的收益与断送未来合作的代价。"""
    try:
        h, _ = _hist_rate(opp_history_decisions)
        recip, recip_n = _reciprocity(opp_history_decisions,
                                      opp_opp_history_decisions)
        local3, local3_n = _recent_rate(opp_decisions, 3)
        if local3_n >= 3 and local3 >= 0.999:
            return "c"
        if local3_n >= 3 and local3 <= 0.001:
            return "b"
        local6, _ = _recent_rate(opp_decisions, 6)
        p_now = 0.5 * h + 0.5 * local6
        index = min(length, len(repay) - 1)
        T, S, R, P = repay[index]
        remaining = 25 - length
        if remaining <= 0:
            return "b"
        gain_now = p_now * (T - R) + (1.0 - p_now) * (P - S)
        cost_future = recip * (remaining - 1) * max(0.0, R - P)
        if recip >= 0.55 and cost_future > gain_now:
            return "c"
        if p_now >= 0.8 and length < 20:
            return "c"
        return "b"
    except Exception:
        return "c"
