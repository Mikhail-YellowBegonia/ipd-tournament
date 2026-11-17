"""多模式混合 + 噪声主题 dev 策略，独立编写，只用标准库。

主题：模式选择、概率门控、周期+条件混合、随机化报复、软硬切换。
每个策略都是九个参数的纯函数式实现，不保存跨调用状态。
"""

import itertools
import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------- 纯工具函数


def _flatten(rows):
    """把 20 场 × 25 步的历史展平成一维元组。"""
    return tuple(itertools.chain.from_iterable(rows))


def _rate(moves):
    """合作率，忽略补位/隐藏的 "m"；没有有效数据时返回 None。"""
    total = 0
    good = 0
    for mv in moves:
        if mv == "c":
            total += 1
            good += 1
        elif mv == "b":
            total += 1
    if total == 0:
        return None
    return good / total


def _last(moves):
    """最后一个可见动作（c/b），没有则返回 None。"""
    for mv in reversed(moves):
        if mv == "c" or mv == "b":
            return mv
    return None


def _streak(moves):
    """末尾连续背叛的步数。"""
    run = 0
    for mv in reversed(moves):
        if mv == "b":
            run += 1
        elif mv == "c":
            break
    return run


def _exploit_share(rows, other_rows):
    """对手占便宜的比例：当对手的对手合作时，对手自己选择背叛的频率。"""
    hit = 0
    total = 0
    for row, other in zip(rows, other_rows):
        for mv, omv in zip(row, other):
            if mv == "c" or mv == "b":
                if omv == "c" or omv == "b":
                    total += 1
                    if omv == "c" and mv == "b":
                        hit += 1
    if total == 0:
        return None
    return hit / total


def _payoff(pair, mine, theirs):
    """单回合得分，pair = (T, S, R, P)。"""
    t, s, r, p = pair
    if mine == "c":
        return r if theirs == "c" else s
    return t if theirs == "c" else p


def _total(repay, mine, theirs):
    """按 repay 表累计 mine 方的总得分。"""
    total = 0
    for idx, (a, b) in enumerate(zip(mine, theirs)):
        total += _payoff(repay[idx], a, b)
    return total


# ---------------------------------------------------------------- 策略


@register("dv_mixa_three_mode_gate", ("mixed", "conditional"))
def three_mode_gate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """按双方组号与比赛阶段在宽容、镜像、记仇三模式间确定性切换。"""
    try:
        mode = (my_id * 5 + opp_id * 3 + length // 4) % 3
        recent = opp_decisions[-2:]
        last = _last(opp_decisions)
        if mode == 0:
            # 宽容模式：连续两手背叛才还手
            return "b" if recent.count("b") >= 2 else "c"
        if mode == 1:
            # 镜像模式：模仿对手上一手
            return "b" if last == "b" else "c"
        # 记仇模式：本场只要被背叛过一次就不再合作
        return "b" if "b" in opp_decisions else "c"
    except Exception:
        return "c"


@register("dv_mixa_prob_gate_mix", ("mixed", "noisy", "conditional"))
def prob_gate_mix(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """用对手合作率做概率门控，在剥削、镜像、宽恕三模式间随机抽签。"""
    try:
        rate = _rate(opp_decisions)
        if rate is None:
            return "c"
        if rate >= 0.9:
            return "c"
        if rate <= 0.12:
            return "b"
        roll = random.random()
        if roll < 0.18:
            # 剥削模式：对手处在可压榨区间就一直背叛
            return "b"
        if roll < 0.55:
            # 镜像模式
            return "b" if _last(opp_decisions) == "b" else "c"
        # 宽恕模式：容忍一手背叛
        return "b" if opp_decisions[-2:].count("b") >= 2 else "c"
    except Exception:
        return "c"


@register("dv_mixa_cycle_override", ("mixed", "conditional"))
def cycle_override(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                   repay, opp_history_decisions, opp_opp_history_decisions):
    """节律模式（三手一循环）与反应模式（近两手多数决）交替，再被合作率覆盖。"""
    try:
        rate = _rate(opp_decisions)
        if rate is not None and rate < 0.3:
            return "b"
        if rate is not None and rate >= 0.7:
            return "c"
        if (length // 5 + my_id + opp_id) % 2 == 0:
            step = (length + my_id + opp_id) % 3
            return "b" if step == 2 else "c"
        window = opp_decisions[-2:]
        return "b" if window.count("b") >= 1 else "c"
    except Exception:
        return "c"


@register("dv_mixa_random_retaliator", ("mixed", "noisy", "conditional"))
def random_retaliator(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                      repay, opp_history_decisions, opp_opp_history_decisions):
    """随机化报复：硬报复、宽恕、掷骰报复三态混合，报复倾向随连败恶化。"""
    try:
        if length == 0:
            return "c"
        rate = _rate(opp_decisions)
        if rate is not None and rate >= 0.9:
            return "c"
        if rate is not None and rate <= 0.12:
            return "b"
        streak = _streak(opp_decisions)
        roll = random.random()
        if streak >= 2:
            # 连续背叛越长，硬报复概率越高
            return "b" if roll < min(0.92, 0.55 + 0.12 * streak) else "c"
        if streak == 1:
            if roll < 0.55:
                return "b"
            if roll < 0.85:
                return "c"
            return "b" if random.random() < 0.5 else "c"
        # 对手刚合作：多数合作，偶尔试探
        return "b" if roll < 0.12 else "c"
    except Exception:
        return "c"


@register("dv_mixa_soft_hard_switch", ("mixed", "conditional", "history"))
def soft_hard_switch(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """读对手历史合作率与占便宜倾向，在软、硬两种模式间切换。"""
    try:
        long_rate = _rate(_flatten(opp_history_decisions))
        exploit = _exploit_share(opp_history_decisions, opp_opp_history_decisions)
        in_rate = _rate(opp_decisions)
        if length == 0:
            if long_rate is not None and long_rate < 0.35:
                return "b"
            return "c"
        hard = False
        if long_rate is not None and long_rate < 0.55:
            hard = True
        if exploit is not None and exploit > 0.18:
            hard = True
        if in_rate is not None and in_rate < 0.35:
            hard = True
        if hard:
            # 硬模式：只有对手本场明显软化才局部松手
            if in_rate is not None and in_rate >= 0.75:
                return "c"
            if _last(opp_decisions) == "b":
                return "b"
            return "c" if length % 4 == 3 else "b"
        # 软模式：合作优先，容忍一手背叛
        return "b" if opp_decisions[-2:].count("b") >= 2 else "c"
    except Exception:
        return "c"


@register("dv_mixa_endgame_noise", ("mixed", "noisy", "endgame"))
def endgame_noise(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """开局偏合作、中盘三模式随机混合、终局背叛概率递增直到末手必叛。"""
    try:
        if length >= 24:
            return "b"
        if length >= 20:
            p_defect = 0.45 + 0.12 * (length - 20)
            return "b" if random.random() < p_defect else "c"
        if length < 3:
            return "c" if random.random() < 0.8 else "b"
        mode_roll = random.random()
        if mode_roll < 0.35:
            move = "b" if _last(opp_decisions) == "b" else "c"
        elif mode_roll < 0.70:
            move = "b" if "b" in opp_decisions else "c"
        else:
            move = "b" if random.random() < 0.3 else "c"
        flip = 0.10 + 0.05 * (length // 5)
        if random.random() < flip:
            move = "c" if move == "b" else "b"
        return move
    except Exception:
        return "c"


@register("dv_mixa_gated_pavlov", ("mixed", "noisy", "conditional"))
def gated_pavlov(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """赢留输变的 Pavlov 与记仇、宽容三模式随机门控，附带探索噪声。"""
    try:
        rate = _rate(opp_decisions)
        if rate is None:
            return "c"
        if rate >= 0.92:
            return "c"
        if rate <= 0.10:
            return "b"
        n = len(opp_decisions)
        pair = repay[n - 1]
        mine = my_decisions[n - 1]
        gained = _payoff(pair, mine, opp_decisions[n - 1])
        won = gained >= pair[2]
        roll = random.random()
        if roll < 0.60:
            # Pavlov：赢了重复，输了换招
            move = mine if won else ("b" if mine == "c" else "c")
        elif roll < 0.85:
            # 记仇模式
            move = "b" if "b" in opp_decisions else "c"
        else:
            # 宽容模式：只有连续两手背叛才还手
            move = "b" if opp_decisions[-2:].count("b") >= 2 else "c"
        if random.random() < 0.10:
            move = "c" if move == "b" else "b"
        return move
    except Exception:
        return "c"


@register("dv_mixa_periodic_jitter", ("mixed", "noisy"))
def periodic_jitter(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                    repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 3 或 4 的节律与反向节律、随机抽签三模式混合，再叠加抖动。"""
    try:
        period = 3 if (my_id + opp_id) % 2 == 0 else 4
        offset = (my_id * 7 + opp_id * 5) % period
        step = (length + offset) % period
        roll = random.random()
        if roll < 0.50:
            base = "b" if step == period - 1 else "c"
        elif roll < 0.80:
            base = "c" if step == period - 1 else "b"
        else:
            base = "c" if random.random() < 0.65 else "b"
        jitter = 0.18 + 0.05 * (length // 5)
        if random.random() < jitter:
            base = "c" if base == "b" else "b"
        return base
    except Exception:
        return "c"


@register("dv_mixa_history_exploit_mix", ("mixed", "history", "conditional"))
def history_exploit_mix(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                        repay, opp_history_decisions, opp_opp_history_decisions):
    """读历史合作率与占便宜倾向选慷慨/中间/戒备模式，并按本场分差微调。"""
    try:
        long_rate = _rate(_flatten(opp_history_decisions))
        exploit = _exploit_share(opp_history_decisions, opp_opp_history_decisions)
        in_rate = _rate(opp_decisions)
        last = _last(opp_decisions)
        if long_rate is None:
            return "b" if last == "b" else "c"
        recent_two = opp_decisions[-2:]
        if long_rate >= 0.68 and (exploit is None or exploit <= 0.12):
            # 慷慨模式：先手示好，只有明显被压榨才还手
            if in_rate is not None and in_rate <= 0.2:
                return "b"
            return "b" if recent_two.count("b") >= 2 else "c"
        if long_rate <= 0.45 or (exploit is not None and exploit >= 0.22):
            # 戒备模式：以牙还牙，分差落后时加码
            if length == 0:
                return "b" if long_rate <= 0.3 else "c"
            if last == "b":
                return "b"
            mine_score = _total(repay, my_decisions, opp_decisions)
            theirs = _total(repay, opp_decisions, my_decisions)
            if theirs - mine_score >= 4:
                return "b" if length % 2 == 0 else "c"
            return "b" if in_rate is not None and in_rate < 0.55 else "c"
        # 中间模式：三步窗多数决
        if length == 0:
            return "c"
        if opp_decisions[-3:].count("b") >= 2:
            return "b"
        return "b" if in_rate is not None and in_rate < 0.5 else "c"
    except Exception:
        return "c"


@register("dv_mixa_score_driven_mix", ("mixed", "noisy", "endgame"))
def score_driven_mix(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                     repay, opp_history_decisions, opp_opp_history_decisions):
    """按本场分差切换落后/领先/均势模式，均势时三子模式抽签，末两手必叛。"""
    try:
        if length >= 23:
            return "b"
        if length == 0:
            return "c" if random.random() < 0.7 else "b"
        mine_score = _total(repay, my_decisions, opp_decisions)
        theirs = _total(repay, opp_decisions, my_decisions)
        delta = mine_score - theirs
        last = _last(opp_decisions)
        roll = random.random()
        if delta <= -4:
            # 落后模式：惩罚加码，落后越多越硬
            p_defect = 0.45 + 0.05 * (length // 5)
            if last == "b":
                p_defect += 0.25
        elif delta >= 4:
            # 领先模式：锁定分差，尽量合作
            p_defect = 0.10 if last != "b" else 0.35
        else:
            # 均势模式：三种子模式抽签
            if roll < 0.40:
                p_defect = 0.60 if last == "b" else 0.05
            elif roll < 0.70:
                p_defect = 0.25
            else:
                p_defect = 0.85
        if random.random() < 0.10:
            p_defect = 1.0 - p_defect
        return "b" if random.random() < p_defect else "c"
    except Exception:
        return "c"
