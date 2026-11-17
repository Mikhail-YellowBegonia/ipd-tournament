"""留出池 basic_a：经典无条件 / 周期性 / 随机策略（独立编写，只用标准库）。

按 CONTRACT.md 第 3 节的主题组织，共 13 个彼此不同的策略：
- 无条件基线：AllC、AllD；
- Random(p)：p = 0.1 / 0.3 / 0.5 / 0.7 / 0.9。五档刻意使用不同的抽样结构
  （阈值比较、加权抽样、随机比特、十面骰、均匀区间），而不是同一行代码换数字；
- 周期模式：cb、ccb、cbb、cccb、bcc、bbc，每档用不同的相位实现方式。

说明：题面列举 2 + 5 + 7 = 14 个候选，但硬性要求恰好 13 个（检查器亦然），
因此保留 AllC/AllD、完整 p 网格和 6 个周期模式，略去 bc —— 它与 cb 属于同一
条周期规则，只在相位上差一手。

这些策略都是"无条件"的：不读本场对手历史，也不使用跨场历史，所以没有
conditional / history / endgame / mixed 标签；随机策略才带 noisy。
"""

import itertools
import random

STRATEGIES = []


def register(name, tags):
    def deco(fn):
        STRATEGIES.append((name, tuple(sorted(tags)), fn))
        return fn
    return deco


# ---------------------------------------------------------------- 无条件基线

@register("ho_basica_all_c", ("basic", "cooperative"))
def all_cooperate(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """无条件合作：每一手都出 c，作为最合作的经典基线。"""
    try:
        return "c"
    except Exception:
        return "c"


@register("ho_basica_all_d", ("basic", "hostile"))
def all_defect(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """无条件背叛：每一手都出 b，作为最敌对的经典基线（也是被报复测试的靶子）。"""
    try:
        return "b"
    except Exception:
        return "c"


# ------------------------------------------------------------ 随机策略 Random(p)

@register("ho_basica_rand_p10", ("basic", "hostile", "noisy"))
def random_p10(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 0.1：每手独立以 0.1 概率出 c（阈值比较），偏敌对。"""
    try:
        draw = random.random()
        if draw < 0.1:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basica_rand_p30", ("basic", "hostile", "noisy"))
def random_p30(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 0.3：每手用加权抽样（3:7）取出 c/b，偏敌对。"""
    try:
        return random.choices(("c", "b"), weights=(3, 7), k=1)[0]
    except Exception:
        return "c"


@register("ho_basica_rand_p50", ("basic", "noisy"))
def random_p50(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 0.5：取一个随机比特当公平硬币，逐手五五开。"""
    try:
        if random.getrandbits(1):
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basica_rand_p70", ("basic", "cooperative", "noisy"))
def random_p70(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 0.7：掷十面骰，点数 < 7 出 c，偏合作。"""
    try:
        die = random.randrange(10)
        if die < 7:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basica_rand_p90", ("basic", "cooperative", "noisy"))
def random_p90(rs, length, my_id, my_decisions, opp_id, opp_decisions,
               repay, opp_history_decisions, opp_opp_history_decisions):
    """随机 0.9：取 [0,1) 均匀数，大于 0.1 出 c，偏合作。"""
    try:
        draw = random.uniform(0.0, 1.0)
        if draw > 0.1:
            return "c"
        return "b"
    except Exception:
        return "c"


# ------------------------------------------------------------- 周期模式 periodic

@register("ho_basica_period_cb", ("basic", "cooperative"))
def periodic_cb(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 2：c,b,c,b…（相位由已完成手数的奇偶决定；25 手为 13c/12b）。"""
    try:
        if int(length) % 2 == 0:
            return "c"
        return "b"
    except Exception:
        return "c"


@register("ho_basica_period_ccb", ("basic", "cooperative"))
def periodic_ccb(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 3：c,c,b 循环，用元组按下标取相位（25 手中 17c/8b）。"""
    try:
        rotation = ("c", "c", "b")
        return rotation[int(length) % 3]
    except Exception:
        return "c"


@register("ho_basica_period_cbb", ("basic", "hostile"))
def periodic_cbb(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 3：c,b,b 循环，用字符串下标取相位（25 手中 16b/9c，偏敌对）。"""
    try:
        pattern = "cbb"
        return pattern[int(length) % 3]
    except Exception:
        return "c"


@register("ho_basica_period_cccb", ("basic", "cooperative"))
def periodic_cccb(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                  repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 4：c,c,c,b 循环，模式串由重复拼接生成（25 手中 19c/6b）。"""
    try:
        pattern = "c" * 3 + "b"
        return pattern[int(length) % 4]
    except Exception:
        return "c"


@register("ho_basica_period_bcc", ("basic", "cooperative"))
def periodic_bcc(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 3：b,c,c 循环（开手背叛后转合作），用 cycle+islice 取当前相位。"""
    try:
        phase = int(length) % 3
        return next(itertools.islice(itertools.cycle("bcc"), phase, phase + 1))
    except Exception:
        return "c"


@register("ho_basica_period_bbc", ("basic", "hostile"))
def periodic_bbc(rs, length, my_id, my_decisions, opp_id, opp_decisions,
                 repay, opp_history_decisions, opp_opp_history_decisions):
    """周期 3：b,b,c 循环（连续两手背叛后回合作），用 divmod 求相位；偏敌对。"""
    try:
        _quotient, phase = divmod(int(length), 3)
        return "bbc"[phase]
    except Exception:
        return "c"
