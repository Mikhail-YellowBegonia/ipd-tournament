"""L3：读对手历史的包装器。

每个包装器用检测器库给对手画像，据此改变对待方式，否则交给基础策略。
它们让"我方历史被读取"的代价在模拟中真实出现（ADR 0004）。
"""

from ipd.detectors import profile
from ipd.sim import rules

C, D = "c", "b"


def _profiler():
    """同一场 25 次调用收到的是同一个历史元组对象，画像只算一次。

    这是对手池为了模拟速度做的跨调用缓存；提交物禁止这样做。
    持有元组引用，保证 is 比较不会因对象回收而误命中。
    """
    last = [None, None, None]

    def get(oh, ooh):
        if last[0] is not oh or last[1] is not ooh:
            last[:] = [oh, ooh, profile(oh, ooh)]
        return last[2]
    return get


def reputation_gate(base, threshold=0.25):
    """对手无端背叛或剥削合作者的比率超过阈值时，全程背叛。"""
    get_profile = _profiler()

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        p = get_profile(oh, ooh)
        bad = max(p.rate("unprovoked", prior=0.05, strength=5),
                  p.rate("exploit", prior=0.05, strength=5))
        if bad > threshold:
            return D
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def endgame_preempt(base, margin=1, min_rate=0.5):
    """对手常在终局无端背叛时，比它的典型背叛长度再提前 margin 手开始背叛。"""
    get_profile = _profiler()

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        p = get_profile(oh, ooh)
        if p.trials("endgame") >= 2 and p.rate("endgame", prior=0.0, strength=1) >= min_rate:
            if length >= rules.TURNS - p.endgame_k() - margin:
                return D
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def exploit_soft(base, threshold=0.2):
    """对手被背叛后很少报复时，全程背叛剥削它。"""
    get_profile = _profiler()

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        p = get_profile(oh, ooh)
        if p.trials("retaliate") >= 3 and p.rate("retaliate") < threshold:
            return D
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s


def opening_mirror(base, threshold=0.5):
    """对手常以背叛起手时，自己也以背叛起手，之后交给基础策略。"""
    get_profile = _profiler()

    def s(rs, length, my_id, my, opp_id, opp, repay, oh, ooh):
        if length == 0:
            p = get_profile(oh, ooh)
            return D if p.rate("opening_defect", prior=0.0, strength=1) > threshold else C
        return base(rs, length, my_id, my, opp_id, opp, repay, oh, ooh)
    return s
