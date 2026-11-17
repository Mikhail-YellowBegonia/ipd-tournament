"""独立提交版手工基线：对手条件响应建模 + 剩余回合动态规划。

这是项目手工基线的独立打包版本，只依赖标准库，可直接作为 `strategy.py`
的兜底实现；保留考试九参数 `strategy` 函数名。本文件不导入任何项目模块，
原始参照尺的实现思路与注释保留在仓库的基线模块中。

模型
----
对手模型假设：对手本回合动作只依赖于**上一回合的动作对**
(对手上一手, 我方上一手) 以及当前收益阶段。数据来自对手最近 20 场历史
（`opp_history_decisions` 是它的动作，`opp_opp_history_decisions` 是它当时
对手的动作）以及本场已有动作对；含 "m"（未知/隐藏）的位置跳过。

合作概率用 Beta 先验平滑，并带两个结构假设：
- 互惠先验：我方上一手合作时对手更可能合作，我方上一手背叛时更可能报复
  （RPD 的"善意/报复"经验，见 RECIPROCITY_PRIOR）；
- 层次平滑：先按动作对（不分阶段）估计，再作为阶段内统计的先验中心，
  样本少时向动作对级估计收缩；本场证据的权重高于历史（对手可能按本场行为调整）。
首手只有 Beta(1,1) 均匀先验，并按"先善意"规则决策（见 OPENING_MIN_COOP）。

决策
----
把收益表 `repay` 当作已知，对剩余回合做有限期动态规划：
状态 = 上一回合动作对（4 种），终局价值为 0，逐回合倒推，取当前回合
期望收益最大的动作。这样会自动处理终局诱惑（最后几手背叛的收益 T-P
对比继续合作的 R）以及"先合作换后续合作"的动机。

约束（与提交物一致）：只用标准库；不保存跨调用可变状态；不修改传入数据；
单次决策目标 ≤ 10 ms；整个决策包在 try/except 内，异常时返回合作。

扩展点：issue #1/#3 提到的可选信誉项 λ（在收益中扣除我方本场行为被读取的
代价，近似值后续可由检测器库提供）可加在 `_expectations` 的收益项上；
v1 只做概率模型，不启用。
"""

C, D, M = "c", "b", "m"
TURNS = 25
MISSING_ROW = (M,) * TURNS

# ---- 对手模型超参数 ----
# 首手没有"上一手动作对"，对未知对手的 DP 首手会因 S=-6 的损失而过度悲观；
# 按 RPD 惯例先合作（"善意"），只有历史显示对手惯于背叛起手时才防御。
OPENING_MIN_COOP = 0.5
HISTORY_WEIGHT = 1.0  # 历史中每步观测的权重
CURRENT_WEIGHT = 3.0  # 本场每步观测的权重
OPENING_PRIOR_STRENGTH = 2.0  # 首手 Beta(1,1) 先验强度（虚拟观测数）
# 互惠结构先验：我方上一手合作时，对手下一手更可能合作；我方上一手背叛时更可能报复。
# 这是 RPD 的通用"善意/报复"经验，作为 Beta 先验中心使用（强度 PAIR_PRIOR_STRENGTH）。
RECIPROCITY_PRIOR = {C: 0.8, D: 0.2}
PAIR_PRIOR_STRENGTH = 3.0
PHASE_PRIOR_STRENGTH = 4.0  # 阶段内统计向动作对级估计收缩的强度
DEFAULT_RATE = 0.5  # 无观测时的合作概率（Beta(1,1) 均值）

_INDEX = {C: 0, D: 1}
# 回合 -> 收益阶段编号（0-9、10-19、20-24 三个阶段）。
_PHASE = tuple(0 if t < 10 else (1 if t < 20 else 2) for t in range(TURNS))


def _empty_counts():
    """动作对级 / 阶段级 / 首手 的 [命中, 试验] 计数（加权浮点）。"""
    pair = [[0.0, 0.0] for _ in range(4)]
    phase = [[[0.0, 0.0] for _ in range(4)] for _ in range(3)]
    opening = [0.0, 0.0]
    return pair, phase, opening


def _accumulate(pair, phase, opening, their_moves, facing_moves, weight):
    """把一段对局记入对手模型。

    Args:
        their_moves: 对手（历史主体）自己的动作序列。
        facing_moves: 该对手当时的对手的动作序列（即"我方"在相似局面下会扮演的角色）。
        weight: 这段证据的权重。
    """
    n = min(len(their_moves), len(facing_moves))
    if n == 0:
        return

    first = their_moves[0]
    if first == C or first == D:
        opening[1] += weight
        if first == C:
            opening[0] += weight

    for t in range(1, n):
        theirs_prev = their_moves[t - 1]
        facing_prev = facing_moves[t - 1]
        theirs_now = their_moves[t]
        if theirs_prev == M or facing_prev == M or theirs_now == M:
            continue
        # 状态索引：我方上一手 * 2 + 对手上一手
        s = _INDEX[facing_prev] * 2 + _INDEX[theirs_prev]
        pair[s][1] += weight
        phase[_PHASE[t]][s][1] += weight
        if theirs_now == C:
            pair[s][0] += weight
            phase[_PHASE[t]][s][0] += weight


def _build_model(opp_history_decisions, opp_opp_history_decisions, my_decisions, opp_decisions):
    """汇总历史与本场观测，返回 (pair, phase, opening) 计数。"""
    pair, phase, opening = _empty_counts()
    for theirs, facing in zip(opp_history_decisions, opp_opp_history_decisions):
        if theirs == MISSING_ROW:
            continue  # 整场隐藏或补位
        _accumulate(pair, phase, opening, theirs, facing, HISTORY_WEIGHT)
    # 本场：对手的动作是 their_moves，我方动作是 facing_moves
    _accumulate(pair, phase, opening, opp_decisions, my_decisions, CURRENT_WEIGHT)
    return pair, phase, opening


def _opening_rate(opening):
    """首手合作概率：Beta(1,1) 平滑。"""
    hits, trials = opening
    return (hits + DEFAULT_RATE * OPENING_PRIOR_STRENGTH) / (
        trials + OPENING_PRIOR_STRENGTH
    )


def _pair_rate(pair, s):
    """动作对级合作概率：互惠结构先验 + 观测平滑。"""
    prior = RECIPROCITY_PRIOR[C if s < 2 else D]
    hits, trials = pair[s]
    return (hits + prior * PAIR_PRIOR_STRENGTH) / (trials + PAIR_PRIOR_STRENGTH)


def _rate(pair, phase, phase_idx, s):
    """阶段内合作概率：以动作对级估计为先验中心做层次平滑。"""
    hits, trials = phase[phase_idx][s]
    base = _pair_rate(pair, s)
    return (hits + base * PHASE_PRIOR_STRENGTH) / (trials + PHASE_PRIOR_STRENGTH)


def _payoff_matrix(row):
    """repay 行 (T, S, R, P) -> payoff[我方动作下标][对手动作下标]。"""
    t, s, r, p = row
    return ((r, s), (t, p))


def _expectations(q, pay, value):
    """对手合作概率 q 下，我方合作/背叛的期望收益(含后续价值)。"""
    q0 = q
    q1 = 1.0 - q
    # 下一状态索引 = 我方动作 * 2 + 对手动作
    expect_c = q0 * (pay[0][0] + value[0]) + q1 * (pay[0][1] + value[1])
    expect_d = q0 * (pay[1][0] + value[2]) + q1 * (pay[1][1] + value[3])
    return expect_c, expect_d


def _best_action(q, pay, value):
    """给定对手合作概率 q 与下一步价值 value，返回当前最优动作（平局取合作）。"""
    expect_c, expect_d = _expectations(q, pay, value)
    return C if expect_c >= expect_d else D


def _choose_action(length, my_decisions, opp_decisions, pair, phase, opening, repay):
    """对剩余回合倒推动态规划，返回当前回合动作。"""
    if length == 0:
        # 首手没有"上一手动作对"可用，DP 对未知对手会因 S=-6 过度悲观；
        # 按 RPD 惯例先善意合作，只有历史显示对手常以背叛起手时才防御。
        return C if _opening_rate(opening) >= OPENING_MIN_COOP else D

    # value[s] = 从下一回合起、状态为 s 时的最优期望剩余收益；终局为 0
    value = [0.0, 0.0, 0.0, 0.0]
    for t in range(TURNS - 1, length, -1):
        pay = _payoff_matrix(repay[t])
        phase_idx = _PHASE[t]
        nxt = [0.0, 0.0, 0.0, 0.0]
        for s in range(4):
            expect_c, expect_d = _expectations(_rate(pair, phase, phase_idx, s), pay, value)
            nxt[s] = expect_c if expect_c >= expect_d else expect_d
        value = nxt

    s = _INDEX[my_decisions[-1]] * 2 + _INDEX[opp_decisions[-1]]
    q0 = _rate(pair, phase, _PHASE[length], s)
    return _best_action(q0, _payoff_matrix(repay[length]), value)


def strategy(
    rs,
    length,
    my_id,
    my_decisions,
    opp_id,
    opp_decisions,
    repay,
    opp_history_decisions,
    opp_opp_history_decisions,
):
    """手工基线策略（考试九参数接口）。异常时返回 "c"。"""
    try:
        pair, phase, opening = _build_model(
            opp_history_decisions, opp_opp_history_decisions, my_decisions, opp_decisions
        )
        return _choose_action(length, my_decisions, opp_decisions, pair, phase, opening, repay)
    except Exception:
        return C


__all__ = ["strategy"]
