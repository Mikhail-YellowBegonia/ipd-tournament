"""比赛规则常量，来源：docs/reference/course_rules.md。"""

COOPERATE = "c"
DEFECT = "b"
MISSING = "m"
LEGAL_MOVES = (COOPERATE, DEFECT)

TURNS = 25
ROUNDS = 8
HISTORY_MATCHES = 20

# (起始 length, 结束 length（不含）, (T, S, R, P))
PHASES = (
    (0, 10, (4, -6, 2, -1)),
    (10, 20, (5, -7, 3, -2)),
    (20, 25, (6, -8, 4, -3)),
)


def build_repay():
    """返回 25 × 4 的收益表元组，repay[length] = (T, S, R, P)。"""
    rows = []
    for start, end, row in PHASES:
        rows.extend([row] * (end - start))
    return tuple(rows)


REPAY = build_repay()

MISSING_ROW = (MISSING,) * TURNS


def phase_of(length):
    """返回回合索引所在的阶段编号 0/1/2。"""
    for i, (start, end, _) in enumerate(PHASES):
        if start <= length < end:
            return i
    raise ValueError(f"length out of range: {length}")


def payoff(me, opp, row):
    """单回合我方得分。row = (T, S, R, P)。"""
    t, s, r, p = row
    if me == COOPERATE:
        return r if opp == COOPERATE else s
    return t if opp == COOPERATE else p
