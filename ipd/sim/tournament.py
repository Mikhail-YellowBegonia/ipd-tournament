import itertools
import random
import time
from collections import deque
from dataclasses import dataclass, field

from ipd.sim import rules
from ipd.sim.config import TournamentConfig


@dataclass
class MatchRecord:
    rs: int
    a: int
    b: int
    moves_a: tuple
    moves_b: tuple
    score_a: int
    score_b: int


@dataclass
class TournamentResult:
    names: list
    scores: list
    # pair_scores[i][j]：i 对阵 j 的累计得分
    pair_scores: list
    faults: list
    matches: list = field(default_factory=list)

    def ranking(self):
        """按总分降序返回 (名次, 名字, 总分)；同分同名次。"""
        order = sorted(range(len(self.names)), key=lambda i: -self.scores[i])
        out = []
        for pos, i in enumerate(order):
            if pos and self.scores[i] == self.scores[order[pos - 1]]:
                rank = out[-1][0]
            else:
                rank = pos + 1
            out.append((rank, self.names[i], self.scores[i]))
        return out


class _Player:
    def __init__(self, agent, player_id, history_matches):
        self.agent = agent
        self.id = player_id
        # 每条记录：(自己的 25 步, 对手的 25 步)，最旧在前
        self.history = deque(maxlen=history_matches)
        self.faults = 0


def _history_view(records, cfg, rng):
    """把记录转成考试接口的两份 20 × 25 元组，按配置补位与隐藏。"""
    pad = cfg.history_matches - len(records)
    own_rows = [rules.MISSING_ROW] * pad
    opp_rows = [rules.MISSING_ROW] * pad
    for own, opp in records:
        if cfg.hide_prob == 0.0:
            own_rows.append(own)
            opp_rows.append(opp)
        elif cfg.hide_scope == "match":
            hidden = rng.random() < cfg.hide_prob
            own_rows.append(rules.MISSING_ROW if hidden else own)
            opp_rows.append(rules.MISSING_ROW if hidden else opp)
        else:
            mask = [rng.random() < cfg.hide_prob for _ in range(rules.TURNS)]
            own_rows.append(tuple(rules.MISSING if h else x for h, x in zip(mask, own)))
            opp_rows.append(tuple(rules.MISSING if h else x for h, x in zip(mask, opp)))
    return tuple(own_rows), tuple(opp_rows)


def _decide(player, cfg, args):
    """调用策略；异常、超时、非法返回值都按合作处理并计一次故障。"""
    if player.faults >= cfg.fault_limit:
        return rules.COOPERATE
    start = time.perf_counter()
    try:
        move = player.agent.fn(*args)
    except Exception:
        move = None
    elapsed = time.perf_counter() - start
    if move not in rules.LEGAL_MOVES or (cfg.timeout is not None and elapsed > cfg.timeout):
        player.faults += 1
        return rules.COOPERATE
    return move


def play_match(pa, pb, rs, cfg, rng):
    """进行一场 25 回合对局，返回双方动作序列与得分。不写入历史。"""
    # 历史视图在开场时生成一次，整场不变
    hist_b_own, hist_b_opp = _history_view(pb.history, cfg, rng)
    hist_a_own, hist_a_opp = _history_view(pa.history, cfg, rng)
    moves_a, moves_b = [], []
    score_a = score_b = 0
    for length in range(rules.TURNS):
        da, db = tuple(moves_a), tuple(moves_b)
        ma = _decide(pa, cfg, (rs, length, pa.id, da, pb.id, db, rules.REPAY, hist_b_own, hist_b_opp))
        mb = _decide(pb, cfg, (rs, length, pb.id, db, pa.id, da, rules.REPAY, hist_a_own, hist_a_opp))
        row = rules.REPAY[length]
        score_a += rules.payoff(ma, mb, row)
        score_b += rules.payoff(mb, ma, row)
        moves_a.append(ma)
        moves_b.append(mb)
    return tuple(moves_a), tuple(moves_b), score_a, score_b


def run_tournament(agents, seed=0, cfg=None, keep_matches=False):
    """按课程规则进行一次完整锦标赛（一个种子）。

    策略内部使用的全局 random 会在开始时以 seed 重置；
    赛程与历史隐藏使用独立的随机数生成器，两者互不干扰。
    """
    cfg = cfg or TournamentConfig()
    names = [a.name for a in agents]
    if len(set(names)) != len(names):
        raise ValueError("agent names must be unique")

    n = len(agents)
    random.seed(seed)
    rng = random.Random(f"sim-{seed}")
    players = [_Player(a, i + 1, cfg.history_matches) for i, a in enumerate(agents)]
    scores = [0] * n
    pair_scores = [[0] * n for _ in range(n)]
    matches = []

    for rs in range(cfg.rounds):
        pairs = list(itertools.combinations(range(n), 2))
        if cfg.shuffle_schedule:
            rng.shuffle(pairs)
            pairs = [(j, i) if rng.random() < 0.5 else (i, j) for i, j in pairs]
        pending = []
        for i, j in pairs:
            pa, pb = players[i], players[j]
            moves_a, moves_b, sa, sb = play_match(pa, pb, rs, cfg, rng)
            scores[i] += sa
            scores[j] += sb
            pair_scores[i][j] += sa
            pair_scores[j][i] += sb
            update = [(pa, (moves_a, moves_b)), (pb, (moves_b, moves_a))]
            if cfg.history_update == "match":
                for p, rec in update:
                    p.history.append(rec)
            else:
                pending.extend(update)
            if keep_matches:
                matches.append(MatchRecord(rs, i, j, moves_a, moves_b, sa, sb))
        for p, rec in pending:
            p.history.append(rec)

    return TournamentResult(
        names=names,
        scores=scores,
        pair_scores=pair_scores,
        faults=[p.faults for p in players],
        matches=matches,
    )
