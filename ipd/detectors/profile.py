"""从历史记录中统计可被他人识别的行为模式。

"主体"指历史的主人：own 是主体的动作，opp 是主体当时对手的动作。
读对手历史时传入 (opp_history_decisions, opp_opp_history_decisions)；
给我方本场打分（信誉塑形）时传入 [(my_decisions, opp_decisions)]。

所有统计都是 (命中次数, 试验次数)，含 "m" 的位置不计入试验；
平滑与阈值由调用方决定（见 rate）。单次遍历，便于在模拟中每回合重算。
"""

from ipd.sim import rules

C, D, M = rules.COOPERATE, rules.DEFECT, rules.MISSING
LAST = rules.TURNS - 1
# 终局阶段之前的回合才计入"无端背叛"，避免与终局背叛重复
ENDGAME_START = rules.PHASES[-1][0]
# 判定"对手几乎总合作"的阈值与最少有效步数
COOPERATOR_RATE = 0.9
COOPERATOR_MIN_MOVES = 10

# 每个统计项的含义（命中 / 试验）
EVENTS = {
    "coop": "主体合作的步数 / 有效步数",
    "opening_defect": "首手背叛的场数 / 首手有效的场数",
    "last_defect": "末手背叛 / 末手有效且对手倒数第二手合作",
    "endgame": "以无端背叛开头、持续到末手的背叛段的场数 / 末段有效的场数",
    "unprovoked": "上一手双方合作时背叛 / 此类机会（仅终局前）",
    "retaliate": "对手上一手背叛时背叛 / 此类机会",
    "copy": "动作等于对手上一手 / 前一手双方都有效的步数",
    "exploit": "对几乎总合作的对手背叛的步数 / 这类对局的有效步数",
    "grim": "对手首次背叛后再未合作的场数 / 对手在倒数第二手前背叛过的场数",
}


def _empty():
    return {name: [0, 0] for name in EVENTS}


def _scan(own, opp, counts, endgame_runs):
    n = min(len(own), len(opp))
    if n == 0:
        return
    valid = [own[t] != M and opp[t] != M for t in range(n)]
    if not any(valid):
        return

    coop = moves = opp_coop = 0
    for t in range(n):
        if valid[t]:
            moves += 1
            coop += own[t] == C
            opp_coop += opp[t] == C
    counts["coop"][0] += coop
    counts["coop"][1] += moves

    if valid[0]:
        counts["opening_defect"][1] += 1
        counts["opening_defect"][0] += own[0] == D

    for t in range(1, n):
        if not (valid[t] and valid[t - 1]):
            continue
        counts["copy"][1] += 1
        counts["copy"][0] += own[t] == opp[t - 1]
        if opp[t - 1] == D:
            counts["retaliate"][1] += 1
            counts["retaliate"][0] += own[t] == D
        elif own[t - 1] == C and t < ENDGAME_START:
            counts["unprovoked"][1] += 1
            counts["unprovoked"][0] += own[t] == D

    if n == rules.TURNS and valid[LAST] and valid[LAST - 1] and opp[LAST - 1] == C:
        counts["last_defect"][1] += 1
        counts["last_defect"][0] += own[LAST] == D

    # 终局背叛段：从末手往前的连续背叛，且段首前一手对手合作（不是报复）
    if n == rules.TURNS and valid[LAST]:
        start = LAST + 1
        while start > 0 and valid[start - 1] and own[start - 1] == D:
            start -= 1
        if start > 0 and valid[start - 1]:
            counts["endgame"][1] += 1
            if start <= LAST and opp[start - 1] == C:
                counts["endgame"][0] += 1
                endgame_runs.append(rules.TURNS - start)

    if moves >= COOPERATOR_MIN_MOVES and opp_coop >= COOPERATOR_RATE * moves:
        counts["exploit"][1] += moves
        counts["exploit"][0] += moves - coop

    first_d = next((t for t in range(n - 2) if valid[t] and opp[t] == D), None)
    if first_d is not None:
        after = [own[t] for t in range(first_d + 1, n) if valid[t]]
        if after:
            counts["grim"][1] += 1
            counts["grim"][0] += C not in after


class Profile:
    """一份历史的行为统计。"""

    __slots__ = ("counts", "matches", "endgame_runs")

    def __init__(self, counts, matches, endgame_runs):
        self.counts = counts
        self.matches = matches
        self.endgame_runs = endgame_runs

    def hits(self, name):
        return self.counts[name][0]

    def trials(self, name):
        return self.counts[name][1]

    def rate(self, name, prior=0.5, strength=2.0):
        """带 Beta 先验平滑的比率：(命中 + 先验 × 强度) / (试验 + 强度)。"""
        h, n = self.counts[name]
        return (h + prior * strength) / (n + strength)

    def endgame_k(self):
        """终局背叛段的典型长度（中位数）；没有观测时返回 0。"""
        if not self.endgame_runs:
            return 0
        runs = sorted(self.endgame_runs)
        return runs[len(runs) // 2]

    def as_dict(self):
        out = {name: tuple(v) for name, v in self.counts.items()}
        out["matches"] = self.matches
        out["endgame_k"] = self.endgame_k()
        return out


def profile(own_rows, opp_rows):
    """统计主体在多场记录中的行为。own_rows、opp_rows 一一对应。"""
    counts = _empty()
    runs = []
    matches = 0
    for own, opp in zip(own_rows, opp_rows):
        if own == rules.MISSING_ROW:
            continue
        before = counts["coop"][1]
        _scan(own, opp, counts, runs)
        matches += counts["coop"][1] > before
    return Profile(counts, matches, runs)


def match_profile(my_decisions, opp_decisions):
    """单场（可未结束）的统计，用于给我方本场行为打信誉分。"""
    return profile((my_decisions,), (opp_decisions,))
