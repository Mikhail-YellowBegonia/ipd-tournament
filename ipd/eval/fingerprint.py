"""行为指纹：用固定探针刻画策略在受控对局中的合作率序列。

为什么要指纹
------------
对手池里同一族的策略（如 TFT 与各种 GTFT/带噪声变体）行为接近，直接按
对手个数平均会让某些族主导评估。指纹把每个策略映射成固定长度向量：

- 对每个固定探针（AllC、AllD、TFT、TF2T、Random(0.5)、两种周期模式），
  记录 25 手合作率序列；
- 每个探针再配 4 种历史情境（无历史、合作历史、被剥削历史、交替历史），
  用来区分"读历史"的策略；
- 随机策略用多个固定种子取平均，保证结果可复现。

向量长度 = 探针数 × 25。距离定义为各分量绝对差的平均（0 表示行为相同，
1 表示每手都相反）。聚类采用确定性的平均连接层次聚类，阈值内的策略
归为一簇；簇代表取 medoid，评估时可以按 1/簇大小 给每个策略加权。

只用标准库；不修改 `ipd/sim/`。
"""

from dataclasses import dataclass
import random
from typing import Callable

from ipd.sim import rules
from ipd.sim.agent import Agent

C, D, M = rules.COOPERATE, rules.DEFECT, rules.MISSING
TURNS = rules.TURNS
HISTORY_MATCHES = rules.HISTORY_MATCHES

# 探针策略：给定 (length, 自己此前动作, 对手此前动作, 局部随机数)，返回本手动作。
ProbePolicy = Callable[[int, tuple, tuple, random.Random], str]
HistoryRows = tuple[tuple[str, ...], ...]

DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_CLUSTER_THRESHOLD = 0.12
# 这些历史情境只用于生成"对手的过去 20 场"，当前场总是从头开始。
HISTORY_SCENARIOS = ("empty", "coop", "defect", "mixed")

# 合作率序列的判定阈值：与 AllC/AllD 探针对局时的平均合作率。
COOPERATIVE_ALL_C = 0.7
COOPERATIVE_ALL_D = 0.25
HOSTILE_ALL_C = 0.25


@dataclass(frozen=True)
class Probe:
    """一个固定探针及其 20 场历史。"""

    name: str
    base: str
    scenario: str
    policy: ProbePolicy
    # 探针自己的历史动作（20 × 25），即被测策略看到的 opp_history_decisions
    history: HistoryRows
    # 探针的对手的历史动作（20 × 25），即 opp_opp_history_decisions
    counterpart_history: HistoryRows


def _fixed(move):
    def policy(length, mine, theirs, rng):
        return move
    return policy


def _tft(length, mine, theirs, rng):
    return theirs[-1] if length else C


def _tf2t(length, mine, theirs, rng):
    if length >= 2 and theirs[-1] == D and theirs[-2] == D:
        return D
    return C


def _random_half(length, mine, theirs, rng):
    return C if rng.random() < 0.5 else D


def _periodic(pattern):
    def policy(length, mine, theirs, rng):
        return pattern[length % len(pattern)]
    return policy


_BASE_POLICIES = {
    "all_c": _fixed(C),
    "all_d": _fixed(D),
    "tft": _tft,
    "tf2t": _tf2t,
    "random50": _random_half,
    "periodic_cb": _periodic("cb"),
    "periodic_ccb": _periodic("ccb"),
}


def _counterpart_move(scenario, length):
    if scenario == "coop":
        return C
    if scenario == "defect":
        return D
    if scenario == "mixed":
        return "cb"[length % 2]
    raise ValueError(f"unknown history scenario: {scenario!r}")


def _build_history(policy, scenario):
    """模拟探针与固定脚本对手的 20 场历史，返回两份 20 × 25 元组。"""
    if scenario == "empty":
        row = rules.MISSING_ROW
        return (row,) * HISTORY_MATCHES, (row,) * HISTORY_MATCHES
    if scenario not in HISTORY_SCENARIOS:
        raise ValueError(f"unknown history scenario: {scenario!r}")

    rng = random.Random(f"probe-history:{scenario}")
    own_rows, opp_rows = [], []
    for _ in range(HISTORY_MATCHES):
        own, opp = [], []
        for length in range(TURNS):
            other = _counterpart_move(scenario, length)
            move = policy(length, tuple(own), tuple(opp), rng)
            if move not in rules.LEGAL_MOVES:
                move = C
            own.append(move)
            opp.append(other)
        own_rows.append(tuple(own))
        opp_rows.append(tuple(opp))
    return tuple(own_rows), tuple(opp_rows)


def _build_probes():
    probes = []
    for base, policy in _BASE_POLICIES.items():
        for scenario in HISTORY_SCENARIOS:
            history, counterpart = _build_history(policy, scenario)
            probes.append(
                Probe(
                    name=f"{base}@{scenario}",
                    base=base,
                    scenario=scenario,
                    policy=policy,
                    history=history,
                    counterpart_history=counterpart,
                )
            )
    return tuple(probes)


PROBES = _build_probes()
PROBE_INDEX = {probe.name: i for i, probe in enumerate(PROBES)}


@dataclass(frozen=True)
class Fingerprint:
    """一个策略的固定长度行为向量。"""

    name: str
    family: str
    layer: str
    vector: tuple[float, ...]
    faults: int = 0
    seeds: tuple[int, ...] = DEFAULT_SEEDS

    def cooperation(self, probe_name):
        """某个探针下 25 手的平均合作率。"""
        start = PROBE_INDEX[probe_name] * TURNS
        return sum(self.vector[start:start + TURNS]) / TURNS

    def probe_sequence(self, probe_name):
        """某个探针下 25 手的合作率序列。"""
        start = PROBE_INDEX[probe_name] * TURNS
        return self.vector[start:start + TURNS]


def _decide(agent, length, my, opp, probe):
    """调用策略；异常或非法返回值按规则记为合作，并计一次故障。"""
    args = (
        0,
        length,
        1,
        tuple(my),
        2,
        tuple(opp),
        rules.REPAY,
        probe.history,
        probe.counterpart_history,
    )
    try:
        move = agent.fn(*args)
    except Exception:
        return C, True
    if move not in rules.LEGAL_MOVES:
        return C, True
    return move, False


def fingerprint(agent, seeds=DEFAULT_SEEDS, probes=PROBES):
    """计算一个策略的行为指纹。

    Args:
        agent: 被测策略（`Agent`）。
        seeds: 固定种子；对随机策略取多种子平均。每个探针开始时重置
            全局 `random` 到该种子，因此确定性策略的结果与种子无关。
        probes: 探针序列；默认 `PROBES`，测试可以传入更小的集合。

    Returns:
        `Fingerprint`。向量按"探针 × 25 手"展平，值为平均合作率。
    """
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    if not probes:
        raise ValueError("need at least one probe")

    sums = [0.0] * (len(probes) * TURNS)
    faults = 0
    state = random.getstate()
    try:
        for seed in seeds:
            for probe_index, probe in enumerate(probes):
                # 每个探针从同一随机种子开始，保证不同探针可比。
                random.seed(seed)
                rng = random.Random(f"probe:{seed}:{probe.name}")
                my, opp = [], []
                for length in range(TURNS):
                    move, fault = _decide(agent, length, my, opp, probe)
                    faults += fault
                    if move == C:
                        sums[probe_index * TURNS + length] += 1.0
                    opp_move = probe.policy(length, tuple(my), tuple(opp), rng)
                    if opp_move not in rules.LEGAL_MOVES:
                        opp_move = C
                    my.append(move)
                    opp.append(opp_move)
    finally:
        random.setstate(state)

    total = float(len(seeds))
    vector = tuple(value / total for value in sums)
    return Fingerprint(
        name=agent.name,
        family=agent.family,
        layer=agent.layer,
        vector=vector,
        faults=faults,
        seeds=seeds,
    )


def distance(a, b):
    """两个指纹的平均绝对距离（0–1）。"""
    if len(a.vector) != len(b.vector):
        raise ValueError("fingerprints have different lengths")
    if not a.vector:
        return 0.0
    return sum(abs(x - y) for x, y in zip(a.vector, b.vector)) / len(a.vector)


@dataclass
class Cluster:
    """一簇行为相近的策略。"""

    label: str
    members: list
    representative: Fingerprint

    def __post_init__(self):
        if not self.label:
            self.label = "c00"

    @property
    def size(self):
        return len(self.members)

    @property
    def names(self):
        return tuple(member.name for member in self.members)


def _cluster_label(index):
    return f"c{index:02d}"


def cluster_fingerprints(fingerprints, threshold=DEFAULT_CLUSTER_THRESHOLD):
    """确定性平均连接层次聚类。

    反复合并平均距离最小且不超过 `threshold` 的两簇；距离相同时按簇的
    原始下标决定，保证结果只依赖输入顺序。簇按代表策略名字排序并编号。

    Args:
        fingerprints: `Fingerprint` 序列。
        threshold: 合并阈值，平均绝对距离。

    Returns:
        `Cluster` 列表，每个簇有 medoid 代表与稳定编号。
    """
    items = list(fingerprints)
    n = len(items)
    if n == 0:
        return []
    if threshold < 0:
        raise ValueError("threshold must be non-negative")

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(items[i], items[j])
            dist[i][j] = dist[j][i] = d

    groups = [[i] for i in range(n)]
    while len(groups) > 1:
        best = None
        best_distance = float("inf")
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                total = 0.0
                for i in groups[a]:
                    for j in groups[b]:
                        total += dist[i][j]
                average = total / (len(groups[a]) * len(groups[b]))
                if average < best_distance - 1e-12:
                    best_distance = average
                    best = (a, b)
        if best is None or best_distance > threshold:
            break
        a, b = best
        groups[a] = groups[a] + groups[b]
        del groups[b]

    clusters = []
    for group in groups:
        rep = min(
            group,
            key=lambda i: (sum(dist[i][j] for j in group), items[i].name, i),
        )
        clusters.append(
            Cluster(label="", members=[items[i] for i in group], representative=items[rep])
        )
    clusters.sort(key=lambda cluster: cluster.representative.name)
    for index, cluster in enumerate(clusters):
        cluster.label = _cluster_label(index)
    return clusters


def dedupe(fingerprints, threshold=DEFAULT_CLUSTER_THRESHOLD):
    """每簇保留一个代表（medoid），用于去重后的对手池。"""
    return [cluster.representative for cluster in cluster_fingerprints(fingerprints, threshold)]


def cluster_weights(clusters):
    """簇内每个策略的权重：1 / 簇大小。同簇策略合计权重为 1。"""
    weights = {}
    for cluster in clusters:
        weight = 1.0 / cluster.size
        for member in cluster.members:
            weights[member.name] = weight
    return weights


def classify_fingerprint(fp):
    """按对 AllC / AllD 探针的合作率把行为粗分为三类。

    - cooperative：对 AllC 与 AllD 都较合作（宽容/无条件合作）；
    - hostile：即使对 AllC 也很少合作（敌对/剥削）；
    - shrewd：其余，即会根据对手行为调整（条件合作/报复/随机）。
    """
    coop_all_c = fp.cooperation("all_c@empty")
    coop_all_d = fp.cooperation("all_d@empty")
    if coop_all_c >= COOPERATIVE_ALL_C and coop_all_d >= COOPERATIVE_ALL_D:
        return "cooperative"
    if coop_all_c <= HOSTILE_ALL_C:
        return "hostile"
    return "shrewd"


def classify_clusters(clusters):
    """给每个簇标一个行为类别，返回 {label: category}。"""
    return {cluster.label: classify_fingerprint(cluster.representative) for cluster in clusters}


def mean_vector(fingerprints):
    """一组指纹的逐分量均值，便于报告簇中心。"""
    items = list(fingerprints)
    if not items:
        raise ValueError("need at least one fingerprint")
    length = len(items[0].vector)
    if any(len(item.vector) != length for item in items):
        raise ValueError("fingerprints have different lengths")
    return tuple(sum(item.vector[i] for item in items) / len(items) for i in range(length))


def probe_names():
    """默认探针名字，按向量中的顺序。"""
    return tuple(probe.name for probe in PROBES)


def fingerprint_many(agents, seeds=DEFAULT_SEEDS, probes=PROBES):
    """批量计算指纹。"""
    return [fingerprint(agent, seeds=seeds, probes=probes) for agent in agents]


__all__ = [
    "Cluster",
    "DEFAULT_CLUSTER_THRESHOLD",
    "DEFAULT_SEEDS",
    "Fingerprint",
    "HISTORY_SCENARIOS",
    "PROBES",
    "PROBE_INDEX",
    "Probe",
    "classify_clusters",
    "classify_fingerprint",
    "cluster_fingerprints",
    "cluster_weights",
    "dedupe",
    "distance",
    "fingerprint",
    "fingerprint_many",
    "mean_vector",
    "probe_names",
]
