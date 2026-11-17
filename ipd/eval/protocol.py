"""多配比评估协议：指纹去重 + 层配比 + 行为类型配比（issue #2）。

协议先用固定探针给候选池里的每个策略算行为指纹，做确定性聚类并每簇保留
medoid 代表，避免同族策略在平均分里重复加权；然后构造两组配比：

- 层配比：固定 L0/L1/L2 的相对比例，把 L3（读历史）占比扫描为 0、25%、50%；
- 类型配比：按指纹簇的行为类别（cooperative / hostile / shrewd）各构造一个
  "某一类为主"的配比。

每个配比用 5 个固定种子跑完整的 8 轮锦标赛，报告候选策略的每场平均分、
在配比内（候选 + 对手）的排名，以及所有配比中的最差分数。

用法::

    python -m ipd.eval.protocol --agent tft gradual "end_defect(k=1)[tft]"
    python -m ipd.eval.protocol --agent ipd.agents.baseline:baseline_agent

只用标准库；不修改 `ipd/sim/`。
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import random
import time

from ipd.eval.fingerprint import (
    DEFAULT_CLUSTER_THRESHOLD,
    DEFAULT_SEEDS as FINGERPRINT_SEEDS,
    classify_clusters,
    cluster_fingerprints,
    fingerprint_many,
)
from ipd.eval.spec import load_agent
from ipd.pool import agents_by_layer, uniquify
from ipd.sim import Agent, TournamentConfig, run_tournament

DEFAULT_PROTOCOL_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_POOL_SIZE = 20
# 用多个阶段池种子生成候选池，让指纹/聚类看到更多不同的参数变体。
DEFAULT_POOL_SEEDS = (0, 1, 2)
DEFAULT_VARIANTS = 2
L3_SHARES = (0.0, 0.25, 0.5)
ARCHETYPES = ("cooperative", "hostile", "shrewd")
ARCHETYPE_LABELS = {
    "cooperative": "coop-heavy",
    "hostile": "hostile-heavy",
    "shrewd": "shrewd-heavy",
}
# 去掉 L3 后，L0/L1/L2 的相对比例（来自 DEFAULT_MIX 的 0.10/0.35/0.30）。
LAYER_WEIGHTS = {"L0": 0.10, "L1": 0.35, "L2": 0.30}
ARCHETYPE_TARGET = 0.5


def allocate_counts(total, weights):
    """按最大余数法把 total 个名额分给 weights 里的键，和恰好为 total。"""
    if total < 0:
        raise ValueError("total must be non-negative")
    keys = list(weights)
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("weights must have positive sum")
    quotas = {key: total * weights[key] / weight_sum for key in keys}
    counts = {key: int(quotas[key]) for key in keys}
    left = total - sum(counts.values())
    order = sorted(keys, key=lambda key: (-(quotas[key] - counts[key]), keys.index(key)))
    for key in order[:left]:
        counts[key] += 1
    return counts


@dataclass(frozen=True)
class Mix:
    """一个评估配比：候选策略之外的对手集合。"""

    name: str
    kind: str
    description: str
    agents: tuple
    l3_share: float = None
    archetype: str = ""

    @property
    def size(self):
        return len(self.agents)

    def layer_counts(self):
        return Counter(agent.layer for agent in self.agents)


@dataclass(frozen=True)
class DedupedPool:
    """按指纹去重后的候选池。"""

    reservoir: tuple
    fingerprints: tuple
    clusters: tuple
    categories: dict
    threshold: float

    @property
    def representatives(self):
        """每簇 medoid 对应的 Agent（按簇顺序）。"""
        by_name = {agent.name: agent for agent in self.reservoir}
        return tuple(by_name[cluster.representative.name] for cluster in self.clusters)

    @property
    def representative_fingerprints(self):
        return tuple(cluster.representative for cluster in self.clusters)

    def summary(self):
        counts = Counter(self.categories.values())
        return {
            "reservoir": len(self.reservoir),
            "clusters": len(self.clusters),
            "representatives": len(self.representatives),
            "categories": dict(counts),
        }


def build_deduped_pool(
    pool_seeds=DEFAULT_POOL_SEEDS,
    variants=DEFAULT_VARIANTS,
    threshold=DEFAULT_CLUSTER_THRESHOLD,
    fingerprint_seeds=FINGERPRINT_SEEDS,
):
    """生成候选池 -> 指纹 -> 聚类 -> 每簇 medoid 代表。"""
    reservoir = []
    for seed in pool_seeds:
        for layer in ("L0", "L1", "L2", "L3"):
            reservoir.extend(agents_by_layer(layer, seed=seed, variants=variants))
    reservoir = uniquify(reservoir)
    fingerprints = fingerprint_many(reservoir, seeds=fingerprint_seeds)
    clusters = cluster_fingerprints(fingerprints, threshold=threshold)
    categories = classify_clusters(clusters)
    return DedupedPool(
        reservoir=tuple(reservoir),
        fingerprints=tuple(fingerprints),
        clusters=tuple(clusters),
        categories=categories,
        threshold=threshold,
    )


def _sample_from(rng, agents, count):
    pool = sorted(agents, key=lambda agent: agent.name)
    if count > len(pool):
        raise ValueError(f"need {count} agents but only {len(pool)} available")
    return rng.sample(pool, count)


def _fill(picked, candidates, size, rng):
    """把 candidates 中未选过的补到 size 个；保持名字唯一。"""
    picked = list(picked)
    chosen = {agent.name for agent in picked}
    remaining = sorted(
        (agent for agent in candidates if agent.name not in chosen),
        key=lambda agent: agent.name,
    )
    need = size - len(picked)
    if need < 0:
        raise ValueError(f"picked {len(picked)} agents for mix of size {size}")
    if need > len(remaining):
        raise ValueError(f"need {need} more agents but only {len(remaining)} available")
    picked.extend(rng.sample(remaining, need))
    return picked


def build_layer_mixes(pool, size=DEFAULT_POOL_SIZE, seed=0):
    """构造 L3 占比 0/25/50% 的层配比；对手全部来自去重后的簇代表。"""
    by_layer = defaultdict(list)
    for agent in pool.representatives:
        by_layer[agent.layer].append(agent)
    mixes = []
    for share in L3_SHARES:
        n3 = int(share * size + 0.5)
        counts = allocate_counts(size - n3, LAYER_WEIGHTS)
        counts["L3"] = n3
        rng = random.Random(f"layer:{seed}:{share}:{size}")
        picked = []
        for layer in ("L0", "L1", "L2", "L3"):
            count = counts[layer]
            if count:
                picked.extend(_sample_from(rng, by_layer[layer], count))
        picked = _fill(picked, pool.representatives, size, rng)
        picked.sort(key=lambda agent: agent.name)
        actual = Counter(agent.layer for agent in picked)
        description = " ".join(
            f"{layer}={actual[layer]}" for layer in ("L0", "L1", "L2", "L3")
        )
        mixes.append(
            Mix(
                name=f"L3={share:.0%}",
                kind="layer",
                description=description,
                agents=tuple(picked),
                l3_share=share,
            )
        )
    return mixes


def build_archetype_mixes(pool, size=DEFAULT_POOL_SIZE, seed=0):
    """按指纹簇类别构造 cooperative/hostile/shrewd 为主的配比。"""
    reps = list(pool.representatives)
    by_name = {agent.name: agent for agent in reps}
    by_category = defaultdict(list)
    for cluster in pool.clusters:
        by_category[pool.categories[cluster.label]].append(by_name[cluster.representative.name])
    mixes = []
    for category in ARCHETYPES:
        group = sorted(by_category[category], key=lambda agent: agent.name)
        others = sorted(
            (agent for agent in reps if agent.name not in {x.name for x in group}),
            key=lambda agent: agent.name,
        )
        # 某一型不足总规模一半时缩小该配比，保证该型确实"为主"而不是被淹没。
        mix_size = min(size, max(2, 2 * len(group)))
        target = min(int(ARCHETYPE_TARGET * mix_size + 0.5), len(group))
        rng = random.Random(f"archetype:{seed}:{category}:{size}")
        chosen = _sample_from(rng, group, target)
        picked = _fill(chosen, others, mix_size, rng)
        picked.sort(key=lambda agent: agent.name)
        actual = Counter(agent.layer for agent in picked)
        l3_count = actual["L3"]
        mix_name = ARCHETYPE_LABELS[category]
        description = (
            f"{category} {len(chosen)}/{mix_size}"
            f"（候选簇 {len(group)}） L3={l3_count}"
        )
        mixes.append(
            Mix(
                name=mix_name,
                kind="archetype",
                description=description,
                agents=tuple(picked),
                archetype=category,
            )
        )
    return mixes


def build_mixes(
    pool,
    size=DEFAULT_POOL_SIZE,
    seed=0,
    include_layer=True,
    include_archetype=True,
):
    """按需构造全部配比。"""
    mixes = []
    if include_layer:
        mixes.extend(build_layer_mixes(pool, size=size, seed=seed))
    if include_archetype:
        mixes.extend(build_archetype_mixes(pool, size=size, seed=seed))
    return mixes


@dataclass(frozen=True)
class MixResult:
    """候选策略在一个配比下的结果。"""

    name: str
    kind: str
    description: str
    per_match: float
    total: int
    rank: int
    n_agents: int
    seed_scores: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class ProtocolResult:
    """候选策略在全部配比下的结果。"""

    agent_name: str
    mixes: tuple

    @property
    def worst_mix(self):
        return min(self.mixes, key=lambda result: result.per_match)

    @property
    def mean_score(self):
        return sum(result.per_match for result in self.mixes) / len(self.mixes)

    def table(self):
        lines = [
            f"agent: {self.agent_name}",
            f"{'mix':<15} {'composition':<38} {'avg/match':>9} {'rank':>8}",
        ]
        for result in self.mixes:
            rank = f"{result.rank}/{result.n_agents}"
            lines.append(
                f"{result.name:<15} {result.description:<38} "
                f"{result.per_match:>9.2f} {rank:>8}"
            )
        worst = self.worst_mix
        lines.append(
            f"summary: mean {self.mean_score:.2f}, "
            f"worst {worst.per_match:.2f} ({worst.name})"
        )
        return "\n".join(lines)


def _candidate_name(opponents):
    used = {agent.name for agent in opponents}
    name = "candidate"
    suffix = 1
    while name in used:
        suffix += 1
        name = f"candidate#{suffix}"
    return name


def _competition_rank(totals, name):
    """并列同名次（1, 1, 3）。"""
    score = totals[name]
    return 1 + sum(1 for other, value in totals.items() if other != name and value > score)


def evaluate_mix(agent, mix, seeds=DEFAULT_PROTOCOL_SEEDS, cfg=None):
    """在单个配比上跑 5 个种子，返回候选策略的平均分与排名。"""
    cfg = cfg or TournamentConfig()
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    candidate = Agent(
        name=_candidate_name(mix.agents),
        fn=agent.fn,
        family=agent.family,
        layer=agent.layer,
        params=agent.params,
    )
    agents = [candidate] + list(mix.agents)
    totals = {item.name: 0 for item in agents}
    seed_scores = []
    for seed in seeds:
        result = run_tournament(agents, seed=seed, cfg=cfg)
        for name, score in zip(result.names, result.scores):
            totals[name] += score
        seed_scores.append(result.scores[0])
    per_match = totals[candidate.name] / (len(seeds) * cfg.rounds * len(mix.agents))
    return MixResult(
        name=mix.name,
        kind=mix.kind,
        description=mix.description,
        per_match=per_match,
        total=totals[candidate.name],
        rank=_competition_rank(totals, candidate.name),
        n_agents=len(agents),
        seed_scores=tuple(seed_scores),
    )


def evaluate_agent(agent, mixes, seeds=DEFAULT_PROTOCOL_SEEDS, cfg=None):
    """在全部配比上评估一个候选策略。"""
    if not mixes:
        raise ValueError("need at least one mix")
    results = tuple(evaluate_mix(agent, mix, seeds=seeds, cfg=cfg) for mix in mixes)
    return ProtocolResult(agent_name=agent.name, mixes=results)


def evaluate_agents(agents, mixes, seeds=DEFAULT_PROTOCOL_SEEDS, cfg=None):
    """批量评估；配比只构造一次。"""
    return [evaluate_agent(agent, mixes, seeds=seeds, cfg=cfg) for agent in agents]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--agent",
        nargs="+",
        required=True,
        help='族名（tft）、族+参数/基础（end_defect(k=1)[tft]）或模块路径',
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_PROTOCOL_SEEDS)
    )
    parser.add_argument("--pool-size", type=int, default=DEFAULT_POOL_SIZE)
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_CLUSTER_THRESHOLD,
        help="指纹聚类阈值（平均绝对距离）",
    )
    parser.add_argument("--no-layer", action="store_true", help="不跑层配比")
    parser.add_argument("--no-archetype", action="store_true", help="不跑类型配比")
    args = parser.parse_args()
    if args.no_layer and args.no_archetype:
        parser.error("--no-layer and --no-archetype cannot both be set")

    start = time.perf_counter()
    pool = build_deduped_pool(
        pool_seeds=tuple(args.pool_seed + i for i in range(len(DEFAULT_POOL_SEEDS))),
        threshold=args.threshold,
    )
    summary = pool.summary()
    print(
        f"pool: {summary['reservoir']} agents -> {summary['clusters']} clusters, "
        f"{summary['representatives']} representatives "
        f"({summary['categories']}), threshold={args.threshold}"
    )
    mixes = build_mixes(
        pool,
        size=args.pool_size,
        seed=args.pool_seed,
        include_layer=not args.no_layer,
        include_archetype=not args.no_archetype,
    )
    print(f"max mix size: {args.pool_size} opponents, seeds={tuple(args.seeds)}")
    print("mixes: " + "; ".join(f"{mix.name} [{mix.description}]" for mix in mixes))
    print()

    agents = [load_agent(spec) for spec in args.agent]
    results = evaluate_agents(agents, mixes, seeds=args.seeds)
    for result in results:
        print(result.table())
        print()
    if len(results) > 1:
        print(f"{'agent':<36} {'mean':>8} {'worst':>8}  worst mix")
        for result in results:
            worst = result.worst_mix
            print(
                f"{result.agent_name:<36} {result.mean_score:>8.2f} "
                f"{worst.per_match:>8.2f}  {worst.name}"
            )
    print(f"\nelapsed: {time.perf_counter() - start:.1f}s")


__all__ = [
    "ARCHETYPES",
    "DEFAULT_POOL_SEEDS",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_PROTOCOL_SEEDS",
    "DedupedPool",
    "L3_SHARES",
    "Mix",
    "MixResult",
    "ProtocolResult",
    "allocate_counts",
    "build_archetype_mixes",
    "build_deduped_pool",
    "build_layer_mixes",
    "build_mixes",
    "evaluate_agent",
    "evaluate_agents",
    "evaluate_mix",
]


if __name__ == "__main__":
    main()
