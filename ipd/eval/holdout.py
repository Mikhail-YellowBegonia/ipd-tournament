"""留出测试池评估入口（issue #10，sub-issue 11.6）。

留出池只用于最终验收，不进入训练。本模块用**纯标准库**（与评测端一致）
从 130 个留出策略里按种子抽样，和候选策略、基线一起跑完整 8 轮锦标赛，
报告每场平均分、池内名次、逐种子分数与标签覆盖。

用法::

    python -m ipd.eval.holdout --check
    python -m ipd.eval.holdout --agent ipd.agents.baseline:baseline_agent
    python -m ipd.eval.holdout --agent tft --size 40 --seeds 0,1,2

默认每个种子抽 30 个留出对手（真实考场 n 通常 20–40），跑 5 个固定种子。
规模设成 20–40 是为了控制耗时；`--size 130` 会用整个池子（约 8 倍耗时）。
"""

import argparse
import json
import random

from ipd.eval.spec import load_agent
from ipd.pool.holdout import (
    HOLDOUT_AGENTS,
    MODULE_NAMES,
    TAG_DESCRIPTIONS,
    module_counts,
    tag_counts,
)
from ipd.sim import Agent, TournamentConfig, run_tournament

DEFAULT_SIZE = 30
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_BASELINE = "ipd.agents.baseline:baseline_agent"
# issue #10 的行为覆盖下限。
TAG_QUOTAS = {
    "history": 0.20,
    "conditional": 0.30,
    "endgame": 0.10,
    "noisy": 0.10,
}


def sample_agents(seed, size, pool=None):
    """按种子无放回抽样；返回 Agent 列表（顺序稳定）。"""
    pool = list(HOLDOUT_AGENTS if pool is None else pool)
    if size <= 0:
        raise ValueError("size must be positive")
    if size >= len(pool):
        return list(pool)
    rng = random.Random(f"holdout-sample-{seed}-{size}")
    indices = sorted(rng.sample(range(len(pool)), size))
    return [pool[index] for index in indices]


def tag_coverage(agents):
    """返回每个标签在给定 Agent 列表里的数量与占比。"""
    counts = {}
    for agent in agents:
        for tag in agent.params.get("tags", ()):
            counts[tag] = counts.get(tag, 0) + 1
    total = len(agents) or 1
    return {
        tag: {"count": counts.get(tag, 0), "share": counts.get(tag, 0) / total}
        for tag in sorted(set(counts) | set(TAG_QUOTAS))
    }


def check_pool(pool=None):
    """检查池子规模与行为覆盖；返回 `(passed, report)`。"""
    pool = list(HOLDOUT_AGENTS if pool is None else pool)
    counts = tag_counts() if pool is HOLDOUT_AGENTS else tag_coverage(pool)
    total = len(pool)
    coverage = {
        tag: {
            "count": counts.get(tag, {}).get("count", 0)
            if isinstance(counts.get(tag), dict)
            else counts.get(tag, 0),
            "share": (
                counts.get(tag, {}).get("share", 0.0)
                if isinstance(counts.get(tag), dict)
                else counts.get(tag, 0) / (total or 1)
            ),
            "minimum": quota,
        }
        for tag, quota in TAG_QUOTAS.items()
    }
    size_ok = 100 <= total <= 200
    coverage_ok = all(item["share"] >= item["minimum"] for item in coverage.values())
    report = {
        "total": total,
        "size_ok": size_ok,
        "modules": module_counts(),
        "tags": tag_counts(),
        "coverage": coverage,
        "coverage_ok": coverage_ok,
    }
    return size_ok and coverage_ok, report


def _rank(totals, name):
    score = totals[name]
    return 1 + sum(1 for other, value in totals.items() if other != name and value > score)


def evaluate_holdout(
    candidate,
    baseline=None,
    size=DEFAULT_SIZE,
    seeds=DEFAULT_SEEDS,
    cfg=None,
    pool=None,
    verbose=False,
):
    """在留出池抽样上与基线比较。

    Args:
        candidate: 候选 `Agent`。
        baseline: 基线 `Agent`；None 表示不比较。
        size: 每个种子抽样的留出对手数。
        seeds: 评估种子。
        cfg: `TournamentConfig`；None 用默认（8 轮、hide_prob=0.1）。
        pool: 留出 Agent 列表；None 用完整留出池。
        verbose: 是否打印每个种子的进度。

    Returns:
        指标字典（JSON 友好）。
    """
    cfg = cfg or TournamentConfig()
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("need at least one seed")
    pool = list(HOLDOUT_AGENTS if pool is None else pool)
    if size <= 0:
        raise ValueError("size must be positive")

    names = {agent.name for agent in pool}
    candidate_name = "candidate"
    suffix = 1
    while candidate_name in names:
        suffix += 1
        candidate_name = f"candidate#{suffix}"
    # 候选/基线的名字在池子里可能重复；重新包装为唯一名字。
    players = [
        Agent(
            name=candidate_name,
            fn=candidate.fn,
            family=candidate.family,
            layer=candidate.layer,
            params=candidate.params,
        )
    ]
    if baseline is not None:
        baseline_name = "baseline"
        suffix = 1
        while baseline_name in names or baseline_name == candidate_name:
            suffix += 1
            baseline_name = f"baseline#{suffix}"
        players.append(
            Agent(
                name=baseline_name,
                fn=baseline.fn,
                family=baseline.family,
                layer=baseline.layer,
                params=baseline.params,
            )
        )
    else:
        baseline_name = None

    totals = {agent.name: 0.0 for agent in players}
    per_seed = []
    seed_ranks = []
    coverages = []
    faults = 0
    for seed in seeds:
        sampled = sample_agents(seed, size, pool)
        opponents = [
            Agent(
                name=f"{agent.name}",
                fn=agent.fn,
                family=agent.family,
                layer=agent.layer,
                params=agent.params,
            )
            for agent in sampled
        ]
        roster = players + opponents
        if len({agent.name for agent in roster}) != len(roster):
            raise RuntimeError("holdout evaluation roster has duplicate names")
        result = run_tournament(roster, seed=seed, cfg=cfg)
        faults += sum(result.faults)
        for name, score in zip(result.names, result.scores):
            if name in totals:
                totals[name] += score
        seed_ranks.append(
            1 + sum(1 for score in result.scores if score > result.scores[0])
        )
        per_seed.append(
            {
                "seed": seed,
                "candidate": result.scores[0],
                "baseline": (
                    result.scores[1] if baseline_name is not None else None
                ),
                "rank": seed_ranks[-1],
            }
        )
        coverages.append(tag_coverage(sampled))
        if verbose:
            print(
                f"seed {seed}: candidate total {result.scores[0]}, "
                f"{'baseline ' + str(result.scores[1]) if baseline_name else 'no baseline'}"
            )

    opponent_count = size + (1 if baseline is not None else 0)
    divisor = len(seeds) * cfg.rounds * opponent_count
    candidate_per_match = totals[candidate_name] / divisor
    baseline_per_match = (
        totals[baseline_name] / divisor if baseline_name is not None else None
    )
    rank = sum(seed_ranks) / len(seed_ranks)
    return {
        "candidate": candidate_name,
        "baseline": baseline_name,
        "size": size,
        "seeds": list(seeds),
        "n_agents": len(players) + size,
        "faults": faults,
        "candidate_per_match": candidate_per_match,
        "baseline_per_match": baseline_per_match,
        "gap": (
            candidate_per_match - baseline_per_match
            if baseline_per_match is not None
            else None
        ),
        "rank": rank,
        "per_seed": per_seed,
        "coverage": coverages,
        "pool_size": len(pool),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="留出测试池评估（issue #10）")
    parser.add_argument("--agent", help="候选策略写法，见 ipd.eval.spec")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--check", action="store_true", help="只检查池子规模与覆盖")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    if args.check:
        passed, report = check_pool()
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"holdout pool: {report['total']} strategies (100–200 ok: {report['size_ok']})")
            print(f"modules: {report['modules']}")
            print("tags:")
            for tag, count in report["tags"].items():
                print(f"  {tag:<12} {count:>3}  {TAG_DESCRIPTIONS.get(tag, '')}")
            print("coverage:")
            for tag, item in report["coverage"].items():
                print(
                    f"  {tag:<12} {item['count']:>3} ({item['share']:.1%}) "
                    f"minimum {item['minimum']:.0%}"
                )
            print("RESULT:", "OK" if passed else "FAIL")
        return 0 if passed else 1

    if not args.agent:
        parser.error("--agent is required unless --check is used")
    candidate = load_agent(args.agent)
    baseline = None
    if not args.no_baseline:
        baseline = load_agent(args.baseline)
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())
    result = evaluate_holdout(
        candidate,
        baseline=baseline,
        size=args.size,
        seeds=seeds,
        verbose=not args.json,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(
            f"holdout pool: {result['pool_size']} strategies; "
            f"sample {result['size']}/seed; "
            f"{result['n_agents']} players"
        )
        print(
            f"candidate {result['candidate']}: "
            f"{result['candidate_per_match']:.2f} per match, "
            f"mean rank {result['rank']:.1f}/{result['n_agents']}"
        )
        if result["baseline_per_match"] is not None:
            print(
                f"baseline {result['baseline']}: "
                f"{result['baseline_per_match']:.2f} per match; "
                f"gap {result['gap']:+.2f}"
            )
        print(f"faults: {result['faults']}")
        for seed in result["per_seed"]:
            print(
                f"  seed {seed['seed']}: candidate {seed['candidate']} "
                f"baseline {seed['baseline']}"
            )
        for index, coverage in enumerate(result["coverage"]):
            shares = ", ".join(
                f"{tag} {item['share']:.0%}" for tag, item in coverage.items()
            )
            print(f"  coverage[{index}]: {shares}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
