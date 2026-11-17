"""全池诊断/评测入口（plan 0001 / T2）。

与 ``ipd.eval.holdout`` 的区别：

- 默认跑**全池**（130 个留出策略），而不是每种子抽 30 个；
- 同时放**候选、基线、基线克隆**三个席位，克隆-基线的 gap 作为同场噪声对照；
- 输出逐对手得分、尾部、按 tag 聚合、逐种子 gap、配置变体；
- 决策用全池 5 seeds；30 个/种子只做快速筛查（实验 0009：该协议自身噪声 ±3）。

用法::

    # 全池 5 seeds，候选 vs 基线 vs 克隆（约 2.5 分钟）
    python -m ipd.eval.diagnose --agent ipd.agents.baseline_guard:baseline_guard_agent

    # 与 ipd.eval.holdout 同口径的 30/种子对照（噪声大，仅筛查）
    python -m ipd.eval.diagnose --agent tft --size 30 --seeds 0,1,2,3,4 --no-clone

    # 只跑候选（对齐实验 0009 的诊断：130 个对手、seed 0）
    python -m ipd.eval.diagnose --agent ipd.agents.baseline:baseline_agent \
        --no-baseline --no-clone --seeds 0

    # 四种 config 变体（默认 seed 0）
    python -m ipd.eval.diagnose --agent ... --variants

    # CI/脚本用：门槛检查（全池均值 >= 基线 +1.0，最差种子 >= 基线 -1.0）
    python -m ipd.eval.diagnose --agent ... --gate
"""

import argparse
import json
import random
import time
from dataclasses import replace

from ipd.eval.holdout import sample_agents
from ipd.eval.spec import load_agent
from ipd.pool.holdout import HOLDOUT_AGENTS, TAG_DESCRIPTIONS
from ipd.pool.registry import canonical
from ipd.sim import Agent, TournamentConfig, run_tournament

DEFAULT_BASELINE = "ipd.agents.baseline:baseline_agent"
DEFAULT_SEEDS = "0,1,2,3,4"
DEFAULT_TAIL = 10
POOL_CHOICES = ("holdout", "canonical")

# plan 0001 的四道门之一：配置变体各跑一次（默认 seed 0）。
CONFIG_VARIANTS = (
    ("history_update=round", {"history_update": "round"}),
    ("hide_scope=step", {"hide_scope": "step"}),
    ("shuffle_schedule=False", {"shuffle_schedule": False}),
    ("hide_prob=0.0", {"hide_prob": 0.0}),
)


def _with_name(agent, name):
    """复制 Agent 并改名；fn/family/layer/params 保留。"""
    return Agent(
        name=name,
        fn=agent.fn,
        family=agent.family,
        layer=agent.layer,
        params=dict(agent.params),
    )


def _sample(pool, seed, size, pool_name):
    """按种子抽样；holdout 复用官方协议，其它池用本地 RNG。"""
    pool = list(pool)
    if size is None or size >= len(pool):
        return pool
    if pool_name == "holdout":
        return sample_agents(seed, size, pool)
    rng = random.Random(f"diagnose-{seed}-{size}")
    indices = sorted(rng.sample(range(len(pool)), size))
    return [pool[index] for index in indices]


def _run_seed(candidate, baseline, include_clone, sampled, seed, cfg):
    """跑一个种子的锦标赛，返回该种子的原始统计。"""
    roster = []
    roles = {}

    def add(agent, role, name):
        roles[role] = len(roster)
        roster.append(_with_name(agent, name))

    add(candidate, "candidate", "candidate")
    if baseline is not None:
        add(baseline, "baseline", "baseline")
        if include_clone:
            add(baseline, "clone", "baseline#clone")
    first_opponent = len(roster)
    roster.extend(sampled)
    opponent_indices = range(first_opponent, len(roster))

    result = run_tournament(roster, seed=seed, cfg=cfg)
    n_agents = len(roster)
    divisor = cfg.rounds * (n_agents - 1)
    candidate_idx = roles["candidate"]
    candidate_total = result.scores[candidate_idx]
    baseline_idx = roles.get("baseline")
    clone_idx = roles.get("clone")

    per_opponent = []
    for index in opponent_indices:
        row = {
            "name": roster[index].name,
            "tags": tuple(roster[index].params.get("tags", ())),
            "candidate_total": result.pair_scores[candidate_idx][index],
            "baseline_total": (
                result.pair_scores[baseline_idx][index]
                if baseline_idx is not None
                else None
            ),
        }
        per_opponent.append(row)

    baseline_total = (
        result.scores[baseline_idx] if baseline_idx is not None else None
    )
    clone_total = result.scores[clone_idx] if clone_idx is not None else None
    rank = 1 + sum(1 for score in result.scores if score > candidate_total)
    return {
        "seed": seed,
        "candidate_total": candidate_total,
        "baseline_total": baseline_total,
        "clone_total": clone_total,
        "candidate_per_match": candidate_total / divisor,
        "baseline_per_match": (
            baseline_total / divisor if baseline_total is not None else None
        ),
        "clone_per_match": (
            clone_total / divisor if clone_total is not None else None
        ),
        "rank": rank,
        "n_agents": n_agents,
        "faults": sum(result.faults),
        "per_opponent": per_opponent,
    }


def evaluate_diagnose(
    candidate,
    baseline=None,
    clone=True,
    pool=None,
    pool_name="holdout",
    size=None,
    seeds=(0,),
    cfg=None,
    variants=False,
    tail=DEFAULT_TAIL,
):
    """全池/抽样诊断的主函数；返回可直接 JSON 序列化的报告。

    Args:
        candidate: 候选 ``Agent``。
        baseline: 基线 ``Agent``；None 表示不设基线（只看候选绝对分）。
        clone: 是否加基线克隆席位（噪声对照）；无基线时自动关闭。
        pool: 对手池；None 用留出池。
        pool_name: ``holdout``/``canonical``/``custom``；影响抽样方式。
        size: 每种子对手数；None 表示全池。
        seeds: 种子序列。
        cfg: ``TournamentConfig``；默认 8 轮、hide_prob 0.1。
        variants: 是否跑四种 config 变体（默认 seed）。
        tail: 尾部列表长度。
    """
    cfg = cfg or TournamentConfig()
    pool = list(HOLDOUT_AGENTS if pool is None else pool)
    if size is not None and size <= 0:
        raise ValueError("size must be positive or None")
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    include_clone = bool(clone and baseline is not None)

    seed_rows = []
    opponent_rows = {}
    totals = {"candidate": 0.0, "baseline": 0.0, "clone": 0.0}
    faults = 0
    started = time.perf_counter()

    for seed in seeds:
        sampled = _sample(pool, seed, size, pool_name)
        row = _run_seed(candidate, baseline, include_clone, sampled, seed, cfg)
        seed_rows.append({key: value for key, value in row.items() if key != "per_opponent"})
        totals["candidate"] += row["candidate_per_match"]
        if row["baseline_per_match"] is not None:
            totals["baseline"] += row["baseline_per_match"]
        if row["clone_per_match"] is not None:
            totals["clone"] += row["clone_per_match"]
        faults += row["faults"]
        for item in row["per_opponent"]:
            entry = opponent_rows.setdefault(
                item["name"],
                {
                    "name": item["name"],
                    "tags": item["tags"],
                    "candidate_total": 0,
                    "baseline_total": 0,
                    "appearances": 0,
                },
            )
            entry["candidate_total"] += item["candidate_total"]
            entry["baseline_total"] += item["baseline_total"] or 0
            entry["appearances"] += 1

    elapsed = time.perf_counter() - started
    n_seeds = len(seeds)
    n_opponents = len(_sample(pool, seeds[0], size, pool_name))
    n_agents = n_opponents + 1 + (1 if baseline is not None else 0) + (
        1 if include_clone else 0
    )

    per_opponent = []
    for entry in opponent_rows.values():
        divisor = entry["appearances"] * cfg.rounds
        candidate_per_match = entry["candidate_total"] / divisor
        baseline_per_match = (
            entry["baseline_total"] / divisor if baseline is not None else None
        )
        per_opponent.append(
            {
                "name": entry["name"],
                "tags": list(entry["tags"]),
                "candidate": candidate_per_match,
                "baseline": baseline_per_match,
                "delta": (
                    candidate_per_match - baseline_per_match
                    if baseline_per_match is not None
                    else None
                ),
                "appearances": entry["appearances"],
            }
        )
    per_opponent.sort(key=lambda item: -item["candidate"])

    by_tag = {}
    if baseline is not None:
        for item in per_opponent:
            for tag in item["tags"]:
                bucket = by_tag.setdefault(
                    tag, {"n": 0, "candidate": 0.0, "baseline": 0.0}
                )
                bucket["n"] += 1
                bucket["candidate"] += item["candidate"]
                bucket["baseline"] += item["baseline"]
        for tag, bucket in by_tag.items():
            count = bucket["n"]
            bucket["candidate"] /= count
            bucket["baseline"] /= count
            bucket["delta"] = bucket["candidate"] - bucket["baseline"]
            bucket["description"] = TAG_DESCRIPTIONS.get(tag, "")

    candidate_per_match = totals["candidate"] / n_seeds
    baseline_per_match = totals["baseline"] / n_seeds if baseline is not None else None
    clone_per_match = totals["clone"] / n_seeds if include_clone else None
    gap = (
        candidate_per_match - baseline_per_match
        if baseline_per_match is not None
        else None
    )
    per_seed_gaps = [
        row["candidate_per_match"] - row["baseline_per_match"]
        for row in seed_rows
        if row["baseline_per_match"] is not None
    ]
    per_seed_noise = [
        row["clone_per_match"] - row["baseline_per_match"]
        for row in seed_rows
        if row["clone_per_match"] is not None
    ]

    negative = [item for item in per_opponent if item["candidate"] < 0]
    worst_delta = sorted(
        (item for item in per_opponent if item["delta"] is not None),
        key=lambda item: item["delta"],
    )[:tail]

    report = {
        "candidate": "candidate",
        "baseline": "baseline" if baseline is not None else None,
        "clone": include_clone,
        "pool": pool_name,
        "pool_size": len(pool),
        "size": n_opponents,
        "n_agents": n_agents,
        "seeds": list(seeds),
        "rounds": cfg.rounds,
        "faults": faults,
        "elapsed": elapsed,
        "candidate_per_match": candidate_per_match,
        "baseline_per_match": baseline_per_match,
        "clone_per_match": clone_per_match,
        "gap": gap,
        "worst_seed_gap": min(per_seed_gaps) if per_seed_gaps else None,
        "clone_noise": (
            clone_per_match - baseline_per_match
            if clone_per_match is not None
            else None
        ),
        "worst_seed_noise": min(per_seed_noise) if per_seed_noise else None,
        "rank": sum(row["rank"] for row in seed_rows) / n_seeds,
        "per_seed": seed_rows,
        "per_opponent": per_opponent,
        "tails": {
            "negative": negative,
            "worst_delta": worst_delta,
            "tail_size": tail,
        },
        "by_tag": by_tag,
    }

    if variants:
        variant_rows = []
        for label, overrides in CONFIG_VARIANTS:
            sub = evaluate_diagnose(
                candidate,
                baseline=baseline,
                clone=clone,
                pool=pool,
                pool_name=pool_name,
                size=size,
                seeds=(seeds[0],),
                cfg=replace(cfg, **overrides),
                variants=False,
                tail=tail,
            )
            variant_rows.append(
                {
                    "name": label,
                    "candidate_per_match": sub["candidate_per_match"],
                    "baseline_per_match": sub["baseline_per_match"],
                    "clone_per_match": sub["clone_per_match"],
                    "gap": sub["gap"],
                    "clone_noise": sub["clone_noise"],
                    "faults": sub["faults"],
                }
            )
        report["variants"] = variant_rows

    return report


def check_gate(report, min_gain=1.0, max_worst_drop=1.0):
    """按 plan 0001 的 T1 门槛检查；返回 ``(passed, reasons)``。"""
    reasons = []
    if report.get("gap") is None:
        return False, ["no baseline: cannot check gap"]
    if report["gap"] < min_gain:
        reasons.append(
            f"mean gap {report['gap']:+.2f} < required {min_gain:+.2f}"
        )
    worst = report.get("worst_seed_gap")
    if worst is not None and worst < -max_worst_drop:
        reasons.append(
            f"worst seed gap {worst:+.2f} < allowed {-max_worst_drop:+.2f}"
        )
    return not reasons, reasons


def _format_seed_table(report):
    lines = []
    for row in report["per_seed"]:
        base = row["baseline_per_match"]
        clone = row["clone_per_match"]
        base_text = "n/a" if base is None else f"{base:.2f}"
        clone_text = "n/a" if clone is None else f"{clone:.2f}"
        gap = (
            f"{row['candidate_per_match'] - base:+.2f}"
            if base is not None
            else "n/a"
        )
        noise = (
            f"{clone - base:+.2f}"
            if clone is not None and base is not None
            else "n/a"
        )
        lines.append(
            f"  seed {row['seed']}: candidate {row['candidate_per_match']:.2f} "
            f"baseline {base_text} clone {clone_text} "
            f"gap {gap} noise {noise}"
        )
    return lines


def format_report(report):
    """把报告渲染成终端可读的文本。"""
    lines = []
    lines.append(
        f"pool: {report['pool']} ({report['pool_size']} strategies); "
        f"sample {report['size']}/seed; {report['n_agents']} players; "
        f"seeds {report['seeds']}"
    )
    lines.append(
        f"candidate: {report['candidate_per_match']:.2f} per match, "
        f"mean rank {report['rank']:.1f}/{report['n_agents']}"
    )
    if report["baseline_per_match"] is not None:
        lines.append(
            f"baseline: {report['baseline_per_match']:.2f} per match; "
            f"gap {report['gap']:+.2f}; worst seed {report['worst_seed_gap']:+.2f}"
        )
        if report["clone_per_match"] is not None:
            lines.append(
                f"clone: {report['clone_per_match']:.2f} per match; "
                f"noise {report['clone_noise']:+.2f}; "
                f"worst noise {report['worst_seed_noise']:+.2f}"
            )
    lines.append(f"faults: {report['faults']}; elapsed: {report['elapsed']:.1f} s")
    lines.append("per seed:")
    lines.extend(_format_seed_table(report))

    lines.append(f"worst {report['tails']['tail_size']} by candidate score:")
    for item in sorted(report["per_opponent"], key=lambda x: x["candidate"])[
        : report["tails"]["tail_size"]
    ]:
        tags = ",".join(item["tags"]) or "-"
        lines.append(f"  {item['candidate']:8.2f}  {item['name']:<38} [{tags}]")

    worst_delta = report["tails"]["worst_delta"]
    if worst_delta:
        lines.append(f"worst {report['tails']['tail_size']} by delta vs baseline:")
        for item in worst_delta:
            tags = ",".join(item["tags"]) or "-"
            lines.append(
                f"  {item['delta']:+7.2f}  {item['name']:<38} [{tags}]"
            )

    if report["by_tag"]:
        lines.append("by tag:")
        lines.append(f"  {'tag':<14}{'n':>4}{'candidate':>11}{'baseline':>10}{'delta':>9}")
        for tag, bucket in sorted(
            report["by_tag"].items(), key=lambda pair: pair[1]["delta"]
        ):
            lines.append(
                f"  {tag:<14}{bucket['n']:>4}{bucket['candidate']:>11.2f}"
                f"{bucket['baseline']:>10.2f}{bucket['delta']:>+9.2f}"
            )

    for row in report.get("variants", ()):
        gap = "n/a" if row["gap"] is None else f"{row['gap']:+.2f}"
        noise = (
            "n/a" if row["clone_noise"] is None else f"{row['clone_noise']:+.2f}"
        )
        lines.append(
            f"variant {row['name']:<24} gap {gap} noise {noise}"
        )

    return "\n".join(lines)


def _parse_seeds(text):
    seeds = tuple(int(part) for part in text.split(",") if part.strip())
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def main(argv=None):
    parser = argparse.ArgumentParser(description="全池诊断/评测（plan 0001 / T2）")
    parser.add_argument("--agent", required=True, help="候选策略写法，见 ipd.eval.spec")
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--no-clone", action="store_true", help="不加基线克隆席位")
    parser.add_argument("--pool", choices=POOL_CHOICES, default="holdout")
    parser.add_argument(
        "--size", type=int, default=None, help="每种子对手数；默认全池"
    )
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--variants", action="store_true", help="跑四种 config 变体")
    parser.add_argument("--tail", type=int, default=DEFAULT_TAIL)
    parser.add_argument("--hide-prob", type=float, default=0.1)
    parser.add_argument("--hide-scope", choices=("match", "step"), default="match")
    parser.add_argument("--history-update", choices=("match", "round"), default="match")
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--gate", action="store_true", help="按 T1 门槛检查并返回非零")
    parser.add_argument("--min-gain", type=float, default=1.0)
    parser.add_argument("--max-worst-drop", type=float, default=1.0)
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args(argv)

    candidate = load_agent(args.agent)
    baseline = None if args.no_baseline else load_agent(args.baseline)
    pool = list(HOLDOUT_AGENTS) if args.pool == "holdout" else canonical()
    cfg = TournamentConfig(
        hide_prob=args.hide_prob,
        hide_scope=args.hide_scope,
        history_update=args.history_update,
        shuffle_schedule=not args.no_shuffle,
    )
    report = evaluate_diagnose(
        candidate,
        baseline=baseline,
        clone=not args.no_clone,
        pool=pool,
        pool_name=args.pool,
        size=args.size,
        seeds=_parse_seeds(args.seeds),
        cfg=cfg,
        variants=args.variants,
        tail=args.tail,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_report(report))

    if args.gate:
        passed, reasons = check_gate(report, args.min_gain, args.max_worst_drop)
        print("GATE:", "PASS" if passed else "FAIL")
        for reason in reasons:
            print("  -", reason)
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
