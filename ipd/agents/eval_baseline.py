"""在 issue #3 规定的池子中评估手工基线。

池子与 `python -m ipd.sim.run --sample 80` 一致（canonical + 80 个随机采样对手），
再加入基线自身；默认跑 5 个种子（0–4），与实验 0001–0003 可比。

用法::

    python -m ipd.agents.eval_baseline
    python -m ipd.agents.eval_baseline --sample 40 --seeds 0 1 2
    python -m ipd.agents.eval_baseline --timing-reps 200

输出：基线在池中的名次与每场平均分、基线对 AllC/AllD/TFT/Gradual 的每场平均分、
基线单次决策耗时（完整 20×25 历史下的最坏/平均/中位数）。
"""

import argparse
import random
import time

from ipd.agents.baseline import baseline_agent, strategy
from ipd.pool import canonical, sample_population, uniquify
from ipd.sim import TournamentConfig, rules, run_tournament

# 实验记录 0004 需要单独报告的标准对手
FOCUS = ("all_c", "all_d", "tft", "gradual")


def build_pool(sample, pool_seed):
    """canonical + sample 个随机对手 + 基线；名字唯一。"""
    agents = canonical() + sample_population(random.Random(pool_seed), sample)
    agents.append(baseline_agent())
    return uniquify(agents)


def benchmark_decision(reps=200, seed=0):
    """在完整 20×25 历史下测单次决策耗时（秒），返回 (worst, mean, median)。"""
    rng = random.Random(seed)
    turns = rules.TURNS
    full = tuple(
        tuple(rng.choice("cb") for _ in range(turns)) for _ in range(rules.HISTORY_MATCHES)
    )
    # 逐步隐藏的完整历史（"m" 出现在任意位置）
    masked = tuple(
        tuple("m" if rng.random() < 0.1 else move for move in row) for row in full
    )
    empty = (rules.MISSING_ROW,) * rules.HISTORY_MATCHES
    # (length, my_decisions, opp_decisions, opp_history, opp_opp_history)
    cases = [
        (turns - 1, full[0][: turns - 1], full[0][1:], full, masked),
        (turns - 2, full[1][: turns - 2], full[1][2:], masked, full),
        (0, (), (), empty, empty),
    ]
    # 先用最坏情况把解释器状态预热，避免把首次调用的开销计入统计
    for case in cases:
        strategy(0, case[0], 1, case[1], 2, case[2], rules.REPAY, case[3], case[4])
    times = []
    for _ in range(reps):
        for case in cases:
            start = time.perf_counter()
            strategy(0, case[0], 1, case[1], 2, case[2], rules.REPAY, case[3], case[4])
            times.append(time.perf_counter() - start)
    times.sort()
    return times[-1], sum(times) / len(times), times[len(times) // 2]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample", type=int, default=80, help="随机采样对手数")
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--top", type=int, default=15, help="显示前若干名，0 为全部")
    parser.add_argument("--timing-reps", type=int, default=100)
    args = parser.parse_args()

    agents = build_pool(args.sample, args.pool_seed)
    cfg = TournamentConfig()
    me = next(i for i, a in enumerate(agents) if a.family == "baseline")

    totals = [0] * len(agents)
    faults = [0] * len(agents)
    pair_totals = [[0] * len(agents) for _ in range(len(agents))]
    start = time.perf_counter()
    for seed in args.seeds:
        result = run_tournament(agents, seed=seed, cfg=cfg)
        totals = [t + s for t, s in zip(totals, result.scores)]
        faults = [f + x for f, x in zip(faults, result.faults)]
        pair_totals = [
            [a + b for a, b in zip(ra, rb)] for ra, rb in zip(pair_totals, result.pair_scores)
        ]
    elapsed = time.perf_counter() - start

    n = len(agents)
    per_match = len(args.seeds) * cfg.rounds * (n - 1)
    order = sorted(range(n), key=lambda i: -totals[i])
    baseline_rank = order.index(me) + 1
    shown = order[: args.top] if args.top else order
    print(f"{n} agents, seeds={args.seeds}, pool-seed={args.pool_seed}, {elapsed:.1f}s")
    print(f"{'rank':>4}  {'avg/match':>9}  {'faults':>6}  name")
    for rank, i in enumerate(shown, 1):
        mark = "  <- baseline" if i == me else ""
        print(f"{rank:>4}  {totals[i] / per_match:>9.2f}  {faults[i]:>6}  {agents[i].name}{mark}")
    if args.top:
        print(f"\nbaseline: rank {baseline_rank}/{n}, avg/match {totals[me] / per_match:.2f}")

    print("\nbaseline vs standard opponents (avg score per match):")
    for name in FOCUS:
        try:
            j = agents.index(next(a for a in agents if a.name == name))
        except StopIteration:
            continue
        avg = pair_totals[me][j] / (len(args.seeds) * cfg.rounds)
        print(f"  {name:10s} {avg:8.2f}")

    worst, mean, median = benchmark_decision(args.timing_reps)
    print(
        f"\nbaseline decision time (full 20x25 history, {3 * args.timing_reps} calls): "
        f"worst {worst * 1e3:.3f} ms, mean {mean * 1e3:.3f} ms, median {median * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
