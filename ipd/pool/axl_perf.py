"""测量 Axelrod 适配策略的单次决策耗时，列出最慢的若干个。

    PYTHONPATH=. .venv/bin/python -m ipd.pool.axl_perf [--long] [--top 15]
"""

import argparse
import random
import time

from ipd.pool.axl import axelrod_agents
from ipd.sim import rules


def time_agent(agent, matches=4, seed=0):
    """对随机对手打若干场，返回 (平均毫秒, 最坏毫秒)。"""
    rng = random.Random(seed)
    random.seed(seed)
    times = []
    for rs in range(matches):
        my, opp = [], []
        for length in range(rules.TURNS):
            start = time.perf_counter()
            move = agent.fn(rs, length, 1, tuple(my), 2, tuple(opp), rules.REPAY, None, None)
            times.append((time.perf_counter() - start) * 1000)
            my.append(move)
            opp.append(rng.choice("cb"))
    return sum(times) / len(times), max(times)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--long", action="store_true", help="包含 long_run_time 策略")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    results = sorted(((a.name, *time_agent(a)) for a in axelrod_agents(args.long)),
                     key=lambda r: -r[2])
    print(f"{len(results)} strategies")
    print(f"{'mean ms':>8}  {'max ms':>8}  name")
    for name, mean, worst in results[: args.top]:
        print(f"{mean:>8.3f}  {worst:>8.3f}  {name}")
    means = sorted(r[1] for r in results)
    print(f"median of means: {means[len(means) // 2]:.3f} ms")


if __name__ == "__main__":
    main()
