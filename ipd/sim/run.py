"""本地跑一次联赛并打印排名。

    python -m ipd.sim.run                     # 标准版本对手
    python -m ipd.sim.run --sample 30         # 再加 30 个随机采样的对手
    .venv/bin/python -m ipd.sim.run --axl     # 再加全部 axelrod 策略（需训练端环境）
"""

import argparse
import random
import time

from ipd.pool import canonical, sample_population, uniquify
from ipd.sim import TournamentConfig, run_tournament


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--sample", type=int, default=0, help="额外随机采样的对手数")
    parser.add_argument("--pool-seed", type=int, default=0)
    parser.add_argument("--no-canonical", action="store_true")
    parser.add_argument("--axl", action="store_true", help="加入 axelrod 策略（需安装 axelrod）")
    parser.add_argument("--top", type=int, default=0, help="只显示前若干名，0 为全部")
    args = parser.parse_args()

    agents = [] if args.no_canonical else canonical()
    agents += sample_population(random.Random(args.pool_seed), args.sample)
    if args.axl:
        from ipd.pool.axl import axelrod_agents
        agents += axelrod_agents()
    agents = uniquify(agents)
    cfg = TournamentConfig()

    totals = [0] * len(agents)
    faults = [0] * len(agents)
    start = time.perf_counter()
    for seed in args.seeds:
        result = run_tournament(agents, seed=seed, cfg=cfg)
        totals = [t + s for t, s in zip(totals, result.scores)]
        faults = [f + x for f, x in zip(faults, result.faults)]
    elapsed = time.perf_counter() - start

    n = len(agents)
    per_match = len(args.seeds) * cfg.rounds * (n - 1)
    order = sorted(range(n), key=lambda i: -totals[i])
    if args.top:
        order = order[: args.top]
    print(f"{n} agents, seeds={args.seeds}, {elapsed:.1f}s")
    print(f"{'rank':>4}  {'avg/match':>9}  {'faults':>6}  name")
    for rank, i in enumerate(order, 1):
        print(f"{rank:>4}  {totals[i] / per_match:>9.2f}  {faults[i]:>6}  {agents[i].name}")


if __name__ == "__main__":
    main()
