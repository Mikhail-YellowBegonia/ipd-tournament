"""测量 Axelrod 适配策略的性能。

报告每个策略的平均每决策时间，以及最慢的策略。
"""

import time
import statistics

try:
    import axelrod as axl
    AXELROD_AVAILABLE = True
except ImportError:
    AXELROD_AVAILABLE = False
    print("axelrod library not available")
    exit(0)

from ipd.pool.axl import axelrod_agents


def measure_strategy_performance(agent, num_decisions=100):
    """测量单个策略的平均决策时间（毫秒）。

    模拟一场完整对局（25 回合），重复多次以获得稳定测量。
    """
    timings = []

    for trial in range(num_decisions // 25 + 1):
        rs = trial
        for length in range(25):
            # 构造简单的历史（交替合作/背叛）
            my = tuple(["c", "b"][i % 2] for i in range(length))
            opp = tuple(["b", "c"][i % 2] for i in range(length))

            start = time.perf_counter()
            try:
                agent.fn(rs, length, 1, my, 2, opp, None, None, None)
                elapsed = time.perf_counter() - start
                timings.append(elapsed * 1000)  # 转换为毫秒
            except Exception:
                # 策略崩溃，记录为高时间
                timings.append(1000.0)

    if not timings:
        return float('inf')

    return statistics.mean(timings)


def main():
    print("正在加载 Axelrod 策略...")
    agents = axelrod_agents(include_long_run_time=False)
    print(f"已加载 {len(agents)} 个策略（排除 long_run_time）\n")

    print("测量性能（每策略 ~100 决策）...")
    results = []

    for i, agent in enumerate(agents):
        if (i + 1) % 10 == 0:
            print(f"  进度: {i + 1}/{len(agents)}")

        avg_time_ms = measure_strategy_performance(agent)
        results.append((agent.name, avg_time_ms))

    # 按时间排序
    results.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "="*70)
    print("性能报告：最慢的 20 个策略")
    print("="*70)
    print(f"{'策略名称':<40} {'平均时间 (ms)':<20}")
    print("-"*70)

    for name, avg_time_ms in results[:20]:
        print(f"{name:<40} {avg_time_ms:>15.4f}")

    print("\n" + "="*70)
    print(f"最快策略: {results[-1][0]} ({results[-1][1]:.4f} ms)")
    print(f"最慢策略: {results[0][0]} ({results[0][1]:.4f} ms)")
    print(f"中位数: {statistics.median([t for _, t in results]):.4f} ms")
    print("="*70)

    # 如果有超过 10ms 的策略，警告
    slow_strategies = [(n, t) for n, t in results if t > 10.0]
    if slow_strategies:
        print(f"\n警告: {len(slow_strategies)} 个策略平均决策时间 > 10ms")


if __name__ == "__main__":
    main()
