"""历史编码器性能基准（issue #5）。

    .venv/bin/python -m ipd.agents.bench_encoder
    .venv/bin/python -m ipd.agents.bench_encoder --runs 3 --reps 200 --threads 1

默认：CPU、batch=1、完整 20×25 历史、length=24；预热后测 `runs` 组 × `reps` 次，
报告每组的最坏/中位/平均耗时。只用训练端环境（需要 torch）。
"""

import argparse
import gc
import time

import torch

from ipd.agents.encoder import HistoryEncoder


def _inputs(batch, hidden, seed):
    generator = torch.Generator().manual_seed(seed)
    own = torch.randint(1, 3, (batch, 20, 25), generator=generator)
    opp = torch.randint(1, 3, (batch, 20, 25), generator=generator)
    if hidden:
        mask = torch.rand((batch, 20, 25), generator=generator) < hidden
        own = own.masked_fill(mask, 0)
        opp = opp.masked_fill(mask, 0)
    my = torch.randint(1, 3, (batch, 24), generator=generator)
    oppd = torch.randint(1, 3, (batch, 24), generator=generator)
    return own, opp, my, oppd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=float, default=0.0, help="逐步隐藏概率")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    model = HistoryEncoder().eval()
    own, opp, my, oppd = _inputs(args.batch, args.hidden, args.seed)
    print(
        f"HistoryEncoder dim={model.embed_dim}, params={model.num_parameters()}, "
        f"batch={args.batch}, hidden={args.hidden}, threads={args.threads}"
    )
    with torch.inference_mode():
        for _ in range(100):
            model(own, opp, my, oppd, 0, 24)
        for run in range(args.runs):
            times = []
            gc.disable()
            try:
                for _ in range(args.reps):
                    start = time.perf_counter()
                    model(own, opp, my, oppd, 0, 24)
                    times.append(time.perf_counter() - start)
            finally:
                gc.enable()
            times.sort()
            print(
                f"run {run}: worst {times[-1] * 1e3:.3f} ms, "
                f"median {times[len(times) // 2] * 1e3:.3f} ms, "
                f"mean {sum(times) / len(times) * 1e3:.3f} ms"
            )


if __name__ == "__main__":
    main()
