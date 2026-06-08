#!/usr/bin/env python3
"""Distributed matmul + collective benchmark for 2 NDs on trn2.48xlarge.

Simulates tensor-parallel workload: each ND computes a large matmul then
performs a collective (all_reduce / reduce_scatter) on the result.
This saturates both TensorE (compute) and CC/NeuronLink (communication).

Launched via torchrun: torchrun --nproc_per_node=2 collective_benchmark.py

Reports:
  - matmul-only TFLOPS per ND
  - matmul+collective TFLOPS per ND (effective throughput with comm overhead)
  - collective overhead as % of total time
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
import torch.nn as nn
import torch_neuronx  # noqa: F401


def _sync():
    torch.neuron.synchronize()


PEAK_PER_ND = 167.0  # TFLOPS bf16 per ND (1 logical NC with lnc=2)


class MatmulAllReduce(nn.Module):
    def __init__(self, size: int, group):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
        self.group = group

    def forward(self, x):
        y = self.linear(x)
        return funcol.all_reduce(y, reduceOp="sum", group=self.group)


class MatmulReduceScatter(nn.Module):
    def __init__(self, size: int, group):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
        self.group = group

    def forward(self, x):
        y = self.linear(x)
        return funcol.reduce_scatter_tensor(y, "sum", scatter_dim=0, group=self.group)


class MatmulOnly(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.linear(x)


def bench_one(
    name: str,
    mod: nn.Module,
    x: torch.Tensor,
    flops: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    torch._dynamo.reset()
    compiled = torch.compile(mod, backend="neuron", dynamic=False)

    for _ in range(warmup):
        compiled(x)
        _sync()

    dist.barrier()

    times_us: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x)
        _sync()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    med_us = statistics.median(times_us)
    achieved_tflops = flops / med_us / 1e6
    mfu_pct = achieved_tflops / PEAK_PER_ND * 100.0

    return {
        "name": name,
        "median_us": med_us,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "achieved_tflops": achieved_tflops,
        "mfu_pct": mfu_pct,
        "times_us": times_us,
    }


def bench_sustain(
    mod: nn.Module,
    x: torch.Tensor,
    duration_s: float,
    rank: int,
) -> dict[str, Any]:
    """Run compiled model in a tight loop for a fixed duration."""
    torch._dynamo.reset()
    compiled = torch.compile(mod, backend="neuron", dynamic=False)

    # Warmup
    for _ in range(5):
        compiled(x)
        _sync()

    dist.barrier()

    count = 0
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < duration_s:
        compiled(x)
        _sync()
        count += 1

    elapsed = time.perf_counter() - t_start
    return {"count": count, "elapsed_s": elapsed, "iters_per_sec": count / elapsed}


def main() -> None:
    p = argparse.ArgumentParser(description="Distributed matmul + collective benchmark")
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[4096, 8192, 16384],
                   help="Square matrix sizes (M=K=N)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--sustain", type=int, default=0,
                   help="Seconds to run sustained load (0=off). Runs largest size in a tight loop.")
    p.add_argument("--output", default="/tmp/collective_bench.json")
    args = p.parse_args()

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # With lnc=2 and 2 NDs: 4 logical NCs total (0,1 on ND0; 2,3 on ND1).
    # torchrun --nproc_per_node=4 gives ranks 0-3, each maps 1:1 to a logical NC.
    torch.neuron.set_device(rank)

    device = f"neuron:{rank}"
    _ = torch.zeros(1, device=device)

    if rank == 0:
        print(f"[bench] hostname={socket.gethostname()}")
        print(f"[bench] python={sys.version.split()[0]}"
              f"  torch={torch.__version__}"
              f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
        print(f"[bench] world_size={world_size}  backend=neuron")
        print(f"[bench] sizes={args.sizes}  warmup={args.warmup}  reps={args.reps}")
        print(f"[bench] peak_per_nd={PEAK_PER_ND} TFLOPS bf16")

    group = dist.group.WORLD
    results: list[dict[str, Any]] = []

    for size in args.sizes:
        flops = 2 * size * size * size
        x = torch.randn(size, size, dtype=torch.bfloat16, device=device)

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"[bench] SIZE={size}x{size}  flops={flops/1e12:.2f} TFLOPS")
            print(f"{'=' * 70}")

        # 1) Matmul only — baseline compute
        mod_compute = MatmulOnly(size).to(device)
        r_compute = bench_one("matmul_only", mod_compute, x, flops,
                              args.warmup, args.reps)
        r_compute["size"] = size
        results.append(r_compute)
        if rank == 0:
            print(f"[bench]   matmul_only:        {r_compute['median_us']:>8.0f} us  "
                  f"{r_compute['achieved_tflops']:.1f} TF/s  MFU={r_compute['mfu_pct']:.1f}%")

        # 2) Matmul + all_reduce
        mod_ar = MatmulAllReduce(size, group).to(device)
        r_ar = bench_one("matmul+all_reduce", mod_ar, x, flops,
                         args.warmup, args.reps)
        r_ar["size"] = size
        overhead = (r_ar["median_us"] - r_compute["median_us"]) / r_ar["median_us"] * 100
        r_ar["comm_overhead_pct"] = overhead
        results.append(r_ar)
        if rank == 0:
            print(f"[bench]   matmul+all_reduce:  {r_ar['median_us']:>8.0f} us  "
                  f"{r_ar['achieved_tflops']:.1f} TF/s  MFU={r_ar['mfu_pct']:.1f}%  "
                  f"comm_overhead={overhead:.1f}%")

        # 3) Matmul + reduce_scatter
        mod_rs = MatmulReduceScatter(size, group).to(device)
        r_rs = bench_one("matmul+reduce_scatter", mod_rs, x, flops,
                         args.warmup, args.reps)
        r_rs["size"] = size
        overhead = (r_rs["median_us"] - r_compute["median_us"]) / r_rs["median_us"] * 100
        r_rs["comm_overhead_pct"] = overhead
        results.append(r_rs)
        if rank == 0:
            print(f"[bench]   matmul+red_scatter: {r_rs['median_us']:>8.0f} us  "
                  f"{r_rs['achieved_tflops']:.1f} TF/s  MFU={r_rs['mfu_pct']:.1f}%  "
                  f"comm_overhead={overhead:.1f}%")

    # Sustained load: compile once, hammer for --sustain seconds
    if args.sustain > 0:
        size = args.sizes[-1]
        x = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        flops = 2 * size * size * size

        mod_ar = MatmulAllReduce(size, group).to(device)
        if rank == 0:
            print(f"\n[bench] SUSTAINED LOAD: size={size} for {args.sustain}s "
                  f"(matmul+all_reduce) — watch neuron-top now")
        s = bench_sustain(mod_ar, x, args.sustain, rank)
        if rank == 0:
            tflops = flops * s["iters_per_sec"] / 1e12
            print(f"[bench]   {s['count']} iters in {s['elapsed_s']:.1f}s  "
                  f"= {s['iters_per_sec']:.1f} it/s  "
                  f"= {tflops:.1f} TF/s  MFU={tflops/PEAK_PER_ND*100:.1f}%")

    if rank == 0:
        print(f"\n\n{'=' * 90}")
        print(f"{'NAME':<24} {'SIZE':>6} {'LAT(us)':>9} {'TF/s':>7} {'MFU%':>6} {'COMM%':>6}")
        print(f"{'-' * 90}")
        for r in results:
            comm = f"{r['comm_overhead_pct']:.1f}" if "comm_overhead_pct" in r else "—"
            print(f"{r['name']:<24} {r['size']:>6} "
                  f"{r['median_us']:>9.0f} "
                  f"{r['achieved_tflops']:>7.1f} "
                  f"{r['mfu_pct']:>6.1f} "
                  f"{comm:>6}")
        print(f"{'=' * 90}")

        payload = {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
            "world_size": world_size,
            "peak_per_nd_tflops": PEAK_PER_ND,
            "warmup": args.warmup,
            "reps": args.reps,
            "sizes": args.sizes,
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[bench] wrote {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
