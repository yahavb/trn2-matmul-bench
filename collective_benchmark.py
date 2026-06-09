#!/usr/bin/env python3
"""Unified matmul + collective benchmark for trn2.

Three modes (all use torch.compile backend="neuron", bf16, lnc=2):
  1) single-core:  1 LNC, no TP, no collectives — pure compute baseline
  2) single-device: 4 LNCs on 1 ND, TP=4, collectives over die-to-die (400 GB/s)
  3) multi-device:  8 LNCs on 2 NDs, TP=8, collectives cross NeuronLink (128 GB/s)

Peak per LNC = 632 BF16 TFLOPS/chip ÷ 4 LNCs = 158 TFLOPS.

Usage:
  # Mode 1: single core
  python collective_benchmark.py --mode single-core --sizes 4096 8192 16384

  # Mode 2: single device, TP=4
  torchrun --nproc_per_node=4 collective_benchmark.py --mode single-device --sizes 4096 8192 16384

  # Mode 3: multi device, TP=8
  torchrun --nproc_per_node=8 collective_benchmark.py --mode multi-device --sizes 4096 8192 16384
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

PEAK_PER_LNC = 158.0  # 632 BF16 TFLOPS/chip ÷ 4 LNCs (lnc=2)


def _sync():
    torch.neuron.synchronize()


class MatmulAllReduce(nn.Module):
    def __init__(self, size: int, group):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
        self.group = group

    def forward(self, x):
        y = self.linear(x)
        return funcol.all_reduce(y, reduceOp="sum", group=self.group)


class MatmulAllGather(nn.Module):
    def __init__(self, size: int, group):
        super().__init__()
        self.linear = nn.Linear(size, size, bias=False, dtype=torch.bfloat16)
        self.group = group

    def forward(self, x):
        y = self.linear(x)
        return funcol.all_gather_tensor(y, gather_dim=0, group=self.group)


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
    use_dist: bool,
) -> dict[str, Any]:
    torch._dynamo.reset()
    compiled = torch.compile(mod, backend="neuron", dynamic=False)

    for _ in range(warmup):
        compiled(x)
        _sync()

    if use_dist:
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
    mfu_pct = achieved_tflops / PEAK_PER_LNC * 100.0

    return {
        "name": name,
        "median_us": med_us,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "achieved_tflops": achieved_tflops,
        "mfu_pct": mfu_pct,
        "times_us": times_us,
    }


def run_single_core(args) -> None:
    """Mode 1: single LNC, no distributed, pure compute."""
    os.environ.setdefault("NEURON_RT_NUM_CORES", "1")
    device = "neuron:0"
    _ = torch.zeros(1, device=device)

    print(f"[bench] MODE=single-core (1 LNC, no TP)")
    print(f"[bench] hostname={socket.gethostname()}")
    print(f"[bench] python={sys.version.split()[0]}"
          f"  torch={torch.__version__}"
          f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
    print(f"[bench] peak_per_lnc={PEAK_PER_LNC} TFLOPS bf16")
    print(f"[bench] sizes={args.sizes}  warmup={args.warmup}  reps={args.reps}")

    results: list[dict[str, Any]] = []

    for size in args.sizes:
        flops = 2 * size * size * size
        x = torch.randn(size, size, dtype=torch.bfloat16, device=device)

        print(f"\n{'=' * 70}")
        print(f"[bench] SIZE={size}x{size}  flops={flops/1e12:.2f} TFLOPS")
        print(f"{'=' * 70}")

        mod = MatmulOnly(size).to(device)
        r = bench_one("matmul_only", mod, x, flops, args.warmup, args.reps, use_dist=False)
        r["size"] = size
        results.append(r)
        print(f"[bench]   matmul_only:        {r['median_us']:>8.0f} us  "
              f"{r['achieved_tflops']:.1f} TF/s  MFU={r['mfu_pct']:.1f}%")

    _print_summary(results)
    _save_results(args, results, mode="single-core", world_size=1)


def run_distributed(args, mode: str) -> None:
    """Mode 2 & 3: multi-core with collectives."""
    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.neuron.set_device(rank)

    device = f"neuron:{rank}"
    _ = torch.zeros(1, device=device)

    if rank == 0:
        print(f"[bench] MODE={mode} (TP={world_size})")
        print(f"[bench] hostname={socket.gethostname()}")
        print(f"[bench] python={sys.version.split()[0]}"
              f"  torch={torch.__version__}"
              f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
        print(f"[bench] world_size={world_size}  backend=neuron")
        print(f"[bench] peak_per_lnc={PEAK_PER_LNC} TFLOPS bf16")
        print(f"[bench] sizes={args.sizes}  warmup={args.warmup}  reps={args.reps}")

    group = dist.group.WORLD
    results: list[dict[str, Any]] = []

    for size in args.sizes:
        flops = 2 * size * size * size
        x = torch.randn(size, size, dtype=torch.bfloat16, device=device)

        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"[bench] SIZE={size}x{size}  flops={flops/1e12:.2f} TFLOPS")
            print(f"{'=' * 70}")

        # matmul only — baseline
        mod_compute = MatmulOnly(size).to(device)
        r_compute = bench_one("matmul_only", mod_compute, x, flops,
                              args.warmup, args.reps, use_dist=True)
        r_compute["size"] = size
        results.append(r_compute)
        if rank == 0:
            print(f"[bench]   matmul_only:        {r_compute['median_us']:>8.0f} us  "
                  f"{r_compute['achieved_tflops']:.1f} TF/s  MFU={r_compute['mfu_pct']:.1f}%")

        # all_reduce
        mod_ar = MatmulAllReduce(size, group).to(device)
        r_ar = bench_one("matmul+all_reduce", mod_ar, x, flops,
                         args.warmup, args.reps, use_dist=True)
        r_ar["size"] = size
        overhead = (r_ar["median_us"] - r_compute["median_us"]) / r_ar["median_us"] * 100
        r_ar["comm_overhead_pct"] = overhead
        results.append(r_ar)
        if rank == 0:
            print(f"[bench]   matmul+all_reduce:  {r_ar['median_us']:>8.0f} us  "
                  f"{r_ar['achieved_tflops']:.1f} TF/s  MFU={r_ar['mfu_pct']:.1f}%  "
                  f"comm_overhead={overhead:.1f}%")

        # all_gather
        mod_ag = MatmulAllGather(size, group).to(device)
        r_ag = bench_one("matmul+all_gather", mod_ag, x, flops,
                         args.warmup, args.reps, use_dist=True)
        r_ag["size"] = size
        overhead = (r_ag["median_us"] - r_compute["median_us"]) / r_ag["median_us"] * 100
        r_ag["comm_overhead_pct"] = overhead
        results.append(r_ag)
        if rank == 0:
            print(f"[bench]   matmul+all_gather:  {r_ag['median_us']:>8.0f} us  "
                  f"{r_ag['achieved_tflops']:.1f} TF/s  MFU={r_ag['mfu_pct']:.1f}%  "
                  f"comm_overhead={overhead:.1f}%")

        # reduce_scatter (skip 16384 with TP=4 — exceeds RDH buffer)
        if not (size >= 16384 and world_size == 4):
            mod_rs = MatmulReduceScatter(size, group).to(device)
            r_rs = bench_one("matmul+reduce_scatter", mod_rs, x, flops,
                             args.warmup, args.reps, use_dist=True)
            r_rs["size"] = size
            overhead = (r_rs["median_us"] - r_compute["median_us"]) / r_rs["median_us"] * 100
            r_rs["comm_overhead_pct"] = overhead
            results.append(r_rs)
            if rank == 0:
                print(f"[bench]   matmul+red_scatter: {r_rs['median_us']:>8.0f} us  "
                      f"{r_rs['achieved_tflops']:.1f} TF/s  MFU={r_rs['mfu_pct']:.1f}%  "
                      f"comm_overhead={overhead:.1f}%")
        elif rank == 0:
            print(f"[bench]   matmul+red_scatter: SKIPPED (16384 + TP=4 exceeds RDH buffer)")

    if rank == 0:
        _print_summary(results)
        _save_results(args, results, mode=mode, world_size=world_size)

    dist.destroy_process_group()


def _print_summary(results: list[dict[str, Any]]) -> None:
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


def _save_results(args, results, mode, world_size) -> None:
    payload = {
        "mode": mode,
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
        "world_size": world_size,
        "peak_per_lnc_tflops": PEAK_PER_LNC,
        "warmup": args.warmup,
        "reps": args.reps,
        "sizes": args.sizes,
        "results": results,
    }
    output = args.output.replace(".json", f"_{mode}.json")
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[bench] wrote {output}")


def main() -> None:
    p = argparse.ArgumentParser(description="Unified matmul + collective benchmark")
    p.add_argument("--mode", choices=["single-core", "single-device", "multi-device"],
                   required=True)
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[4096, 8192, 16384])
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--output", default="/tmp/collective_bench.json")
    args = p.parse_args()

    if args.mode == "single-core":
        run_single_core(args)
    else:
        run_distributed(args, mode=args.mode)


if __name__ == "__main__":
    main()
