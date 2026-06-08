#!/usr/bin/env python3
"""Neuron collective ops microbenchmark — measures latency and bandwidth of
all_reduce, all_gather, and reduce_scatter across 2 NDs on trn2.48xlarge.

Uses torch.compile(backend="neuron") with functional collectives.
Launched via torchrun: torchrun --nproc_per_node=2 collective_benchmark.py

Reports per-op latency (us), algorithm bandwidth (GB/s), and bus bandwidth (GB/s).
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
import torch_neuronx  # noqa: F401


def _sync():
    torch.neuron.synchronize()


def _msg_sizes_bytes(start_exp: int, end_exp: int) -> list[int]:
    return [2**e for e in range(start_exp, end_exp + 1)]


def _make_all_reduce(group):
    def fn(x):
        return funcol.all_reduce(x, reduceOp="sum", group=group)
    return fn


def _make_all_gather(group):
    def fn(x):
        return funcol.all_gather_tensor(x, gather_dim=0, group=group)
    return fn


def _make_reduce_scatter(group):
    def fn(x):
        return funcol.reduce_scatter_tensor(x, "sum", scatter_dim=0, group=group)
    return fn


OPS = {
    "all_reduce": _make_all_reduce,
    "all_gather": _make_all_gather,
    "reduce_scatter": _make_reduce_scatter,
}


def _algbw(size_bytes: int, latency_s: float) -> float:
    """Algorithm bandwidth in GB/s."""
    return size_bytes / latency_s / 1e9 if latency_s > 0 else 0.0


def _busbw(op: str, size_bytes: int, latency_s: float, world_size: int) -> float:
    """Bus bandwidth in GB/s — corrects for ring/tree protocol overhead."""
    n = world_size
    if n <= 1:
        return 0.0
    algbw = _algbw(size_bytes, latency_s)
    if op == "all_reduce":
        return algbw * 2 * (n - 1) / n
    elif op in ("all_gather", "reduce_scatter"):
        return algbw * (n - 1) / n
    return algbw


def bench_collective(
    op_name: str,
    size_bytes: int,
    rank: int,
    world_size: int,
    warmup: int,
    reps: int,
) -> dict[str, Any]:
    """Benchmark a single collective op at a given message size."""
    device = f"neuron:{rank}"
    num_elements = size_bytes // 2  # bf16 = 2 bytes per element

    if op_name == "reduce_scatter":
        num_elements = (num_elements // world_size) * world_size

    x = torch.randn(num_elements, dtype=torch.bfloat16, device=device)
    actual_bytes = num_elements * 2

    group = dist.group.WORLD
    op_fn = OPS[op_name](group)
    compiled_fn = torch.compile(op_fn, backend="neuron", dynamic=False)

    for _ in range(warmup):
        compiled_fn(x)
        _sync()

    dist.barrier()

    times_us: list[float] = []
    for _ in range(reps):
        dist.barrier()
        t0 = time.perf_counter()
        compiled_fn(x)
        _sync()
        t1 = time.perf_counter()
        times_us.append((t1 - t0) * 1e6)

    med_us = statistics.median(times_us)
    med_s = med_us / 1e6
    algbw = _algbw(actual_bytes, med_s)
    busbw = _busbw(op_name, actual_bytes, med_s, world_size)

    return {
        "op": op_name,
        "size_bytes": actual_bytes,
        "num_elements": num_elements,
        "dtype": "bf16",
        "median_us": med_us,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "algbw_gbps": algbw,
        "busbw_gbps": busbw,
        "times_us": times_us,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Neuron collective ops benchmark")
    p.add_argument("--ops", nargs="+", default=list(OPS.keys()),
                   choices=list(OPS.keys()))
    p.add_argument("--min-exp", type=int, default=10,
                   help="Min message size as 2^N bytes (default: 10 = 1KB)")
    p.add_argument("--max-exp", type=int, default=30,
                   help="Max message size as 2^N bytes (default: 30 = 1GB)")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--output", default="/tmp/collective_bench.json")
    args = p.parse_args()

    dist.init_process_group("neuron")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.neuron.set_device(rank)

    device = f"neuron:{rank}"
    _ = torch.zeros(1, device=device)

    if rank == 0:
        print(f"[coll] hostname={socket.gethostname()}")
        print(f"[coll] python={sys.version.split()[0]}"
              f"  torch={torch.__version__}"
              f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
        print(f"[coll] world_size={world_size}  backend=neuron")
        print(f"[coll] ops={args.ops}  sizes=2^{args.min_exp}..2^{args.max_exp} bytes")
        print(f"[coll] warmup={args.warmup}  reps={args.reps}")

    sizes = _msg_sizes_bytes(args.min_exp, args.max_exp)
    results: list[dict[str, Any]] = []

    for op_name in args.ops:
        if rank == 0:
            print(f"\n{'=' * 70}")
            print(f"[coll] OP: {op_name}")
            print(f"{'=' * 70}")

        for size_bytes in sizes:
            try:
                r = bench_collective(op_name, size_bytes, rank, world_size,
                                     args.warmup, args.reps)
                results.append(r)
                if rank == 0:
                    print(f"[coll]   {size_bytes:>12,} B  "
                          f"lat={r['median_us']:>9.1f} us  "
                          f"algbw={r['algbw_gbps']:>7.2f} GB/s  "
                          f"busbw={r['busbw_gbps']:>7.2f} GB/s")
            except Exception as exc:
                results.append({
                    "op": op_name, "size_bytes": size_bytes,
                    "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                })
                if rank == 0:
                    print(f"[coll]   {size_bytes:>12,} B  FAIL: {exc}")

    if rank == 0:
        print(f"\n\n{'=' * 90}")
        print(f"{'OP':<16} {'SIZE':>12} {'LAT(us)':>10} {'ALGBW(GB/s)':>12} {'BUSBW(GB/s)':>12}")
        print(f"{'-' * 90}")
        for r in results:
            if "median_us" not in r:
                print(f"{r['op']:<16} {r['size_bytes']:>12,}  FAIL")
                continue
            print(f"{r['op']:<16} {r['size_bytes']:>12,} "
                  f"{r['median_us']:>10.1f} "
                  f"{r['algbw_gbps']:>12.2f} "
                  f"{r['busbw_gbps']:>12.2f}")
        print(f"{'=' * 90}")

        payload = {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "torch": torch.__version__,
            "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
            "world_size": world_size,
            "warmup": args.warmup,
            "reps": args.reps,
            "ops": args.ops,
            "sizes_bytes": sizes,
            "results": results,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[coll] wrote {args.output}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
