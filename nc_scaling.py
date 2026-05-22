#!/usr/bin/env python3
"""NC scaling efficiency — measures aggregate throughput across 1-4 logical NCs.

For each NC count (1, 2-same-die, 2-cross-die, 4), runs a matrix size sweep to
produce a full picture: how MFU and scaling efficiency vary with both compute
load (matrix size) and parallelism (NC count).

Reports:
  - Per-config absolute TFLOPS and MFU%
  - Scaling efficiency = aggregate_tflops / (N × single_nc_tflops)
  - Bandwidth-limited vs compute-limited regime identification

Each NC runs in its own OS process with NEURON_RT_VISIBLE_CORES pinned.

Usage:
    python3 nc_scaling.py [--sizes 4096 8192 16384] [--warmup 3] [--reps 10]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import socket
import statistics
import sys
import time
from typing import Any


# Die topology for trn2.3xlarge with lnc=2:
#   Die 0: LNC 0, LNC 1
#   Die 1: LNC 2, LNC 3
CONFIGS = {
    "1nc":            [0],
    "2nc_same_die":   [0, 1],
    "2nc_cross_die":  [0, 2],
    "3nc":            [0, 1, 2],
    "4nc_full_chip":  [0, 1, 2, 3],
}

_VISIBLE_CORES_BY_SLOT = ["0", "1", "2", "3"]

PEAK_PER_LNC = 167.0  # TFLOPS bf16 per logical NC


def _worker(
    slot: int,
    size: int,
    ready_q: "mp.Queue[int]",
    go_ev: "mp.Event",
    result_q: "mp.Queue[dict[str, Any]]",
    warmup: int,
    reps: int,
    backend: str,
) -> None:
    os.environ["NEURON_RT_VISIBLE_CORES"] = _VISIBLE_CORES_BY_SLOT[slot]

    import torch
    import torch.nn as nn
    import torch_neuronx  # noqa: F401

    dev = torch.device("neuron:0")
    _ = torch.zeros(1, device=dev)

    mod = nn.Linear(size, size, bias=False, dtype=torch.bfloat16).to(dev)
    compiled = torch.compile(mod, backend=backend, dynamic=False)
    x = torch.randn(size, size, dtype=torch.bfloat16, device=dev)

    for _ in range(max(warmup, 2)):
        compiled(x)
        torch_neuronx.synchronize()

    ready_q.put(slot)
    go_ev.wait()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x)
        torch_neuronx.synchronize()
        times.append(time.perf_counter() - t0)

    flops = 2 * size * size * size
    med = statistics.median(times)
    result_q.put({
        "slot": slot,
        "visible_cores": _VISIBLE_CORES_BY_SLOT[slot],
        "size": size,
        "median_us": med * 1e6,
        "achieved_tflops": flops / med / 1e12,
        "times_ms": [t * 1e3 for t in times],
    })


def run_config(
    config_name: str,
    slots: list[int],
    size: int,
    warmup: int,
    reps: int,
    backend: str,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    ready_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()
    go_ev = ctx.Event()

    procs = [
        ctx.Process(
            target=_worker,
            args=(s, size, ready_q, go_ev, result_q, warmup, reps, backend),
            daemon=True,
        )
        for s in slots
    ]
    for p in procs:
        p.start()

    seen: set[int] = set()
    deadline = time.time() + 900
    while len(seen) < len(slots):
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("workers did not reach ready state")
        s = ready_q.get(timeout=remaining)
        seen.add(s)

    go_ev.set()

    results: list[dict[str, Any]] = []
    for _ in slots:
        results.append(result_q.get(timeout=300))
    for p in procs:
        p.join(timeout=60)

    results.sort(key=lambda r: r["slot"])
    aggregate = sum(r["achieved_tflops"] for r in results)
    return {
        "config": config_name,
        "slots": slots,
        "size": size,
        "num_ncs": len(slots),
        "per_nc_results": results,
        "aggregate_tflops": aggregate,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="NC scaling efficiency sweep")
    p.add_argument("--sizes", nargs="+", type=int,
                   default=[2048, 4096, 8192, 16384],
                   help="Square matrix sizes to sweep")
    p.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                   choices=list(CONFIGS.keys()),
                   help="NC configurations to test")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--peak-per-lnc", type=float, default=PEAK_PER_LNC, dest="peak_per_lnc")
    p.add_argument("--compile-backend", default="neuron", dest="backend")
    p.add_argument("--output", default="/tmp/nc_scaling.json")
    args = p.parse_args()

    print(f"[scaling] hostname={socket.gethostname()}")
    print(f"[scaling] python={sys.version.split()[0]}")
    print(f"[scaling] sizes={args.sizes} configs={args.configs}")
    print(f"[scaling] peak_per_lnc={args.peak_per_lnc} TFLOPS")
    print(f"[scaling] warmup={args.warmup} reps={args.reps} backend={args.backend}")

    all_results: list[dict[str, Any]] = []

    for size in args.sizes:
        print(f"\n{'=' * 60}")
        print(f"[scaling] SIZE = {size}x{size}x{size}")
        print(f"{'=' * 60}")

        for config_name in args.configs:
            slots = CONFIGS[config_name]
            print(f"\n[scaling]   config={config_name} NCs={slots} ...")
            try:
                r = run_config(config_name, slots, size, args.warmup, args.reps, args.backend)
                all_results.append(r)
                per = ", ".join(f"{x['achieved_tflops']:.1f}" for x in r["per_nc_results"])
                print(f"[scaling]   aggregate={r['aggregate_tflops']:.1f} TF/s  per=[{per}]")
            except Exception as exc:
                print(f"[scaling]   FAIL: {type(exc).__name__}: {exc}")
                all_results.append({
                    "config": config_name, "slots": slots, "size": size,
                    "num_ncs": len(slots), "status": "fail",
                    "error": f"{type(exc).__name__}: {exc}"[:512],
                })

    # Compute scaling efficiency relative to single-NC baseline at each size
    for r in all_results:
        if "aggregate_tflops" not in r:
            continue
        # Find single-NC result at the same size
        baseline = next(
            (x["aggregate_tflops"] for x in all_results
             if x.get("config") == "1nc" and x.get("size") == r["size"]
             and "aggregate_tflops" in x),
            None,
        )
        num_ncs = r["num_ncs"]
        r["ideal_tflops"] = args.peak_per_lnc * num_ncs
        r["mfu_pct"] = r["aggregate_tflops"] / r["ideal_tflops"] * 100.0
        if baseline and baseline > 0:
            r["scaling_efficiency"] = r["aggregate_tflops"] / (num_ncs * baseline)
            r["speedup_vs_1nc"] = r["aggregate_tflops"] / baseline
        else:
            r["scaling_efficiency"] = None
            r["speedup_vs_1nc"] = None

    # Summary table
    print(f"\n\n{'=' * 90}")
    print(f"{'CONFIG':<18} {'SIZE':>6} {'NCs':>3} {'AGG TF/s':>9} {'IDEAL':>7} "
          f"{'MFU%':>6} {'SPEEDUP':>8} {'EFFICIENCY':>10}")
    print(f"{'-' * 90}")
    for r in all_results:
        if "aggregate_tflops" not in r:
            print(f"{r['config']:<18} {r['size']:>6} {r['num_ncs']:>3} "
                  f"{'FAIL':>9}")
            continue
        eff = f"{r['scaling_efficiency']:.0%}" if r.get("scaling_efficiency") else "—"
        spd = f"{r['speedup_vs_1nc']:.2f}x" if r.get("speedup_vs_1nc") else "—"
        print(f"{r['config']:<18} {r['size']:>6} {r['num_ncs']:>3} "
              f"{r['aggregate_tflops']:>9.1f} {r['ideal_tflops']:>7.0f} "
              f"{r['mfu_pct']:>6.1f} {spd:>8} {eff:>10}")
    print(f"{'=' * 90}")

    # Identify bottleneck regime per size
    print(f"\n[scaling] REGIME ANALYSIS:")
    for size in args.sizes:
        size_results = [r for r in all_results
                        if r.get("size") == size and "aggregate_tflops" in r]
        if not size_results:
            continue
        single = next((r for r in size_results if r["config"] == "1nc"), None)
        full = next((r for r in size_results if r["config"] == "4nc_full_chip"), None)
        if single and full:
            eff = full["scaling_efficiency"]
            if eff and eff > 0.9:
                regime = "COMPUTE-BOUND (scales well)"
            elif eff and eff > 0.75:
                regime = "MIXED (partial HBM contention)"
            else:
                regime = "BANDWIDTH-LIMITED (HBM contention dominates)"
            print(f"  size={size}: 4NC efficiency={eff:.0%} → {regime}")

    payload = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "backend": args.backend,
        "peak_per_lnc_tflops": args.peak_per_lnc,
        "warmup": args.warmup,
        "reps": args.reps,
        "sizes": args.sizes,
        "configs": {k: v for k, v in CONFIGS.items() if k in args.configs},
        "results": all_results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[scaling] wrote {args.output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
