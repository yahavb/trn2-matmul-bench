#!/usr/bin/env python3
"""MFU headroom analysis — size scaling on a single logical NC.

Sweeps matrix sizes from 512 to 32768 (powers of two + intermediate steps) to
identify the compute saturation knee and quantify headroom (peak − achieved).

Produces a summary table and JSON with per-size MFU, achieved TFLOPS, and the
size at which diminishing returns begin (knee detection via second derivative).

Usage:
    python3 mfu_headroom.py [--min-size 512] [--max-size 32768] [--steps-per-octave 2]
    python3 mfu_headroom.py --sizes 1024 2048 4096 8192 16384
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch_neuronx

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False

_HAS_TRACE = hasattr(torch_neuronx, "trace")

_FORCE_NEURON_DEVICE = False


def _get_device() -> torch.device:
    if _FORCE_NEURON_DEVICE:
        return torch.device("neuron")
    return torch_xla.device() if _XLA_AVAILABLE else torch.device("neuron:0")


def _sync_device() -> None:
    if _FORCE_NEURON_DEVICE:
        torch.neuron.synchronize()
    elif hasattr(torch_neuronx, "synchronize"):
        torch_neuronx.synchronize()
    elif _XLA_AVAILABLE:
        torch_xla.sync()
        xm.wait_device_ops()
    else:
        raise RuntimeError("no sync API available")


def _generate_sizes(min_size: int, max_size: int, steps_per_octave: int) -> list[int]:
    sizes = []
    log_min = math.log2(min_size)
    log_max = math.log2(max_size)
    step = 1.0 / steps_per_octave
    val = log_min
    while val <= log_max + 1e-9:
        s = int(round(2 ** val))
        s = max(128, (s + 63) // 64 * 64)
        if s not in sizes:
            sizes.append(s)
        val += step
    return sizes


def bench_size(
    size: int, device: torch.device, compiled: Any, warmup: int, reps: int
) -> dict[str, Any]:
    x = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    for _ in range(warmup):
        compiled(x)
        _sync_device()

    times_us: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x)
        _sync_device()
        times_us.append((time.perf_counter() - t0) * 1e6)

    flops = 2 * size * size * size
    med_us = statistics.median(times_us)
    achieved = flops / med_us / 1e6  # TFLOPS
    return {
        "size": size,
        "flops": flops,
        "median_us": med_us,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "achieved_tflops": achieved,
        "times_us": times_us,
    }


def find_knee(results: list[dict[str, Any]]) -> int | None:
    """Find the knee point where MFU gains flatten (max second derivative of MFU)."""
    if len(results) < 3:
        return None
    mfu = [r["mfu_pct"] for r in results]
    d2 = []
    for i in range(1, len(mfu) - 1):
        d2.append(mfu[i + 1] - 2 * mfu[i] + mfu[i - 1])
    # Knee = point of maximum negative second derivative (concavity change)
    min_idx = 0
    for i, v in enumerate(d2):
        if v < d2[min_idx]:
            min_idx = i
    return results[min_idx + 1]["size"]


def main() -> None:
    p = argparse.ArgumentParser(description="MFU headroom — size scaling sweep")
    p.add_argument("--sizes", nargs="+", type=int, default=None,
                   help="Explicit list of square matrix sizes to test")
    p.add_argument("--min-size", type=int, default=512, dest="min_size")
    p.add_argument("--max-size", type=int, default=32768, dest="max_size")
    p.add_argument("--steps-per-octave", type=int, default=2, dest="steps_per_octave",
                   help="Intermediate sizes per power-of-two (1=powers only, 2=adds midpoints)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--peak-tflops", type=float, default=167.0, dest="peak_tflops",
                   help="Peak bf16 TFLOPS for 1 logical NC (default: 167 for trn2 lnc=2)")
    p.add_argument("--compile-backend", default=None, dest="compile_backend")
    p.add_argument("--output", default="/tmp/mfu_headroom.json")
    args = p.parse_args()

    global _FORCE_NEURON_DEVICE
    backend = args.compile_backend
    if backend is None:
        backend = "neuron" if not _HAS_TRACE else "openxla"
    if backend == "neuron":
        _FORCE_NEURON_DEVICE = True

    sizes = args.sizes if args.sizes else _generate_sizes(
        args.min_size, args.max_size, args.steps_per_octave
    )

    print(f"[headroom] python={sys.version.split()[0]} torch={torch.__version__}")
    print(f"[headroom] hostname={socket.gethostname()}")
    print(f"[headroom] backend={backend} peak={args.peak_tflops} TFLOPS")
    print(f"[headroom] sizes={sizes}")
    print(f"[headroom] warmup={args.warmup} reps={args.reps}")

    device = _get_device()
    _ = torch.zeros(1, device=device)
    print(f"[headroom] device initialized: {device}")

    # Compile a single nn.Linear for the largest size; dynamic=False means we
    # recompile per size. For size sweeps this is necessary since neuronx-cc
    # optimizes tiling per shape.
    results: list[dict[str, Any]] = []

    for size in sizes:
        print(f"\n[headroom] size={size}x{size}x{size} ...")
        mod = nn.Linear(size, size, bias=False, dtype=torch.bfloat16).to(device)
        compiled = torch.compile(mod, backend=backend, dynamic=False)

        # Warmup triggers compilation for this shape
        x_warmup = torch.randn(size, size, dtype=torch.bfloat16, device=device)
        try:
            for _ in range(args.warmup):
                compiled(x_warmup)
                _sync_device()
        except Exception as exc:
            print(f"[headroom]   COMPILE/WARMUP FAIL: {type(exc).__name__}: {exc}")
            results.append({"size": size, "status": "fail",
                            "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue

        try:
            r = bench_size(size, device, compiled, warmup=1, reps=args.reps)
        except Exception as exc:
            print(f"[headroom]   RUN FAIL: {type(exc).__name__}: {exc}")
            results.append({"size": size, "status": "fail",
                            "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue

        r["mfu_pct"] = r["achieved_tflops"] / args.peak_tflops * 100.0
        r["headroom_tflops"] = args.peak_tflops - r["achieved_tflops"]
        r["headroom_pct"] = 100.0 - r["mfu_pct"]
        r["status"] = "ok"
        del r["times_us"]
        results.append(r)
        print(f"[headroom]   {r['achieved_tflops']:.1f} TF/s  "
              f"MFU={r['mfu_pct']:.1f}%  headroom={r['headroom_pct']:.1f}%")

    # Summary
    ok_results = [r for r in results if r["status"] == "ok"]
    knee = find_knee(ok_results) if ok_results else None
    peak_achieved = max((r["achieved_tflops"] for r in ok_results), default=0)
    peak_mfu = max((r["mfu_pct"] for r in ok_results), default=0)

    print("\n" + "=" * 72)
    print(f"{'SIZE':>8} {'TF/s':>8} {'MFU%':>7} {'HEADROOM%':>10} {'STATUS'}")
    print("-" * 72)
    for r in results:
        if r["status"] == "ok":
            print(f"{r['size']:>8} {r['achieved_tflops']:>8.1f} {r['mfu_pct']:>7.1f} "
                  f"{r['headroom_pct']:>10.1f}   ok")
        else:
            print(f"{r['size']:>8} {'—':>8} {'—':>7} {'—':>10}   FAIL")
    print("-" * 72)
    print(f"Peak achieved: {peak_achieved:.1f} TF/s ({peak_mfu:.1f}% MFU)")
    if knee:
        print(f"Saturation knee: size={knee} (gains flatten beyond this point)")
    print(f"Hardware peak: {args.peak_tflops:.1f} TF/s")
    print(f"Max headroom: {args.peak_tflops - peak_achieved:.1f} TF/s "
          f"({100.0 - peak_mfu:.1f}%)")
    print("=" * 72)

    payload = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
        "backend": backend,
        "peak_tflops": args.peak_tflops,
        "warmup": args.warmup,
        "reps": args.reps,
        "sizes": sizes,
        "results": results,
        "summary": {
            "peak_achieved_tflops": peak_achieved,
            "peak_mfu_pct": peak_mfu,
            "saturation_knee_size": knee,
            "max_headroom_tflops": args.peak_tflops - peak_achieved,
            "max_headroom_pct": 100.0 - peak_mfu,
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[headroom] wrote {args.output}")


if __name__ == "__main__":
    main()
