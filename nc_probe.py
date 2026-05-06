#!/usr/bin/env python3
"""
Trainium 2 NC topology probe.

The NRT (Neuron RunTime) allocates ALL logical NCs to the first process that
initialises it (Requested:4 Available:0 in the multi-process attempt).
To run N processes in parallel each on its own logical NC we must partition the
chip before NRT initialises by setting NEURON_RT_VISIBLE_CORES *before* any
neuron import in each spawned child.

Hardware: trn2.3xlarge = 1 chip, 2 dies, 4 physical NCs per die = 8 physical NCs.
lnc=2 → 2 logical NCs per die → 4 logical NCs per chip.
chip peak bf16 ≈ 667 TF → per logical NC ≈ 167 TF.

NEURON_RT_VISIBLE_CORES uses logical NC group indices (0-based).
With lnc=2 the chip has 4 logical NC groups (indices 0-3); groups ≥4 map to
non-existent physical NCs and fail at NRT init.
  slot 0 → VISIBLE_CORES=0  (logical NC group 0)
  slot 1 → VISIBLE_CORES=1  (logical NC group 1)
  slot 2 → VISIBLE_CORES=2  (logical NC group 2)
  slot 3 → VISIBLE_CORES=3  (logical NC group 3)

Expected outcomes (if neuron:0 = 1 logical NC at ~167 TF peak):
  1 process  (lnc 0)         → ~134 TF/s
  2 processes (lnc 0 + 1)    → ~268 TF/s  (same die, independent)
  2 processes (lnc 0 + 2)    → ~268 TF/s  (different dies, independent)
  4 processes (lnc 0+1+2+3)  → ~537 TF/s  (full chip)

If we get the same TFLOPS regardless of N → all NCs share an execution unit.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import socket
import statistics
import sys
import time
from typing import Any

M, K, N = 16384, 16384, 16384
FLOPS = 2 * M * K * N

# NEURON_RT_VISIBLE_CORES uses logical NC group indices (0-based).
# With lnc=2, the chip has 4 logical NC groups (0-3).
# Slot i → VISIBLE_CORES value that restricts NRT to exactly that logical NC group.
# Groups 0-3 exhaust the chip; groups 4+ map to non-existent physical NCs.
_VISIBLE_CORES_BY_SLOT = ["0", "1", "2", "3"]


def _worker(
    slot: int,
    ready_q: "mp.Queue[int]",
    go_ev: "mp.Event",
    result_q: "mp.Queue[dict[str, Any]]",
    warmup: int,
    reps: int,
) -> None:
    # Set NEURON_RT_VISIBLE_CORES BEFORE any neuron/torch import so NRT only
    # initialises against the assigned physical NC pair.
    import os
    visible = _VISIBLE_CORES_BY_SLOT[slot]
    os.environ["NEURON_RT_VISIBLE_CORES"] = visible
    print(f"[probe worker:{slot}] NEURON_RT_VISIBLE_CORES={visible}", flush=True)

    import torch
    import torch.nn as nn
    import torch_neuronx  # noqa: F401

    class _Lin(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Linear(N, N, bias=False, dtype=torch.bfloat16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.w(x)

    dev = torch.device("neuron:0")  # the only device in the restricted NRT view
    mod = _Lin().to(dev)
    compiled = torch.compile(mod, backend="neuron", dynamic=False)
    x = torch.randn(M, K, dtype=torch.bfloat16).to(dev)

    print(f"[probe worker:{slot}] compiling + warming up …", flush=True)
    for _ in range(max(warmup, 2)):
        compiled(x)
        torch_neuronx.synchronize()
    print(f"[probe worker:{slot}] ready (NCs {visible})", flush=True)
    ready_q.put(slot)

    go_ev.wait()

    times: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x)
        torch_neuronx.synchronize()
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    result_q.put({
        "slot": slot,
        "visible_cores": visible,
        "median_ms": med * 1e3,
        "tflops": FLOPS / med / 1e12,
        "times_ms": [t * 1e3 for t in times],
    })


def run_config(
    label: str, slots: list[int], warmup: int, reps: int
) -> dict[str, Any]:
    print(f"\n[probe] ─── {label}: slots {slots} "
          f"(NCs {[_VISIBLE_CORES_BY_SLOT[s] for s in slots]}) ───", flush=True)
    ctx = mp.get_context("spawn")
    ready_q: mp.Queue = ctx.Queue()
    result_q: mp.Queue = ctx.Queue()
    go_ev = ctx.Event()

    procs = [
        ctx.Process(
            target=_worker,
            args=(s, ready_q, go_ev, result_q, warmup, reps),
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
            raise TimeoutError("workers did not reach ready state in time")
        s = ready_q.get(timeout=remaining)
        seen.add(s)
        print(f"[probe]   slot {s} ready ({len(seen)}/{len(slots)})", flush=True)

    print("[probe]   → go (all workers firing simultaneously)", flush=True)
    go_ev.set()

    results: list[dict[str, Any]] = []
    for _ in slots:
        r = result_q.get(timeout=300)
        results.append(r)
    for p in procs:
        p.join(timeout=60)

    results.sort(key=lambda r: r["slot"])
    for r in results:
        print(
            f"[probe]   slot {r['slot']} (NCs {r['visible_cores']})  "
            f"median={r['median_ms']:.0f} ms  {r['tflops']:.1f} TF/s"
        )
    total = sum(r["tflops"] for r in results)
    print(f"[probe]   AGGREGATE = {total:.1f} TF/s", flush=True)
    return {"label": label, "slots": slots, "results": results, "total_tflops": total}


def main() -> None:
    p = argparse.ArgumentParser(description="TRN2 NC topology probe")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--reps",   type=int, default=5)
    p.add_argument("--output", default="/tmp/nc_probe.json")
    args = p.parse_args()

    print(f"[probe] hostname={socket.gethostname()}", flush=True)
    print(f"[probe] M={M} K={K} N={N}  warmup={args.warmup}  reps={args.reps}", flush=True)
    print(f"[probe] NC partitioning: {_VISIBLE_CORES_BY_SLOT}", flush=True)

    # Configs: (label, list of slot indices to run in parallel)
    configs: list[tuple[str, list[int]]] = [
        ("single_lnc0",          [0]),         # 1 logical NC baseline
        ("pair_lnc0_lnc1",       [0, 1]),      # 2 LNCs, same die
        ("pair_lnc0_lnc2",       [0, 2]),      # 2 LNCs, different dies
        ("all4_lnc0123",         [0, 1, 2, 3]),# full chip
    ]

    all_results: list[dict[str, Any]] = []
    for label, slots in configs:
        try:
            r = run_config(label, slots, args.warmup, args.reps)
            all_results.append(r)
        except Exception as exc:
            print(f"[probe] {label} FAILED: {type(exc).__name__}: {exc}")
            all_results.append({"label": label, "slots": slots, "error": str(exc)})

    print("\n[probe] ═══ SUMMARY ═══")
    baseline = next(
        (r["results"][0]["tflops"] for r in all_results
         if r.get("label") == "single_lnc0" and "results" in r),
        None,
    )
    for r in all_results:
        if "error" in r:
            print(f"  {r['label']:25s}: ERROR {r['error'][:100]}")
        else:
            per = ", ".join(f"{x['tflops']:.1f}" for x in r["results"])
            scale = f"  ({r['total_tflops'] / baseline:.2f}×)" if baseline else ""
            print(f"  {r['label']:25s}: {r['total_tflops']:6.1f} TF/s  per=[{per}]{scale}")

    payload: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "M": M, "K": K, "N": N,
        "warmup": args.warmup,
        "reps": args.reps,
        "nc_partition": _VISIBLE_CORES_BY_SLOT,
        "results": all_results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[probe] wrote {args.output}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
