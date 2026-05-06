"""Trainium 2 matmul peak microbenchmark — bf16 / fp8 / mxfp8.

Two compilation paths are measured independently for each shape × dtype:
  trace   — torch_neuronx.trace(): AOT XLA compilation, fixed input shape baked
             into a .neff artifact; trace time is reported separately.
  compile — torch.compile(backend=...): JIT, compilation happens on the
             first warmup call; backend auto-detected or set via --compile-backend.
             Defaults to "neuron" (Beta 2 torch_neuron_eager) or "openxla" (public).

Square matmuls from 1024^3 → 32768^3 (powers of two) plus WAN-shaped GEMMs.
Reports achieved TFLOP/s and MFU% (vs --peak-bf16-tflops / --peak-fp8-tflops).

Target: single Trainium 2 die (trn2.3xlarge = 4 NeuronCores, 1 device).
Set NEURON_RT_NUM_CORES=1 — libneuronxla PJRT auto-detects logical-neuroncore-
config=2 from hardware and passes 2 to nrt_allocate_neuron_cores; TRN2 NRT
requires 1 or a multiple of 8, so 2 is rejected.
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

_HAS_TRACE = hasattr(torch_neuronx, "trace")

# Beta 2 (torch_neuron_eager) registers torch.device("neuron") — no torch_xla.
# Public torch-neuronx uses XLA as the device backend.
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


def _get_device() -> torch.device:
    if _XLA_AVAILABLE:
        return torch_xla.device()
    return torch.device("neuron:0")


def _sync_device() -> None:
    """Sync and wait for all in-flight device ops.

    Prefers torch_neuronx.synchronize() (Beta 2) over the torch_xla pair.
    """
    if hasattr(torch_neuronx, "synchronize"):
        torch_neuronx.synchronize()
    elif _XLA_AVAILABLE:
        torch_xla.sync()
        xm.wait_device_ops()
    else:
        raise RuntimeError("no sync API available")


MATMUL_SHAPES: dict[str, tuple[int, int, int]] = {
    # Square: powers of two 1k → 32k
    "matmul.sq_1024":             (1024,  1024,  1024),
    "matmul.sq_2048":             (2048,  2048,  2048),
    "matmul.sq_4096":             (4096,  4096,  4096),
    "matmul.sq_8192":             (8192,  8192,  8192),
    "matmul.sq_16384":            (16384, 16384, 16384),
    "matmul.sq_32768":            (32768, 32768, 32768),
    # WAN-shaped GEMMs: 1.3B @ 480p
    "matmul.wan_qkv_1.3b_480p":  (1590, 1536,  4608),
    "matmul.wan_o_1.3b_480p":    (1590, 1536,  1536),
    "matmul.wan_ffn1_1.3b_480p": (1590, 1536,  8960),
    "matmul.wan_ffn2_1.3b_480p": (1590, 8960,  1536),
    # WAN-shaped GEMMs: 14B @ 480p / 720p
    "matmul.wan_qkv_14b_480p":   (1590, 5120, 15360),
    "matmul.wan_o_14b_480p":     (1590, 5120,  5120),
    "matmul.wan_ffn1_14b_480p":  (1590, 5120, 13824),
    "matmul.wan_ffn2_14b_480p":  (1590, 13824, 5120),
    "matmul.wan_qkv_14b_720p":   (3600, 5120, 15360),
    "matmul.wan_ffn1_14b_720p":  (3600, 5120, 13824),
    "matmul.wan_ffn2_14b_720p":  (3600, 13824, 5120),
}

_PEAK_BF16_DEFAULT = 325.0   # TFLOPS BF16 (1 TRN2 die)
_PEAK_FP8_DEFAULT  = 650.0   # TFLOPS FP8  (1 TRN2 die, 2× bf16)
_MXFP8_BLOCK       = 32      # OCP MXFP8 group / block size


# ---------------------------------------------------------------------------
# Module definitions
# ---------------------------------------------------------------------------

class _NoOp(nn.Module):
    """Passthrough — zero FLOPs, measures pure dispatch+sync overhead."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _BF16Linear(nn.Module):
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.l = nn.Linear(k, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l(x)


class _FP8Linear(nn.Module):
    """Both activations and weight cast to float8_e4m3fn before the matmul.

    torch_neuronx is expected to lower torch.mm(fp8, fp8) to native TRN2 FP8
    hardware ops.  If the compiler version does not yet support this, compilation
    will raise — the benchmark catches and records the failure.
    """
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        w = torch.randn(n, k, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.register_buffer("w_fp8", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp8 = x.to(torch.float8_e4m3fn)
        return torch.mm(x_fp8, self.w_fp8.t()).to(torch.bfloat16)


class _MXFP8Linear(nn.Module):
    """Microscaling FP8 (OCP MXFP8) — per-group-32 scaled float8_e4m3fn.

    Weight dequantised to bf16 before the matmul — models the expected TRN2
    compute path until native MXFP8 compiler support lands.
    """
    def __init__(self, k: int, n: int, block: int = _MXFP8_BLOCK) -> None:
        super().__init__()
        k_pad = math.ceil(k / block) * block
        w = torch.zeros(n, k_pad, dtype=torch.bfloat16)
        nn.init.normal_(w[:, :k])
        w_g = w.reshape(n, -1, block)
        amax = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = amax / fp8_max
        self.register_buffer("w_q",   (w_g / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.squeeze(-1))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dq = (
            self.w_q.to(torch.bfloat16) * self.scale.unsqueeze(-1)
        ).reshape(self.w_q.shape[0], -1)[:, : self.k]
        return x @ w_dq.t()


_DTYPE_MODULES = {
    "bf16":   _BF16Linear,
    "fp8":    _FP8Linear,
    "mxfp8":  _MXFP8Linear,
}


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def _compile_trace(mod: nn.Module, x: torch.Tensor) -> tuple[Any, float]:
    """AOT compile via torch_neuronx.trace(); returns (callable, trace_s)."""
    t0 = time.perf_counter()
    compiled = torch_neuronx.trace(mod, (x,))
    return compiled, time.perf_counter() - t0


_COMPILE_BACKEND: str = "openxla"  # overridden in main() after arg parse


def _compile_jit(mod: nn.Module, x: torch.Tensor) -> tuple[Any, float]:
    """JIT compile via torch.compile; backend controlled by --compile-backend."""
    compiled = torch.compile(mod, backend=_COMPILE_BACKEND)
    return compiled, 0.0


_METHODS: dict[str, Any] = {
    "trace":   _compile_trace,
    "compile": _compile_jit,
}


# ---------------------------------------------------------------------------
# Timing loop (shared by both methods and the overhead case)
# ---------------------------------------------------------------------------

def _time_compiled(
    compiled: Any,
    x_dev: torch.Tensor,
    warmup: int,
    reps: int,
) -> list[float]:
    for _ in range(warmup):
        compiled(x_dev)
        _sync_device()

    times_us: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x_dev)
        _sync_device()
        times_us.append((time.perf_counter() - t0) * 1e6)
    return times_us


# ---------------------------------------------------------------------------
# Benchmark entry points
# ---------------------------------------------------------------------------

def bench_overhead(warmup: int, reps: int, methods: list[str]) -> list[dict[str, Any]]:
    """No-op passthrough: zero FLOPs, measures pure dispatch+sync overhead."""
    case = "matmul.overhead"
    x = torch.randn(1, 1, dtype=torch.bfloat16)
    device = _get_device()
    x_dev = x.to(device)
    out: list[dict[str, Any]] = []

    for method in methods:
        print(f"[bench] {case}  method={method}  (no-op passthrough)")
        compile_fn = _METHODS[method]
        try:
            compiled, trace_s = compile_fn(_NoOp(), x)
            if trace_s:
                print(f"[bench]   traced in {trace_s:.1f}s")
        except Exception as exc:
            print(f"[bench]   COMPILE FAIL: {exc}")
            out.append({"case": case, "method": method, "dtype": "bf16",
                        "M": 1, "K": 0, "N": 0, "flops": 0,
                        "status": "compile_fail",
                        "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue

        try:
            times_us = _time_compiled(compiled, x_dev, warmup, reps)
        except Exception as exc:
            print(f"[bench]   RUN FAIL: {exc}")
            out.append({"case": case, "method": method, "dtype": "bf16",
                        "M": 1, "K": 0, "N": 0, "flops": 0,
                        "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue

        med = statistics.median(times_us)
        print(f"[bench]   overhead median {med:.0f}µs  "
              f"min {min(times_us):.0f}µs  max {max(times_us):.0f}µs")
        out.append({
            "case": case, "method": method, "dtype": "bf16",
            "M": 1, "K": 0, "N": 0, "flops": 0,
            "trace_s": trace_s,
            "median_us": med, "min_us": min(times_us), "max_us": max(times_us),
            "achieved_tflops": 0.0, "mfu_pct": 0.0, "status": "ok",
        })
    return out


def bench_one(
    case: str,
    M: int, K: int, N: int,
    dtype: str,
    method: str,
    warmup: int,
    reps: int,
    peak_bf16: float,
    peak_fp8: float,
) -> dict[str, Any]:
    flops = 2 * M * K * N
    base = {"case": case, "method": method, "M": M, "K": K, "N": N,
            "dtype": dtype, "flops": flops}
    print(f"[bench] {case}  method={method}  dtype={dtype}  M={M} K={K} N={N}")

    mod = _DTYPE_MODULES[dtype](K, N)
    if dtype == "bf16":
        mod = mod.to(torch.bfloat16)
    x = torch.randn(M, K, dtype=torch.bfloat16)

    compile_fn = _METHODS[method]
    try:
        compiled, trace_s = compile_fn(mod, x)
        if trace_s:
            print(f"[bench]   compiled (trace) in {trace_s:.1f}s")
    except Exception as exc:
        print(f"[bench]   COMPILE FAIL: {type(exc).__name__}: {exc}")
        return {**base, "status": "compile_fail",
                "error": f"{type(exc).__name__}: {exc}"[:512]}

    device = _get_device()
    x_dev = x.to(device)
    # compile path: Dynamo traces on the first forward call and must see consistent
    # devices for inputs and weights. trace path needs CPU model for XLA AOT.
    if method != "trace":
        mod.to(device)

    try:
        times_us = _time_compiled(compiled, x_dev, warmup, reps)
    except Exception as exc:
        print(f"[bench]   RUN FAIL: {type(exc).__name__}: {exc}")
        return {**base, "trace_s": trace_s, "status": "fail",
                "error": f"{type(exc).__name__}: {exc}"[:512]}

    med = statistics.median(times_us)
    achieved = flops / med / 1e6  # TFLOP/s
    peak = peak_fp8 if dtype in ("fp8", "mxfp8") else peak_bf16
    mfu = achieved / peak * 100.0

    print(f"[bench]   median {med:.0f}µs · {achieved:.1f} TF/s · MFU={mfu:.1f}%")
    return {
        **base,
        "trace_s": trace_s,
        "median_us": med,
        "min_us": min(times_us),
        "max_us": max(times_us),
        "achieved_tflops": achieved,
        "mfu_pct": mfu,
        "status": "ok",
    }


def main() -> None:
    p = argparse.ArgumentParser(description="TRN2 matmul peak benchmark")
    p.add_argument("--cases",   nargs="+", default=list(MATMUL_SHAPES.keys()),
                   help="Shape keys to benchmark (default: all)")
    p.add_argument("--dtypes",  nargs="+", default=["bf16", "fp8", "mxfp8"],
                   choices=list(_DTYPE_MODULES))
    _default_backend = "neuron" if not _HAS_TRACE else "openxla"
    p.add_argument("--compile-backend", default=_default_backend, dest="compile_backend",
                   help="torch.compile backend for the 'compile' method "
                        "(auto-detected: 'neuron' for Beta 2, 'openxla' for public)")
    _default_methods = ["compile"] if not _HAS_TRACE else ["trace", "compile"]
    p.add_argument("--methods", nargs="+", default=_default_methods,
                   choices=list(_METHODS),
                   help="Compilation paths: trace=torch_neuronx.trace, "
                        "compile=torch.compile(backend=compile_backend)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps",   type=int, default=10)
    p.add_argument("--peak-bf16-tflops", type=float, default=_PEAK_BF16_DEFAULT,
                   dest="peak_bf16", metavar="TF")
    p.add_argument("--peak-fp8-tflops",  type=float, default=_PEAK_FP8_DEFAULT,
                   dest="peak_fp8",  metavar="TF")
    p.add_argument("--output", default="/tmp/trn2_matmul_bench.json")
    args = p.parse_args()

    global _COMPILE_BACKEND
    _COMPILE_BACKEND = args.compile_backend
    if "trace" in args.methods and not _HAS_TRACE:
        print("[bench] WARNING: 'trace' requested but torch_neuronx.trace not available; skipping")
        args.methods = [m for m in args.methods if m != "trace"]

    print(f"[bench] python={sys.version.split()[0]}"
          f"  torch={torch.__version__}"
          f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
    print(f"[bench] hostname={socket.gethostname()}")
    print(f"[bench] methods={args.methods}  dtypes={args.dtypes}  compile_backend={_COMPILE_BACKEND}")
    print(f"[bench] peak_bf16={args.peak_bf16} TFLOPS  peak_fp8={args.peak_fp8} TFLOPS")

    results: list[dict[str, Any]] = bench_overhead(args.warmup, args.reps, args.methods)

    for case in args.cases:
        if case not in MATMUL_SHAPES:
            print(f"[bench] unknown case, skipping: {case}")
            continue
        M, K, N = MATMUL_SHAPES[case]
        for method in args.methods:
            for dtype in args.dtypes:
                try:
                    results.append(
                        bench_one(case, M, K, N, dtype, method,
                                  args.warmup, args.reps,
                                  args.peak_bf16, args.peak_fp8)
                    )
                except Exception as exc:
                    results.append({
                        "case": case, "method": method,
                        "M": M, "K": K, "N": N, "dtype": dtype,
                        "flops": 2 * M * K * N, "status": "fail",
                        "error": f"{type(exc).__name__}: {exc}"[:512],
                    })

    payload = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
        "warmup": args.warmup,
        "reps": args.reps,
        "methods": args.methods,
        "peak_bf16_tflops": args.peak_bf16,
        "peak_fp8_tflops": args.peak_fp8,
        "shapes": {k: list(v) for k, v in MATMUL_SHAPES.items() if k in args.cases},
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
