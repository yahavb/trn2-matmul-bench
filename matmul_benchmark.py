"""Trainium 2/3 matmul peak microbenchmark — bf16 / fp8 / mxfp8.

Two compilation paths are measured for each shape × dtype:
  trace   — torch_neuronx.trace(): AOT XLA compilation (public torch-neuronx only).
  compile — torch.compile(backend=...): JIT; backend auto-detected from the installed
             package ("neuron" for Beta 2 torch_neuron_eager, "openxla" for public).

Square matmuls 1024³ → 32768³ (powers of two) plus WAN-shaped GEMMs.
Reports achieved TFLOP/s and MFU% vs peak_bf16 / peak_fp8.

Hardware target: single Trainium 2 die (trn2.3xlarge — 4 NeuronCores, 96 GB HBM).
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

# Beta 2 (torch_neuron_eager) registers torch.device("neuron:0") and has no
# torch_xla; public torch-neuronx uses XLA as the device backend.
_HAS_TRACE = hasattr(torch_neuronx, "trace")
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _XLA_AVAILABLE = True
except ImportError:
    _XLA_AVAILABLE = False


_FORCE_NEURON_DEVICE = False  # set True when using backend="neuron" (eager path)


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

_PEAK_BF16_DEFAULT = 325.0  # TFLOPS bf16, 1 TRN2 die
_PEAK_FP8_DEFAULT  = 650.0  # TFLOPS fp8,  1 TRN2 die (2× bf16)
_MXFP8_BLOCK       = 32     # OCP MXFP8 block size


class _NoOp(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _BF16Linear(nn.Module):
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.l = nn.Linear(k, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l(x)


class _FP8Linear(nn.Module):
    """Activations and weight cast to float8_e4m3fn; expects native FP8 hardware lowering."""
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.register_buffer("w_fp8",
            torch.randn(n, k, dtype=torch.bfloat16).to(torch.float8_e4m3fn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mm(x.to(torch.float8_e4m3fn), self.w_fp8.t()).to(torch.bfloat16)


class _MXFP8Linear(nn.Module):
    """OCP MXFP8 — per-group-32 scaled float8_e4m3fn, dequantised to bf16 before matmul."""
    def __init__(self, k: int, n: int, block: int = _MXFP8_BLOCK) -> None:
        super().__init__()
        k_pad = math.ceil(k / block) * block
        w = torch.zeros(n, k_pad, dtype=torch.bfloat16)
        nn.init.normal_(w[:, :k])
        w_g   = w.reshape(n, -1, block)
        scale = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / torch.finfo(torch.float8_e4m3fn).max
        self.register_buffer("w_q",   (w_g / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.squeeze(-1))
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dq = (self.w_q.to(torch.bfloat16) * self.scale.unsqueeze(-1)).reshape(self.w_q.shape[0], -1)[:, :self.k]
        return x @ w_dq.t()


_DTYPE_MODULES = {"bf16": _BF16Linear, "fp8": _FP8Linear, "mxfp8": _MXFP8Linear}


def _compile_trace(mod: nn.Module, x: torch.Tensor) -> tuple[Any, float]:
    t0 = time.perf_counter()
    return torch_neuronx.trace(mod, (x,)), time.perf_counter() - t0


def _make_compile_jit(backend: str) -> Any:
    # dynamic=False: neuronx-cc rejects unbounded dynamism; Dynamo sometimes marks
    # batch dims as dynamic by default, causing compile failures for shapes ≥ 2048².
    def _compile_jit(mod: nn.Module, x: torch.Tensor) -> tuple[Any, float]:
        return torch.compile(mod, backend=backend, dynamic=False), 0.0
    return _compile_jit


def _time_compiled(compiled: Any, x_dev: torch.Tensor, warmup: int, reps: int) -> list[float]:
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


def bench_overhead(warmup: int, reps: int, methods: dict[str, Any]) -> list[dict[str, Any]]:
    case = "matmul.overhead"
    x = torch.randn(1, 1, dtype=torch.bfloat16)
    x_dev = x.to(_get_device())
    out: list[dict[str, Any]] = []

    for name, compile_fn in methods.items():
        print(f"[bench] {case}  method={name}  (no-op passthrough)")
        try:
            compiled, trace_s = compile_fn(_NoOp(), x)
            if trace_s:
                print(f"[bench]   traced in {trace_s:.1f}s")
        except Exception as exc:
            out.append({"case": case, "method": name, "dtype": "bf16",
                        "M": 1, "K": 0, "N": 0, "flops": 0,
                        "status": "compile_fail", "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue
        try:
            times_us = _time_compiled(compiled, x_dev, warmup, reps)
        except Exception as exc:
            out.append({"case": case, "method": name, "dtype": "bf16",
                        "M": 1, "K": 0, "N": 0, "flops": 0,
                        "status": "fail", "error": f"{type(exc).__name__}: {exc}"[:512]})
            continue
        med = statistics.median(times_us)
        print(f"[bench]   overhead median {med:.0f}µs  min {min(times_us):.0f}µs  max {max(times_us):.0f}µs")
        out.append({"case": case, "method": name, "dtype": "bf16",
                    "M": 1, "K": 0, "N": 0, "flops": 0, "trace_s": trace_s,
                    "median_us": med, "min_us": min(times_us), "max_us": max(times_us),
                    "achieved_tflops": 0.0, "mfu_pct": 0.0, "status": "ok"})
    return out


def bench_one(
    case: str, M: int, K: int, N: int,
    dtype: str, method: str, compile_fn: Any,
    warmup: int, reps: int,
    peak_bf16: float, peak_fp8: float,
) -> dict[str, Any]:
    flops = 2 * M * K * N
    base  = {"case": case, "method": method, "M": M, "K": K, "N": N, "dtype": dtype, "flops": flops}
    print(f"[bench] {case}  method={method}  dtype={dtype}  M={M} K={K} N={N}")

    mod = _DTYPE_MODULES[dtype](K, N)
    if dtype == "bf16":
        mod = mod.to(torch.bfloat16)
    x = torch.randn(M, K, dtype=torch.bfloat16)

    try:
        compiled, trace_s = compile_fn(mod, x)
        if trace_s:
            print(f"[bench]   compiled (trace) in {trace_s:.1f}s")
    except Exception as exc:
        print(f"[bench]   COMPILE FAIL: {type(exc).__name__}: {exc}")
        return {**base, "status": "compile_fail", "error": f"{type(exc).__name__}: {exc}"[:512]}

    device = _get_device()
    x_dev  = x.to(device)
    # torch.compile wraps the module by reference; move weights to device now so
    # Dynamo's first-call tracing sees consistent devices. trace() handles this itself.
    if method != "trace":
        mod.to(device)

    try:
        times_us = _time_compiled(compiled, x_dev, warmup, reps)
    except Exception as exc:
        print(f"[bench]   RUN FAIL: {type(exc).__name__}: {exc}")
        return {**base, "trace_s": trace_s, "status": "fail", "error": f"{type(exc).__name__}: {exc}"[:512]}

    med      = statistics.median(times_us)
    achieved = flops / med / 1e6
    peak     = peak_fp8 if dtype in ("fp8", "mxfp8") else peak_bf16
    mfu      = achieved / peak * 100.0
    print(f"[bench]   median {med:.0f}µs · {achieved:.1f} TF/s · MFU={mfu:.1f}%")
    return {**base, "trace_s": trace_s,
            "median_us": med, "min_us": min(times_us), "max_us": max(times_us),
            "achieved_tflops": achieved, "mfu_pct": mfu, "status": "ok"}


def main() -> None:
    p = argparse.ArgumentParser(description="TRN2/3 matmul peak benchmark")
    p.add_argument("--cases",   nargs="+", default=list(MATMUL_SHAPES.keys()))
    p.add_argument("--dtypes",  nargs="+", default=["bf16", "fp8", "mxfp8"],
                   choices=list(_DTYPE_MODULES))
    p.add_argument("--methods", nargs="+",
                   default=["compile"] if not _HAS_TRACE else ["trace", "compile"],
                   choices=["trace", "compile"])
    p.add_argument("--compile-backend", dest="compile_backend",
                   default="neuron" if not _HAS_TRACE else "openxla",
                   help="backend for torch.compile (auto-detected if omitted)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps",   type=int, default=10)
    p.add_argument("--peak-bf16-tflops", type=float, default=_PEAK_BF16_DEFAULT,
                   dest="peak_bf16", metavar="TF")
    p.add_argument("--peak-fp8-tflops",  type=float, default=_PEAK_FP8_DEFAULT,
                   dest="peak_fp8",  metavar="TF")
    p.add_argument("--output", default="/tmp/trn2_matmul_bench.json")
    args = p.parse_args()

    if "trace" in args.methods and not _HAS_TRACE:
        print("[bench] WARNING: 'trace' unavailable (Beta 2 package); skipping")
        args.methods = [m for m in args.methods if m != "trace"]

    # When using the "neuron" backend (eager path), force torch.device("neuron")
    # instead of XLA device, even if torch_xla is importable.
    global _FORCE_NEURON_DEVICE
    if args.compile_backend == "neuron":
        _FORCE_NEURON_DEVICE = True

    methods: dict[str, Any] = {}
    if "trace"   in args.methods: methods["trace"]   = _compile_trace
    if "compile" in args.methods: methods["compile"] = _make_compile_jit(args.compile_backend)

    print(f"[bench] python={sys.version.split()[0]}"
          f"  torch={torch.__version__}"
          f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
    print(f"[bench] hostname={socket.gethostname()}")
    print(f"[bench] methods={args.methods}  dtypes={args.dtypes}  compile_backend={args.compile_backend}")
    print(f"[bench] peak_bf16={args.peak_bf16} TFLOPS  peak_fp8={args.peak_fp8} TFLOPS")

    # Initialize the Neuron device stream pool before any benchmark runs.
    # Without this, torch.compile may fail with "Stream pool not initialized"
    # because the device hasn't been touched before the first compiled execution.
    device = _get_device()
    _ = torch.zeros(1, device=device)
    print(f"[bench] device initialized: {device}")

    results: list[dict[str, Any]] = bench_overhead(args.warmup, args.reps, methods)

    for case in args.cases:
        if case not in MATMUL_SHAPES:
            print(f"[bench] unknown case, skipping: {case}")
            continue
        M, K, N = MATMUL_SHAPES[case]
        for name, compile_fn in methods.items():
            for dtype in args.dtypes:
                try:
                    results.append(bench_one(case, M, K, N, dtype, name, compile_fn,
                                             args.warmup, args.reps,
                                             args.peak_bf16, args.peak_fp8))
                except Exception as exc:
                    results.append({"case": case, "method": name,
                                    "M": M, "K": K, "N": N, "dtype": dtype,
                                    "flops": 2 * M * K * N, "status": "fail",
                                    "error": f"{type(exc).__name__}: {exc}"[:512]})

    payload = {
        "hostname": socket.gethostname(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_neuronx": getattr(torch_neuronx, "__version__", "?"),
        "warmup": args.warmup, "reps": args.reps,
        "methods": args.methods, "compile_backend": args.compile_backend,
        "peak_bf16_tflops": args.peak_bf16, "peak_fp8_tflops": args.peak_fp8,
        "shapes": {k: list(v) for k, v in MATMUL_SHAPES.items() if k in args.cases},
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[bench] wrote {args.output}")


if __name__ == "__main__":
    main()
