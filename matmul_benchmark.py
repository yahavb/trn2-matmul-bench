"""Trainium 2 matmul peak microbenchmark — bf16 / fp8 / mxfp8, via XLA path.

Square matmuls from 1024^3 → 32768^3 (powers of two) plus WAN-shaped GEMMs.
Each shape × dtype combination is traced with torch_neuronx and timed on-device.
Reports achieved TFLOP/s and MFU% (vs --peak-bf16-tflops / --peak-fp8-tflops).

Target: single Trainium 2 die (trn2.3xlarge = 4 NeuronCores, 1 device).
Leave NEURON_RT_NUM_CORES unset to use all NCs on the die (NRT default).
Valid values on TRN2 are 1 (single NC) or multiples of 8 (multi-die).
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
import torch_xla
import torch_xla.core.xla_model as xm


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

# Single TRN2 die = 2 NeuronCores.  Adjust via --peak-{bf16,fp8}-tflops.
_PEAK_BF16_DEFAULT = 325.0   # TFLOPS BF16 (1 die)
_PEAK_FP8_DEFAULT  = 650.0   # TFLOPS FP8  (1 die, 2× bf16)
_MXFP8_BLOCK       = 32      # OCP MXFP8 group / block size


# ---------------------------------------------------------------------------
# Module definitions
# ---------------------------------------------------------------------------

class _BF16Linear(nn.Module):
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        self.l = nn.Linear(k, n, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l(x)


class _FP8Linear(nn.Module):
    """Both activations and weight cast to float8_e4m3fn before the matmul.

    torch_neuronx is expected to lower torch.mm(fp8, fp8) to native TRN2 FP8
    hardware ops.  If the compiler version does not yet support this, tracing
    will raise — the benchmark catches and records the failure.
    """
    def __init__(self, k: int, n: int) -> None:
        super().__init__()
        w = torch.randn(n, k, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        self.register_buffer("w_fp8", w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_fp8 = x.to(torch.float8_e4m3fn)
        # Direct fp8×fp8 — let the compiler lower to native hardware ops.
        return torch.mm(x_fp8, self.w_fp8.t()).to(torch.bfloat16)


class _MXFP8Linear(nn.Module):
    """Microscaling FP8 (OCP MXFP8) linear — per-group-32 scaled float8_e4m3fn.

    Weight is stored quantised with per-block scale factors.  At forward time
    the weight is dequantised to bf16 and the matmul runs in bf16 — this models
    the expected TRN2 compute path until native MXFP8 compiler support lands.
    """
    def __init__(self, k: int, n: int, block: int = _MXFP8_BLOCK) -> None:
        super().__init__()
        k_pad = math.ceil(k / block) * block
        w = torch.zeros(n, k_pad, dtype=torch.bfloat16)
        nn.init.normal_(w[:, :k])
        w_g = w.reshape(n, -1, block)                          # (N, G, B)
        amax = w_g.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        scale = amax / fp8_max                                  # (N, G, 1)
        self.register_buffer("w_q",   (w_g / scale).to(torch.float8_e4m3fn))
        self.register_buffer("scale", scale.squeeze(-1))        # (N, G)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantise weight: (N, G, B) × (N, G, 1) → (N, k_pad) → (N, K)
        w_dq = (
            self.w_q.to(torch.bfloat16) * self.scale.unsqueeze(-1)
        ).reshape(self.w_q.shape[0], -1)[:, : self.k]
        return x @ w_dq.t()


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

_DTYPE_MODULES = {
    "bf16":   _BF16Linear,
    "fp8":    _FP8Linear,
    "mxfp8":  _MXFP8Linear,
}


def bench_one(
    case: str,
    M: int, K: int, N: int,
    dtype: str,
    warmup: int,
    reps: int,
    peak_bf16: float,
    peak_fp8: float,
) -> dict[str, Any]:
    flops = 2 * M * K * N
    print(f"[bench] {case}  dtype={dtype}  M={M} K={K} N={N}")

    try:
        mod = _DTYPE_MODULES[dtype](K, N)
        if dtype == "bf16":
            mod = mod.to(torch.bfloat16)
        x = torch.randn(M, K, dtype=torch.bfloat16)

        t0 = time.perf_counter()
        compiled = torch_neuronx.trace(mod, (x,))
        trace_s = time.perf_counter() - t0
        print(f"[bench]   traced in {trace_s:.1f}s")
    except Exception as exc:
        print(f"[bench]   TRACE FAIL: {type(exc).__name__}: {exc}")
        return {
            "case": case, "M": M, "K": K, "N": N, "dtype": dtype,
            "flops": flops, "status": "trace_fail",
            "error": f"{type(exc).__name__}: {exc}"[:512],
        }

    device = torch_xla.device()
    x_dev = x.to(device)

    for _ in range(warmup):
        compiled(x_dev)
        torch_xla.sync()
        xm.wait_device_ops()

    times_us: list[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        compiled(x_dev)
        torch_xla.sync()
        xm.wait_device_ops()
        times_us.append((time.perf_counter() - t0) * 1e6)

    med = statistics.median(times_us)
    achieved = flops / med / 1e6  # TFLOP/s
    peak = peak_fp8 if dtype in ("fp8", "mxfp8") else peak_bf16
    mfu = achieved / peak * 100.0

    print(f"[bench]   median {med:.0f}µs · {achieved:.1f} TF/s · MFU={mfu:.1f}%")
    return {
        "case": case, "M": M, "K": K, "N": N, "dtype": dtype,
        "flops": flops,
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
    p.add_argument("--cases",  nargs="+", default=list(MATMUL_SHAPES.keys()),
                   help="Shape keys to benchmark (default: all)")
    p.add_argument("--dtypes", nargs="+", default=["bf16", "fp8", "mxfp8"],
                   choices=list(_DTYPE_MODULES), help="Dtype regimes to measure")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps",   type=int, default=10)
    p.add_argument("--peak-bf16-tflops", type=float, default=_PEAK_BF16_DEFAULT,
                   dest="peak_bf16", metavar="TF",
                   help="BF16 peak for MFU%% (default: %(default)s, 1 TRN2 die)")
    p.add_argument("--peak-fp8-tflops",  type=float, default=_PEAK_FP8_DEFAULT,
                   dest="peak_fp8",  metavar="TF",
                   help="FP8 peak for MFU%% (default: %(default)s, 1 TRN2 die)")
    p.add_argument("--output", default="/tmp/trn2_matmul_bench.json")
    args = p.parse_args()

    print(f"[bench] python={sys.version.split()[0]}"
          f"  torch={torch.__version__}"
          f"  torch_neuronx={getattr(torch_neuronx, '__version__', '?')}")
    print(f"[bench] hostname={socket.gethostname()}")
    print(f"[bench] peak_bf16={args.peak_bf16} TFLOPS  peak_fp8={args.peak_fp8} TFLOPS")

    results: list[dict[str, Any]] = []
    for case in args.cases:
        if case not in MATMUL_SHAPES:
            print(f"[bench] unknown case, skipping: {case}")
            continue
        M, K, N = MATMUL_SHAPES[case]
        for dtype in args.dtypes:
            try:
                results.append(
                    bench_one(case, M, K, N, dtype, args.warmup, args.reps,
                              args.peak_bf16, args.peak_fp8)
                )
            except Exception as exc:
                results.append({
                    "case": case, "M": M, "K": K, "N": N, "dtype": dtype,
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
