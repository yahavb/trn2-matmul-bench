"""Compile MXFP8 matmul NEFF with pre-quantized inputs — for neuron-profile capture.

This script uses the NKI compiler API directly (same as the test framework)
to produce a NEFF that takes pre-quantized MXFP8 inputs. No torch involved.

Usage:
    python3 mxfp8_compile_neff.py --size 16384
    # Then profile:
    NEURON_RT_VISIBLE_CORES=62-63 neuron-profile capture -n output_neff/file.neff -s profile.ntff
    NEURON_RT_VISIBLE_CORES=62-63 neuron-profile view -n output_neff/file.neff -s profile.ntff --output-format summary-text
"""
import os
import sys
import argparse

os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + \
    " --target=trn3pre --internal-backend-options=--enable-mx-alternative-emax"

import numpy as np
from neuronxcc.nki._private.private_api import float8_e4m3fn_x4
from neuronxcc.nki._private.test import mx_util

from nki.compiler.driver import compile_to_bir
from nki.compiler.frontend import ParserFrontend, TracerFrontend
from nki.compiler.ncc_driver import CompileOptions, compile_bir_to_neff

import nki
import nki.language as nl

from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_generic_kernel import matmul_mxfp8


# ─── Constants ────────────────────────────────────────────────────────

INTERLEAVE_FACTOR = 4


def _get_mx_max_exp(dst_dtype):
    return {float8_e4m3fn_x4: 7}[dst_dtype]


# ─── Swizzle ──────────────────────────────────────────────────────────

def swizzle(src, tile_p=512):
    """[K, F] -> [K//4, F*4] interleaved layout."""
    P, F = src.shape
    assert P % INTERLEAVE_FACTOR == 0
    remainder = P % tile_p
    if remainder != 0 and remainder % 128 == 0:
        full_p = P - remainder
        parts = []
        if full_p > 0:
            parts.append(swizzle(src[:full_p], tile_p))
        off = full_p
        rem = remainder
        if rem >= 256:
            parts.append(swizzle(src[off:off + 256], tile_p=256))
            off += 256
            rem -= 256
        if rem >= 128:
            parts.append(swizzle(src[off:off + 128], tile_p=128))
        return np.concatenate(parts, axis=0)

    dst = np.zeros((P // INTERLEAVE_FACTOR, F * INTERLEAVE_FACTOR), dtype=src.dtype)
    num_tiles = (P + tile_p - 1) // tile_p
    sub_tile = tile_p // INTERLEAVE_FACTOR
    for tp in range(num_tiles):
        cur_size = min(tile_p, P - tp * tile_p)
        cur_sub = cur_size // INTERLEAVE_FACTOR
        for s in range(INTERLEAVE_FACTOR):
            for p in range(cur_sub):
                src_p = tp * tile_p + s * cur_sub + p
                if src_p < P:
                    for f in range(F):
                        dst[tp * sub_tile + p, f * INTERLEAVE_FACTOR + s] = src[src_p, f]
    return dst


def resize_scales_compact_to_oversized(compact_scales):
    """[P//8, F//4] -> [P, F//4] oversized layout."""
    compact_k, f_div4 = compact_scales.shape
    tile_k = compact_k * 8
    oversized = np.zeros((tile_k, f_div4), dtype=compact_scales.dtype)
    for idx in range(compact_k):
        hbm_idx = (idx // 4) * 32 + (idx % 4)
        oversized[hbm_idx, :] = compact_scales[idx, :]
    return oversized


# ─── NKI kernel (same as what the test uses) ──────────────────────────

def matmul_mxfp8_kernel(lhs, rhs, lhs_scales, rhs_scales):
    """Pre-quantized MXFP8 matmul kernel — pure TensorE workload."""
    return matmul_mxfp8(
        lhs=lhs,
        rhs=rhs,
        lhs_scales=lhs_scales,
        rhs_scales=rhs_scales,
        float8_dtype="float8_e4m3fn",
        output_dtype=nl.bfloat16,
        run_with_lnc2=True,
        lhs_is_swizzled=True,
        rhs_is_swizzled=True,
    )


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compile MXFP8 matmul NEFF for profiling")
    parser.add_argument("--size", type=int, default=16384)
    parser.add_argument("--output-dir", type=str, default="./mxfp8_neff_output")
    parser.add_argument("--cache-dir", type=str, default="./mxfp8_input_cache")
    args = parser.parse_args()

    M = K = N = args.size
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "artifacts"), exist_ok=True)

    print(f"Compiling MXFP8 MatMul NEFF: {M}x{K}x{N}")
    print(f"Output: {output_dir}")

    # ─── Prepare pre-quantized inputs (same as test framework) ────────

    cache_file = os.path.join(args.cache_dir, f"mxfp8_x4_inputs_{M}x{K}x{N}.npz")
    if os.path.exists(cache_file):
        print(f"Loading cached inputs from {cache_file}...")
        cached = np.load(cache_file, allow_pickle=True)
        # np.savez loses custom dtype — restore it by viewing raw bytes as x4
        lhs_data = cached["lhs_data"].view(float8_e4m3fn_x4)
        rhs_data = cached["rhs_data"].view(float8_e4m3fn_x4)
        lhs_scales = cached["lhs_scales"]
        rhs_scales = cached["rhs_scales"]
    else:
        print("Generating inputs...")
        print("  Generating random [K, M] and [K, N] data...")
        lhs_fp32 = np.random.randn(K, M).astype(np.float32)
        rhs_fp32 = np.random.randn(K, N).astype(np.float32)

        print("  Swizzling...")
        lhs_sw = swizzle(lhs_fp32)  # [K//4, M*4]
        rhs_sw = swizzle(rhs_fp32)  # [K//4, N*4]

        print("  Quantizing to MXFP8...")
        lhs_data_x4, lhs_scales_compact = mx_util.quantize_mx_golden(
            lhs_sw, float8_e4m3fn_x4, custom_mx_max_exp=_get_mx_max_exp
        )
        rhs_data_x4, rhs_scales_compact = mx_util.quantize_mx_golden(
            rhs_sw, float8_e4m3fn_x4, custom_mx_max_exp=_get_mx_max_exp
        )

        print("  Converting scales to oversized layout...")
        lhs_scales = resize_scales_compact_to_oversized(lhs_scales_compact)
        rhs_scales = resize_scales_compact_to_oversized(rhs_scales_compact)

        # Data stays as float8_e4m3fn_x4 dtype — this is the key!
        lhs_data = lhs_data_x4
        rhs_data = rhs_data_x4

        os.makedirs(args.cache_dir, exist_ok=True)
        np.savez(cache_file,
                 lhs_data=lhs_data, rhs_data=rhs_data,
                 lhs_scales=lhs_scales, rhs_scales=rhs_scales)
        print(f"  Saved to {cache_file}")

    print(f"  LHS data:   {lhs_data.shape} dtype={lhs_data.dtype}")
    print(f"  LHS scales: {lhs_scales.shape} dtype={lhs_scales.dtype}")
    print(f"  RHS data:   {rhs_data.shape} dtype={rhs_data.dtype}")
    print(f"  RHS scales: {rhs_scales.shape} dtype={rhs_scales.dtype}")

    # ─── Compile using NKI compiler API (same as test framework) ──────

    print("\nCompiling kernel to NEFF...")

    # Build inputs dict — numpy arrays with correct dtypes
    # This is exactly what build_matmul_inputs returns in the test
    kernel_inputs = {
        "lhs": lhs_data,          # float8_e4m3fn_x4, shape [K//4, M]
        "rhs": rhs_data,          # float8_e4m3fn_x4, shape [K//4, N]
        "lhs_scales": lhs_scales, # uint8, shape [K//4, M]
        "rhs_scales": rhs_scales, # uint8, shape [K//4, N]
    }

    neff_path = os.path.abspath(os.path.join(output_dir, "file.neff"))
    artifacts_path = os.path.abspath(os.path.join(output_dir, "artifacts"))

    compile_opts = CompileOptions(
        target="trn3pre",
        lnc=2,
        output_path=neff_path,
        artifacts_dir=artifacts_path,
        neuronx_cc_args=("--internal-backend-options=--enable-mx-alternative-emax",),
    )
    compile_opts = compile_opts.disable_backend_optimizations()

    frontend = TracerFrontend()

    print("  Running compile_to_bir...")
    result = compile_to_bir(
        kernel_func=matmul_mxfp8_kernel,
        frontend=frontend,
        inputs=kernel_inputs,
        compile_opts=compile_opts,
        output_names=["out"],
    )

    print("  Running compile_bir_to_neff...")
    argument_names = [s.name for s in result.descriptor.input_specs]
    output_names = [s.name for s in result.descriptor.output_specs]

    compiled = compile_bir_to_neff(compile_opts, result, [], argument_names, output_names)

    print(f"\n  NEFF compiled successfully!")
    print(f"  NEFF path: {compiled.neff_path}")
    print(f"  MLIR time: {compiled.mlir_time:.1f}s")
    print(f"  NEFF time: {compiled.neuronx_cc_time:.1f}s")

    print(f"\nTo profile:")
    print(f"  NEURON_RT_VISIBLE_CORES=62-63 neuron-profile capture \\")
    print(f"    -n {compiled.neff_path} \\")
    print(f"    -s profile_mxfp8_{M}.ntff")
    print(f"")
    print(f"  NEURON_RT_VISIBLE_CORES=62-63 neuron-profile view \\")
    print(f"    -n {compiled.neff_path} \\")
    print(f"    -s profile_mxfp8_{M}.ntff \\")
    print(f"    --output-format summary-text")


if __name__ == "__main__":
    sys.exit(main() or 0)
