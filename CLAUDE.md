# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A standalone matmul peak microbenchmark for AWS Trainium 2/3 chips. Measures bf16/fp8/mxfp8 GEMM performance across square powers-of-two and WAN-shaped matrices. Reports achieved TFLOP/s and MFU% against logical-NC peak.

## Running

All benchmarks run inside Docker containers on Trainium instances (trn2.3xlarge). There is no local test suite — the code is validated by running on hardware.

```bash
# Public path (torch_neuronx trace, 1 physical NC)
docker build -f Dockerfile -t matmul-bench .
docker run --privileged matmul-bench

# Torch Neuron Beta 2 path (torch.compile backend="neuron", 1 logical NC = 2 physical NCs)
docker build -f Dockerfile.beta2 -t matmul-bench-beta2 .
docker run --privileged matmul-bench-beta2

# NC topology probe (multi-process parallel benchmark)
docker build -f Dockerfile.probe -t matmul-bench-probe .
docker run --privileged matmul-bench-probe
```

Production runs use AWS Batch in sa-east-1 (odyssey-trainium-queue). Results upload to `s3://ody-trainium-cache-sae1/matmul-bench/`.

## Architecture

Two compilation paths in `matmul_benchmark.py`:
- **trace** — `torch_neuronx.trace()`: AOT XLA compilation. Uses public torch-neuronx package. Targets 1 physical NC (~83 TFLOPS bf16 peak).
- **compile** — `torch.compile(backend="neuron")`: JIT via Beta 2 torch_neuron_eager. Targets 1 logical NC (~167 TFLOPS bf16 peak).

The script auto-detects which path is available based on `hasattr(torch_neuronx, "trace")` and `_XLA_AVAILABLE`.

`matmul_benchmark_trn3.py` is a separate script that compiles MXFP8 matmul NEFFs via the NKI compiler API directly (no torch), targeting trn3pre hardware for neuron-profile capture.

`nc_probe.py` verifies logical NC independence by spawning separate OS processes with `NEURON_RT_VISIBLE_CORES` pinned to individual logical NCs and measuring aggregate throughput.

## Key Environment Variables

- `NEURON_RT_NUM_CORES` — NRT accepts 1 or multiples of 8 on trn2
- `NEURON_RT_VISIBLE_CORES` — restricts NRT to specific logical NC group indices (0-3 on trn2.3xlarge with lnc=2)
- `NEURON_LOGICAL_NC_CONFIG` — set to 1 to expose 8 physical NCs instead of 4 logical NCs
- `S3_PREFIX` — where results/logs upload (default: `s3://ody-trainium-cache-sae1/matmul-bench`)

## Dependencies

Managed via `uv` with `pyproject.toml` + `uv.lock`. Uses custom pip indices:
- `https://pip.repos.neuron.amazonaws.com` for neuronx packages
- `https://download.pytorch.org/whl/cpu` for CPU-only torch wheels

## Hardware Context

trn2.3xlarge = 1 chip, 2 dies, 8 physical NCs (4 per die). With lnc=2 (default): 4 logical NCs, each ~167 TFLOPS bf16.
- LNCs 0,1 share die 0 (HBM contention when both active)
- LNCs 2,3 share die 1
- Cross-die pairs (0+2, 1+3) scale independently
