# trn2-matmul-bench

Standalone matmul peak microbenchmark for a single Trainium 2 die (`trn2.3xlarge`).
Measures bf16 / fp8 / mxfp8 across square powers-of-two and WAN-shaped GEMMs.
Reports achieved TFLOP/s and MFU% against the 325 TFLOPS bf16 full-die peak.

## Hardware

`trn2.3xlarge` — 1 Trainium 2 die:

| | |
|---|---|
| Physical NeuronCores | 4 (NeuronCore-v3) |
| Logical NeuronCore config | 2 (lnc=2, each = 2 physical NCs) |
| HBM | 96 GB |
| Peak bf16 | 325 TFLOPS |
| Peak fp8 | 650 TFLOPS |

## Two benchmark paths

| | Public (`matmul-bench`) | Beta 2 (`matmul-bench-beta2`) |
|---|---|---|
| Package | `torch-neuronx==2.9.0.2.13.24727` (public Neuron pip index) | `torch_neuron_eager` from `/workspace/` in Anthropic Beta 2 base image |
| Compilation | `torch_neuronx.trace()` — AOT XLA, single-NC NEFF | `torch.compile(backend="neuron")` — JIT, full die |
| NeuronCores used | 1 physical NC | 4 physical NCs (full die) |
| Device | `torch_xla.device()` (XLA) | `torch.device("neuron:0")` |
| Sync | `torch_xla.sync()` + `xm.wait_device_ops()` | `torch_neuronx.synchronize()` |
| torch version | `torch>=2.1` (CPU wheel) | `torch==2.10.0+cpu` |

## Measured results — bf16

Hardware: `trn2.3xlarge` · Warmup 3 · Reps 10 · 2026-05-06

| Shape | M | K | N | pub/trace TF/s (MFU) | b2/neuron TF/s (MFU) |
|---|---:|---:|---:|---:|---:|
| sq_1024 | 1024 | 1024 | 1024 | 6.9 (2%) | 9.9 (3%) |
| sq_2048 | 2048 | 2048 | 2048 | 18.3 (6%) | 28.5 (9%) |
| sq_4096 | 4096 | 4096 | 4096 | 25.3 (8%) | 91.3 (28%) |
| sq_8192 | 8192 | 8192 | 8192 | 31.6 (10%) | 117.0 (36%) |
| **sq_16384** | 16384 | 16384 | 16384 | **46.9 (14%)** | **134.4 (41%)** |
| sq_32768 | 32768 | 32768 | 32768 | 26.8 (8%) | FAIL¹ |
| wan_qkv_1.3b_480p | 1590 | 1536 | 4608 | 14.1 (4%) | 32.5 (10%) |
| wan_o_1.3b_480p | 1590 | 1536 | 1536 | 11.1 (3%) | 13.4 (4%) |
| wan_ffn1_1.3b_480p | 1590 | 1536 | 8960 | 14.7 (5%) | 47.2 (15%) |
| wan_ffn2_1.3b_480p | 1590 | 8960 | 1536 | 42.0 (13%) | 51.3 (16%) |
| wan_qkv_14b_480p | 1590 | 5120 | 15360 | 30.3 (9%) | 84.9 (26%) |
| wan_o_14b_480p | 1590 | 5120 | 5120 | 41.9 (13%) | 57.6 (18%) |
| wan_ffn1_14b_480p | 1590 | 5120 | 13824 | 30.0 (9%) | 84.4 (26%) |
| wan_ffn2_14b_480p | 1590 | 13824 | 5120 | 71.7 (22%) | 80.7 (25%) |
| wan_qkv_14b_720p | 3600 | 5120 | 15360 | 22.8 (7%) | 99.8 (31%) |
| wan_ffn1_14b_720p | 3600 | 5120 | 13824 | 24.0 (7%) | 97.8 (30%) |
| **wan_ffn2_14b_720p** | 3600 | 13824 | 5120 | **58.7 (18%)** | **104.4 (32%)** |

MFU% = achieved / 325 TFLOPS full-die peak.

**fp8 / mxfp8:** all shapes fail on both paths.
Public `torch-neuronx` trace does not yet lower `float8_e4m3fn` ops to hardware.
Beta 2 `neuronx-cc` rejects `f8e4m3fn` at compile time (`Unsupported element type`).

¹ sq_32768 bf16: public path runs but performance degrades (shape too large for single-NC tiling);
Beta 2 neuronx-cc hits a compiler limit.

## Why Beta 2 is faster

The public XLA path compiles a single-NC NEFF — it is designed for multi-process tensor
parallelism where each rank owns one NeuronCore.
Beta 2 `torch_neuron_eager` drives all 4 physical NeuronCores in a single process via
`torch.compile(backend="neuron")`, giving ~2.9× higher TFLOPS at sq_16384
(134 vs 47 TFLOPS) and up to ~4× for larger square shapes.

The 41% MFU ceiling at sq_16384 reflects Beta 2 compiler immaturity rather than hardware
limits — `neuronx-cc` in the public path achieves 58% MFU on a single NC.

## Files

| File | Purpose |
|---|---|
| `matmul_benchmark.py` | Benchmark script — runs on either path |
| `pyproject.toml` / `uv.lock` | Public path deps (torch-neuronx, libneuronxla, neuronx-cc) |
| `Dockerfile` | Public path image |
| `entrypoint.sh` | Public path entrypoint |
| `Dockerfile.beta2` | Beta 2 image (requires Anthropic Beta 2 base image) |
| `entrypoint.beta2.sh` | Beta 2 entrypoint |

## Running

```bash
# Public path (requires trn2 instance)
docker build -f Dockerfile -t matmul-bench .
docker run --privileged matmul-bench

# Beta 2 path (requires access to Anthropic Beta 2 base image)
docker build -f Dockerfile.beta2 -t matmul-bench-beta2 .
docker run --privileged matmul-bench-beta2
```

Results are written to S3 (`$S3_PREFIX/results/`) and logs to `$S3_PREFIX/logs/`.

### AWS Batch (sa-east-1, odyssey-trainium-queue)

| Job definition | Image tag | Path |
|---|---|---|
| `trn2-matmul-bench:3` | `matmul-bench` | Public |
| `trn2-matmul-bench:4` | `matmul-bench-beta2` | Beta 2 |
