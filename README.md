# trn2-matmul-bench

Standalone matmul peak microbenchmark for a single Trainium 2 chip (`trn2.3xlarge`).
Measures bf16 / fp8 / mxfp8 across square powers-of-two and WAN-shaped GEMMs.
Reports achieved TFLOP/s and MFU% against the logical-NC peak.

## Hardware

`trn2.3xlarge` — 1 Trainium 2 chip, 2 dies:

| | |
|---|---|
| Neuron devices (PCI) | 1 |
| Dies per chip | 2 |
| **Physical NeuronCores** | **8** (4 per die, NeuronCore-v3) |
| Logical NeuronCore config | lnc=2 (default) → **4 logical NCs** (each = 2 physical NCs) |
| HBM | 96 GB (24 GB per logical NC) |
| Peak bf16 — full chip | ~667 TFLOPS |
| Peak bf16 — per die | ~333 TFLOPS |
| **Peak bf16 — per logical NC** | **~167 TFLOPS** |
| Peak fp8 — per logical NC | ~334 TFLOPS |

`torch_neuronx.device_count()` returns the count of **logical NCs at the current lnc setting**:

| env | `device_count()` | meaning |
|---|---:|---|
| (default) | 4 | 4 logical NCs (lnc=2) |
| `NEURON_LOGICAL_NC_CONFIG=1` | 8 | 8 physical NCs (no grouping) |

The default `neuron-ls` output displays the *logical* view (`| 0 | 4 | 0-3 | 96 GB |`)
which hides the underlying 8 physical cores; setting `NEURON_LOGICAL_NC_CONFIG=1`
reveals `| 0 | 8 | 0-7 | 96 GB |`.

## NC topology — verified experimentally

`nc_probe.py` ran sq_16384 bf16 on 1–4 logical NCs simultaneously, using separate
OS processes each restricted to one LNC via `NEURON_RT_VISIBLE_CORES` (logical NC index).

| Config | VISIBLE_CORES | Aggregate TF/s | Scaling |
|---|---|---:|---|
| 1 LNC | `0` | 134.7 | 1.00× baseline |
| 2 LNCs, same die (0+1) | `0`, `1` | 228.5 | 1.70× — intra-die HBM contention |
| 2 LNCs, different dies (0+2) | `0`, `2` | **268.2** | **2.00×** — fully independent |
| 4 LNCs (full chip) | `0`,`1`,`2`,`3` | 455.1 | 3.38× — 2 contended pairs |

The (0,1) vs (0,2) asymmetry pins down the die layout:
LNCs 0,1 sit on die 0 and LNCs 2,3 on die 1.

**Key findings:**

- A single `torch.compile(backend="neuron")` process uses exactly **1 logical NC** (2 physical NCs, ~167 TFLOPS peak).
  134.7 / 167 = **80.7% MFU** at sq_16384.
- Cross-die LNCs are fully independent (2.00× scaling at sq_16384).
- Same-die LNCs share HBM bandwidth — each drops to ~114 TF/s (~85% efficiency).
- Full-chip aggregate at sq_16384: **455 TF/s** (not 4×134=537), limited by intra-die contention.

## Two benchmark paths

| | Public (`matmul-bench`) | Torch Neuron Beta 2 (`matmul-bench-beta2`) |
|---|---|---|
| Package | `torch-neuronx==2.9.0.2.13.24727` (public Neuron pip index) | `torch_neuron_eager` from `/workspace/` in Torch Neuron Beta 2 base image |
| Compilation | `torch_neuronx.trace()` — AOT XLA | `torch.compile(backend="neuron")` — JIT |
| NeuronCores used | **1 physical NC** | **1 logical NC (2 physical NCs)** |
| Effective peak (bf16) | **~83 TFLOPS** (1/8 chip) | **~167 TFLOPS** (1/4 chip) |
| Device | `torch_xla.device()` (XLA) | `torch.device("neuron:0")` |
| Sync | `torch_xla.sync()` + `xm.wait_device_ops()` | `torch_neuronx.synchronize()` |
| torch version | `torch>=2.1` (CPU wheel) | `torch==2.10.0+cpu` |

## Measured results — bf16

Hardware: `trn2.3xlarge` · Warmup 3 · Reps 10 · 2026-05-06

MFU% = achieved / effective NC peak (~83 TFLOPS for public 1-physical-NC path; ~167 TFLOPS for Beta 2 1-logical-NC path).

| Shape | M | K | N | pub/trace TF/s (MFU%) | b2/neuron TF/s (MFU%) |
|---|---:|---:|---:|---:|---:|
| sq_1024 | 1024 | 1024 | 1024 | 6.9 (8%) | 9.9 (6%) |
| sq_2048 | 2048 | 2048 | 2048 | 18.3 (22%) | 28.5 (17%) |
| sq_4096 | 4096 | 4096 | 4096 | 25.3 (30%) | 91.3 (55%) |
| sq_8192 | 8192 | 8192 | 8192 | 31.6 (38%) | 117.0 (70%) |
| **sq_16384** | 16384 | 16384 | 16384 | **46.9 (56%)** | **134.4 (81%)** |
| sq_32768 | 32768 | 32768 | 32768 | 26.8 (32%) | FAIL¹ |
| wan_qkv_1.3b_480p | 1590 | 1536 | 4608 | 14.1 (17%) | 32.5 (19%) |
| wan_o_1.3b_480p | 1590 | 1536 | 1536 | 11.1 (13%) | 13.4 (8%) |
| wan_ffn1_1.3b_480p | 1590 | 1536 | 8960 | 14.7 (18%) | 47.2 (28%) |
| wan_ffn2_1.3b_480p | 1590 | 8960 | 1536 | 42.0 (50%) | 51.3 (31%) |
| wan_qkv_14b_480p | 1590 | 5120 | 15360 | 30.3 (36%) | 84.9 (51%) |
| wan_o_14b_480p | 1590 | 5120 | 5120 | 41.9 (50%) | 57.6 (35%) |
| wan_ffn1_14b_480p | 1590 | 5120 | 13824 | 30.0 (36%) | 84.4 (51%) |
| wan_ffn2_14b_480p | 1590 | 13824 | 5120 | 71.7 (86%) | 80.7 (48%) |
| wan_qkv_14b_720p | 3600 | 5120 | 15360 | 22.8 (27%) | 99.8 (60%) |
| wan_ffn1_14b_720p | 3600 | 5120 | 13824 | 24.0 (29%) | 97.8 (59%) |
| **wan_ffn2_14b_720p** | 3600 | 13824 | 5120 | **58.7 (70%)** | **104.4 (63%)** |

**fp8 / mxfp8:** all shapes fail on both paths.
Public `torch-neuronx` trace does not yet lower `float8_e4m3fn` ops to hardware.
Torch Neuron Beta 2 `neuronx-cc` rejects `f8e4m3fn` at compile time (`Unsupported element type`).

¹ sq_32768 bf16: public path runs but performance degrades (shape too large for single-NC tiling);
Beta 2 neuronx-cc hits a compiler limit.

## Why Beta 2 is faster

The public XLA path compiles a single-NC NEFF (~83 TFLOPS peak).
The Torch Neuron Beta 2 path uses `torch.compile(backend="neuron")` which targets a full
logical NC (2 physical NCs, ~167 TFLOPS peak), giving ~2.9× higher achieved TFLOPS at
sq_16384 (134 vs 47 TFLOPS).

Both paths approach their respective NC peaks at large shapes:
the public path reaches 86% MFU at wan_ffn2_14b_480p; Beta 2 reaches 81% MFU at sq_16384.

## Files

| File | Purpose |
|---|---|
| `matmul_benchmark.py` | Benchmark script — runs on either path |
| `nc_probe.py` | NC topology probe — verifies logical NC independence via parallel processes |
| `device_check.py` | Prints device count and properties with no env overrides |
| `pyproject.toml` / `uv.lock` | Public path deps (torch-neuronx, libneuronxla, neuronx-cc) |
| `Dockerfile` | Public path image |
| `entrypoint.sh` | Public path entrypoint |
| `Dockerfile.beta2` | Torch Neuron Beta 2 image (requires Beta 2 base image) |
| `entrypoint.beta2.sh` | Torch Neuron Beta 2 entrypoint |
| `Dockerfile.probe` | NC topology probe image (extends matmul-bench-beta2) |
| `entrypoint.probe.sh` | NC topology probe entrypoint |
| `entrypoint.device_check.sh` | Device check entrypoint |

## Running

```bash
# Public path (requires trn2 instance)
docker build -f Dockerfile -t matmul-bench .
docker run --privileged matmul-bench

# Torch Neuron Beta 2 path (requires access to Beta 2 base image)
docker build -f Dockerfile.beta2 -t matmul-bench-beta2 .
docker run --privileged matmul-bench-beta2

# NC topology probe (extends matmul-bench-beta2)
docker build -f Dockerfile.probe -t matmul-bench-probe .
docker run --privileged matmul-bench-probe
```

Results are written to S3 (`$S3_PREFIX/results/`) and logs to `$S3_PREFIX/logs/`.

### AWS Batch (sa-east-1, odyssey-trainium-queue)

| Job definition | Image tag | Path |
|---|---|---|
| `trn2-matmul-bench:3` | `matmul-bench` | Public |
| `trn2-matmul-bench:4` | `matmul-bench-beta2` | Torch Neuron Beta 2 |
| `trn2-matmul-bench:6` | `matmul-bench-probe` | NC topology probe / device check |
