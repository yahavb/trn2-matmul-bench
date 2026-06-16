# trn2 Collective Benchmark Results

**Date:** 2026-06-09  
**Hardware:** trn2 (1 chip = 2 dies = 8 physical NCs = 4 LNCs with lnc=2)  
**Peak:** 632 BF16 TFLOPS/chip = 158 TFLOPS/LNC  
**Matrix:** 16384×16384 bf16 (8.80 TFLOPS per matmul)  
**Compilation:** `torch.compile(backend="neuron")`  

## Three Topologies

| # | Mode | Description | TP | Interconnect |
|---|------|-------------|---:|---|
| 1 | single-core | 1 LNC, no collectives | 1 | N/A |
| 2 | single-device | 4 LNCs on 1 chip | 4 | Die-to-die (400 GB/s) |
| 3 | multi-device | 8 LNCs on 2 chips | 8 | NeuronLink (128 GB/s) |

## Results

| Mode | Op | Latency (us) | TF/s | MFU% | Comm Overhead |
|------|-----|---:|---:|---:|---:|
| single-core | matmul_only | 63,607 | 138.3 | 87.5% | — |
| single-device (TP=4) | matmul_only | 75,593 | 116.4 | 73.6% | — |
| single-device (TP=4) | all_reduce | 128,846 | 68.3 | 43.2% | 41.3% |
| single-device (TP=4) | all_gather | 93,792 | 93.8 | 59.4% | 19.4% |
| single-device (TP=4) | reduce_scatter | — | — | — | RDH buffer overflow |
| multi-device (TP=8) | matmul_only | 75,549 | 116.4 | 73.7% | — |
| multi-device (TP=8) | all_reduce | 130,716 | 67.3 | 42.6% | 42.2% |
| multi-device (TP=8) | all_gather | 118,975 | 73.9 | 46.8% | 36.5% |
| multi-device (TP=8) | reduce_scatter | 81,696 | 107.7 | 68.1% | 7.5% |

## Key Findings

1. **Compute entitlement:** Single LNC achieves 87.5% MFU — the ceiling for this compilation path.

2. **HBM contention penalty:** Going from 1 LNC to 4 LNCs on the same chip drops matmul_only from 87.5% → 73.6% MFU (14% loss). Multiple LNCs sharing HBM stacks (2 LNCs per 716 GB/s stack) creates bandwidth contention even without collectives.

3. **Cross-device adds no compute penalty:** matmul_only is identical at TP=4 (73.6%) and TP=8 (73.7%). The inter-chip link only matters for collectives.

4. **all_gather is link-sensitive:** Communication overhead jumps from 19.4% (on-chip D2D, 400 GB/s) to 36.5% (cross-chip NeuronLink, 128 GB/s) — a +17% penalty from the 3× slower link.

5. **all_reduce is link-insensitive:** ~41-43% overhead regardless of topology. The ring algorithm or runtime overlap hides the slower link.

6. **reduce_scatter is the most efficient collective:** Only 7.5% overhead at TP=8, sending less data per rank.

## Interconnect Bandwidth Reference

| Link | Bandwidth | Where |
|------|---:|---|
| Die-to-die (D2D) | 400 GB/s | Between 2 dies on same chip |
| NeuronLink (per link) | 128 GB/s | Between chips (4× PCIe Gen5 x8) |
| HBM3 stack | 716 GB/s | Shared by 2 LNCs |
