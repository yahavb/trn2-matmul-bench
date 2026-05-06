#!/bin/bash
# Trainium 2 matmul benchmark entrypoint — Beta 2 torch_neuron_eager path.
# Uses torch.compile(backend="neuron") + torch_neuronx.synchronize().
set -eo pipefail

export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"

# NEURON_RT_NUM_CORES: Beta-2 NRT (2.x.47730.0) accepts only 1 or multiples of 8.
[[ -n "${NEURON_RT_NUM_CORES:-}" ]] && export NEURON_RT_NUM_CORES
echo "[bench] NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-(unset, NRT default)}"

echo "=== neuron-ls ==="
neuron-ls 2>&1 || echo "(neuron-ls not found)"

PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

mkdir -p /host-tmp
export TMPDIR=/host-tmp

TS="$(date -u '+%Y%m%d-%H%M%SZ')"
OUTPUT="${OUTPUT:-/host-tmp/trn2_matmul_bench_${TS}.json}"
LOG_FILE="trn2_matmul_bench_${TS}.log"

# Beta 2: torch_neuron_eager has no trace(); use compile backend="neuron".
# matmul_benchmark.py auto-detects _HAS_TRACE=False and defaults to these,
# but we pass them explicitly for clarity.
/bench/.venv/bin/python3 /bench/matmul_benchmark.py \
    --warmup "${WARMUP:-3}" \
    --reps   "${REPS:-10}" \
    --methods compile \
    --compile-backend neuron \
    --output "$OUTPUT" \
    2>&1 | tee "$LOG_FILE"

S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"
aws s3 cp "$LOG_FILE" "${S3_PREFIX}/logs/${LOG_FILE}"              --region "$REGION"
[[ -f "$OUTPUT" ]] && aws s3 cp "$OUTPUT" "${S3_PREFIX}/results/$(basename "$OUTPUT")" --region "$REGION"
echo "[bench] uploaded: ${S3_PREFIX}/results/$(basename "$OUTPUT")"
