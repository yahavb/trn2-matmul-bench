#!/bin/bash
# Trainium 2 matmul benchmark entrypoint.
# Mirrors the pattern in the odyssey trainium entrypoint_trn_op_benchmark.sh.
set -eo pipefail

export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"
# NEURON_RT_NUM_CORES note (TRN2 / Beta-2 NRT 2.x.47730.0):
#   The Beta-2 NRT rejects NEURON_RT_NUM_CORES=2 with:
#     "must request one core, or the whole device (multiple of 8)"
#   trn2.3xlarge has 4 physical NeuronCores in 2 logical NCs (lnc=2).
#   With =1: NRT accepts; bench runs on 1 physical NC (~47 TFLOPS for sq_16384).
#   With UNSET: NRT should default to the full logical NC; prior runs showed
#     ~130 TFLOPS for sq_16384 — hypothesis being tested overnight.
#   Override via the Batch job definition env or the NEURON_RT_NUM_CORES env var.
[[ -n "${NEURON_RT_NUM_CORES:-}" ]] && export NEURON_RT_NUM_CORES
echo "[bench] NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-(unset, NRT default)}"

echo "=== neuron-ls ==="
neuron-ls 2>&1 || echo "(neuron-ls not found)"

# Patch libneuronxla to resolve libnrt from /usr/local/lib rather than the
# hardcoded /opt/aws/neuron/lib path baked into the pre-built wheel.
LIBNRT_PY=$(ls /bench/.venv/lib/python3.*/site-packages/libneuronxla/libnrt.py 2>/dev/null | head -1)
[[ -f "$LIBNRT_PY" ]] && sed -i 's|/opt/aws/neuron/lib/libnrt|/usr/local/lib/libnrt|g' "$LIBNRT_PY"

PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

mkdir -p /host-tmp
export TMPDIR=/host-tmp

TS="$(date -u '+%Y%m%d-%H%M%SZ')"
OUTPUT="${OUTPUT:-/host-tmp/trn2_matmul_bench_${TS}.json}"
LOG_FILE="trn2_matmul_bench_${TS}.log"

/bench/.venv/bin/python3 /bench/matmul_benchmark.py \
    --warmup "${WARMUP:-3}" \
    --reps   "${REPS:-10}" \
    --output "$OUTPUT" \
    2>&1 | tee "$LOG_FILE"

S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"
aws s3 cp "$LOG_FILE" "${S3_PREFIX}/logs/${LOG_FILE}"              --region "$REGION"
[[ -f "$OUTPUT" ]] && aws s3 cp "$OUTPUT" "${S3_PREFIX}/results/$(basename "$OUTPUT")" --region "$REGION"
echo "[bench] uploaded: ${S3_PREFIX}/results/$(basename "$OUTPUT")"
