#!/bin/bash
# Runs inside the Beta 2 Docker container on a Trainium 2 node.
# Sets up the Python environment from the Beta 2 workspace, then runs the
# benchmark and uploads results to S3.
set -euo pipefail

# Restrict to a single TRN2 die (2 NeuronCores).
export NEURON_RT_NUM_CORES="${NEURON_RT_NUM_CORES:-2}"

# ---- environment setup -------------------------------------------------------
python3.12 -m venv --system-site-packages /venv
source /venv/bin/activate

# Install Beta 2 private wheels from the image's /workspace directory.
pip install --quiet \
    /workspace/neuron_torch_mlir_wheels/*.whl \
    /workspace/nki_wheels/*.whl \
    /workspace/neuronx_cc_wheels/*.whl
pip install --quiet --no-deps -e /workspace/torch_neuron_eager

# ---- run benchmark -----------------------------------------------------------
TS=$(date -u '+%Y%m%d-%H%M%SZ')
OUTPUT="${OUTPUT:-/tmp/trn2_matmul_bench_${TS}.json}"
LOG="/tmp/trn2_matmul_bench_${TS}.log"

python3 /bench/matmul_benchmark.py \
    --warmup "${WARMUP:-3}" \
    --reps   "${REPS:-10}" \
    --output "$OUTPUT" \
    2>&1 | tee "$LOG"

# ---- upload to S3 ------------------------------------------------------------
S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"

aws s3 cp "$LOG"    "${S3_PREFIX}/logs/$(basename "$LOG")"       --region "$REGION" || true
aws s3 cp "$OUTPUT" "${S3_PREFIX}/results/$(basename "$OUTPUT")" --region "$REGION" || true
echo "[bench] uploaded to ${S3_PREFIX}/results/$(basename "$OUTPUT")"
