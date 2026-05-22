#!/bin/bash
# MFU headroom + NC scaling analysis entrypoint.
# Runs both scripts sequentially: size sweep then multi-NC scaling.
set -eo pipefail

export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"

[[ -n "${NEURON_RT_NUM_CORES:-}" ]] && export NEURON_RT_NUM_CORES
echo "[headroom] NEURON_RT_NUM_CORES=${NEURON_RT_NUM_CORES:-(unset, NRT default)}"

echo "=== neuron-ls ==="
neuron-ls 2>&1 || echo "(neuron-ls not found)"

PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

mkdir -p /host-tmp
export TMPDIR=/host-tmp

TS="$(date -u '+%Y%m%d-%H%M%SZ')"

# --- Phase 1: Single-NC size sweep (headroom analysis) ---
HEADROOM_OUTPUT="/host-tmp/mfu_headroom_${TS}.json"
HEADROOM_LOG="mfu_headroom_${TS}.log"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 1: MFU Headroom (single-NC size sweep)"
echo "═══════════════════════════════════════════════════════"

/bench/.venv/bin/python3 /bench/mfu_headroom.py \
    --warmup "${WARMUP:-3}" \
    --reps   "${REPS:-10}" \
    --peak-tflops "${PEAK_TFLOPS:-167.0}" \
    --compile-backend "${COMPILE_BACKEND:-neuron}" \
    --output "$HEADROOM_OUTPUT" \
    ${SIZES:+--sizes $SIZES} \
    2>&1 | tee "$HEADROOM_LOG"

# --- Phase 2: Multi-NC scaling sweep ---
SCALING_OUTPUT="/host-tmp/nc_scaling_${TS}.json"
SCALING_LOG="nc_scaling_${TS}.log"

echo ""
echo "═══════════════════════════════════════════════════════"
echo " Phase 2: NC Scaling Efficiency (multi-NC sweep)"
echo "═══════════════════════════════════════════════════════"

/bench/.venv/bin/python3 /bench/nc_scaling.py \
    --warmup "${WARMUP:-3}" \
    --reps   "${REPS:-10}" \
    --peak-per-lnc "${PEAK_TFLOPS:-167.0}" \
    --compile-backend "${COMPILE_BACKEND:-neuron}" \
    --output "$SCALING_OUTPUT" \
    ${SCALING_SIZES:+--sizes $SCALING_SIZES} \
    ${SCALING_CONFIGS:+--configs $SCALING_CONFIGS} \
    2>&1 | tee "$SCALING_LOG"

# --- Upload results ---
S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"

aws s3 cp "$HEADROOM_LOG"    "${S3_PREFIX}/logs/${HEADROOM_LOG}"         --region "$REGION"
aws s3 cp "$SCALING_LOG"     "${S3_PREFIX}/logs/${SCALING_LOG}"          --region "$REGION"
[[ -f "$HEADROOM_OUTPUT" ]] && aws s3 cp "$HEADROOM_OUTPUT" "${S3_PREFIX}/results/$(basename "$HEADROOM_OUTPUT")" --region "$REGION"
[[ -f "$SCALING_OUTPUT" ]]  && aws s3 cp "$SCALING_OUTPUT"  "${S3_PREFIX}/results/$(basename "$SCALING_OUTPUT")"  --region "$REGION"

echo ""
echo "[headroom] uploaded results to ${S3_PREFIX}/results/"
echo "[headroom] done."
