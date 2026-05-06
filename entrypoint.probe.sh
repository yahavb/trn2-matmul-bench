#!/bin/bash
# Trainium 2 NC topology probe entrypoint.
set -eo pipefail

export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"

PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

mkdir -p /host-tmp
export TMPDIR=/host-tmp

echo "=== neuron-ls ==="
neuron-ls 2>&1 || echo "(neuron-ls not found)"

TS="$(date -u '+%Y%m%d-%H%M%SZ')"
OUTPUT="${OUTPUT:-/host-tmp/nc_probe_${TS}.json}"
LOG_FILE="nc_probe_${TS}.log"

/bench/.venv/bin/python3 /bench/nc_probe.py \
    --warmup "${WARMUP:-2}" \
    --reps   "${REPS:-5}" \
    --output "$OUTPUT" \
    2>&1 | tee "$LOG_FILE"

S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"
aws s3 cp "$LOG_FILE" "${S3_PREFIX}/logs/${LOG_FILE}"              --region "$REGION"
[[ -f "$OUTPUT" ]] && aws s3 cp "$OUTPUT" "${S3_PREFIX}/results/$(basename "$OUTPUT")" --region "$REGION"
echo "[probe] uploaded: ${S3_PREFIX}/results/$(basename "$OUTPUT")"
