#!/bin/bash
set -eo pipefail
export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"
PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

echo "=== env ==="
env | grep -i neuron || echo "(no NEURON_* vars set)"

echo "=== neuron-ls ==="
neuron-ls 2>&1 || true

echo "=== device_check ==="
/bench/.venv/bin/python3 /bench/device_check.py

TS="$(date -u '+%Y%m%d-%H%M%SZ')"
S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
/bench/.venv/bin/python3 /bench/device_check.py \
    > /tmp/device_check_${TS}.json
aws s3 cp /tmp/device_check_${TS}.json \
    "${S3_PREFIX}/results/device_check_${TS}.json" \
    --region "${AWS_DEFAULT_REGION:-sa-east-1}"
