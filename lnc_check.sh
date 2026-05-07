#!/bin/bash
# Probe what neuron-ls and torch_neuronx report under different lnc / env settings.
set +e
export PATH="/bench/.venv/bin:/opt/aws/neuron/bin:$PATH"
PYTHON_LIBDIR=$(/bench/.venv/bin/python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH="/usr/local/lib:${PYTHON_LIBDIR}:${LD_LIBRARY_PATH:-}"

dump() {
    echo "================================================================"
    echo "$1"
    echo "================================================================"
    shift
    "$@"
    echo
}

dump "neuron-ls (default, no env)" \
    neuron-ls

dump "neuron-ls --help" \
    neuron-ls --help

dump "neuron-ls -l (if supported)" \
    neuron-ls -l

dump "neuron-ls --topology (if supported)" \
    neuron-ls --topology

dump "NEURON_RT_LOGICAL_NC_CONFIG=1 neuron-ls" \
    env NEURON_RT_LOGICAL_NC_CONFIG=1 neuron-ls

dump "NEURON_LOGICAL_NC_CONFIG=1 neuron-ls" \
    env NEURON_LOGICAL_NC_CONFIG=1 neuron-ls

dump "NEURON_RT_NUM_CORES=8 neuron-ls" \
    env NEURON_RT_NUM_CORES=8 neuron-ls

dump "ls /sys/devices/* (PCI Neuron devices)" \
    bash -c 'ls -la /sys/bus/pci/devices/ 2>/dev/null | grep -i 0000 | head'

dump "lspci | grep -i neuron" \
    bash -c 'lspci 2>/dev/null | grep -i -E "neuron|amazon|annap" || echo "(no matches)"'

dump "neuron-monitor --help (if available)" \
    bash -c 'which neuron-monitor && neuron-monitor --help 2>&1 | head -30 || echo "(neuron-monitor not found)"'

dump "ls /opt/aws/neuron/bin" \
    ls /opt/aws/neuron/bin

dump "torch_neuronx.device_count() under various lnc env" \
    /bench/.venv/bin/python3 - <<'PY'
import os, json, subprocess
configs = [
    {},
    {"NEURON_RT_LOGICAL_NC_CONFIG": "1"},
    {"NEURON_LOGICAL_NC_CONFIG":    "1"},
    {"NEURON_RT_LOGICAL_NEURON_CORES": "1"},
]
for env_overrides in configs:
    e = {**os.environ, **env_overrides}
    out = subprocess.run(
        ["/bench/.venv/bin/python3", "-c",
         "import torch_neuronx; print(torch_neuronx.device_count())"],
        env=e, capture_output=True, text=True, timeout=120,
    )
    print(f"env={env_overrides}  device_count -> stdout={out.stdout.strip()!r}  "
          f"stderr={out.stderr.strip()[-200:]!r}  returncode={out.returncode}")
PY

TS="$(date -u '+%Y%m%d-%H%M%SZ')"
LOG_FILE="lnc_check_${TS}.log"
S3_PREFIX="${S3_PREFIX:-s3://ody-trainium-cache-sae1/matmul-bench}"
REGION="${AWS_DEFAULT_REGION:-sa-east-1}"
# This script writes everything to stdout already; CloudWatch + S3 capture it.
echo "[done] timestamp=${TS}"
