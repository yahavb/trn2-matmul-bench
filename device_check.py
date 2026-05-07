#!/usr/bin/env python3
"""Print torch_neuronx device count and properties with no env overrides."""
import json, os, socket, sys
import torch
import torch_neuronx

n = torch_neuronx.device_count()
devs = {}
for i in range(n):
    props = torch_neuronx.get_device_properties(i)
    devs[f"neuron:{i}"] = str(props)

result = {
    "hostname": socket.gethostname(),
    "NEURON_RT_VISIBLE_CORES": os.environ.get("NEURON_RT_VISIBLE_CORES", "(unset)"),
    "NEURON_RT_NUM_CORES":     os.environ.get("NEURON_RT_NUM_CORES",     "(unset)"),
    "device_count": n,
    "devices": devs,
}
print(json.dumps(result, indent=2))
