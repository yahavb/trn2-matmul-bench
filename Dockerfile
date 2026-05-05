# TRN2 matmul benchmark — built on the Neuron Beta 2 base image.
# The base image ships /workspace with torch_neuron_eager, NKI wheels, and
# neuronx_cc wheels; the venv is assembled from those at container start time
# (see entrypoint.sh) so there are no public PyPI pulls for the Neuron stack.
FROM 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

ENV PYTHONUNBUFFERED=1

COPY matmul_benchmark.py /bench/matmul_benchmark.py
COPY entrypoint.sh       /bench/entrypoint.sh
RUN chmod +x /bench/entrypoint.sh

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bench/entrypoint.sh"]
