FROM 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest

ENV PYTHONUNBUFFERED=1
# Override the Beta 2 image's UV_PROJECT_ENVIRONMENT which points to
# /workspace/torch_neuron_eager/.venv and would cause uv sync to install
# packages there instead of /bench/.venv.
ENV UV_PROJECT_ENVIRONMENT=/bench/.venv

# The Beta 2 image ships Neuron runtime packages (NRT) pre-installed via apt.
# Copy the .so files to /usr/local/lib so they are discoverable without
# adding /opt/aws/neuron/lib to LD_LIBRARY_PATH explicitly; ldconfig registers
# them. This mirrors the pattern in the odyssey trainium Dockerfile.
RUN find /opt/aws/neuron/lib -maxdepth 1 -name '*.so*' \
        | xargs -I{} cp -P {} /usr/local/lib/ && \
    ldconfig

# uv manages the Python environment end-to-end.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /bench

# Install Python deps at build time from the locked requirements.
# torch: CPU wheel (device dispatch handled by XLA/Neuron, not CUDA).
# torch-neuronx / libneuronxla / neuronx-cc: from the public Neuron pip index.
COPY pyproject.toml uv.lock ./
RUN /usr/local/bin/uv sync --no-install-project --frozen

COPY matmul_benchmark.py entrypoint.sh ./
RUN chmod +x entrypoint.sh

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/bench/entrypoint.sh"]
