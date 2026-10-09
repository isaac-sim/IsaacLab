FROM isaac-lab-base:latest

USER root

# Cloud readiness checks require wget.
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# Set NVIDIA paths
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"

# Link NVIDIA binaries
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin

RUN uv pip install --no-cache --python ${VIRTUAL_ENV}/bin/python \
    "ray[default,tune]==2.31.0" optuna bayesian-optimization mlflow

USER isaaclab
