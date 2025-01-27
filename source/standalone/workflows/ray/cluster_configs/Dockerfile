FROM isaac-lab-base:latest

# WGet is needed so that GCS or other cloud providers can mark the container as ready.
# Otherwise the Ray liveliness checks fail.
RUN apt-get update && apt-get install wget

# Set NVIDIA paths
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"

# Link NVIDIA binaries
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin

# Install Ray and configure it
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install "ray[default, tune]"==2.31.0 && \
sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
/isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray

# Install tuning dependencies
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install optuna bayesian-optimization

# Install MLflow for logging
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install mlflow
