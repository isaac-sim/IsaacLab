# Install Ray, as well as the needed things for GKE GPU
FROM isaac-lab-base:latest
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin
RUN /workspace/isaaclab/_isaac_sim/kit/python/bin/python3 -m pip install ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray
