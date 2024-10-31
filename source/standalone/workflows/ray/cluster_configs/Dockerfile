FROM isaac-lab-base:latest
# Set path so that NVIDIA shared object files can be found
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
# Link NVIDIA shared object files to be accessible via bin so they can be found on default path
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin
# Install Ray
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install "ray[default, tune]"==2.31.0 && \
sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" # Make Ray use Isaac Sim Python Executable \
/isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray # Link ray into bin so it can be used with ra" command
# The following is only needed for tuning
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install optuna bayesian-optimization # tuning specific deps
# The following is only needed for tuning logging
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install mlflow
