# This dockerfile only works because of Felix Yu's help #TODO: Add Felix to contributors list
FROM isaac-lab-base:latest
ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"
RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install "ray[default, tune]"==2.31.0 && \
sed -i "1i $(echo "#!/workspace/isaaclab/_isaac_sim/python.sh")" \
/isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray
# The following is only needed for tuning
RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install hpbandster ConfigSpace
