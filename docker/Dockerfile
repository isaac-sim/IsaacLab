# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Nvidia Dockerfiles:
# https://github.com/NVIDIA-Omniverse/IsaacSim-dockerfiles

# Base image
ARG ISAACSIM_VERSION
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION}

# Adds labels to the Dockerfile
LABEL version="1.0"
LABEL description="Dockerfile for building and running the Orbit framework inside Isaac Sim container image."

# Arguments
# Path to Isaac Sim root folder
ARG ISAACSIM_PATH

# Set environment variables
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV ORBIT_PATH=/workspace/orbit

# Install dependencies and remove cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ncurses-term && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

# FIXME: Only necessary for streaming until this fix is properly rolled out by NVIDIA after Isaac Sim2023.1 release
# Ref: https://forums.developer.nvidia.com/t/running-a-standalone-example-with-gui-in-docker-container/248147/3
RUN if [[ "${ISAACSIM_VERSION}" == "2022.2"* ]]; then \
        sed -i 's/\("omni.isaac.quadruped"\s=\s{}\)/"omni.isaac.quadruped" = {order = 10}/g' \
        ${ISAACSIM_PATH}/apps/omni.isaac.sim.python.kit; \
    fi

# Copy the orbit directory
COPY ../ ${ORBIT_PATH}
# Delete the logs directory
RUN rm -rf ${ORBIT_PATH}/logs

# Set up a symbolic link between the installed Isaac Sim root folder and _isaac_sim in the orbit directory
RUN ln -sf ${ISAACSIM_PATH} ${ORBIT_PATH}/_isaac_sim

# installing Orbit dependencies
RUN ${ORBIT_PATH}/orbit.sh --install --extra
# aliasing orbit.sh and python for convenience
RUN echo "alias orbit=${ORBIT_PATH}/orbit.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACSIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACSIM_PATH}/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACSIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACSIM_PATH}/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACSIM_PATH}/python.sh ${ISAACSIM_PATH}/tensorboard'" >> ${HOME}/.bashrc

# make working directory as the orbit directory
# this is the default directory when the container is run
WORKDIR ${ORBIT_PATH}
