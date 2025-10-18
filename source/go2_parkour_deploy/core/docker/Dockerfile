# syntax=docker/dockerfile:experimental
# To pull this image, create an account and login according to
# this SO post:
#    https://stackoverflow.com/a/70970425

# For Jetpack 4.6.1
# (Our Go1 NX has 4.5, but this still works and comes with Pytorch 1.10)
# (See https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-pytorch for list of base images)
FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt update -y
RUN apt install -y --no-install-recommends \
        build-essential \
        vim \
        tree \
        curl \
        sudo \
        less \
        emacs \
        apt-utils \
        tzdata \
        git \
        tmux \
        bash-completion \
        command-not-found \
        libglib2.0-dev \
        libglib2.0-0 \
        liblcm-dev \
        python-dev \
        python3-dev \
        cmake \
        protobuf-compiler \
        dirmngr \
        gpg-agent \
        libssl-dev \
        libcurl4-openssl-dev \
        freeglut3-dev \
        libudev-dev \
        libgl1-mesa-dev \
        libudev1 \
        libudev-dev \
        libgtk2.0-dev 
        
# Install LCM
# (See https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym_deploy/docker/Dockerfile)
RUN apt install -y make gcc-8 g++-8
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
RUN cd /root && git clone https://github.com/lcm-proj/lcm.git
RUN cd /root/lcm && git checkout tags/v1.5.0 && mkdir build && cd build && cmake ..
RUN cd /root/lcm/build make -j
RUN cd /root/lcm/build make install
RUN cd /root/lcm/lcm-python && python3 -m pip install -e .

# Update CMake
# This is necessary for Realsense Python wrapper to build correctly, otherwise importing pyrealsense2 doesn't work
# (See https://github.com/IntelRealSense/librealsense/issues/6980#issuecomment-666858977)
RUN cd /root && wget -O cmake.tar.gz http://www.cmake.org/files/v3.13/cmake-3.13.0.tar.gz
RUN cd /root && tar -xvzf cmake.tar.gz
RUN cd /root && mv cmake-3.13.0 cmake && rm cmake.tar.gz
RUN sudo apt-get install libxinerama-dev
RUN sudo apt-get install libxcursor-dev
RUN sudo apt-get install libjsoncpp-dev 
RUN cd /root/cmake && ./bootstrap --system-curl
RUN cd /root/cmake && make -j4
ENV PATH=/root/cmake/bin:${PATH}

# Build and Install Realsense
# (See https://github.com/35selim/RealSense-Jetson/blob/main/build_pyrealsense2_and_SDK.sh)
# (See https://github.com/IntelRealSense/librealsense/issues/6964#issuecomment-707501049)
# (See https://github.com/IntelRealSense/librealsense/issues/8324)
RUN cd /root && wget -O librealsense.tar.gz https://github.com/IntelRealSense/librealsense/archive/refs/tags/v2.55.1.tar.gz
RUN cd /root && tar -xvzf librealsense.tar.gz
RUN cd /root && mv librealsense-2.55.1 librealsense && rm librealsense.tar.gz
RUN cd /root/librealsense && mkdir build
RUN apt-get install git libssl-dev libusb-1.0-0-dev pkg-config libgtk-3-dev -y
RUN cd /root/librealsense/build && cmake ../ -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS:bool=true -DPYTHON_EXECUTABLE=/usr/bin/python3.8 -DCMAKE_BUILD_TYPE=release -DBUILD_EXAMPLES=true -DBUILD_GRAPHICAL_EXAMPLES=true -DBUILD_WITH_CUDA:bool=true
RUN cd /root/librealsense/build && make -j4
RUN cd /root/librealsense/build && sudo make install

# Needed to find pyrealsense .so files, for some reason they're not in the right place
# (See https://github.com/IntelRealSense/librealsense/issues/10891)
ENV PYTHONPATH=${PYTHONPATH}:/usr/local/lib:/root/librealsense/build/release

RUN python3 -m pip --version
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install opencv-python
RUN python3 -m pip install jetson-stats
ENV PIP_NO_CACHE_DIR=1

RUN apt clean
RUN rm -rf /var/lib/apt/lists/*
