#!/bin/bash

echo "(run_singularity.py): Called on compute node with container profile $1 and arguments ${@:2}"

#==
# Helper functions
#==

setup_directories() {
    # Check and create directories
    for dir in \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/kit" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/ov" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/pip" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/glcache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/cache/computecache" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/logs" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/data" \
        "${CLUSTER_ISAAC_SIM_CACHE_DIR}/documents"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            echo "Created directory: $dir"
        fi
    done
}


#==
# Main
#==


# get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# load variables to set the orbit path on the cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories
# copy all cache files
cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# copy orbit source code
mkdir -p "$CLUSTER_ORBIT_DIR/logs"
touch "$CLUSTER_ORBIT_DIR/logs/.keep"
cp -r $CLUSTER_ORBIT_DIR $TMPDIR

# copy container to the compute node
tar -xf $CLUSTER_SIF_PATH/$1.tar  -C $TMPDIR

# execute command in singularity container
# NOTE: ORBIT_PATH is normally set in `orbit.sh` but we directly call the isaac-sim python because we sync the entire
# orbit directory to the compute node and remote the symbolic link to isaac-sim
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/orbit:/workspace/orbit:rw \
    -B $CLUSTER_ORBIT_DIR/logs:/workspace/orbit/logs:rw \
    --nv --writable --containall $TMPDIR/$1.sif \
    bash -c "export ORBIT_PATH=/workspace/orbit && cd /workspace/orbit && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} ${@:2}"

# copy resulting cache files back to host
cp -r $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

echo "(run_singularity.py): Return"
