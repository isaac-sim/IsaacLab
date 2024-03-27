#!/bin/bash

echo "(run_singularity.py): Called on compute node with arguments $@"

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

# copy singulary image to the compute node
folder="$TMPDIR/isaac-sim.sif"

# Check if the folder exists
if [ -d "$folder" ]; then
    echo "1 (run_singularity.py): Folder was already copied to local SSD."
else
    tar -xf $CLUSTER_SIF_PATH/orbit.tar  -C $TMPDIR
fi

# execute command in singularity container
singularity exec \
    -B $TMPDIR/docker-isaac-sim/cache/kit:${DOCKER_ISAACSIM_PATH}/kit/cache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $TMPDIR/docker-isaac-sim/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $TMPDIR/docker-isaac-sim/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $TMPDIR/docker-isaac-sim/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $TMPDIR/docker-isaac-sim/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $TMPDIR/docker-isaac-sim/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $TMPDIR/docker-isaac-sim/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $TMPDIR/orbit:/workspace/orbit:rw \
    -B $CLUSTER_ORBIT_DIR/logs:/workspace/orbit/logs:rw \
    --nv --writable --containall $TMPDIR/orbit.sif \
    bash -c "cd /workspace/orbit && /isaac-sim/python.sh ${CLUSTER_PYTHON_EXECUTABLE} $@"

# copy resulting cache files back to host
cp -r $TMPDIR/docker-isaac-sim $CLUSTER_ISAAC_SIM_CACHE_DIR/..

echo "(run_singularity.py): Return"
