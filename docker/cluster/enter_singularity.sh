#!/usr/bin/env bash
# source $1/docker/cluster/.env.cluster
# echo "(run_singularity.py): Called on compute node from current isaaclab directory $1 with container profile $2"

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

# load variables to set the Isaac Lab path on the cluster
source $SCRIPT_DIR/.env.cluster
source $SCRIPT_DIR/../.env.base

# make sure that all directories exists in cache directory
setup_directories

# echo "Copying cache files to $TMPDIR"
# # copy all cache files
# cp -r $CLUSTER_ISAAC_SIM_CACHE_DIR $TMPDIR

# make sure logs directory exists (in the permanent isaaclab directory)
mkdir -p "$CLUSTER_ISAACLAB_DIR/logs"
touch "$CLUSTER_ISAACLAB_DIR/logs/.keep"

# copy the temporary isaaclab directory with the latest changes to the compute node
# cp -r $1 $TMPDIR
# Get the directory name
dir_name=$(basename "$CLUSTER_ISAACLAB_DIR")
echo "dir_name: $dir_name"
# # copy container to the compute node
# tar -xf $CLUSTER_SIF_PATH/$2.tar  -C $TMPDIR

# execute command in singularity container
# NOTE: ISAACLAB_PATH is normally set in `isaaclab.sh` but we directly call the isaac-sim python because we sync the entire
# Isaac Lab directory to the compute node and remote the symbolic link to isaac-sim
singularity exec \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/kit:${DOCKER_ISAACSIM_ROOT_PATH}/kit/cache:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw \
    -B $CLUSTER_ISAAC_SIM_CACHE_DIR/documents:${DOCKER_USER_HOME}/Documents:rw \
    -B $CLUSTER_ISAACLAB_DIR:/home/zixuanh/force_tool:rw \
    -B $CLUSTER_ISAACLAB_DIR/logs:/home/zixuanh/force_tool/logs:rw \
    --nv --writable --containall $CLUSTER_SIF_PATH/isaac-lab-base.sif \
    bash

# # copy resulting cache files back to host
# rsync -azPv $CLUSTER_ISAAC_SIM_CACHE_DIR $CLUSTERdocker-isaac-sim_ISAAC_SIM_CACHE_DIR/..

# # if defined, remove the temporary isaaclab directory pushed when the job was submitted
# if $REMOVE_CODE_COPY_AFTER_JOB; then
#     rm -rf $1
# fi

# echo "(run_singularity.py): Return"
