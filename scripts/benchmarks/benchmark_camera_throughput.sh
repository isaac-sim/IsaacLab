#!/bin/bash

# get the path to the script
SCRIPT_PATH=$(realpath $(dirname $0))

# get the path to the IsaacLab sh
ISAACLAB_SH=${SCRIPT_PATH}/../../isaaclab.sh

# run the benchmark script with different configurations

# --low resolution with usd camera
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 12 24 48 96 --tiled_camera --usd_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 12 24 48 96 --tiled_camera --ray_caster_camera --usd_camera --data_type distance_to_image_plane
# --low resolution without usd camera
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 256 512 1024 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 256 512 1024 --tiled_camera --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 2048 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 3072 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64,120x160 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 4096 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 8192 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 64x64 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 4096 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 8192 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 120x160 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane


# --high resolution with usd camera
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320,480x640 --num_envs 96 --usd_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320,480x640 --num_envs 96 --usd_camera --data_type distance_to_image_plane
# --high resolution without usd camera
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320,480x640 --num_envs 96 256 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320,480x640 --num_envs 512 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320,480x640 --num_envs 96 256 512 --tiled_camera --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 1024 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 1024 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 1024 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 1024 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 1024 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 1024 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 2048 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 2048 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 3072 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 3072 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 4096 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 4096 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 8192 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 240x320 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane

# ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 8192 --tiled_camera --data_type rgb
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --resolutions 480x640 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane
