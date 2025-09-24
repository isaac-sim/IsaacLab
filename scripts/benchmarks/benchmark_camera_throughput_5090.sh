#!/bin/bash

# get the path to the script
SCRIPT_PATH=$(realpath $(dirname $0))

# get the path to the IsaacLab sh
ISAACLAB_SH=${SCRIPT_PATH}/../../isaaclab.sh

GPU_ID=0

# GPU watchdog wrapper
run_with_watchdog() {
    CMD="$@"

    echo "Starting: $CMD"
    $CMD &
    MAIN_PID=$!

    # Settings
    UTIL_THRESHOLD=25 # %
    MEM_LIMIT=2000    # MiB
    INTERVAL=10       # seconds
    MAX_LOW_COUNT=$((120 / INTERVAL)) # 2 min
    LOW_COUNT=0

    while kill -0 $MAIN_PID 2>/dev/null; do
        # Query utilization and memory usage
        read -r UTIL <<< "$(nvidia-smi --id=$GPU_ID --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
        
        PROCS=$(nvidia-smi --id=$GPU_ID --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
        
        # Check utilization
        if [ "$UTIL" -lt "$UTIL_THRESHOLD" ]; then
            LOW_COUNT=$((LOW_COUNT+1))
            echo "GPU$GPU_ID utilization low ($UTIL%) - $LOW_COUNT/$MAX_LOW_COUNT"
        else
            LOW_COUNT=0
        fi

        # Kill main process if GPU stays idle too long
        if [ "$LOW_COUNT" -ge "$MAX_LOW_COUNT" ]; then
            echo "GPU$GPU_ID utilization stayed below $UTIL_THRESHOLD% for 2 minutes. Killing main process $MAIN_PID"
            kill -9 $MAIN_PID
            wait $MAIN_PID 2>/dev/null

            # Clean up all remaining GPU processes
            while IFS=',' read -r PID MEM; do
                PID=$(echo "$PID" | xargs)   # trim spaces
                MEM=$(echo "$MEM" | xargs)
                if [ -n "$PID" ] && [ "$MEM" -gt "$MEM_LIMIT" ]; then
                    echo "Killing PID $PID (using ${MEM}MiB on GPU$GPU_ID > ${MEM_LIMIT}MiB)"
                    # Kill entire process group just in case
                    kill -9 "$PID" 2>/dev/null || true
                fi
            done <<< "$PROCS"

            return 1
        fi

        sleep $INTERVAL
    done

    wait $MAIN_PID
    echo "Finished: $CMD"
}

# run the benchmark script with different configurations

# default values
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1  --tiled_camera --ray_caster_camera --usd_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1  --tiled_camera --ray_caster_camera --usd_camera --data_type rgb

# # --low resolution with usd camera
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 12 24 48 96 --tiled_camera --usd_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 12 24 48 96 --tiled_camera --ray_caster_camera --usd_camera --data_type distance_to_image_plane
# # --low resolution without usd camera
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 256 512 1024 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 256 512 1024 --tiled_camera --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 2048 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 3072 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 4096 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 8192 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 4096 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 8192 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 120x160 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane


# # --high resolution with usd camera
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320,480x640 --num_envs 96 --usd_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320,480x640 --num_envs 96 --usd_camera --data_type distance_to_image_plane
# # --high resolution without usd camera
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320,480x640 --num_envs 96 256 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320,480x640 --num_envs 512 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320,480x640 --num_envs 96 256 512 --tiled_camera --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 1024 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 1024 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 1024 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 1024 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 1024 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 1024 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 2048 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

# # run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 2048 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 2048 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 2048 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 3072 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

# # run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 3072 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 3072 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 3072 --ray_caster_camera --data_type distance_to_image_plane

# # run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 4096 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# # run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 4096 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 4096 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 4096 --ray_caster_camera --data_type distance_to_image_plane

# # run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 8192 --tiled_camera --data_type rgb
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 240x320 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane

# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 8192 --tiled_camera --data_type rgb
run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 8192 --tiled_camera --data_type distance_to_image_plane
# run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 480x640 --num_envs 8192 --ray_caster_camera --data_type distance_to_image_plane

# DEBUG __ REMOVE LATER
run_with_watchdog ${ISAACLAB_SH} -p ${SCRIPT_PATH}/benchmark_camera_throughput.py --device cuda:1 --resolutions 64x64,120x160 --num_envs 12 24 48 96 --tiled_camera --usd_camera --data_type rgb