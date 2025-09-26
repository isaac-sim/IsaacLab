#!/bin/bash
set -euo pipefail

PYTHON=/workspace/isaaclab/_isaac_sim/python.sh 
SCRIPT=scripts/benchmarks/benchmark_ray_caster_cli.py  

# ===============================
# Benchmark configurations
# ===============================

NUM_ASSETS_MEMORY=(1 2 4 8 16 32)
NUM_ASSETS=(0 1 2 4 8 16 32)
NUM_ENVS=(32 64 128 256 512 1024 2048 4096)
MESH_SUBDIVISIONS=(0 1 2 3 4 5)
RESOLUTIONS=(0.2 0.1 0.05 0.015)

# ===============================
# Run Benchmarks
# ===============================

echo "=== Running Benchmarks ==="

# Benchmark 1: Num Assets Reference
for num_assets in "${NUM_ASSETS_MEMORY[@]}"; do
    echo "[RUN]: Num Assets Reference | num_assets=${num_assets}"
    $PYTHON $SCRIPT \
        --task ray_caster_benchmark_num_assets_reference \
        --num_assets $num_assets \
        --num_envs 1024 \
        --resolution 0.05
done

# Benchmark 2: Single vs Multi (over envs & resolutions)
for num_envs in "${NUM_ENVS[@]}"; do
    for res in "${RESOLUTIONS[@]}"; do
        echo "[RUN]: Single Raycaster | num_envs=${num_envs}, res=${res}"
        $PYTHON $SCRIPT \
            --task ray_caster_benchmark_single_vs_multi \
            --raycaster_type single \
            --num_envs $num_envs \
            --resolution $res

        echo "[RUN]: Multi Raycaster | num_envs=${num_envs}, res=${res}"
        $PYTHON $SCRIPT \
            --task ray_caster_benchmark_single_vs_multi \
            --raycaster_type multi \
            --num_envs $num_envs \
            --resolution $res
    done
done

# Benchmark 3: Num Assets
for num_assets in "${NUM_ASSETS[@]}"; do
    echo "[RUN]: Num Assets | num_assets=${num_assets}"
    $PYTHON $SCRIPT \
        --task ray_caster_benchmark_num_assets_and_faces \
        --num_assets $num_assets \
        --resolution 0.05 \
        --num_envs 1024
done

# Benchmark 4: Mesh Subdivisions (Num Faces)
for subdiv in "${MESH_SUBDIVISIONS[@]}"; do
    echo "[RUN]: Mesh Subdivisions | subdiv=${subdiv}"
    $PYTHON $SCRIPT \
        --task ray_caster_benchmark_num_faces \
        --mesh_subdivisions $subdiv \
        --resolution 0.05 \
        --num_envs 1024 \
        --num_assets 1
done

echo "=== All Benchmarks Completed ==="
