#!/usr/bin/env bash

ROOT_DIR="${1:-./benchmarks}"
OUTPUT_DIR="${ROOT_DIR}/isaaclab_rsl_rl_training"
TASKS="Isaac-Repose-Cube-Allegro-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Cartpole-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Ant-v0 Isaac-Cartpole-v0 Isaac-Humanoid-v0 Isaac-Velocity-Flat-Unitree-A1-v0 Isaac-Velocity-Flat-Anymal-B-v0 Isaac-Velocity-Flat-Anymal-C-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-Cassie-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-G1-v1 Isaac-Velocity-Flat-Unitree-Go1-v0 Isaac-Velocity-Flat-Unitree-Go2-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Reach-Franka-v0 Isaac-Reach-UR10-v0"
NUM_ENV="4096"
ITERATIONS="500"

for TASK in $TASKS; do
    if [[ $TASK == *"RGB-Camera"* ]] || [[ $TASK == *"Depth-Camera"* ]]; then
        ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py --benchmark_backend omniperf --output_path "$OUTPUT_DIR" --task "$TASK" --num_envs "$NUM_ENV" --headless --max_iterations "$ITERATIONS" --enable_cameras
    else
        ./isaaclab.sh -p scripts/benchmarks/benchmark_rsl_rl.py --benchmark_backend omniperf --output_path "$OUTPUT_DIR" --task "$TASK" --num_envs "$NUM_ENV" --headless --max_iterations "$ITERATIONS"
    fi
done
