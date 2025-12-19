#!/bin/bash

ISAAC_EXEC_SCRIPT=/home/antoiner/Documents/IsaacLab-Internal/isaaclab.sh
RSL_SCRIPT=scripts/benchmarks/benchmark_rsl_rl.py
TASKS="Isaac-Repose-Cube-Allegro-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Cartpole-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Ant-v0 Isaac-Cartpole-v0 Isaac-Humanoid-v0 Isaac-Velocity-Flat-Unitree-A1-v0 Isaac-Velocity-Flat-Anymal-B-v0 Isaac-Velocity-Flat-Anymal-C-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-Cassie-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-G1-v1 Isaac-Velocity-Flat-Unitree-Go1-v0 Isaac-Velocity-Flat-Unitree-Go2-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Reach-Franka-v0 Isaac-Reach-UR10-v0"
SEEDS="1"
TARGET_FOLDER=$1
NUM_ENVS=4096
ITERATIONS=500

for task in $TASKS
    do
    for seed in $SEEDS
      do
        $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --seed $seed --max_iterations $ITERATIONS --num_envs $NUM_ENVS --headless --output_folder $TARGET_FOLDER
      done
  done
