#!/bin/bash

ISAAC_EXEC_SCRIPT=/home/antoiner/Documents/IsaacLab-Internal/isaaclab.sh
RSL_SCRIPT=scripts/benchmarks/benchmark_non_rl.py
TASKS="Isaac-Cartpole-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-H1-v0"
NUM_FRAMES=100
NUM_ENVS="1024 2048 4096 8192 16384"
TARGET_FOLDER=$1

for task in $TASKS
  do
  for num_envs in $NUM_ENVS
    do
      $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --num_frames $NUM_FRAMES --headless --num_envs $num_envs --output_folder $TARGET_FOLDER
    done
  done
