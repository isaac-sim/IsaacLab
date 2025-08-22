#!/bin/bash

ISAAC_EXEC_SCRIPT=/home/antoiner/Documents/IsaacLab-Internal/isaaclab.sh
RSL_SCRIPT=scripts/benchmarks/benchmark_non_rl.py
TASKS="Isaac-Cartpole-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-G1-v1 Isaac-Velocity-Flat-H1-v0"
NUM_FRAMES=100
TARGET_FOLDER=$1
NUM_ENVS=$2

for task in $TASKS
  do
      $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --num_frames $NUM_FRAMES --headless --num_envs $NUM_ENVS --output_folder $TARGET_FOLDER
  done
