#!/bin/bash

ISAAC_EXEC_SCRIPT=/home/antoiner/Documents/IsaacLab-Internal/isaaclab.sh
RSL_SCRIPT=scripts/benchmarks/benchmark_rsl_rl.py
TASKS="Isaac-Cartpole-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-G1-v1 Isaac-Velocity-Flat-H1-v0"
SEEDS="1 2 3 4 5"
TARGET_FOLDER=$1
NUM_ENVS=$2
ITERATIONS=$3

for task in $TASKS
    do
    for seed in $SEEDS
      do
        $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --seed $seed --max_iterations $ITERATIONS --num_envs $NUM_ENVS --headless --output_folder $TARGET_FOLDER
      done
  done
