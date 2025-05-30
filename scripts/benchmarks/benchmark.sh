#!/bin/bash
ISAAC_LAB_PATH=isaaclab.sh
BENCHMARKING_SCRIPT_PATH=scripts/benchmarks/benchmark.py
ENV_LIST='Isaac-Cartpole-Direct-v0 Isaac-Ant-Direct-v0'
NUM_ENVS_LIST='64 128 256 512 1024 2048 4096'
NUM_STEPS=5000
STEP_INTERVAL=100

OUTPUT_DIR=$1
SAVE=true

for ENV in $ENV_LIST
do
  for NUM_ENV in $NUM_ENVS_LIST
  do
    ./$ISAAC_LAB_PATH -p $BENCHMARKING_SCRIPT_PATH --task $ENV --num_envs $NUM_ENV --headless --num_steps $NUM_STEPS --check_every $STEP_INTERVAL --results_dir $OUTPUT_DIR --save_results
  done
done
