#!/bin/bash

ISAAC_EXEC_SCRIPT=/home/kellyg/Documents/isaac/IsaacLab/isaaclab.sh
RSL_SCRIPT=scripts/benchmarks/benchmark_non_rl.py
TASKS="Isaac-Repose-Cube-Allegro-Direct-v0 Isaac-Ant-Direct-v0 Isaac-Cartpole-Direct-v0 Isaac-Cartpole-RGB-Camera-Direct-v0 Isaac-Cartpole-Depth-Camera-Direct-v0 Isaac-Humanoid-Direct-v0 Isaac-Ant-v0 Isaac-Cartpole-v0 Isaac-Humanoid-v0 Isaac-Velocity-Flat-Unitree-A1-v0 Isaac-Velocity-Flat-Anymal-B-v0 Isaac-Velocity-Flat-Anymal-C-v0 Isaac-Velocity-Flat-Anymal-D-v0 Isaac-Velocity-Flat-Cassie-v0 Isaac-Velocity-Flat-G1-v0 Isaac-Velocity-Flat-G1-v1 Isaac-Velocity-Flat-Unitree-Go1-v0 Isaac-Velocity-Flat-Unitree-Go2-v0 Isaac-Velocity-Flat-H1-v0 Isaac-Velocity-Flat-Spot-v0 Isaac-Reach-Franka-v0 Isaac-Reach-UR10-v0"
NUM_FRAMES=100
NUM_ENVS="1024 2048 4096 8192 16384"
TARGET_FOLDER=$1

for task in $TASKS
  do
  for num_envs in $NUM_ENVS
    do
      if [[ $task == *"RGB-Camera"* ]] || [[ $task == *"Depth-Camera"* ]]; then
        $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --num_frames $NUM_FRAMES --headless --num_envs $num_envs --output_folder $TARGET_FOLDER --enable_cameras
      else
        $ISAAC_EXEC_SCRIPT -p $RSL_SCRIPT --task $task --num_frames $NUM_FRAMES --headless --num_envs $num_envs --output_folder $TARGET_FOLDER
      fi
    done
  done
