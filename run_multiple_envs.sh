#!/bin/bash

# Define your task names (environments)
TASKS=(
    "Isaac-Velocity-Flat-Anymal-D-v0"
    "Isaac-Velocity-Rough-Anymal-D-v0"
    "Isaac-Velocity-Flat-G1-v0"
    "Isaac-Velocity-Rough-G1-v0"
    "Isaac-Cartpole-v0"
    "Isaac-Lift-Cube-Franka-v0"
    "Isaac-Open-Drawer-Franka-v0"
    "Isaac-Repose-Cube-Allegro-v0"
    "Isaac-Repose-Cube-Allegro-Direct-v0"
    "Isaac-Repose-Cube-Shadow-Direct-v0"
    "Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0"
)  # Replace with actual task names

# Loop over each task and run it 3 times
for TASK_NAME in "${TASKS[@]}"; do
  for i in {1..3}; do
    echo "Running ${TASK_NAME} - Attempt $i"
    ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py --task "${TASK_NAME}" --headless --run_name seed_fix_${i}
    echo "Completed ${TASK_NAME} - Attempt $i"
  done
done
