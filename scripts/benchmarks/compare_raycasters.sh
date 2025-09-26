#!/bin/bash

# Expects first argument api key, second wandb entity (optional)
export WANDB_API_KEY=$1
export WANDB_ENTITY=$2
echo Running raycaster benchmarks with wandb entity $WANDB_ENTITY
${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/scripts/benchmarks/benchmark_ray_caster.py
${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/scripts/benchmarks/sync_raycast_results.py
