#!/bin/zsh

conda activate isaaclab

# Define the seed and gamma values
seeds=(42 123 456)
gammas=(0.99 1.0)

# Loop over the seeds and gammas
for seed in $seeds; do
  for gamma in $gammas; do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Box-Pushing-Dense-step-Franka-v0 \
    --num_envs 480 --headless --logger wandb \
    --log_project_name box_pushing \
    --log_run_group $gamma \
    --video --video_length 100 --video_interval 10000 --enable_cameras \
    --seed $seed --gamma $gamma
  done
done
