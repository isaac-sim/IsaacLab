#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: $0 <run_name>"
    exit 1
fi

PROJECT=isaac-locomotion-foostep-sigma
ENV=Isaac-Footstep-Sigmaban
TAGS=locomotion
ALGO=ppo
EXP_ID=`python next_run_id.py $ALGO $ENV`
RUN_NAME="${HOSTNAME}_${EXP_ID}_$1"

# killall -9 python

nohup ./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task $ENV-v0 \
    --log_project_name $PROJECT \
    --run_name $RUN_NAME \
    --experiment_name $ALGO \
    --headless  > $HOSTNAME.out &

tail -f $HOSTNAME.out

    # --logger wandb \