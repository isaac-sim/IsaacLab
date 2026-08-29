#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/docs/source/_static/reinforcement-learning"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

for command in uv ffmpeg ffprobe; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Error: ${command} is required to generate reinforcement-learning media." >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

UV_EXTRAS="ovrtx,ovphysx,video"
TASK="Isaac-Velocity-Flat-AnymalD"
CAPTURE_STEPS=300
CHECKPOINT_DIR="${RL_PROGRESS_CHECKPOINT_DIR:-}"
# The "physx" selector automatically chooses the compatible PhysX-family integration for OVRTX.
PHYSICS_SELECTOR="physx"

if [[ -z "${CHECKPOINT_DIR}" ]]; then
    TRAINING_ROOT="${WORK_DIR}/training"
    uv run --frozen --extra "${UV_EXTRAS}" isaaclab train \
        --rl_library rsl_rl \
        --task "${TASK}" \
        --seed 42 \
        --max_iterations 300 \
        --experiment_name "${TRAINING_ROOT}" \
        agent.save_interval=50 \
        physics="${PHYSICS_SELECTOR}"
    CHECKPOINT_DIR="$(find "${TRAINING_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
        | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

CHECKPOINT_DIR="$(cd -- "${CHECKPOINT_DIR}" && pwd)"
# RSL-RL numbers its training loop from zero, so a 300-iteration run ends at model_299.pt.
declare -a ITERATIONS=(0 100 299)

for iteration in "${ITERATIONS[@]}"; do
    checkpoint="${CHECKPOINT_DIR}/model_${iteration}.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Error: expected checkpoint does not exist: ${checkpoint}" >&2
        exit 1
    fi
done

record_checkpoint() {
    local iteration="$1"
    local destination="${WORK_DIR}/iteration_${iteration}"
    mkdir -p "${destination}"

    RL_PROGRESS_VIDEO_DIR="${destination}" \
    RL_PROGRESS_VIDEO_PREFIX="iteration_${iteration}" \
    PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        uv run --frozen --extra "${UV_EXTRAS}" isaaclab play \
        --rl_library rsl_rl \
        --task "${TASK}" \
        --checkpoint "${CHECKPOINT_DIR}/model_${iteration}.pt" \
        --num_envs 1 \
        --seed 42 \
        --video \
        --video_length "${CAPTURE_STEPS}" \
        --viz newton_rtx \
        --external_callback capture_reinforcement_learning.configure_playback \
        physics="${PHYSICS_SELECTOR}"
}

for iteration in "${ITERATIONS[@]}"; do
    record_checkpoint "${iteration}"
done

ffmpeg -y -v error \
    -i "${WORK_DIR}/iteration_0/iteration_0_0000.mp4" \
    -i "${WORK_DIR}/iteration_100/iteration_100_0000.mp4" \
    -i "${WORK_DIR}/iteration_299/iteration_299_0000.mp4" \
    -filter_complex \
    "[0:v]select='not(mod(n,4))',setpts=N/(12*TB),hqdn3d=1.5:1.5:4:4,drawbox=x=12:y=12:w=116:h=32:color=black@0.68:t=fill,drawtext=font='DejaVu Sans':text='Iteration 0':fontcolor=white:fontsize=18:x=20:y=18[early];
     [1:v]select='not(mod(n,4))',setpts=N/(12*TB),hqdn3d=1.5:1.5:4:4,drawbox=x=12:y=12:w=138:h=32:color=black@0.68:t=fill,drawtext=font='DejaVu Sans':text='Iteration 100':fontcolor=white:fontsize=18:x=20:y=18[mid];
     [2:v]select='not(mod(n,4))',setpts=N/(12*TB),hqdn3d=1.5:1.5:4:4,drawbox=x=12:y=12:w=138:h=32:color=black@0.68:t=fill,drawtext=font='DejaVu Sans':text='Iteration 299':fontcolor=white:fontsize=18:x=20:y=18[final];
     [early][mid][final]hstack=inputs=3,split[frames][palette_frames];
     [palette_frames]palettegen=max_colors=192:stats_mode=diff[palette];
     [frames][palette]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
    -loop 0 "${OUTPUT_DIR}/anymal-d-learning-progression.gif"

ffprobe -v error \
    -show_entries stream=width,height,nb_frames,r_frame_rate:format=duration,size \
    -of default=noprint_wrappers=1 \
    "${OUTPUT_DIR}/anymal-d-learning-progression.gif"
