#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/docs/source/_static/quickstart"
WORK_DIR="$(mktemp -d)"

cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

for command in uv ffmpeg ffprobe; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "Error: ${command} is required to generate quickstart media." >&2
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_ROOT}"

UV_EXTRAS="ovrtx,ovphysx,video"
CAPTURE_STEPS=300

record_simple_agent() {
    local policy="$1"
    local output="$2"

    uv run --frozen --extra "${UV_EXTRAS}" python "${SCRIPT_DIR}/capture_quickstart.py" \
        --policy "${policy}" \
        --task Isaac-Open-Drawer-Franka \
        --output-dir "${output}" \
        --video-length "${CAPTURE_STEPS}" \
        physics=physx
}

record_policy() {
    local task="$1"
    local physics="$2"
    local prefix="$3"
    local output="$4"

    QUICKSTART_VIDEO_DIR="${output}" \
    QUICKSTART_VIDEO_PREFIX="${prefix}" \
    PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
        uv run --frozen --extra "${UV_EXTRAS}" python scripts/reinforcement_learning/play.py \
        --task "${task}" \
        --checkpoint pretrained \
        --num_envs 1 \
        --video \
        --video_length "${CAPTURE_STEPS}" \
        --viz newton_rtx \
        --external_callback capture_quickstart.configure_playback \
        "physics=${physics}"
}

record_simple_agent zero "${WORK_DIR}/zero"
record_simple_agent random "${WORK_DIR}/random"
record_policy Isaac-Open-Drawer-Franka physx play "${WORK_DIR}/play"

record_policy Isaac-Cartpole newton_mjwarp cartpole "${WORK_DIR}/cartpole"
record_policy Isaac-Velocity-Flat-G1 newton_mjwarp g1 "${WORK_DIR}/g1"
record_policy Isaac-Lift-KukaAllegro newton_mjwarp kuka_allegro "${WORK_DIR}/kuka_allegro"

ffmpeg -y -v error \
    -i "${WORK_DIR}/zero/zero_0000.mp4" \
    -i "${WORK_DIR}/random/random_0000.mp4" \
    -i "${WORK_DIR}/play/play_0000.mp4" \
    -filter_complex \
    "[0:v]select='not(mod(n,5))',setpts=N/(12*TB),scale=320:240:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=112:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='zero_agent':fontcolor=white:fontsize=16:x=16:y=13[zero];
     [1:v]select='not(mod(n,5))',setpts=N/(12*TB),scale=320:240:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=134:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='random_agent':fontcolor=white:fontsize=16:x=16:y=13[random];
     [2:v]select='not(mod(n,5))',setpts=N/(12*TB),scale=320:240:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=58:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='play':fontcolor=white:fontsize=16:x=16:y=13[play];
     [zero][random][play]hstack=inputs=3,split[frames][palette_frames];
     [palette_frames]palettegen=max_colors=256:stats_mode=diff[palette];
     [frames][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
    -loop 0 "${OUTPUT_DIR}/agent-comparison.gif"

ffmpeg -y -v error \
    -i "${WORK_DIR}/cartpole/cartpole_0000.mp4" \
    -i "${WORK_DIR}/g1/g1_0000.mp4" \
    -i "${WORK_DIR}/kuka_allegro/kuka_allegro_0000.mp4" \
    -filter_complex \
    "[0:v]crop=1280:720:0:120,select='not(mod(n,5))',setpts=N/(12*TB),scale=320:180:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=82:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='Cartpole':fontcolor=white:fontsize=16:x=16:y=13[cartpole];
     [1:v]crop=1280:720:0:120,select='not(mod(n,5))',setpts=N/(12*TB),scale=320:180:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=128:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='G1 locomotion':fontcolor=white:fontsize=16:x=16:y=13[g1];
     [2:v]crop=1280:720:0:120,select='not(mod(n,5))',setpts=N/(12*TB),scale=320:180:flags=lanczos,hqdn3d=1:1:3:3,drawbox=x=8:y=8:w=116:h=28:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='Kuka Allegro':fontcolor=white:fontsize=16:x=16:y=13[kuka];
     [cartpole][g1][kuka]hstack=inputs=3,split[frames][palette_frames];
     [palette_frames]palettegen=max_colors=256:stats_mode=diff[palette];
     [frames][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle" \
    -loop 0 "${OUTPUT_DIR}/task-sampler.gif"

for output in agent-comparison.gif task-sampler.gif; do
    ffprobe -v error \
        -show_entries stream=width,height,nb_frames,r_frame_rate:format=duration,size \
        -of default=noprint_wrappers=1 \
        "${OUTPUT_DIR}/${output}"
done
