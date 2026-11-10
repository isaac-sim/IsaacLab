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
CAPTURE_STEPS=360

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
record_policy Isaac-Velocity-Rough-G1 newton_mjwarp g1 "${WORK_DIR}/g1"
record_policy Isaac-Lift-KukaAllegro newton_mjwarp kuka_allegro "${WORK_DIR}/kuka_allegro"

ffmpeg -y -v error \
    -i "${WORK_DIR}/zero/zero_0000.mp4" \
    -i "${WORK_DIR}/random/random_0000.mp4" \
    -i "${WORK_DIR}/play/play_0000.mp4" \
    -filter_complex \
    "[0:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=336:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='zero_agent':fontcolor=white:fontsize=48:x=48:y=39[zero];
     [1:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=402:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='random_agent':fontcolor=white:fontsize=48:x=48:y=39[random];
     [2:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=174:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='play':fontcolor=white:fontsize=48:x=48:y=39[play];
     [zero][random][play]hstack=inputs=3" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart \
    -threads 8 -g 60 -an "${OUTPUT_DIR}/agent-comparison.mp4"

ffmpeg -y -v error \
    -i "${WORK_DIR}/cartpole/cartpole_0000.mp4" \
    -i "${WORK_DIR}/g1/g1_0000.mp4" \
    -i "${WORK_DIR}/kuka_allegro/kuka_allegro_0000.mp4" \
    -filter_complex \
    "[0:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=246:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='Cartpole':fontcolor=white:fontsize=48:x=48:y=39[cartpole];
     [1:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=480:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='G1 rough terrain':fontcolor=white:fontsize=48:x=48:y=39[g1];
     [2:v]fps=20,scale=1280:720:flags=lanczos,drawbox=x=24:y=24:w=348:h=84:color=black@0.65:t=fill,drawtext=font='DejaVu Sans':text='Kuka Allegro':fontcolor=white:fontsize=48:x=48:y=39[kuka];
     [cartpole][g1][kuka]hstack=inputs=3" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset slow -movflags +faststart \
    -threads 8 -g 60 -an "${OUTPUT_DIR}/task-sampler.mp4"

uv run --frozen --extra "${UV_EXTRAS}" python - "${OUTPUT_DIR}" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

for name in ("agent-comparison.mp4", "task-sampler.mp4"):
    path = Path(sys.argv[1]) / name
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=codec_name,width,height,nb_read_frames,r_frame_rate:format=duration,size",
                "-of", "json", str(path),
            ],
            text=True,
        )
    )
    stream = probe["streams"][0]
    if (
        stream["codec_name"] != "h264"
        or (stream["width"], stream["height"]) != (3840, 720)
        or int(stream["nb_read_frames"]) != 120
        or stream["r_frame_rate"] != "20/1"
    ):
        raise ValueError(f"Unexpected quickstart video format: {path}")
    duration = float(probe["format"]["duration"])
    if abs(duration - 6.0) > 0.01:
        raise ValueError(f"Expected six-second video, got {duration} s: {path}")
    print(
        f"{name}: codec={stream['codec_name']}, size={stream['width']}x{stream['height']}, "
        f"frames={stream['nb_read_frames']}, fps={stream['r_frame_rate']}, "
        f"duration={duration} s, bytes={probe['format']['size']}"
    )
PY
