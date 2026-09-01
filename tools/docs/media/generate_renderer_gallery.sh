#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

if [[ "$#" -ne 1 ]]; then
    echo "Usage: $0 /path/to/renderer-gallery-scene.usda" >&2
    exit 1
fi

SCENE_PATH="$1"
if [[ ! -f "${SCENE_PATH}" ]]; then
    echo "Error: scene does not exist: ${SCENE_PATH}" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
OUTPUT_DIR="${REPO_ROOT}/docs/source/_static/overview/sensors"
CAPTURE_SCRIPT="${SCRIPT_DIR}/capture_renderer_gallery.py"
SIMPLE_SHADING_MODES=(
    simple_shading_constant_diffuse
    simple_shading_diffuse_mdl
    simple_shading_full_mdl
)
COMMON_ARGS=(
    --scene "${SCENE_PATH}"
    --output-dir "${OUTPUT_DIR}"
    --width 640
    --height 360
    --frames 37
    --warmup-steps 24
)

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required to generate the renderer gallery." >&2
    exit 1
fi

if [[ "${OMNI_KIT_ACCEPT_EULA:-${ACCEPT_EULA:-}}" != "Y" ]]; then
    echo "Error: set OMNI_KIT_ACCEPT_EULA=Y or ACCEPT_EULA=Y after reviewing NVIDIA's EULA." >&2
    exit 1
fi

export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=Y

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}"

capture_kitless() {
    local renderer="$1"
    local capture_group="$2"
    uv run --frozen --extra ovrtx python "${CAPTURE_SCRIPT}" \
        --renderer-backend "${renderer}" \
        --capture-group "${capture_group}" \
        "${COMMON_ARGS[@]}"
}

capture_isaac_rtx() {
    local capture_group="$1"
    uv run --frozen --extra isaacsim python "${CAPTURE_SCRIPT}" \
        --renderer-backend isaac_rtx \
        --capture-group "${capture_group}" \
        "${COMMON_ARGS[@]}"
}

capture_kitless newton standard
capture_kitless ovrtx standard
for mode in "${SIMPLE_SHADING_MODES[@]}"; do
    capture_kitless ovrtx "${mode}"
done

capture_isaac_rtx standard
for mode in "${SIMPLE_SHADING_MODES[@]}"; do
    capture_isaac_rtx "${mode}"
done
