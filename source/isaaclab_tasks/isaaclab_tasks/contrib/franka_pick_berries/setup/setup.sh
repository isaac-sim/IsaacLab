#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Dedicated, reproducible Kitless task environment for the franka_pick_berries task; never updates .venv-kitless.
set -euo pipefail
TASK_RENDERER_WHEELS=""
TASK_RENDERER_INTERNAL=false
if [[ "${1:-}" == "--help" && $# == 1 ]]; then
    echo "Usage: bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh [--renderer-internal | --renderer-wheels /authorized/wheel/directory]"
    echo "Live berry rendering requires OVRTX 0.6 / OVStage 0.3 Linux wheels."
    echo "--renderer-internal installs pinned builds from NVIDIA Artifactory (network access required)."
    exit 0
fi
if [[ "${1:-}" == "--renderer-wheels" && $# == 2 ]]; then
    TASK_RENDERER_WHEELS="$(cd -- "$2" && pwd)"
    TASK_OVRTX_WHEEL="$TASK_RENDERER_WHEELS/ovrtx-0.6.0-py3-none-manylinux_2_35_x86_64.whl"
    TASK_OVSTAGE_WHEEL="$TASK_RENDERER_WHEELS/ovstage-0.3.0.0-py3-none-manylinux_2_35_x86_64.whl"
    [[ -f "$TASK_OVRTX_WHEEL" && -f "$TASK_OVSTAGE_WHEEL" ]] || {
        echo "Missing validated OVRTX 0.6 / OVStage 0.3 Linux wheels in $TASK_RENDERER_WHEELS" >&2
        exit 1
    }
elif [[ "${1:-}" == "--renderer-internal" && $# == 1 ]]; then
    TASK_RENDERER_INTERNAL=true
elif [[ $# != 0 ]]; then
    echo "Usage: bash source/isaaclab_tasks/isaaclab_tasks/contrib/franka_pick_berries/setup/setup.sh [--renderer-internal | --renderer-wheels /authorized/wheel/directory]" >&2
    exit 1
fi
TASK_TOOLS_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TASK_LAB_ROOT="$(cd -- "$TASK_TOOLS_ROOT/../../../../../.." && pwd)"
export UV_PROJECT_ENVIRONMENT="$TASK_LAB_ROOT/.venv-tasks"
cd "$TASK_LAB_ROOT"

# Isolated from the default .venv; this branch's own lockfile already pins Newton, Warp,
# and MuJoCo/MuJoCo-Warp, so no separate Newton checkout or dependency overrides are needed
# here. The ``ovrtx`` extra pulls in the public OVRTX/OVStage renderer pins; ``isaaclab-teleop``
# is behind heavier optional extras (mimic/teleop, which pull in Kit XR/robomimic), so it is
# installed directly instead.
uv sync --frozen --python 3.12 --extra ovrtx
uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
    -e "$TASK_LAB_ROOT/source/isaaclab_teleop"
if [[ "$TASK_RENDERER_INTERNAL" == true ]]; then
    uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
        'ovrtx==0.6.0.dev382408+mr50035.92a010ff' ovstage==0.3.0.382327 \
        --index-url https://artifactory.nvidia.com/artifactory/api/pypi/ct-omniverse-pypi/simple
elif [[ -n "$TASK_RENDERER_WHEELS" ]]; then
    uv --no-config pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --no-deps \
        "$TASK_OVRTX_WHEEL" "$TASK_OVSTAGE_WHEEL"
else
    echo "Using the public renderer (ovrtx/ovstage) already pinned in this branch's lockfile."
    echo "Berry rendering requires setup.sh --renderer-internal or --renderer-wheels DIR." >&2
fi
echo "Ready. Export UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT and use uv run --no-sync."
