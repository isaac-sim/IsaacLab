#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly ASSET_DIR="${ISAACLAB_SNAP_CIRCUITS_ASSET_ROOT:-${SCRIPT_DIR}/assets}"
hand="${ISAACLAB_AVP_HAND:-gr1}"
spectator="${ISAACLAB_AVP_SPECTATOR_RTSP:-0}"
mode="${ISAACLAB_AVP_MODE:-avp}"
action="status"
hand_assets=()

while (($#)); do
    case "$1" in
        --hand)
            hand="$2"
            shift 2
            ;;
        --spectator)
            spectator=1
            shift
            ;;
        --mode)
            mode="$2"
            shift 2
            ;;
        --no-spectator)
            spectator=0
            shift
            ;;
        start|stop|restart|status|logs)
            action="$1"
            shift
            ;;
        *)
            echo "Usage: start-demo.sh [--hand gr1|sharpa|prohand] [--mode avp|desktop]" >&2
            echo "                     [--spectator|--no-spectator]" >&2
            echo "                     [start|stop|restart|status|logs]" >&2
            exit 2
            ;;
    esac
done

case "${hand}" in
    gr1)
        task="IsaacContrib-PickPlace-GR1T2-SnapCircuits-Abs"
        ;;
    sharpa)
        task="IsaacContrib-SnapCircuits-SharpaWave-Abs"
        hand_assets=("${ASSET_DIR}/sharpa-urdf-usd-xml/wave_01/dual_sharpa_wave/dual_sharpa_wave.usda")
        ;;
    prohand)
        task="IsaacContrib-SnapCircuits-ProHand-Abs"
        hand_assets=(
            "${ASSET_DIR}/pro-models/assets/usd/gen_1_D_L/gen_1_D_L.usda"
            "${ASSET_DIR}/pro-models/assets/usd/gen_1_D_R/gen_1_D_R.usda"
            "${ASSET_DIR}/pro-models/assets/meshes/prohand_left_with_tips.urdf"
            "${ASSET_DIR}/pro-models/assets/meshes/prohand_right_with_tips.urdf"
        )
        ;;
    *)
        echo "Unknown hand '${hand}'; expected gr1, sharpa, or prohand." >&2
        exit 2
        ;;
esac

if [[ "${mode}" != "avp" && "${mode}" != "desktop" ]]; then
    echo "Unknown mode '${mode}'; expected avp or desktop." >&2
    exit 2
fi
if [[ "${mode}" == "desktop" && "${hand}" != "gr1" ]]; then
    echo "Desktop scripted-hand preview currently supports --hand gr1 only." >&2
    exit 2
fi

if [[ "${action}" == "start" || "${action}" == "restart" ]]; then
    required_assets=("${ASSET_DIR}/prepared/snap_circuits_table.usda")
    if ((${#hand_assets[@]})); then
        required_assets+=("${hand_assets[@]}")
    fi
    for asset in "${required_assets[@]}"; do
        if [[ ! -f "${asset}" ]]; then
            echo "Missing prepared demo asset: ${asset}" >&2
            echo "Run ${SCRIPT_DIR}/setup-assets.sh first." >&2
            exit 1
        fi
    done
fi

if [[ "${action}" == "restart" ]]; then
    "${SCRIPT_DIR}/../avp-teleop.sh" stop
    action="start"
fi

ISAACLAB_AVP_TASK="${task}" ISAACLAB_AVP_SPECTATOR_RTSP="${spectator}" ISAACLAB_AVP_MODE="${mode}" \
    exec "${SCRIPT_DIR}/../avp-teleop.sh" "${action}"
