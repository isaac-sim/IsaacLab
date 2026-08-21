#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -euo pipefail

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
readonly CONTAINER_NAME="${ISAACLAB_AVP_CONTAINER:-isaac-lab-base}"
readonly TELEOP_TASK="${ISAACLAB_AVP_TASK:-IsaacContrib-PickPlace-GR1T2-WaistEnabled-Abs}"
readonly TELEOP_MODE="${ISAACLAB_AVP_MODE:-avp}"
readonly TELEOP_PATTERN="[t]eleop_se3_agent.py|[d]esktop_preview.py"
readonly TELEOP_LOG_CONTAINER="/workspace/isaaclab/docker/apple-vision-pro/teleop.log"
readonly CLOUDXR_EULA_URL="https://developer.nvidia.com/cloudxr-sdk-eula"
readonly SPECTATOR_RTSP="${ISAACLAB_AVP_SPECTATOR_RTSP:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not installed or is not on PATH." >&2
    exit 1
fi

if docker info >/dev/null 2>&1; then
    readonly -a DOCKER=(docker)
    readonly -a CONTAINER_TOOL=("${REPO_DIR}/docker/container.py")
else
    readonly -a DOCKER=(sudo docker)
    readonly -a CONTAINER_TOOL=(sudo "${REPO_DIR}/docker/container.py")
fi

show_status() {
    if ! "${DOCKER[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "container=stopped"
        echo "teleop=stopped"
        echo "cloudxr_signaling=not-listening"
        return
    fi

    "${DOCKER[@]}" ps --filter "name=^/${CONTAINER_NAME}$" \
        --format 'container={{.Names}} status={{.Status}} image={{.Image}}'

    if "${DOCKER[@]}" exec "${CONTAINER_NAME}" pgrep -f "${TELEOP_PATTERN}" >/dev/null 2>&1; then
        echo "teleop=running"
    else
        echo "teleop=stopped"
    fi

    if ss -lnt | grep -q ':48010 '; then
        echo "cloudxr_signaling=listening tcp/48010"
    else
        echo "cloudxr_signaling=not-listening"
    fi

    if ss -lnt | grep -q ':8554 '; then
        echo "spectator_rtsp=listening rtsp://$(hostname -I | awk '{print $1}'):8554/snap-circuits"
    else
        echo "spectator_rtsp=not-listening"
    fi
}

require_headless_config() {
    local config_file="${REPO_DIR}/docker/.container.cfg"

    if [[ ! -f "${config_file}" ]] || ! grep -Eiq '^x11_forwarding_enabled[[:space:]]*=[[:space:]]*0[[:space:]]*$' "${config_file}"; then
        cat >&2 <<EOF
Headless teleoperation requires X11 forwarding to be disabled.
Run this from the IsaacLab repository root, then retry:

  printf '[X11]\\nx11_forwarding_enabled = 0\\n' > docker/.container.cfg
EOF
        exit 1
    fi
}

require_cloudxr_eula() {
    if ! "${DOCKER[@]}" exec "${CONTAINER_NAME}" test -f /root/.cloudxr/run/eula_accepted; then
        cat >&2 <<EOF
The NVIDIA CloudXR SDK EULA has not been accepted in this Docker volume.
Read ${CLOUDXR_EULA_URL}, then run this one-time command interactively:

  $0 accept-eula
EOF
        exit 1
    fi
}

accept_cloudxr_eula() {
    require_headless_config
    cat <<EOF
Review the NVIDIA CloudXR SDK EULA before continuing:
  ${CLOUDXR_EULA_URL}

This writes the acceptance marker to the persistent isaac-cloudxr Docker volume.
EOF
    read -r -p "Type YES to confirm that you accept the EULA: " response
    if [[ "${response}" != "YES" ]]; then
        echo "EULA was not accepted; no marker was written." >&2
        exit 1
    fi

    cd "${REPO_DIR}"
    "${CONTAINER_TOOL[@]}" start
    "${DOCKER[@]}" exec -e CXR_INSTALL_DIR=/root/.cloudxr "${CONTAINER_NAME}" \
        ./isaaclab.sh -p -c \
        'from isaacteleop.cloudxr.runtime import check_eula; check_eula(accept_eula=True)'
    echo "CloudXR EULA acceptance recorded in the persistent Docker volume."
}

start_teleop() {
    local -a docker_environment=()

    require_headless_config
    cd "${REPO_DIR}"
    if ! "${DOCKER[@]}" ps --filter "name=^/${CONTAINER_NAME}$" --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
        "${CONTAINER_TOOL[@]}" start
    fi
    if [[ "${TELEOP_MODE}" == "avp" ]]; then
        require_cloudxr_eula
    fi

    if "${DOCKER[@]}" exec "${CONTAINER_NAME}" pgrep -f "${TELEOP_PATTERN}" >/dev/null 2>&1; then
        echo "Apple Vision Pro teleoperation is already running."
        show_status
        return
    fi

    if [[ "${SPECTATOR_RTSP}" == "1" ]]; then
        docker_environment=(-e ISAACLAB_SPECTATOR_RTSP=1)
    fi

    "${DOCKER[@]}" exec --workdir /workspace/isaaclab "${CONTAINER_NAME}" \
        bash -lc ': > "$1"' _ "${TELEOP_LOG_CONTAINER}"
    if [[ "${TELEOP_MODE}" == "desktop" ]]; then
        "${DOCKER[@]}" exec -d -e PYTHONUNBUFFERED=1 "${docker_environment[@]}" --workdir /workspace/isaaclab "${CONTAINER_NAME}" \
            bash -lc 'log_file=$1; shift; exec "$@" > "${log_file}" 2>&1' _ \
            "${TELEOP_LOG_CONTAINER}" \
            ./isaaclab.sh \
            -p docker/apple-vision-pro/snap-circuits/desktop_preview.py \
            --task "${TELEOP_TASK}"
    else
        "${DOCKER[@]}" exec -d -e PYTHONUNBUFFERED=1 "${docker_environment[@]}" --workdir /workspace/isaaclab "${CONTAINER_NAME}" \
            bash -lc 'log_file=$1; shift; exec "$@" > "${log_file}" 2>&1' _ \
            "${TELEOP_LOG_CONTAINER}" \
            ./isaaclab.sh \
            -p scripts/environments/teleoperation/teleop_se3_agent.py \
            --task "${TELEOP_TASK}" \
            --xr \
            --cloudxr_env avp
    fi

    echo "Isaac Lab ${TELEOP_MODE} mode is starting in the background."
    if [[ "${TELEOP_MODE}" == "desktop" ]]; then
        echo "Run '$0 status' until the RTSP endpoint reports listening."
    else
        echo "Run '$0 status' until tcp/48010 reports listening."
    fi
}

stop_teleop() {
    if ! "${DOCKER[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "Container is not running."
        return
    fi

    "${DOCKER[@]}" exec "${CONTAINER_NAME}" pkill -INT -f "${TELEOP_PATTERN}" || true
    echo "Stop signal sent to the Apple Vision Pro teleoperation process."
}

show_logs() {
    if ! "${DOCKER[@]}" inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        echo "Container is not running." >&2
        exit 1
    fi

    "${DOCKER[@]}" exec "${CONTAINER_NAME}" bash -lc '
        if [ -f /workspace/isaaclab/docker/apple-vision-pro/teleop.log ]; then
            echo "Teleoperation log: /workspace/isaaclab/docker/apple-vision-pro/teleop.log"
            tail -n 200 /workspace/isaaclab/docker/apple-vision-pro/teleop.log
        fi
        latest_kit_log=$(ls -1t /isaac-sim/kit/logs/Kit/IsaacLab/3.0/kit_*.log 2>/dev/null | head -1)
        if [ -n "${latest_kit_log}" ]; then
            echo "Kit log: ${latest_kit_log}"
            tail -n 100 "${latest_kit_log}"
        fi
        latest_cloudxr_log=$(ls -1t /root/.cloudxr/logs/cxr_server.*.log 2>/dev/null | head -1)
        if [ -n "${latest_cloudxr_log}" ]; then
            echo "CloudXR log: ${latest_cloudxr_log}"
            tail -n 100 "${latest_cloudxr_log}"
        fi
    '
}

case "${1:-status}" in
    start)
        start_teleop
        ;;
    stop)
        stop_teleop
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    accept-eula)
        accept_cloudxr_eula
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|accept-eula}" >&2
        exit 2
        ;;
esac
