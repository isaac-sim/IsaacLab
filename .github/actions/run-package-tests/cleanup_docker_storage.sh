#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -u

ISAACSIM_IMAGE="${1:-}"
ISAACSIM_IMAGE_PIN_CONTAINER="${2:-isaacsim-pin-run-package-tests}"
CLEANUP_CONTEXT="${3:-docker-cleanup}"
FAIL_IF_STILL_HIGH="${4:-false}"

DOCKER_STORAGE_PATH="${DOCKER_STORAGE_PATH:-/var/lib/docker}"
DOCKER_CLEANUP_THRESHOLD_PERCENT="${DOCKER_CLEANUP_THRESHOLD_PERCENT:-85}"
DOCKER_AGGRESSIVE_PRUNE_UNTIL="${DOCKER_AGGRESSIVE_PRUNE_UNTIL:-24h}"
DOCKER_CONSERVATIVE_PRUNE_UNTIL="${DOCKER_CONSERVATIVE_PRUNE_UNTIL:-72h}"

get_docker_storage_usage_percent() {
    df -P "${DOCKER_STORAGE_PATH}" 2>/dev/null | awk 'NR == 2 { gsub("%", "", $5); print $5 }'
}

resolve_docker_storage_path() {
    local docker_root_dir
    docker_root_dir="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    if [ -n "${docker_root_dir}" ]; then
        DOCKER_STORAGE_PATH="${docker_root_dir}"
        echo "[${CLEANUP_CONTEXT}] Using Docker root from docker info: ${DOCKER_STORAGE_PATH}"
        return 0
    fi

    echo "[${CLEANUP_CONTEXT}] Could not determine Docker storage path from ${DOCKER_STORAGE_PATH} or docker info." >&2
    return 1
}

print_docker_storage_usage() {
    echo "[${CLEANUP_CONTEXT}] Docker storage usage:"
    df -h "${DOCKER_STORAGE_PATH}" || true
    docker system df || true
}

pin_isaacsim_image_if_present() {
    if [ -z "${ISAACSIM_IMAGE}" ]; then
        echo "[${CLEANUP_CONTEXT}] No IsaacSim base image provided; skipping image pin."
        return
    fi

    if docker image inspect "${ISAACSIM_IMAGE}" >/dev/null 2>&1; then
        echo "[${CLEANUP_CONTEXT}] Pinning IsaacSim base image: ${ISAACSIM_IMAGE}"
        docker create --name "${ISAACSIM_IMAGE_PIN_CONTAINER}" "${ISAACSIM_IMAGE}" true >/dev/null 2>&1 || true
    else
        echo "[${CLEANUP_CONTEXT}] IsaacSim base image is not present locally; skipping image pin."
    fi
}

remove_isaacsim_image_pin() {
    docker rm "${ISAACSIM_IMAGE_PIN_CONTAINER}" >/dev/null 2>&1 || true
}

run_conservative_cleanup() {
    echo "[${CLEANUP_CONTEXT}] Running conservative Docker cleanup."
    echo "[${CLEANUP_CONTEXT}] Removing Docker images older than ${DOCKER_CONSERVATIVE_PRUNE_UNTIL}."
    docker image prune -a -f --filter "until=${DOCKER_CONSERVATIVE_PRUNE_UNTIL}" || true
}

run_aggressive_cleanup() {
    echo "[${CLEANUP_CONTEXT}] Running aggressive Docker cleanup."
    echo "[${CLEANUP_CONTEXT}] Removing stopped Docker containers."
    docker container prune -f --filter "until=${DOCKER_AGGRESSIVE_PRUNE_UNTIL}" || true

    echo "[${CLEANUP_CONTEXT}] Removing unused Docker builder cache older than ${DOCKER_AGGRESSIVE_PRUNE_UNTIL}."
    docker builder prune -a -f --filter "until=${DOCKER_AGGRESSIVE_PRUNE_UNTIL}" || true

    echo "[${CLEANUP_CONTEXT}] Removing Docker images older than ${DOCKER_AGGRESSIVE_PRUNE_UNTIL}."
    docker image prune -a -f --filter "until=${DOCKER_AGGRESSIVE_PRUNE_UNTIL}" || true
}

main() {
    echo "[${CLEANUP_CONTEXT}] Checking Docker storage before cleanup."

    local usage_percent
    usage_percent="$(get_docker_storage_usage_percent)"
    if [ -z "${usage_percent}" ]; then
        resolve_docker_storage_path || true
        usage_percent="$(get_docker_storage_usage_percent)"
    fi

    print_docker_storage_usage

    if [ -z "${usage_percent}" ]; then
        echo "[${CLEANUP_CONTEXT}] Could not determine Docker storage usage; running conservative cleanup."
        pin_isaacsim_image_if_present
        trap remove_isaacsim_image_pin EXIT
        run_conservative_cleanup
        print_docker_storage_usage
        if [ "${FAIL_IF_STILL_HIGH}" = "true" ]; then
            echo "::error::Could not determine Docker storage usage after cleanup. The self-hosted runner needs a measurable Docker storage path before tests can run." >&2
            return 1
        fi
        return 0
    fi

    pin_isaacsim_image_if_present
    trap remove_isaacsim_image_pin EXIT

    if [ "${usage_percent}" -ge "${DOCKER_CLEANUP_THRESHOLD_PERCENT}" ]; then
        echo "[${CLEANUP_CONTEXT}] Docker storage is at ${usage_percent}%, meeting the ${DOCKER_CLEANUP_THRESHOLD_PERCENT}% threshold."
        run_aggressive_cleanup
    else
        echo "[${CLEANUP_CONTEXT}] Docker storage is at ${usage_percent}%, below the ${DOCKER_CLEANUP_THRESHOLD_PERCENT}% threshold."
        run_conservative_cleanup
    fi

    echo "[${CLEANUP_CONTEXT}] Docker storage after cleanup."
    print_docker_storage_usage

    local final_usage_percent
    final_usage_percent="$(get_docker_storage_usage_percent)"
    if [ "${FAIL_IF_STILL_HIGH}" = "true" ] \
        && [ -n "${final_usage_percent}" ] \
        && [ "${final_usage_percent}" -ge "${DOCKER_CLEANUP_THRESHOLD_PERCENT}" ]; then
        echo "::error::Docker storage remains at ${final_usage_percent}% after cleanup. The self-hosted runner needs more free Docker storage before tests can run." >&2
        return 1
    fi
}

main "$@"
