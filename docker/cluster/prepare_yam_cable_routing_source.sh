#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Validate one rsynced source package, then publish it or serve it to peer tasks.

set -euo pipefail

source_sync=/osmo/run/workspace/source-sync
source_acceptance=/osmo/run/workspace/yam-source-accepted.sha256
source_release=/osmo/run/workspace/yam-source-release.sha256
source_serve_port="${OSMO_SOURCE_SERVE_PORT:-}"
image_python="${OSMO_IMAGE_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
output_package=""

install -d -m 0777 "${source_sync}"

if [[ -n "${source_serve_port}" ]]; then
    if [[ ! -x "${image_python}" ]]; then
        echo "Isaac Sim image Python is unavailable: ${image_python}" >&2
        exit 2
    fi
    if [[ ! "${source_serve_port}" =~ ^[0-9]+$ ]] \
        || (( source_serve_port < 1 || source_serve_port > 65535 )); then
        echo "OSMO_SOURCE_SERVE_PORT must be an integer from 1 through 65535." >&2
        exit 2
    fi
elif [[ -z "${OSMO_OUTPUT_DIR:-}" ]]; then
    echo "OSMO_OUTPUT_DIR is required when source serving is disabled." >&2
    exit 2
else
    output_package="${OSMO_OUTPUT_DIR}/source-package"
    mkdir -p "${output_package}"
fi

if [[ ! "${OSMO_SOURCE_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "OSMO_SOURCE_SHA256 is not a lowercase SHA-256 digest." >&2
    exit 2
fi

source_ready=0
for _ in $(seq 1 900); do
    if [[ -f "${source_sync}/source.sha256" \
        && -f "${source_sync}/source.tar.gz" \
        && -f "${source_sync}/source.metadata" \
        && -f "${source_sync}/git-status.txt" ]]; then
        synced_sha="$(tr -d '[:space:]' < "${source_sync}/source.sha256")"
        if [[ "${synced_sha}" == "${OSMO_SOURCE_SHA256}" ]]; then
            actual_sha="$(sha256sum "${source_sync}/source.tar.gz" | awk '{print $1}')"
            status_sha="$(sed -n 's/^git_status_sha256=\([0-9a-f]\{64\}\)$/\1/p' "${source_sync}/source.metadata")"
            actual_status_sha="$(sha256sum "${source_sync}/git-status.txt" | awk '{print $1}')"
            if [[ "${actual_sha}" == "${OSMO_SOURCE_SHA256}" \
                && "${status_sha}" == "${actual_status_sha}" \
                && "$(grep -Fxc "source_sha256=${OSMO_SOURCE_SHA256}" "${source_sync}/source.metadata")" == 1 \
                && "$(grep -Fxc "commit=${OSMO_VCS_REF}" "${source_sync}/source.metadata")" == 1 ]]; then
                source_ready=1
                break
            fi
        fi
    fi
    sleep 2
done
if (( source_ready == 0 )); then
    echo "Timed out waiting for checksum-complete source rsync ${OSMO_SOURCE_SHA256}." >&2
    exit 3
fi

# The injected bootstrap itself must be part of the package whose digest was
# rendered into the workflow. Also require the new replay sampler and its
# SuccessMonitor integration before any expensive GPU task can be scheduled.
if ! tar --extract --gzip --to-stdout \
    --file="${source_sync}/source.tar.gz" \
    docker/cluster/prepare_yam_cable_routing_source.sh \
    | cmp - /tmp/prepare-yam-cable-routing-source.sh; then
    echo "Injected source bootstrap does not match the checksum-gated archive." >&2
    exit 4
fi
for required_source in \
    pyproject.toml \
    source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/mdp/reset_curve_xpbd.py \
    source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/mdp/reset_curves.py \
    source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/mdp/reset_replay.py \
    source/isaaclab_tasks/isaaclab_tasks/contrib/cable_routing/mdp/reset_robot_targets.py \
    docker/cluster/yam_cable_routing_source_fetch.py; do
    if ! tar --list --gzip --file="${source_sync}/source.tar.gz" "${required_source}" >/dev/null; then
        echo "Checksum-gated source is missing ${required_source}." >&2
        exit 4
    fi
done

if [[ -z "${source_serve_port}" ]]; then
    cp --no-preserve=all \
        "${source_sync}/source.tar.gz" \
        "${source_sync}/source.metadata" \
        "${source_sync}/git-status.txt" \
        "${source_sync}/source.sha256" \
        "${output_package}/"
fi

source_acceptance_tmp="${source_acceptance}.tmp"
printf '%s\n' "${OSMO_SOURCE_SHA256}" > "${source_acceptance_tmp}"
mv "${source_acceptance_tmp}" "${source_acceptance}"

# Keep the authenticated rsync tunnel alive until the submitter has downloaded
# the acknowledgement and explicitly released this task.
source_released=0
for _ in $(seq 1 600); do
    if [[ -f "${source_release}" ]] \
        && [[ "$(tr -d '[:space:]' < "${source_release}")" == "${OSMO_SOURCE_SHA256}" ]]; then
        source_released=1
        break
    fi
    sleep 1
done
if (( source_released == 0 )); then
    echo "Timed out waiting for the submitter's source-release acknowledgement." >&2
    exit 5
fi

if [[ -n "${source_serve_port}" ]]; then
    printf \
        'source_package=serving sha256=%s address=0.0.0.0:%s directory=%s\n' \
        "${OSMO_SOURCE_SHA256}" \
        "${source_serve_port}" \
        "${source_sync}"
    exec "${image_python}" -m http.server \
        "${source_serve_port}" \
        --bind 0.0.0.0 \
        --directory "${source_sync}"
fi

printf 'source_package=ready sha256=%s output=%s\n' "${OSMO_SOURCE_SHA256}" "${output_package}"
