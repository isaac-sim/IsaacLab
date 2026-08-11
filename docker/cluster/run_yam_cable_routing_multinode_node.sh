#!/usr/bin/env bash

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Keep the OSMO lead alive until every node in one static torchrun world exits.

set -euo pipefail

runner=/tmp/run-yam-cable-routing-osmo.sh
barrier=/tmp/yam-cable-routing-ddp-barrier.py
source_preparer=/tmp/prepare-yam-cable-routing-source.sh
image_python="${OSMO_IMAGE_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
barrier_pid=""
runner_pid=""
source_server_pid=""
node_status=125
status_reported=0

stop_runner() {
    local termination_attempt
    if [[ -z "${runner_pid}" ]]; then
        return
    fi
    if kill -0 -- "-${runner_pid}" 2>/dev/null; then
        kill -TERM -- "-${runner_pid}" 2>/dev/null || true
        for termination_attempt in $(seq 1 50); do
            if ! kill -0 -- "-${runner_pid}" 2>/dev/null; then
                break
            fi
            sleep 0.1
        done
        if kill -0 -- "-${runner_pid}" 2>/dev/null; then
            kill -KILL -- "-${runner_pid}" 2>/dev/null || true
        fi
    fi
    wait "${runner_pid}" 2>/dev/null || true
    runner_pid=""
}

stop_source_server() {
    if [[ -n "${source_server_pid}" ]]; then
        kill "${source_server_pid}" 2>/dev/null || true
        wait "${source_server_pid}" 2>/dev/null || true
        source_server_pid=""
    fi
}

report_status() {
    local report_status_code
    if (( status_reported != 0 )); then
        return
    fi
    if [[ ! -f "${barrier}" \
        || -z "${OSMO_MASTER_ADDR:-}" \
        || -z "${OSMO_COMPLETION_PORT:-}" \
        || -z "${OSMO_NODE_RANK:-}" \
        || -z "${OSMO_WORKFLOW_ID:-}" ]]; then
        return 1
    fi
    if "${image_python}" "${barrier}" report \
        --host "${OSMO_MASTER_ADDR}" \
        --port "${OSMO_COMPLETION_PORT}" \
        --node-rank "${OSMO_NODE_RANK}" \
        --phase complete \
        --status "${node_status}" \
        --workflow-id "${OSMO_WORKFLOW_ID}" \
        --timeout-seconds 60; then
        report_status_code=0
        status_reported=1
    else
        report_status_code="$?"
    fi
    if (( report_status_code != 0 && node_status == 0 )); then
        node_status=16
    fi
    return "${report_status_code}"
}

cleanup() {
    local exit_status="$?"
    trap - EXIT INT TERM
    if (( node_status == 125 )); then
        node_status="${exit_status}"
        if (( node_status == 0 )); then
            node_status=125
        fi
    fi
    stop_runner
    report_status || true
    if [[ -n "${barrier_pid}" ]]; then
        kill "${barrier_pid}" 2>/dev/null || true
        wait "${barrier_pid}" 2>/dev/null || true
    fi
    stop_source_server
    exit "${node_status}"
}
trap cleanup EXIT
trap 'node_status=143; exit 143' INT TERM

required_environment=(
    OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS
    OSMO_COMPLETION_PORT
    OSMO_MASTER_ADDR
    OSMO_MASTER_PORT
    OSMO_NODE_RANK
    OSMO_NUM_GPU
    OSMO_NUM_NODES
    OSMO_OUTPUT_DIR
    OSMO_RUN_MODE
    OSMO_SOURCE_FETCH_WAIT_SECONDS
    OSMO_SOURCE_SERVE_PORT
    OSMO_SOURCE_SHA256
    OSMO_SOURCE_STARTUP_WAIT_SECONDS
    OSMO_TOTAL_GPU
    OSMO_VCS_REF
    OSMO_WORKFLOW_ID
)
for variable_name in "${required_environment[@]}"; do
    if [[ -z "${!variable_name-}" ]]; then
        echo "Required environment variable ${variable_name} is unset or empty." >&2
        node_status=2
        exit "${node_status}"
    fi
done
if [[ "${OSMO_RUN_MODE}" != multinode ]]; then
    echo "The multi-node wrapper requires OSMO_RUN_MODE=multinode." >&2
    node_status=2
    exit "${node_status}"
fi
for integer_variable in \
    OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS \
    OSMO_COMPLETION_PORT \
    OSMO_MASTER_PORT \
    OSMO_NODE_RANK \
    OSMO_NUM_GPU \
    OSMO_NUM_NODES \
    OSMO_SOURCE_FETCH_WAIT_SECONDS \
    OSMO_SOURCE_SERVE_PORT \
    OSMO_SOURCE_STARTUP_WAIT_SECONDS \
    OSMO_TOTAL_GPU; do
    if [[ ! "${!integer_variable}" =~ ^[0-9]+$ ]]; then
        echo "${integer_variable} must be a non-negative integer." >&2
        node_status=2
        exit "${node_status}"
    fi
done
for port_variable in OSMO_COMPLETION_PORT OSMO_MASTER_PORT OSMO_SOURCE_SERVE_PORT; do
    if (( ${!port_variable} < 1 || ${!port_variable} > 65535 )); then
        echo "${port_variable} must be an integer from 1 through 65535." >&2
        node_status=2
        exit "${node_status}"
    fi
done
if (( OSMO_COMPLETION_PORT == OSMO_MASTER_PORT \
    || OSMO_COMPLETION_PORT == OSMO_SOURCE_SERVE_PORT \
    || OSMO_MASTER_PORT == OSMO_SOURCE_SERVE_PORT )); then
    echo "OSMO_MASTER_PORT, OSMO_COMPLETION_PORT, and OSMO_SOURCE_SERVE_PORT must be distinct." >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS < 1 \
    || OSMO_NUM_GPU < 1 || OSMO_NUM_NODES < 1 || OSMO_TOTAL_GPU < 1 \
    || OSMO_SOURCE_FETCH_WAIT_SECONDS < 1 || OSMO_SOURCE_STARTUP_WAIT_SECONDS < 1 )); then
    echo "GPU, node, and source timeout values must be positive." >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_SOURCE_FETCH_WAIT_SECONDS < OSMO_SOURCE_STARTUP_WAIT_SECONDS + 300 )); then
    echo "OSMO_SOURCE_FETCH_WAIT_SECONDS must exceed lead source startup by at least 300 seconds." >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS <= OSMO_SOURCE_FETCH_WAIT_SECONDS )); then
    echo "OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS must exceed peer source fetch time." >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_NUM_NODES * OSMO_NUM_GPU != OSMO_TOTAL_GPU )); then
    echo "OSMO_NUM_NODES * OSMO_NUM_GPU must equal OSMO_TOTAL_GPU." >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_NODE_RANK < 0 || OSMO_NODE_RANK >= OSMO_NUM_NODES )); then
    echo "OSMO_NODE_RANK is outside the configured node world." >&2
    node_status=2
    exit "${node_status}"
fi
if [[ ! -f "${runner}" || ! -f "${barrier}" ]]; then
    echo "An injected multi-node runtime file is unavailable." >&2
    node_status=2
    exit "${node_status}"
fi
if [[ ! -x "${image_python}" ]]; then
    echo "Isaac Sim image Python is unavailable: ${image_python}" >&2
    node_status=2
    exit "${node_status}"
fi
if (( OSMO_NODE_RANK == 0 )); then
    if [[ ! -f "${source_preparer}" ]]; then
        echo "The injected source preparer is unavailable: ${source_preparer}" >&2
        node_status=2
        exit "${node_status}"
    fi
elif [[ -z "${OSMO_SOURCE_URL:-}" ]]; then
    echo "OSMO_SOURCE_URL is required on every non-lead node." >&2
    node_status=2
    exit "${node_status}"
fi

barrier_output="${OSMO_OUTPUT_DIR}/training-artifacts/run-info/ddp-node-status.tsv"
if (( OSMO_NODE_RANK == 0 )); then
    "${image_python}" "${barrier}" serve \
        --port "${OSMO_COMPLETION_PORT}" \
        --expected-nodes "${OSMO_NUM_NODES}" \
        --workflow-id "${OSMO_WORKFLOW_ID}" \
        --startup-timeout-seconds "${OSMO_BARRIER_STARTUP_TIMEOUT_SECONDS}" \
        --output "${barrier_output}" &
    barrier_pid="$!"
    if ! "${image_python}" "${barrier}" report \
        --host 127.0.0.1 \
        --port "${OSMO_COMPLETION_PORT}" \
        --node-rank 0 \
        --phase probe \
        --workflow-id "${OSMO_WORKFLOW_ID}" \
        --timeout-seconds 30 \
        --retry-seconds 0.1; then
        echo "The lead-node completion barrier did not become ready." >&2
        node_status=18
        exit "${node_status}"
    fi
fi

if (( OSMO_NODE_RANK == 0 )); then
    source_startup_wait_seconds="${OSMO_SOURCE_STARTUP_WAIT_SECONDS}"

    bash "${source_preparer}" &
    source_server_pid="$!"
    source_server_url="http://127.0.0.1:${OSMO_SOURCE_SERVE_PORT}"
    source_server_deadline="$(( SECONDS + source_startup_wait_seconds ))"
    source_server_ready=0
    while (( SECONDS < source_server_deadline )); do
        if ! kill -0 "${source_server_pid}" 2>/dev/null; then
            set +e
            wait "${source_server_pid}"
            source_server_status="$?"
            set -e
            source_server_pid=""
            echo "Source preparation exited before its HTTP server became ready (status ${source_server_status})." >&2
            node_status=19
            exit "${node_status}"
        fi
        if "${image_python}" - "${source_server_url}/source.sha256" "${OSMO_SOURCE_SHA256}" <<'PY'
import sys
import urllib.error
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=2.0) as response:
        payload = response.read(256).decode("utf-8").strip()
except (OSError, UnicodeError, urllib.error.URLError):
    raise SystemExit(1)
raise SystemExit(0 if response.status == 200 and payload == sys.argv[2] else 1)
PY
        then
            source_server_ready=1
            break
        fi
        sleep 1
    done
    if (( source_server_ready == 0 )); then
        echo \
            "Timed out after ${source_startup_wait_seconds}s waiting for the local source server ${source_server_url}." \
            >&2
        node_status=19
        exit "${node_status}"
    fi

    # The lead consumes the already validated local package. Peer nodes retain
    # OSMO_SOURCE_URL and atomically fetch the same package from this server.
    export OSMO_SOURCE_SYNC=/osmo/run/workspace/source-sync
    unset OSMO_SOURCE_URL
    printf 'source_server=ready address=%s sha256=%s\n' "${source_server_url}" "${OSMO_SOURCE_SHA256}"
fi

if (( OSMO_NODE_RANK == 0 )); then
    # Give the trainer and every process it creates a private process group so a
    # failed peer can terminate torchrun, tee, and every local rank together.
    "${image_python}" - "${runner}" <<'PY' &
import os
import sys

os.setsid()
os.execvp("bash", ["bash", sys.argv[1]])
PY
    runner_pid="$!"

    set +e
    completed_pid=""
    wait -n -p completed_pid "${runner_pid}" "${barrier_pid}"
    completed_status="$?"
    set -e

    if [[ "${completed_pid}" == "${barrier_pid}" ]]; then
        barrier_status="${completed_status}"
        barrier_pid=""
        echo "The completion barrier exited before the lead runner (status ${barrier_status})." >&2
        node_status=17
        stop_runner
    else
        node_status="${completed_status}"
        runner_pid=""
        report_status || true
        set +e
        wait "${barrier_pid}"
        barrier_status="$?"
        set -e
        if (( barrier_status != 0 && node_status == 0 )); then
            node_status=17
        fi
        barrier_pid=""
    fi
else
    set +e
    bash "${runner}"
    node_status="$?"
    set -e
    report_status || true
fi

stop_source_server

trap - EXIT INT TERM
exit "${node_status}"
