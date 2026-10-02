#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Run the performance-smoke matrix against one checkout and dependency image.
set -uo pipefail

: "${CHECKOUT_ROOT:?}" "${BENCHMARK_OUTPUT:?}" "${BENCHMARK_LEGS:?}"
: "${CI_IMAGE_TAG:?}" "${JIT_CACHE_ROOT:?}" "${GITHUB_RUN_ID:?}" "${GITHUB_RUN_ATTEMPT:?}"
BENCHMARK_ROLE="${BENCHMARK_ROLE:-current}"

install -d -m 0777 "${JIT_CACHE_ROOT}/warp" "${JIT_CACHE_ROOT}/nv"
chmod -R a+rwX "${JIT_CACHE_ROOT}" || exit 1

container=""
cleanup_container() {
  if [ -n "$container" ]; then
    docker rm -f "$container" >/dev/null 2>&1 || true
  fi
}
trap cleanup_container EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

run_attempt() {
  local attempt="$1"
  container="performance-smoke-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}-${BENCHMARK_ROLE}-${leg}-${sample}-${attempt}"
  cleanup_container
  timeout "${bench_timeout_s}" docker run --name "$container" --gpus all --network=host --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e HOME=/tmp/isaaclab-ci-home \
    -e USER="$(id -un)" \
    -e LOGNAME="$(id -un)" \
    -e XDG_CACHE_HOME=/tmp/isaaclab-ci-home/.cache \
    -e XDG_DATA_HOME=/tmp/isaaclab-ci-home/.local/share \
    -e UV_CACHE_DIR=/tmp/isaaclab-ci-home/.cache/uv \
    -e WARP_CACHE_PATH=/tmp/jit-cache/warp \
    -e CUDA_CACHE_PATH=/tmp/jit-cache/nv \
    -v "${JIT_CACHE_ROOT}:/tmp/jit-cache" \
    -v "${CHECKOUT_ROOT}/source:/workspace/isaaclab/source:rw" \
    -v "${CHECKOUT_ROOT}/scripts:/workspace/isaaclab/scripts:ro" \
    -v "${CHECKOUT_ROOT}/apps:/workspace/isaaclab/apps:ro" \
    --entrypoint bash "$CI_IMAGE_TAG" -c \
    "set -euo pipefail
    mkdir -p /tmp/benchmark-output /tmp/isaaclab-ci-home/.cache /tmp/isaaclab-ci-home/.local/share
    uv run --no-sync isaaclab benchmark runtime \
      --task '$task' \
      --num_envs '$num_envs' \
      --num_steps 200 \
      --warmup_steps 100 \
      --seed 42 \
      --benchmark_formatter schema \
      --output_path /tmp/benchmark-output \
      --visualizer none \
      $args"
  local status=$?
  docker cp "$container:/tmp/benchmark-output/." "${BENCHMARK_OUTPUT}/${leg}/sample-${sample}/" 2>/dev/null || true
  cleanup_container
  container=""
  local -a bundles
  mapfile -t bundles < <(
    find "${BENCHMARK_OUTPUT}/${leg}/sample-${sample}" -maxdepth 1 -type f -name 'benchmark_runtime_*.json'
  )
  if [ "$status" -eq 0 ] && [ "${#bundles[@]}" -ne 1 ]; then
    echo "::error::${leg} exited 0 but did not produce exactly one benchmark JSON"
    return 1
  fi
  return "$status"
}

failed_legs=()
while IFS='|' read -r leg task num_envs bench_timeout_s args; do
  echo "::group::performance-smoke: ${BENCHMARK_ROLE}: ${leg}"
  install -d -m 0777 "${BENCHMARK_OUTPUT}/${leg}"
  leg_ok=true
  for sample in 1 2 3; do
    mkdir -p "${BENCHMARK_OUTPUT}/${leg}/sample-${sample}"
    if ! run_attempt 1; then
      echo "::warning::${leg} sample ${sample} failed; retrying once"
      rm -f "${BENCHMARK_OUTPUT}/${leg}/sample-${sample}/"benchmark_runtime_*.json
      if ! run_attempt 2; then
        echo "::error::${leg} sample ${sample} failed on both attempts"
        leg_ok=false
        break
      fi
    fi
  done
  if [ "$leg_ok" = true ]; then
    echo ok > "${BENCHMARK_OUTPUT}/${leg}/status"
  else
    echo failed > "${BENCHMARK_OUTPUT}/${leg}/status"
    failed_legs+=("$leg")
  fi
  echo "::endgroup::"
done < "$BENCHMARK_LEGS"

if [ "${#failed_legs[@]}" -gt 0 ]; then
  echo "::error::performance-smoke: failed combination(s): ${failed_legs[*]}"
  exit 1
fi
