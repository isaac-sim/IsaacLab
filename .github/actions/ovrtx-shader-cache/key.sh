#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Computes the collection prefix, per-run write key and host directory for each
# of the two OVRTX shader cache trees, and appends them to $GITHUB_OUTPUT.
#
# Every mode of the action runs this, so the collection a job restores from and
# the one it saves to cannot drift. Only values that stay constant for the
# lifetime of a job may be keyed on.
#
# Reads: ISAACSIM_VERSION, GITHUB_WORKSPACE (for uv.lock), RUNNER_OS,
# RUNNER_ARCH, RUNNER_TEMP, GITHUB_SHA, GITHUB_RUN_ID, GITHUB_RUN_ATTEMPT.

set -euo pipefail

: "${ISAACSIM_VERSION:?isaacsim-version input is required}"

# nv_shadercache holds NVIDIA Vulkan driver PSO blobs, which are only valid for
# the GPU architecture and driver version that compiled them, so both gate every
# entry. Missing values fail rather than default: head and tr succeed on empty
# input, so the captured values have to be tested directly.
driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)"
if [ -z "${driver_ver}" ] || [ -z "${gpu_arch}" ]; then
  echo "::error::nvidia-smi reported no driver version or compute capability; cannot key the OVRTX shader cache"
  exit 1
fi

# The uv.lock pin, not the wheel the container resolves from the ovrtx range,
# so the key stays a function of the commit and an upstream release cannot
# re-key every open PR onto a cold collection.
ovrtx_ver="$(grep -A1 '^name = "ovrtx"$' "${GITHUB_WORKSPACE}/uv.lock" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1 || true)"
if [ -z "${ovrtx_ver}" ]; then
  echo "::error::uv.lock has no version entry for ovrtx"
  exit 1
fi

# Cache keys may not contain commas, and the isaacsim tag is only conventionally
# bare.
sanitize() { printf '%s' "$1" | tr -c 'A-Za-z0-9._-' '_'; }

base="v1-${RUNNER_OS}-${RUNNER_ARCH}-sm${gpu_arch}-drv$(sanitize "$driver_ver")"
kit_collection="ovrtx-kit-${base}-isaacsim$(sanitize "$ISAACSIM_VERSION")"
kitless_collection="ovrtx-kitless-${base}-ovrtx$(sanitize "$ovrtx_ver")"
host_dir="${RUNNER_TEMP}/isaaclab-ovrtx-shader-cache"

# Write key is unique per run so each cumulative snapshot is its own immutable
# entry; restore matches the collection prefix, newest first.
emit() {
  echo "$1-collection=$2" >> "$GITHUB_OUTPUT"
  echo "$1-key=$2-${GITHUB_SHA}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" >> "$GITHUB_OUTPUT"
  echo "$1-restore-keys=$2-" >> "$GITHUB_OUTPUT"
  echo "$1-dir=${host_dir}/$1" >> "$GITHUB_OUTPUT"
}
emit kit "$kit_collection"
emit kitless "$kitless_collection"
echo "host-dir=${host_dir}" >> "$GITHUB_OUTPUT"

echo "OVRTX shader cache collections (driver ${driver_ver}, sm${gpu_arch}):"
echo "  kit/      ${kit_collection}"
echo "  kitless/  ${kitless_collection}"
