#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Computes the collection prefix, per-run write key and host directory for each
# of the two OVRTX shader cache trees, and appends them to $GITHUB_OUTPUT.
#
# Every mode of the action runs this, so the restore that reads a collection and
# the save that writes it cannot drift apart. That works only because the result
# is constant for the lifetime of a job: RUNNER_OS, RUNNER_ARCH, the nvidia-smi
# driver version and compute capability, the uv.lock ovrtx pin, ISAACSIM_VERSION
# and GITHUB_SHA/RUN_ID/RUN_ATTEMPT are all fixed once the job starts. Anything
# added here has to hold the same property, or the warmer publishes under a key
# no consumer ever restores.
#
# Reads (environment):
#   ISAACSIM_VERSION  Isaac Sim image tag, identifies the Kit RTX build
#   GITHUB_WORKSPACE  repository checkout, for uv.lock
#   RUNNER_OS, RUNNER_ARCH, RUNNER_TEMP, GITHUB_SHA, GITHUB_RUN_ID,
#   GITHUB_RUN_ATTEMPT, GITHUB_OUTPUT

set -euo pipefail

: "${ISAACSIM_VERSION:?isaacsim-version input is required}"

# nv_shadercache holds NVIDIA Vulkan driver PSO blobs. A blob is valid only for
# the GPU architecture and driver version that compiled it, so both gate every
# entry regardless of which tree it belongs to.
#
# Checked rather than defaulted: every job that uses this action runs on a GPU
# runner, so a driver these queries cannot see is a broken runner rather than a
# case to degrade for. A default is also unsafe here because the key is
# recomputed per mode - a query that fails in one invocation but not another
# would publish the snapshot into a collection nothing reads, leaving the cache
# permanently cold with no failing step to show for it.
#
# Guarding the pipeline's exit status would not catch it: head and tr both
# succeed on empty input, so a '|| echo unknown' tail never fires and the
# components silently come out blank. Test the captured values instead.
driver_ver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' || true)"
if [ -z "${driver_ver}" ] || [ -z "${gpu_arch}" ]; then
  echo "::error::nvidia-smi reported no driver version or compute capability; cannot key the OVRTX shader cache"
  exit 1
fi

# The lockfile pin, not the wheel the container resolves from the ovrtx range:
# the version only decides when a collection is retired, since drv/sm above are
# what gate whether a blob is usable at all. Reading it here keeps the key a
# function of the commit, so a wheel published upstream cannot silently re-key
# every open PR onto a cold collection. uv writes name/version as single-line
# pairs.
ovrtx_ver="$(grep -A1 '^name = "ovrtx"$' "${GITHUB_WORKSPACE}/uv.lock" | sed -n 's/^version = "\(.*\)"$/\1/p' | head -1 || true)"
if [ -z "${ovrtx_ver}" ]; then
  echo "::error::uv.lock has no version entry for ovrtx"
  exit 1
fi

# Cache keys may not contain commas, and the isaacsim tag is only conventionally
# bare, so keep both components to a known-safe alphabet.
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
