#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Runs ON THE HOST runner. Launches ONE container that runs the multi-GPU
# training smoke tests, which spawn real two-rank torchrun jobs.
#
# Invoked by .github/workflows/test-multi-gpu-pytest.yaml's "Multi-GPU training
# smoke" step. Reads from the environment (set by that step):
#   IMAGE_TAG — per-commit CI image to ``docker run``
#
# Deliberately NOT sharded, unlike multi_gpu_host_launcher.sh. That launcher
# pins each shard to its own cuda:N via ISAACLAB_TEST_DEVICES, which models a
# test parametrized over a single device. A two-rank training run owns two GPUs
# at once, so a per-device shard cannot express it: the tests here pick their own
# GPU pair from ``nvidia-smi topo -m`` and pin it with CUDA_VISIBLE_DEVICES.
#
# Every GPU is exposed to the container for that reason -- the pair is selected
# inside, not by the runner.
set -euo pipefail

: "${IMAGE_TAG:?IMAGE_TAG must be set}"

CONTAINER="mgpu-smoke-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-1}"
TEST_PATH="source/isaaclab/test/multi_gpu/test_multi_gpu_training_smoke.py"

cleanup() { docker rm -f "$CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "::group::GPU topology on this runner"
# Recorded because it decides which cases run: a host with no cross-socket pair
# skips the xfail case, and a host with no same-switch pair skips the strict
# camera guard. Without this the skips in the report have no visible cause.
TOPO="$(nvidia-smi topo -m 2>/dev/null || true)"
echo "${TOPO:-topology unavailable}"
echo "::endgroup::"

# A runner whose GPUs all share a switch cannot reproduce NVBUG#6565122 at all.
# That is a correct skip, but silently: three skips among many read as a normal
# green run, so the step would look like it covered the defect when it did not.
# Say so where the run summary shows it.
if [ -z "$TOPO" ]; then
  echo "::warning::GPU topology unavailable -- multi-GPU smoke cases will skip; this run does not cover NVBUG#6565122"
elif ! grep -qw SYS <<<"$TOPO"; then
  echo "::warning::No cross-socket (SYS) GPU pair on this runner -- the NVBUG#6565122 cases will skip; this run does not cover the cross-socket regression"
fi
if ! grep -qwE "PIX|NV[0-9]+" <<<"$TOPO"; then
  echo "::warning::No same-switch (PIX/NVLink) GPU pair on this runner -- the strict camera regression guard will skip"
fi

# --entrypoint bash is required: the image inherits /isaac-sim/runheadless.sh from
# the Isaac Sim base, which would swallow the command and launch Kit instead of
# running pytest. multi_gpu_host_launcher.sh overrides it for the same reason.
# Mount the checkout and run as the host user, exactly as multi_gpu_host_launcher.sh
# does: the image's baked-in /workspace/isaaclab predates this commit, so without
# the mount pytest collects 0 items, and without --user the container writes
# root-owned files into the runner's workspace.
host_uid="$(id -u)"
host_gid="$(id -g)"
docker run --rm --name "$CONTAINER" \
  --entrypoint bash \
  --user "${host_uid}:${host_gid}" \
  -e USER="$(id -un)" \
  -v "$PWD:/workspace/isaaclab:rw" \
  --gpus all --network host --shm-size=16g \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  "$IMAGE_TAG" \
  -lc "cd /workspace/isaaclab && ./isaaclab.sh -p -m pytest ${TEST_PATH} -v -rA --junitxml=/tmp/mgpu-smoke.xml"
