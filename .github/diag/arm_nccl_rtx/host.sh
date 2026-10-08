#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Diagnostic only (do not merge). Runs ON THE HOST runner and starts inner.sh in the CI image.
# Root inside the container so uv can swap NCCL in the image's environment; reports are handed
# back to the runner user so the next checkout can clean the workspace.

set -u
reports="$PWD/reports/arm-nccl-diag"
mkdir -p "$reports"
docker run --rm --gpus all --network=host --entrypoint bash \
  --name "isaac-lab-arm-nccl-diag-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}" \
  -e OMNI_KIT_ACCEPT_EULA=yes \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_DISABLE_CUP=1 \
  -e ISAAC_SIM_HEADLESS=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONIOENCODING=utf-8 \
  -e EXTRA_PIP="${EXTRA_PIP:-}" \
  -e HOST_UID="$(id -u)" \
  -e HOST_GID="$(id -g)" \
  -v "$PWD/.github/diag/arm_nccl_rtx:/diag:ro" \
  -v "$reports:/reports:rw" \
  "$IMAGE_TAG" /diag/inner.sh
rc=$?
if [ -f "$reports/summary.md" ]; then
  cat "$reports/summary.md"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$reports/summary.md" >>"$GITHUB_STEP_SUMMARY"; fi
fi
exit "$rc"
