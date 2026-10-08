#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Diagnostic only (do not merge). Runs ON THE HOST runner and starts inner.sh in the CI image.
# The image's own user owns its Python environment, so inner.sh can swap NCCL there. Reports are
# copied out of the container: on the Spark runner that user cannot write a bind-mounted directory.

set -u
reports="$PWD/reports/arm-nccl-diag"
name="isaac-lab-arm-nccl-diag-${GITHUB_RUN_ID:-local}-${GITHUB_RUN_ATTEMPT:-0}"
mkdir -p "$reports"
docker run --gpus all --network=host --entrypoint bash \
  --name "$name" \
  -e OMNI_KIT_ACCEPT_EULA=yes \
  -e ACCEPT_EULA=Y \
  -e OMNI_KIT_DISABLE_CUP=1 \
  -e ISAAC_SIM_HEADLESS=1 \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONIOENCODING=utf-8 \
  -e EXTRA_PIP="${EXTRA_PIP:-}" \
  -e ONLY="${ONLY:-}" \
  -v "$PWD/.github/diag/arm_nccl_rtx:/diag:ro" \
  "$IMAGE_TAG" /diag/inner.sh
rc=$?
docker cp "$name:/tmp/reports/." "$reports/" || true
docker rm -f "$name" >/dev/null 2>&1 || true
if [ -f "$reports/summary.md" ]; then
  cat "$reports/summary.md"
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then cat "$reports/summary.md" >>"$GITHUB_STEP_SUMMARY"; fi
fi
exit "$rc"
