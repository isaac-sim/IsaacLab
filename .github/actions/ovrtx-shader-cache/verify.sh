#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Host-side gate for the writeback: the warmer exists to publish both trees, so
# an empty one means its selected tests never exercised that render path - a
# silent half-warm that leaves the missing path compiling cold on every
# consumer. Fail rather than let the green check imply both were covered.
# warm-ovrtx-cache is continue-on-error, so this reports without breaking the
# push build.
#
# Not to be confused with tools/verify_ovrtx_shader_cache.py, which runs inside
# the test container and checks that the mounts landed at all.
#
# Reads KIT_FILES / KITLESS_FILES, the per-tree counts report.sh emitted.

set -euo pipefail

status=0
for tree in kit kitless; do
  files="${KIT_FILES:-0}"
  [ "$tree" = "kitless" ] && files="${KITLESS_FILES:-0}"
  if [ "${files:-0}" -eq 0 ]; then
    echo "::error::OVRTX ${tree}/ shader cache is empty - the selected tests did not exercise that render path"
    status=1
  else
    echo "OVRTX ${tree}/ shader cache populated: ${files} file(s)"
  fi
done
exit "$status"
