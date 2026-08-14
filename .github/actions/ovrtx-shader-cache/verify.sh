#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Host-side gate for the writeback: an empty tree means the warmer's selected
# tests never exercised that render path, leaving it to compile cold on every
# consumer. Fail rather than let a green check imply both were covered;
# warm-ovrtx-cache is continue-on-error, so this reports without breaking the
# push build. tools/verify_ovrtx_shader_cache.py is the in-container mount check.
#
# Reads KIT_FILES / KITLESS_FILES, the per-tree counts report.sh emitted.

set -euo pipefail

status=0
check_tree() {
  if [ "$2" -eq 0 ]; then
    echo "::error::OVRTX $1/ shader cache is empty - the selected tests did not exercise that render path"
    status=1
  else
    echo "OVRTX $1/ shader cache populated: $2 file(s)"
  fi
}

check_tree kit "${KIT_FILES:-0}"
check_tree kitless "${KITLESS_FILES:-0}"
exit "$status"
