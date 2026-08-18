#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -u

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 2
fi

readonly max_attempts=3
readonly retry_delay_seconds=3

# Retry whole setup commands so interrupted downloads can resume from the package cache.
for ((attempt = 1; attempt <= max_attempts; attempt++)); do
  if "$@"; then
    exit 0
  else
    status=$?
  fi

  if [ "$attempt" -eq "$max_attempts" ]; then
    exit "$status"
  fi

  echo "::warning::Command failed (attempt ${attempt}/${max_attempts}); retrying in ${retry_delay_seconds} seconds"
  sleep "$retry_delay_seconds"
done
