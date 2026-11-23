#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

set -u

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <pip-or-uv-command> [args...]" >&2
  exit 2
fi

readonly command_attempts=3
readonly package_index_retries=3
readonly retry_delay_seconds=3

# Bound the combined command- and request-level retry budget to avoid excessively long CI stalls.
export PIP_RETRIES="${PIP_RETRIES:-$package_index_retries}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-$package_index_retries}"

for ((attempt = 1; attempt <= command_attempts; attempt++)); do
  if "$@"; then
    exit 0
  else
    exit_code=$?
  fi

  if ((attempt == command_attempts)); then
    exit "$exit_code"
  fi

  echo "Package install attempt $attempt/$command_attempts failed; retrying in $retry_delay_seconds seconds..." >&2
  sleep "$retry_delay_seconds"
done
