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

readonly package_index_retries=12

# Let pip and uv retry failed HTTP requests without repeating successful setup work.
export PIP_RETRIES="${PIP_RETRIES:-$package_index_retries}"
export UV_HTTP_RETRIES="${UV_HTTP_RETRIES:-$package_index_retries}"

exec "$@"
