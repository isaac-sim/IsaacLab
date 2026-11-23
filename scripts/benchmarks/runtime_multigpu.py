# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supported launcher for the multi-GPU runtime benchmark."""

from __future__ import annotations

import sys

from isaaclab.benchmark.entrypoints.multigpu import run_multigpu_benchmark_cli

if __name__ == "__main__":
    raise SystemExit(run_multigpu_benchmark_cli("runtime", sys.argv[1:]))
