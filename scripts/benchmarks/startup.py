# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Supported launcher for :mod:`isaaclab.benchmark.entrypoints.startup`."""

from __future__ import annotations

import sys

from isaaclab.benchmark.entrypoints.startup import run

if __name__ == "__main__":
    run(sys.argv[1:])
