# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility wrapper for the rsl_rl play benchmark implementation."""

from __future__ import annotations

import sys

from isaaclab.benchmark.entrypoints.backends.rsl_rl.play import run

if __name__ == "__main__":
    run(sys.argv[1:])
