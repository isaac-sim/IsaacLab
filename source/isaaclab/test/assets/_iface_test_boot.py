# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Kit and kitless bootstrap for backend interface tests."""

import os

_kitless = "ovphysx" in os.environ.get("LD_PRELOAD", "") or (
    os.environ.get("LD_PRELOAD", "") == "" and "EXP_PATH" not in os.environ
)

if not _kitless:
    from isaaclab.app import AppLauncher

    simulation_app = AppLauncher(headless=True).app
else:
    simulation_app = None
