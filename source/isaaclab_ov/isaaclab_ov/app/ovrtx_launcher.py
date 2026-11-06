# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launcher for the kitless OVRTX runtime."""

from __future__ import annotations

from isaaclab.app.sim_launcher import SimulationLauncher


class OvrtxLauncher(SimulationLauncher):
    """Registers the OVRTX USD schemas before any stage is opened.

    USD discovers schema plugins once per process, so this runs before the scene is built.
    """

    def __init__(self, launcher_args=None):
        import ovrtx

        ovrtx.register_schema_paths()
