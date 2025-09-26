# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Code adapted from https://github.com/leggedrobotics/nav-suite

# Copyright (c) 2025, The Nav-Suite Project Developers (https://github.com/leggedrobotics/nav-suite/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass

from .goal_command import GoalCommandTerm


@configclass
class GoalCommandCfg(CommandTermCfg):
    class_type: type = GoalCommandTerm

    vis_line: bool = True
    """Whether to visualize the line from the robot to the goal."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the commands are generated."""

    path_length_range: list[float] = [2.0, 10.0]
    """Range of the sampled trajectories between start and goal."""

    raycaster_sensor: str = MISSING
    """Name of the raycaster sensor to use for terrain analysis."""

    grid_resolution: float = 0.1
    """Resolution of the grid for the terrain analysis."""

    robot_length: float = 1.0
    """Length of the robot for the terrain analysis."""

    reset_pos_term_name: str = MISSING
    """Name of the reset position term to use for the goal command."""

    edge_threshold: float = 0.5
    """Threshold for the edge detection to define as untraversable."""
