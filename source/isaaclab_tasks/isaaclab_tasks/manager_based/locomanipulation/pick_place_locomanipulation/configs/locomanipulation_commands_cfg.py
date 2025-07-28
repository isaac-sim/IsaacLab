# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import math
from typing import List, Tuple

from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomanipulation.pick_place_locomanipulation.mdp.commands import (
    UniformVelocityAndHeightCommand,
)


@configclass
class UniformVelocityAndHeightCommandCfg(UniformVelocityCommandCfg):
    """Configuration for uniform velocity command generator with height command."""

    class_type: type = UniformVelocityAndHeightCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        """Command ranges."""

        height: tuple[float, float] = (0.5, 0.8)  # min, max [m]

    ranges: Ranges = Ranges()
    command_names: list[str] = ["lin_vel_x", "lin_vel_y", "ang_vel_z", "height"]


@configclass
class LocomotionCommandsCfg:
    """Command terms for the locomotionMDP."""

    base_velocity = UniformVelocityAndHeightCommandCfg(
        asset_name="robot",
        # Make sure the command is only sampled at the start of the episode.
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=UniformVelocityAndHeightCommandCfg.Ranges(
            lin_vel_x=(-0.8, 1.2),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.8, 0.8),
            heading=(-math.pi, math.pi),
            height=(0.4, 0.735),
        ),
    )


@configclass
class StandingCommandsCfg(LocomotionCommandsCfg):
    """Command terms for the standing MDP."""

    base_velocity = UniformVelocityAndHeightCommandCfg(
        asset_name="robot",
        # Make sure the command is only sampled at the start of the episode.
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=1.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=UniformVelocityAndHeightCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
            heading=(-math.pi, math.pi),
            height=(0.735, 0.735),
        ),
    )

