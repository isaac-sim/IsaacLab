# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration of the velocity command of the MicroDuck on a ball."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .commands import BallVelocityCommand


@configclass
class BallVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the ball velocity command term."""

    class_type: type[BallVelocityCommand] | str = "{DIR}.commands:BallVelocityCommand"
    moving_speed: float = 0.03
    """Commanded speed [m/s] above which the robot is expected to step."""
    moving_yaw_rate: float = 0.1
    """Commanded yaw rate [rad/s] above which the robot is expected to step."""
    base_frequency: float = 1.5
    """Frequency of the stepping clock while moving [Hz]."""
    speed_frequency: float = 8.0
    """Extra clock frequency per commanded speed [Hz / (m/s)]."""
    turn_frequency: float = 2.0
    """Extra clock frequency per commanded yaw rate [Hz / (rad/s)]."""
