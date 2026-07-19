# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration classes for Warp-native command terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg as _UniformPoseCommandCfg
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg as _UniformVelocityCommandCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .pose_command import UniformPoseCommand
    from .velocity_command import UniformVelocityCommand


@configclass
class UniformVelocityCommandCfg(_UniformVelocityCommandCfg):
    """Configuration for the Warp-native uniform velocity command generator."""

    class_type: type[UniformVelocityCommand] | str = "{DIR}.velocity_command:UniformVelocityCommand"


@configclass
class UniformPoseCommandCfg(_UniformPoseCommandCfg):
    """Configuration for the Warp-native uniform pose command generator."""

    class_type: type[UniformPoseCommand] | str = "{DIR}.pose_command:UniformPoseCommand"
