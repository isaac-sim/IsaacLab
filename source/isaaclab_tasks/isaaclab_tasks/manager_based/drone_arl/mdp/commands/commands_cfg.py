# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.commands.commands_cfg import UniformPoseCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .drone_pose_command import DroneUniformPoseCommand


@configclass
class DroneUniformPoseCommandCfg(UniformPoseCommandCfg):
    """Configuration for uniform drone pose command generator."""

    class_type: type["DroneUniformPoseCommand"] | str = "{DIR}.drone_pose_command:DroneUniformPoseCommand"
