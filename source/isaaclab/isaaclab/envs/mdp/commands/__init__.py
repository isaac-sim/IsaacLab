# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various command terms that can be used in the environment."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .commands_cfg import NormalVelocityCommandCfg, NullCommandCfg, TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg, UniformPoseCommandCfg, UniformVelocityCommandCfg
    from .null_command import NullCommand
    from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand
    from .pose_command import UniformPoseCommand
    from .velocity_command import NormalVelocityCommand, UniformVelocityCommand

from isaaclab.utils.module import lazy_export

lazy_export(
    ("commands_cfg", [
        "NormalVelocityCommandCfg",
        "NullCommandCfg",
        "TerrainBasedPose2dCommandCfg",
        "UniformPose2dCommandCfg",
        "UniformPoseCommandCfg",
        "UniformVelocityCommandCfg",
    ]),
    ("null_command", "NullCommand"),
    ("pose_2d_command", ["TerrainBasedPose2dCommand", "UniformPose2dCommand"]),
    ("pose_command", "UniformPoseCommand"),
    ("velocity_command", ["NormalVelocityCommand", "UniformVelocityCommand"]),
)
