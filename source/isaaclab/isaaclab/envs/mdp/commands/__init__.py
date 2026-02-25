# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Various command terms that can be used in the environment."""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        "commands_cfg": [
            "NormalVelocityCommandCfg",
            "NullCommandCfg",
            "TerrainBasedPose2dCommandCfg",
            "UniformPose2dCommandCfg",
            "UniformPoseCommandCfg",
            "UniformVelocityCommandCfg",
        ],
        "null_command": ["NullCommand"],
        "pose_2d_command": ["TerrainBasedPose2dCommand", "UniformPose2dCommand"],
        "pose_command": ["UniformPoseCommand"],
        "velocity_command": ["NormalVelocityCommand", "UniformVelocityCommand"],
    },
)
