# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    # observations
    "deformable_com_in_robot_root_frame",
    "DeformableSampledPointsInRobotRootFrame",
    # rewards
    "deformable_lifted",
    "deformable_lifting",
    "deformable_ee_distance",
    "deformable_com_ee_distance",
    "deformable_fingertip_distance",
    "deformable_com_goal_distance",
    "deformable_com_goal_distance_delta",
    "deformable_com_goal_reached",
    # terminations
    "deformable_nodal_vel_above_maximum",
    "deformable_outside_bounds",
    "deformable_state_invalid",
    "joint_vel_out_of_sim_limit",
    # events
    "randomize_deformable_material",
    "reset_deformable_over_support",
    # commands
    "DeformableUniformPoseCommand",
    "DeformableUniformPoseCommandCfg",
    # curriculums
    "modify_gravity_linear",
]

from .curriculums import modify_gravity_linear
from .events import randomize_deformable_material, reset_deformable_over_support
from .observations import (
    DeformableSampledPointsInRobotRootFrame,
    deformable_com_in_robot_root_frame,
)
from .pose_commands import (
    DeformableUniformPoseCommand,
    DeformableUniformPoseCommandCfg,
)
from .rewards import (
    deformable_com_ee_distance,
    deformable_com_goal_distance,
    deformable_com_goal_distance_delta,
    deformable_com_goal_reached,
    deformable_ee_distance,
    deformable_fingertip_distance,
    deformable_lifted,
    deformable_lifting,
)
from .terminations import (
    deformable_nodal_vel_above_maximum,
    deformable_outside_bounds,
    deformable_state_invalid,
    joint_vel_out_of_sim_limit,
)
from isaaclab_tasks.core.lift.mdp import *
