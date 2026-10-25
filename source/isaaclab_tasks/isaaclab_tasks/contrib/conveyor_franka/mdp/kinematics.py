# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent Franka tool-state helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation

    from ..conveyor_franka_env import ConveyorFrankaEnv


def end_effector_pose(env: ConveyorFrankaEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the Franka tool-center position [m] and orientation."""
    robot: Articulation = env.scene["robot"]
    body_id = env._conveyor_hand_body_id
    orientation = robot.data.body_quat_w.torch[:, body_id]
    position = robot.data.body_pos_w.torch[:, body_id]
    position = position + math_utils.quat_apply(orientation, env._conveyor_tool_offset)
    return position, orientation


def tool_velocity(env: ConveyorFrankaEnv) -> torch.Tensor:
    """Return tool-center linear and angular velocity [m/s, rad/s]."""
    robot: Articulation = env.scene["robot"]
    body_id = env._conveyor_hand_body_id
    orientation = robot.data.body_quat_w.torch[:, body_id]
    body_velocity = robot.data.body_link_vel_w.torch[:, body_id]
    offset_world = math_utils.quat_apply(orientation, env._conveyor_tool_offset)
    linear_velocity = body_velocity[:, :3] + torch.linalg.cross(body_velocity[:, 3:], offset_world)
    return torch.cat((linear_velocity, body_velocity[:, 3:]), dim=1)
