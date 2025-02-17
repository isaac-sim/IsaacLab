# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import RigidObject

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_pose(env: ManagerBasedRLEnv, object_name: str = "box") -> torch.Tensor:
    """Get the pose (position and orientation) of an object in the world frame.

    Args:
        env: The environment instance
        object_name: Name of the object to get pose for. Defaults to "box".

    Returns:
        Object pose as [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]. Shape: (num_envs, 7)
    """
    # Get object from scene
    object: RigidObject = env.scene[object_name]

    if not hasattr(object, "data"):
        return torch.zeros(env.num_envs, 7, device=env.device)

    # Get position and orientation from root state
    # Root state contains [pos, quat, lin_vel, ang_vel]
    object_pos = object.data.root_pos_w
    object_quat = object.data.root_quat_w

    # Concatenate position and orientation
    return torch.cat([object_pos, object_quat], dim=1)
