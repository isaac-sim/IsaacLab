# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the Reachy 2 lift task."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def grasp_object(
    env: ManagerBasedRLEnv,
    std: float = 0.05,
    action_name: str = "gripper_action",
    object_cfg_name: str = "object",
    ee_frame_cfg_name: str = "ee_frame",
) -> torch.Tensor:
    """Reward closing the gripper when the end-effector is near the object.

    Bridges the exploration gap between "hover near the cube" and "lift the
    cube": a distance kernel gates a bonus that is only paid while the gripper
    is commanded closed, so the policy is guided to attempt grasps precisely
    when they can succeed.

    Args:
        env: The environment instance.
        std: Kernel width for the end-effector-to-object distance (meters).
        action_name: Name of the binary gripper action term.
        object_cfg_name: Scene name of the object asset.
        ee_frame_cfg_name: Scene name of the end-effector frame sensor.

    Returns:
        Tensor of shape ``(num_envs,)``: ``(1 - tanh(dist/std))`` when the
        gripper is commanded closed, ``0`` otherwise.
    """
    object_pos = env.scene[object_cfg_name].data.root_pos_w
    ee_pos = env.scene[ee_frame_cfg_name].data.target_pos_w[:, 0, :]
    dist = torch.norm(object_pos - ee_pos, dim=1)
    proximity = 1.0 - torch.tanh(dist / std)
    gripper_closed = torch.any(env.action_manager.get_term(action_name).raw_actions < 0.0, dim=1).float()
    return proximity * gripper_closed
