# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def pose_command_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    position_threshold: float,
    orientation_threshold: float,
) -> torch.Tensor:
    """Return whether position and orientation errors are below their thresholds.

    Args:
        env: The environment instance.
        command_name: Name of the pose command term.
        position_threshold: Maximum position error [m].
        orientation_threshold: Maximum rotation-vector error [rad].

    Returns:
        A boolean tensor indicating which environments reached the commanded pose.
    """
    command = env.command_manager.get_term(command_name)
    desired_position_w, desired_orientation_w = combine_frame_transforms(
        command.robot.data.root_pos_w.torch,
        command.robot.data.root_quat_w.torch,
        command.command[:, :3],
        command.command[:, 3:],
    )
    position_error, orientation_error = compute_pose_error(
        desired_position_w,
        desired_orientation_w,
        command.robot.data.body_pos_w.torch[:, command.body_idx],
        command.robot.data.body_quat_w.torch[:, command.body_idx],
    )
    return torch.logical_and(
        torch.linalg.norm(position_error, dim=-1) < position_threshold,
        torch.linalg.norm(orientation_error, dim=-1) < orientation_threshold,
    )
