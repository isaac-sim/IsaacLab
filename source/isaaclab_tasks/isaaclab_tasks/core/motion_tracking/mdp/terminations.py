# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Adapted from HybridRobotics/whole_body_tracking (MIT).
# See docs/licenses/dependencies/motion-tracking-license.txt.

"""Terminate episodes when vertical pose or tilt tracking is lost."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Check if the anchor position is beyond the threshold in the z-axis."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    """Check if the anchor orientation is beyond the threshold."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)

    # compute the projected gravity vector for the motion and the robot
    # PhysX exposes a unit direction; Newton exposes acceleration in m/s^2.
    # The orientation threshold is defined on the projected unit gravity.
    gravity_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, gravity_w)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, gravity_w)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str]
) -> torch.Tensor:
    """Check if the body position is beyond the threshold in the z-axis."""
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indices = [command.cfg.body_names.index(name) for name in body_names]

    error = torch.abs(command.body_pos_relative_w[:, body_indices, -1] - command.robot_body_pos_w[:, body_indices, -1])
    return torch.any(error > threshold, dim=-1)
