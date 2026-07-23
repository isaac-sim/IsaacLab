# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for state-based in-hand reorientation tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_angle_axis, quat_mul

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def sample_joint_positions_within_limits(
    default_position: torch.Tensor,
    limits: torch.Tensor,
    noise_scale: float,
) -> torch.Tensor:
    """Sample reset positions between each joint's default position and limits.

    Args:
        default_position: Default joint positions [m or rad, depending on joint type], shape ``(..., J)``.
        limits: Lower and upper joint-position limits [m or rad, depending on joint type], shape ``(..., J, 2)``.
        noise_scale: Dimensionless interpolation scale from the default position toward the sampled limits.

    Returns:
        Sampled joint positions [m or rad, depending on joint type], shape ``(..., J)``.

    Raises:
        ValueError: If :paramref:`noise_scale` is outside ``[0, 1]``.
    """
    if not 0.0 <= noise_scale <= 1.0:
        raise ValueError(f"Expected noise_scale in [0, 1], got {noise_scale}.")
    position_sample = math_utils.sample_uniform(
        -1.0,
        1.0,
        default_position.shape,
        device=default_position.device,
    )
    position_fraction = 0.5 * (position_sample + 1.0)
    position_delta = limits[..., 0] - default_position
    position_delta = position_delta + (limits[..., 1] - limits[..., 0]) * position_fraction
    joint_position = default_position + noise_scale * position_delta
    return torch.clamp(joint_position, min=limits[..., 0], max=limits[..., 1])


def random_xy_rotation(count: int, device: str | torch.device) -> torch.Tensor:
    """Sample the Direct tasks' sequential random X/Y rotation.

    Args:
        count: Number of rotations to sample.
        device: Device on which to sample.

    Returns:
        Sampled ``(x, y, z, w)`` unit quaternions, shape ``(count, 4)``.
    """
    random_values = math_utils.sample_uniform(-1.0, 1.0, (count, 2), device=device)
    x_unit = torch.tensor([1.0, 0.0, 0.0], device=device).repeat(count, 1)
    y_unit = torch.tensor([0.0, 1.0, 0.0], device=device).repeat(count, 1)
    return math_utils.quat_mul(
        math_utils.quat_from_angle_axis(random_values[:, 0] * torch.pi, x_unit),
        math_utils.quat_from_angle_axis(random_values[:, 1] * torch.pi, y_unit),
    )


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    """Compose ``[-pi, pi]``-scaled random X- and Y-axis rotations into ``(x, y, z, w)`` quaternions."""
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


def reset_reorient_state(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    position_noise: float,
    joint_position_noise: float,
    joint_velocity_noise: float,
    action_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the object and hand with the Direct task's distributions.

    Args:
        env: Environment containing the robot and object.
        env_ids: Environment indices to reset.
        position_noise: Object-position noise half-width [m].
        joint_position_noise: Scale applied to sampled joint-position deltas.
        joint_velocity_noise: Joint-velocity noise half-width [rad/s].
        action_name: Action term whose terminal raw action is retained in the reset observation.
        robot_cfg: Robot scene entity.
        object_cfg: Object scene entity.
    """
    raw_action = env.action_manager.get_term(action_name).raw_actions
    if not hasattr(env, "_reorient_reset_action"):
        env._reorient_reset_action = torch.zeros_like(raw_action)
        env._reorient_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=raw_action.device)
    env._reorient_reset_action[env_ids] = raw_action[env_ids]
    env._reorient_reset_step[env_ids] = env.common_step_counter

    object_asset: Articulation | RigidObject = env.scene[object_cfg.name]
    object_pose = object_asset.data.default_root_pose.torch[env_ids].clone()
    object_velocity = torch.zeros_like(object_asset.data.default_root_vel.torch[env_ids])
    position_delta = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), 3), device=env.device)
    object_pose[:, :3] += position_noise * position_delta + env.scene.env_origins[env_ids]
    object_pose[:, 3:7] = random_xy_rotation(len(env_ids), env.device)
    object_asset.write_root_pose_to_sim_index(root_pose=object_pose, env_ids=env_ids)
    object_asset.write_root_velocity_to_sim_index(root_velocity=object_velocity, env_ids=env_ids)

    robot: Articulation = env.scene[robot_cfg.name]
    default_position = robot.data.default_joint_pos.torch[env_ids]
    limits = robot.data.joint_limits.torch[env_ids]
    joint_position = sample_joint_positions_within_limits(default_position, limits, joint_position_noise)
    velocity_sample = math_utils.sample_uniform(-1.0, 1.0, (len(env_ids), robot.num_joints), device=env.device)
    joint_velocity = robot.data.default_joint_vel.torch[env_ids] + joint_velocity_noise * velocity_sample
    robot.set_joint_position_target_index(target=joint_position, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=joint_position, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocity, env_ids=env_ids)
