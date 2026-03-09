# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.assets import ContactSensor, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def action_rate_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1).clamp(-1000, 1000)


def action_l2_clamped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1).clamp(-1000, 1000)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 1.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance with contact bonus.

    The reward is close to 1 when the distance is small. The reward is scaled by contact:
    - Full reward (1x) when good contact (thumb + finger)
    - Half reward (0.5x) when no contact

    Args:
        env: The environment instance.
        std: Standard deviation for tanh kernel.
        thumb_name: Name of the thumb contact sensor.
        finger_names: Names of the finger contact sensors.
        contact_threshold: Contact force magnitude threshold.
        object_cfg: Configuration for the object.
        asset_cfg: Configuration for the robot asset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    asset_pos = wp.to_torch(asset.data.body_pos_w)[:, asset_cfg.body_ids]
    object_pos = wp.to_torch(obj.data.root_pos_w)
    distance = torch.linalg.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    contact_bonus = contacts(env, contact_threshold, thumb_name, finger_names).float().clamp(0.1, 1.0)
    return (1 - torch.tanh(distance / std)) * contact_bonus


def _contact_force_mag(sensor: ContactSensor, num_envs: int) -> torch.Tensor:
    """Extract per-environment contact force magnitude from a sensor's force_matrix_w."""
    force = wp.to_torch(sensor.data.force_matrix_w).view(num_envs, 3)
    return torch.linalg.norm(force, dim=-1)


def contacts(env: ManagerBasedRLEnv, threshold: float, thumb_name: str, finger_names: list[str]) -> torch.Tensor:
    """Reward for good contact: thumb + at least one finger above threshold.

    Args:
        env: The environment instance.
        threshold: Contact force magnitude threshold.
        thumb_name: Name of the thumb contact sensor in the scene.
        finger_names: Names of the finger contact sensors in the scene.

    Returns:
        Boolean tensor indicating good contact condition per environment.
    """
    thumb_mag = _contact_force_mag(env.scene.sensors[thumb_name], env.num_envs)

    any_finger_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for finger_name in finger_names:
        finger_mag = _contact_force_mag(env.scene.sensors[finger_name], env.num_envs)
        any_finger_contact = any_finger_contact | (finger_mag > threshold)

    return (thumb_mag > threshold) & any_finger_contact


def contact_count(env: ManagerBasedRLEnv, threshold: float, sensor_names: list[str]) -> torch.Tensor:
    """Count the number of contact sensors with force above threshold.

    For each sensor that detects contact above the threshold, add 1 to the total.
    This provides a reward proportional to the number of fingers in contact.

    Args:
        env: The environment instance.
        threshold: Contact force magnitude threshold.
        sensor_names: Names of the contact sensors in the scene.

    Returns:
        Tensor of shape (num_envs,) with the count of sensors in contact per environment.
    """
    count = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for sensor_name in sensor_names:
        mag = _contact_force_mag(env.scene.sensors[sensor_name], env.num_envs)
        count += (mag > threshold).float()
    return count / len(sensor_names)


class success_reward(ManagerTermBase):
    """Reward success by comparing commanded pose to the object pose using tanh kernels on error.

    The reward is gated by contact: only given when thumb + at least one finger are in contact.

    Maintains a sticky ``succeeded`` boolean tensor per environment that flips to ``True`` once
    the success condition is met during an episode and resets to ``False`` on environment reset.

    Args:
        cfg: Configuration object specifying term parameters.
        env: The manager-based RL environment.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            self.succeeded[:] = False
        else:
            self.succeeded[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        pos_std: float,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.01,
        rot_std: float | None = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, des_quat_w = combine_frame_transforms(
            wp.to_torch(asset.data.root_pos_w),
            wp.to_torch(asset.data.root_quat_w),
            command[:, :3],
            command[:, 3:7],
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w,
            des_quat_w,
            wp.to_torch(obj.data.root_pos_w),
            wp.to_torch(obj.data.root_quat_w),
        )
        pos_dist = torch.linalg.norm(pos_err, dim=1)
        contact_mask = contacts(env, contact_threshold, thumb_name, finger_names)

        if rot_std:
            rot_dist = torch.linalg.norm(rot_err, dim=1)
            reward = (1 - torch.tanh(pos_dist / pos_std)) * (1 - torch.tanh(rot_dist / rot_std)) * contact_mask.float()
            self.succeeded |= contact_mask & (pos_dist < pos_std) & (rot_dist < rot_std)
        else:
            reward = ((1 - torch.tanh(pos_dist / pos_std)) ** 2) * contact_mask.float()
            self.succeeded |= contact_mask & (pos_dist < pos_std)

        return reward


def position_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward tracking of commanded position using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(
        wp.to_torch(asset.data.root_pos_w),
        wp.to_torch(asset.data.root_quat_w),
        des_pos_b,
    )
    distance = torch.linalg.norm(wp.to_torch(obj.data.root_pos_w) - des_pos_w, dim=1)
    return (1 - torch.tanh(distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()


def orientation_command_error_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    align_asset_cfg: SceneEntityCfg,
    thumb_name: str,
    finger_names: list[str],
    contact_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence."""

    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_b = command[:, 3:7]
    root_state = wp.to_torch(asset.data.root_state_w)
    des_quat_w = math_utils.quat_mul(root_state[:, 3:7], des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(wp.to_torch(obj.data.root_quat_w), des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()
