# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


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
    no_contact_scale: float = 0.1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching the object using a tanh-kernel on end-effector distance with contact bonus.

    The reward is close to 1 when the distance is small. The reward is scaled by contact:
    - Full reward (1x) when good contact (thumb + finger)
    - Reduced reward (``no_contact_scale`` x) when no contact

    Args:
        env: The environment instance.
        std: Standard deviation for tanh kernel.
        thumb_name: Name of the thumb contact sensor.
        finger_names: Names of the finger contact sensors.
        contact_threshold: Contact force magnitude threshold.
        no_contact_scale: Reward multiplier while there is no good contact (the pre-grasp reach gate).
        object_cfg: Configuration for the object.
        asset_cfg: Configuration for the robot asset.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    asset_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    object_pos = obj.data.root_pos_w.torch
    distance = torch.linalg.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    contact_bonus = contacts(env, contact_threshold, thumb_name, finger_names).float().clamp(no_contact_scale, 1.0)
    return (1 - torch.tanh(distance / std)) * contact_bonus


def _contact_force_mag(sensor: ContactSensor, num_envs: int) -> torch.Tensor:
    """Extract per-environment contact force magnitude from a sensor's force_matrix_w."""
    force = sensor.data.force_matrix_w.torch.view(num_envs, 3)
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
            env_ids = slice(None)
        # Episode success = whether the success condition was met at any point this episode.
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = self.succeeded[env_ids].float().mean().item()
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
            asset.data.root_pos_w.torch,
            asset.data.root_quat_w.torch,
            command[:, :3],
            command[:, 3:7],
        )
        pos_err, rot_err = compute_pose_error(
            des_pos_w,
            des_quat_w,
            obj.data.root_pos_w.torch,
            obj.data.root_quat_w.torch,
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
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
        des_pos_b,
    )
    distance = torch.linalg.norm(obj.data.root_pos_w.torch - des_pos_w, dim=1)
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
    des_quat_w = math_utils.quat_mul(asset.data.root_link_quat_w.torch, des_quat_b)
    quat_distance = math_utils.quat_error_magnitude(obj.data.root_quat_w.torch, des_quat_w)

    return (1 - torch.tanh(quat_distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()


class delivery_progress(ManagerTermBase):
    """Contact-gated reward for net progress of the object toward the commanded goal.

    Returns ``clamp((d0 - d) / d0, 0, 1) * grasp``, with ``d`` the current object-to-goal distance
    and ``d0`` that distance when the goal was set. Linear and non-negative, so off-target or
    backward motion earns nothing. ``d0`` is (re)captured on reset and on goal resample; scene state
    is read in ``__call__``, never ``__init__``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._d0 = torch.ones(env.num_envs, device=env.device)  # start-distance to the current goal [m]
        self._goal_w = torch.zeros(env.num_envs, 3, device=env.device)  # last goal position [m], world
        self._need_capture = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._need_capture[slice(None) if env_ids is None else env_ids] = True

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.1,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            asset.data.root_pos_w.torch, asset.data.root_quat_w.torch, command[:, :3]
        )
        d = torch.linalg.norm(obj.data.root_pos_w.torch - des_pos_w, dim=1)
        # (Re)capture d0 on reset or goal resample. Updated unconditionally (vectorised) to avoid a
        # per-step host-device sync; unflagged envs keep their stored d0/goal via the masked where.
        capture = self._need_capture | (torch.linalg.norm(des_pos_w - self._goal_w, dim=1) > 1e-4)
        self._d0 = torch.where(capture, d.clamp(min=1e-3), self._d0)
        self._goal_w = torch.where(capture.unsqueeze(-1), des_pos_w, self._goal_w)
        self._need_capture &= ~capture
        progress = ((self._d0 - d) / self._d0).clamp(0.0, 1.0)
        return progress * contacts(env, contact_threshold, thumb_name, finger_names).float()


class _GraspAgeDecay(ManagerTermBase):
    """Base for grip rewards that fade with time since the episode's first grasp.

    After the first grasp the reward is scaled by a factor decaying linearly from 1.0 to
    ``decay_floor`` over ``decay_steps`` control steps, then held at the floor. The grasp step is
    latched on first contact only (monotonic per episode), so release-and-regrip cannot reset it.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._t_grasp = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        self._t_grasp[slice(None) if env_ids is None else env_ids] = -1

    def _decay_factor(self, env: ManagerBasedRLEnv, grasped: torch.Tensor, decay_steps: float, decay_floor: float):
        now = env.episode_length_buf
        self._t_grasp = torch.where(grasped & (self._t_grasp < 0), now, self._t_grasp)
        age = (now - self._t_grasp).clamp(min=0).float()
        factor = (1.0 - (1.0 - decay_floor) * age / decay_steps).clamp(decay_floor, 1.0)
        return torch.where(self._t_grasp >= 0, factor, torch.ones_like(factor))


class good_finger_contact_decay(_GraspAgeDecay):
    """:func:`contacts` reward (thumb + finger) with time-since-grasp decay (see :class:`_GraspAgeDecay`)."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        thumb_name: str,
        finger_names: list[str],
        decay_steps: float = 200.0,
        decay_floor: float = 0.3,
    ) -> torch.Tensor:
        grasped = contacts(env, threshold, thumb_name, finger_names)
        return grasped.float() * self._decay_factor(env, grasped, decay_steps, decay_floor)


class contact_count_decay(_GraspAgeDecay):
    """:func:`contact_count` reward with time-since-grasp decay, anchored on the thumb+finger grasp event."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        threshold: float,
        sensor_names: list[str],
        thumb_name: str,
        finger_names: list[str],
        grasp_threshold: float = 0.1,
        decay_steps: float = 200.0,
        decay_floor: float = 0.3,
    ) -> torch.Tensor:
        grasped = contacts(env, grasp_threshold, thumb_name, finger_names)
        return contact_count(env, threshold, sensor_names) * self._decay_factor(env, grasped, decay_steps, decay_floor)


class object_ee_distance_decay(_GraspAgeDecay):
    """:func:`object_ee_distance` (reach/hold) reward with *post-grasp* time-since-grasp decay.

    Pre-grasp the decay factor is 1.0 (the approach pull is untouched); after the first grasp it fades
    so that keeping the hand on a held-in-place object stops being a large constant reward.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 1.0,
        no_contact_scale: float = 0.1,
        grasp_threshold: float = 0.1,
        decay_steps: float = 200.0,
        decay_floor: float = 0.3,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        base = object_ee_distance(
            env,
            std=std,
            thumb_name=thumb_name,
            finger_names=finger_names,
            contact_threshold=contact_threshold,
            no_contact_scale=no_contact_scale,
            object_cfg=object_cfg,
            asset_cfg=asset_cfg,
        )
        grasped = contacts(env, grasp_threshold, thumb_name, finger_names)
        return base * self._decay_factor(env, grasped, decay_steps, decay_floor)
