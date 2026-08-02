# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
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
    - Reduced reward (0.1x) when no contact

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
    asset_pos = asset.data.body_pos_w.torch[:, asset_cfg.body_ids]
    object_pos = obj.data.root_pos_w.torch
    distance = torch.linalg.norm(asset_pos - object_pos[:, None, :], dim=-1).max(dim=-1).values
    contact_bonus = contacts(env, contact_threshold, thumb_name, finger_names).float().clamp(0.1, 1.0)
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
            self.succeeded |= (pos_dist < pos_std) & (rot_dist < rot_std) & contact_mask
        else:
            reward = ((1 - torch.tanh(pos_dist / pos_std)) ** 2) * contact_mask.float()
            self.succeeded |= (pos_dist < pos_std) & contact_mask

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
    """Reward tracking of commanded position using tanh kernel, gated by contact presence.

    .. deprecated::
        Use :class:`position_command_progress`, which pays per increment of ground gained on the
        best error so far instead of paying every step the object is near the goal. Replace
        ``std`` with ``min_improvement``.
    """
    warnings.warn(
        "The reward term 'position_command_error_tanh' is deprecated. Use 'position_command_progress' instead,"
        " replacing 'std' with 'min_improvement'.",
        DeprecationWarning,
        stacklevel=2,
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_w, _ = combine_frame_transforms(
        asset.data.root_pos_w.torch,
        asset.data.root_quat_w.torch,
        command[:, :3],
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
    """Reward tracking of commanded orientation using tanh kernel, gated by contact presence.

    .. deprecated::
        Use :class:`orientation_command_progress`, which pays per increment of ground gained on the
        best error so far instead of paying every step the object is near the goal. Replace
        ``std`` with ``min_improvement``.
    """
    warnings.warn(
        "The reward term 'orientation_command_error_tanh' is deprecated. Use 'orientation_command_progress' instead,"
        " replacing 'std' with 'min_improvement'.",
        DeprecationWarning,
        stacklevel=2,
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    obj: RigidObject = env.scene[align_asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_quat_w = math_utils.quat_mul(asset.data.root_link_quat_w.torch, command[:, 3:7])
    quat_distance = math_utils.quat_error_magnitude(obj.data.root_quat_w.torch, des_quat_w)
    return (1 - torch.tanh(quat_distance / std)) * contacts(env, contact_threshold, thumb_name, finger_names).float()


class _ProgressReward(ManagerTermBase):
    """Base class for rewards that only pay out when a tracking error reaches a new episode best.

    The term holds a per-environment bar equal to the smallest error credited so far in the episode.
    A fixed reward of ``1.0`` is paid on every step that pushes the error below that bar by at least
    ``min_improvement``, and nothing is paid otherwise. Since the bar moves only when a reward is
    paid, ground that was already credited cannot be farmed again by backing off and re-approaching,
    and the episodic reward sum counts how many improvements the policy actually made.

    The bar is seeded with the error measured on the first step of an episode, so holding the starting
    pose earns nothing. Progress made while the gating condition is false does not move the bar, so it
    stays claimable once the condition is met again. The bar is measured against the command in force,
    so it is re-seeded whenever the command resamples and never carries across goals.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # inf marks an environment whose bar has not been seeded yet against its current command
        self.best_error = torch.full((env.num_envs,), float("inf"), device=env.device)
        self._prev_command: torch.Tensor | None = None

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.best_error[env_ids] = float("inf")

    def _progress(
        self, error: torch.Tensor, gate: torch.Tensor, min_improvement: float, command: torch.Tensor
    ) -> torch.Tensor:
        """Return 1.0 for the environments that beat their best error under the command in force.

        Args:
            error: Current tracking error per environment.
            gate: Environments allowed to be credited this step.
            min_improvement: Amount the error must beat the bar by to be paid again.
            command: Command being tracked; a change re-seeds the bar for that environment.

        Returns:
            Tensor of shape ``(num_envs,)`` that is ``1.0`` where a payout is due.
        """
        # a resampled command changes the error's reference, so the bar it was measured against no
        # longer applies; dropping it to inf re-seeds from the first error under the new command
        if self._prev_command is None:
            self._prev_command = command.clone()
        else:
            self.best_error[(self._prev_command != command).any(dim=1)] = float("inf")
            self._prev_command.copy_(command)
        unseeded = torch.isinf(self.best_error)
        self.best_error[unseeded] = error[unseeded]
        improved = gate & (error < self.best_error - min_improvement)
        self.best_error[improved] = error[improved]
        return improved.float()


class position_command_progress(_ProgressReward):
    """Reward every step that brings the object closer to the commanded position than ever before.

    See :class:`_ProgressReward` for the progress bookkeeping. The reward is gated by contact, so
    only positional gains made while the thumb and at least one finger are touching are credited.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.1,
        min_improvement: float = 0.01,
    ) -> torch.Tensor:
        """Compute the progress reward.

        Args:
            env: The environment instance.
            command_name: Name of the pose command to track.
            asset_cfg: Configuration for the asset the command is expressed relative to.
            align_asset_cfg: Configuration for the asset that must reach the command.
            thumb_name: Name of the thumb contact sensor.
            finger_names: Names of the finger contact sensors.
            contact_threshold: Contact force magnitude threshold [N].
            min_improvement: Distance [m] the object must gain on the episode best to be paid again.
        """
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_pos_w, _ = combine_frame_transforms(
            asset.data.root_pos_w.torch,
            asset.data.root_quat_w.torch,
            command[:, :3],
        )
        distance = torch.linalg.norm(obj.data.root_pos_w.torch - des_pos_w, dim=1)
        gate = contacts(env, contact_threshold, thumb_name, finger_names)
        return self._progress(distance, gate, min_improvement, command)


class orientation_command_progress(_ProgressReward):
    """Reward every step that brings the object closer to the commanded orientation than ever before.

    See :class:`_ProgressReward` for the progress bookkeeping. The reward is gated by contact, so
    only rotational gains made while the thumb and at least one finger are touching are credited.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        asset_cfg: SceneEntityCfg,
        align_asset_cfg: SceneEntityCfg,
        thumb_name: str,
        finger_names: list[str],
        contact_threshold: float = 0.1,
        min_improvement: float = 0.05,
    ) -> torch.Tensor:
        """Compute the progress reward.

        Args:
            env: The environment instance.
            command_name: Name of the pose command to track.
            asset_cfg: Configuration for the asset the command is expressed relative to.
            align_asset_cfg: Configuration for the asset that must reach the command.
            thumb_name: Name of the thumb contact sensor.
            finger_names: Names of the finger contact sensors.
            contact_threshold: Contact force magnitude threshold [N].
            min_improvement: Angle [rad] the object must gain on the episode best to be paid again.
        """
        asset: RigidObject = env.scene[asset_cfg.name]
        obj: RigidObject = env.scene[align_asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        des_quat_w = math_utils.quat_mul(asset.data.root_link_quat_w.torch, command[:, 3:7])
        quat_distance = math_utils.quat_error_magnitude(obj.data.root_quat_w.torch, des_quat_w)
        gate = contacts(env, contact_threshold, thumb_name, finger_names)
        return self._progress(quat_distance, gate, min_improvement, command)
