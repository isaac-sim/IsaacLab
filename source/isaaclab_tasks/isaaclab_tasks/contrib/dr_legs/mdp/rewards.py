# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DR-Legs-specific reward terms with no stock Isaac Lab equivalent.

Includes the posture/effort terms used by the hold-pose task and the gait/contact/feet
terms used by the walk task. The exponential kernels follow the convention
``exp(-error / (2 * sigma**2))`` and the quaternion error is the squared norm of the
imaginary part of the relative quaternion (scalar-last ``(x, y, z, w)``, matching
:mod:`isaaclab.utils.math`).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "base_lin_vel_l2",
    "base_ang_vel_l2",
    "joint_pd_command_l2",
    "joint_pos_tracking_exp",
    "ActionRate2L2",
    "contact_matching",
    "foot_clearance",
    "feet_parallel",
    "feet_flat",
    "feet_touchdown_vel",
    "root_orientation_exp",
    "survival_success_rate",
    "walk_success_rate",
]


##
# Kernel and gait helpers (private).
##


def _exp_se(error: torch.Tensor, sigma: float) -> torch.Tensor:
    return torch.exp(-error / (2.0 * sigma * sigma))


def _quat_inv_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    return math_utils.quat_mul(math_utils.quat_conjugate(q1), q2)


def _exp_quat_err(q1: torch.Tensor, q2: torch.Tensor, sigma: float) -> torch.Tensor:
    err_imag = _quat_inv_mul(q1, q2)[:, :3]
    return _exp_se(torch.sum(torch.square(err_imag), dim=1), sigma)


def _gait_phase(env: ManagerBasedRLEnv, gait_period: float) -> torch.Tensor:
    episode_time = env.episode_length_buf.float() * env.step_dt
    return (episode_time % gait_period) / gait_period


def _reference_contacts(phase: torch.Tensor) -> torch.Tensor:
    left = (phase < 0.5).float()
    right = (phase >= 0.5).float()
    return torch.stack([left, right], dim=1)


##
# Posture / effort terms.
##


def joint_pos_tracking_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    error = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    return torch.exp(-torch.sum(torch.square(error), dim=1) / std**2)


def base_lin_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)


def base_ang_vel_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :3]), dim=1)


def joint_pd_command_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stiffness: float = 5.0,
    damping: float = 0.2,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos[:, joint_ids]
    joint_vel = asset.data.joint_vel[:, joint_ids]
    joint_pos_target = asset.data.joint_pos_target[:, joint_ids]
    pd_command = stiffness * (joint_pos_target - joint_pos) - damping * joint_vel
    return torch.sum(torch.square(pd_command), dim=1)


class ActionRate2L2(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        dim = env.action_manager.total_action_dim
        self._prev_action = torch.zeros((env.num_envs, dim), device=env.device)
        self._prev_prev_action = torch.zeros((env.num_envs, dim), device=env.device)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self._prev_action[env_ids] = 0.0
        self._prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_action = env.action_manager.action
        jerk = current_action - 2.0 * self._prev_action + self._prev_prev_action
        reward = torch.sum(torch.square(jerk), dim=1)
        self._prev_prev_action[:] = self._prev_action
        self._prev_action[:] = current_action
        return reward


##
# Gait / feet / orientation terms (walk task).
##


def contact_matching(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
    gait_period: float = 1.0,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    observed = torch.linalg.norm(net_forces, dim=-1) > threshold
    reference = _reference_contacts(_gait_phase(env, gait_period)).bool()
    return torch.sum(observed == reference, dim=1).float() - 1.0


def foot_clearance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float = 0.065,
    sigma: float = 0.01,
    gait_period: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_ids = asset_cfg.body_ids
    foot_z = asset.data.body_link_pos_w.torch[:, foot_ids, 2]
    phase = _gait_phase(env, gait_period)
    two_pi = 2.0 * math.pi
    left_target = target_height * torch.clamp(torch.sin(two_pi * (phase - 0.5)), min=0.0)
    right_target = target_height * torch.clamp(torch.sin(two_pi * phase), min=0.0)
    left_swing = (left_target > 0.0).float()
    right_swing = (right_target > 0.0).float()
    left_reward = left_swing * _exp_se(torch.square(left_target - foot_z[:, 0]), sigma)
    right_reward = right_swing * _exp_se(torch.square(right_target - foot_z[:, 1]), sigma)
    return left_reward + right_reward


def feet_parallel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.2,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    left_yaw = math_utils.yaw_quat(foot_quat[:, 0])
    right_yaw = math_utils.yaw_quat(foot_quat[:, 1])
    return _exp_quat_err(left_yaw, right_yaw, sigma)


def feet_flat(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sigma: float = 0.2,
) -> torch.Tensor:
    """Reward keeping each foot level (small roll/pitch); foot yaw is ignored. Mean over feet."""
    asset = env.scene[asset_cfg.name]
    foot_quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    num_envs, num_feet = foot_quat.shape[0], foot_quat.shape[1]
    quat = foot_quat.reshape(-1, 4)
    tilt = _quat_inv_mul(math_utils.yaw_quat(quat), quat)
    tilt_sq = torch.sum(torch.square(tilt[:, :3]), dim=1).reshape(num_envs, num_feet)
    return _exp_se(tilt_sq, sigma).mean(dim=1)


def feet_touchdown_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_z_vel = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids, 2]
    net_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    contact = (torch.linalg.norm(net_forces, dim=-1) > threshold).float()
    downward_vel = torch.clamp(-foot_z_vel, min=0.0)
    return torch.sum(contact * downward_vel, dim=-1)


def root_orientation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.2,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    root_quat = asset.data.root_link_quat_w.torch
    tilt = _quat_inv_mul(math_utils.yaw_quat(root_quat), root_quat)
    return _exp_se(torch.sum(torch.square(tilt[:, :3]), dim=1), sigma)


class survival_success_rate(ManagerTermBase):
    """Logs ``Metrics/success_rate`` = fraction of environments that survived the full episode."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)

    def reset(self, env_ids: torch.Tensor):
        survived = self._env.termination_manager.time_outs[env_ids]
        self._env.extras.setdefault("log", {})["Metrics/success_rate"] = survived.float().mean().item()

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        return torch.zeros(env.num_envs, device=env.device)


class walk_success_rate(ManagerTermBase):
    """Episode-mean velocity-tracking + gait-contact success metric for the walk task."""

    def __init__(self, env: ManagerBasedRLEnv, cfg: RewardTermCfg):
        super().__init__(cfg, env)
        self._err_xy_sum = torch.zeros(env.num_envs, device=env.device)
        self._err_yaw_sum = torch.zeros(env.num_envs, device=env.device)
        self._contact_sum = torch.zeros(env.num_envs, device=env.device)
        self._steps = torch.zeros(env.num_envs, device=env.device)
        self._vel_xy_threshold = 0.15
        self._vel_yaw_threshold = 0.4
        self._contact_match_threshold = 0.7

    def reset(self, env_ids: torch.Tensor):
        denom = self._steps[env_ids].clamp_min(1.0)
        err_xy = self._err_xy_sum[env_ids] / denom
        err_yaw = self._err_yaw_sum[env_ids] / denom
        contact = self._contact_sum[env_ids] / denom
        survived = self._env.termination_manager.time_outs[env_ids]
        success = (
            survived
            & (err_xy < self._vel_xy_threshold)
            & (err_yaw < self._vel_yaw_threshold)
            & (contact >= self._contact_match_threshold)
        )
        log = self._env.extras.setdefault("log", {})
        log["Metrics/success_rate"] = success.float().mean().item()
        log["Metrics/error_vel_xy"] = err_xy.mean().item()
        log["Metrics/error_vel_yaw"] = err_yaw.mean().item()
        log["Metrics/contact_match_rate"] = contact.mean().item()
        self._err_xy_sum[env_ids] = 0.0
        self._err_yaw_sum[env_ids] = 0.0
        self._contact_sum[env_ids] = 0.0
        self._steps[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        sensor_cfg: SceneEntityCfg,
        gait_period: float,
        contact_threshold: float,
        vel_xy_threshold: float,
        vel_yaw_threshold: float,
        contact_match_threshold: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        self._vel_xy_threshold = vel_xy_threshold
        self._vel_yaw_threshold = vel_yaw_threshold
        self._contact_match_threshold = contact_match_threshold
        command = env.command_manager.get_command(command_name)
        asset = env.scene[asset_cfg.name]
        self._err_xy_sum += torch.linalg.norm(command[:, :2] - asset.data.root_lin_vel_b.torch[:, :2], dim=1)
        self._err_yaw_sum += torch.abs(command[:, 2] - asset.data.root_ang_vel_b.torch[:, 2])
        contact_sensor = env.scene.sensors[sensor_cfg.name]
        net_forces = contact_sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
        observed = torch.linalg.norm(net_forces, dim=-1) > contact_threshold
        reference = _reference_contacts(_gait_phase(env, gait_period)).bool()
        self._contact_sum += (observed == reference).float().mean(dim=1)
        self._steps += 1.0
        return torch.zeros(env.num_envs, device=env.device)
