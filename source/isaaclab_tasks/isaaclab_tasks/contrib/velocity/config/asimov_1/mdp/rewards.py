# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 rewards."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.string import resolve_matching_names_values

from .observations import foot_pos_w, foot_vel_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

CONTACT_FORCE_THRESHOLD = 1.0


def _contact_flags(sensor: ContactSensor) -> torch.Tensor:
    return sensor.data.net_forces_w.torch.norm(dim=-1) > CONTACT_FORCE_THRESHOLD


def _command_active(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (cmd_norm > threshold).float()


def track_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


def flat_orientation(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_ids and asset_cfg.body_ids != slice(None):
        body_quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids, :].squeeze(1)
        gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
        projected = math_utils.quat_apply_inverse(body_quat, gravity_dir)
        xy_squared = torch.sum(torch.square(projected[:, :2]), dim=1)
    else:
        xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b.torch[:, :2]), dim=1)
    return torch.exp(-xy_squared / std**2)


def body_angular_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :].squeeze(1)
    return torch.sum(torch.square(ang_vel[:, :2]), dim=1)


class angular_momentum_penalty(ManagerTermBase):
    """Penalize whole-body angular momentum about the center of mass."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG).name]
        self._masses = asset.data.default_mass.torch.to(env.device)
        self._total_mass = self._masses.sum(dim=1, keepdim=True)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        pos = asset.data.body_com_pos_w.torch
        vel = asset.data.body_com_lin_vel_w.torch
        m = self._masses.unsqueeze(-1)
        com = (m * pos).sum(dim=1) / self._total_mass
        com_vel = (m * vel).sum(dim=1) / self._total_mass
        r = pos - com.unsqueeze(1)
        v = vel - com_vel.unsqueeze(1)
        angmom = (m * torch.cross(r, v, dim=-1)).sum(dim=1)
        angmom_sq = torch.sum(torch.square(angmom), dim=-1)
        env.extras.setdefault("log", {})["Metrics/angular_momentum_mean"] = torch.mean(torch.sqrt(angmom_sq))
        return angmom_sq


def joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str = "twist",
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    standing = 1.0 - _command_active(env, command_name, command_threshold)
    return torch.sum(torch.abs(angle), dim=1) * standing


def feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    threshold_min: float = 0.05,
    threshold_max: float = 0.5,
    command_name: str | None = None,
    command_threshold: float = 0.5,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    current_air_time = sensor.data.current_air_time.torch
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    in_air = current_air_time > 0
    mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(torch.sum(in_air.float()), min=1)
    env.extras.setdefault("log", {})["Metrics/air_time_mean"] = mean_air_time
    if command_name is not None:
        reward = reward * _command_active(env, command_name, command_threshold)
    return reward


def feet_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str | None = None,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    foot_z = foot_pos_w(env, asset_cfg)[:, :, 2]
    vel_norm = torch.norm(foot_vel_w(env, asset_cfg)[:, :, :2], dim=-1)
    cost = torch.sum(torch.abs(foot_z - target_height) * vel_norm, dim=1)
    if command_name is not None:
        cost = cost * _command_active(env, command_name, command_threshold)
    return cost


class feet_swing_height(ManagerTermBase):
    """Penalize deviations from the desired peak swing-foot height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        n_feet = len(cfg.params["asset_cfg"].body_names)
        self._peak_heights = torch.zeros((env.num_envs, n_feet), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_name: str,
        target_height: float,
        command_name: str,
        command_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        foot_heights = foot_pos_w(env, asset_cfg)[:, :, 2]
        in_air = ~_contact_flags(sensor)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, foot_heights), self._peak_heights)
        first_contact = sensor.compute_first_contact(dt=env.step_dt).torch
        active = _command_active(env, command_name, command_threshold)
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
        num_landings = torch.clamp(torch.sum(first_contact.float()), min=1)
        env.extras.setdefault("log", {})["Metrics/peak_height_mean"] = (
            torch.sum(self._peak_heights * first_contact.float()) / num_landings
        )
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    in_contact = _contact_flags(sensor).float()
    vel_xy_norm = torch.norm(foot_vel_w(env, asset_cfg)[:, :, :2], dim=-1)
    cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1)
    cost = cost * _command_active(env, command_name, command_threshold)
    num_in_contact = torch.clamp(torch.sum(in_contact), min=1)
    env.extras.setdefault("log", {})["Metrics/slip_velocity_mean"] = (
        torch.sum(vel_xy_norm * in_contact) / num_in_contact
    )
    return cost


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    force_magnitude = sensor.data.net_forces_w.torch.norm(dim=-1)
    first_contact = sensor.compute_first_contact(dt=env.step_dt).torch
    landing_impact = force_magnitude * first_contact.float()
    cost = torch.sum(landing_impact, dim=1)
    num_landings = torch.clamp(torch.sum(first_contact.float()), min=1)
    env.extras.setdefault("log", {})["Metrics/landing_force_mean"] = torch.sum(landing_impact) / num_landings
    if command_name is not None:
        cost = cost * _command_active(env, command_name, command_threshold)
    return cost


def feet_orientation_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_quats = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    penalty = torch.zeros(env.num_envs, device=env.device)
    for i in range(body_quats.shape[1]):
        grav_local = math_utils.quat_apply_inverse(body_quats[:, i], gravity.expand(env.num_envs, -1))
        penalty += torch.norm(grav_local[:, :2], dim=-1)
    return penalty


def feet_stumble(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    ratio_threshold: float = 4.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    forces = sensor.data.net_forces_w.torch
    force_horizontal = torch.norm(forces[:, :, :2], dim=-1)
    force_vertical = torch.abs(forces[:, :, 2])
    stumble = force_horizontal > (ratio_threshold * force_vertical)
    stumble &= _contact_flags(sensor)
    any_stumble = torch.any(stumble, dim=1).float()
    env.extras.setdefault("log", {})["Metrics/stumble_rate"] = torch.mean(any_stumble)
    return any_stumble


def feet_contact_force_limit(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    max_force: float = 350.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    force_z = torch.abs(sensor.data.net_forces_w.torch[:, :, 2])
    total_excess = torch.sum(torch.clamp(force_z - max_force, min=0.0), dim=1)
    env.extras.setdefault("log", {})["Metrics/max_contact_force"] = torch.max(force_z)
    return total_excess


def self_collision_cost(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    force_matrix = sensor.data.force_matrix_w.torch
    hit = force_matrix.norm(dim=-1) > force_threshold
    return hit.sum(dim=(1, 2)).float()


class variable_posture(ManagerTermBase):
    """Reward command-dependent posture tracking with disturbance tolerance."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        joint_ids, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)
        self._joint_ids = joint_ids
        _, _, std_standing = resolve_matching_names_values(data=cfg.params["std_standing"], list_of_strings=joint_names)
        _, _, std_walking = resolve_matching_names_values(data=cfg.params["std_walking"], list_of_strings=joint_names)
        _, _, std_running = resolve_matching_names_values(data=cfg.params["std_running"], list_of_strings=joint_names)
        self._std_standing = torch.tensor(std_standing, device=env.device)
        self._std_walking = torch.tensor(std_walking, device=env.device)
        self._std_running = torch.tensor(std_running, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std_standing,
        std_walking,
        std_running,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        walking_threshold: float = 0.1,
        running_threshold: float = 1.5,
        pose_weight_velocity_threshold: float | None = None,
        pose_weight_slow: float = 1.0,
        pose_weight_fast: float = 1.0,
        disturbance_std_scale: float = 1.0,
        disturbance_velocity_threshold: float = 0.5,
        disturbance_velocity_full: float = 1.5,
        disturbance_tilt_threshold: float = 0.15,
        disturbance_tilt_full: float = 0.6,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running

        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=1)
        total_speed = linear_speed + torch.abs(command[:, 2])

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        std = (
            self._std_standing * standing_mask.unsqueeze(1)
            + self._std_walking * walking_mask.unsqueeze(1)
            + self._std_running * running_mask.unsqueeze(1)
        )

        if disturbance_std_scale > 1.0:
            actual_lin_vel = asset.data.root_lin_vel_b.torch
            commanded_lin_vel = torch.zeros_like(actual_lin_vel)
            commanded_lin_vel[:, :2] = command[:, :2]
            lin_vel_error = torch.norm(actual_lin_vel - commanded_lin_vel, dim=1)
            velocity_den = max(disturbance_velocity_full - disturbance_velocity_threshold, 1e-6)
            velocity_severity = torch.clamp((lin_vel_error - disturbance_velocity_threshold) / velocity_den, 0.0, 1.0)
            tilt = torch.norm(asset.data.projected_gravity_b.torch[:, :2], dim=1)
            tilt_den = max(disturbance_tilt_full - disturbance_tilt_threshold, 1e-6)
            tilt_severity = torch.clamp((tilt - disturbance_tilt_threshold) / tilt_den, 0.0, 1.0)
            severity = torch.maximum(velocity_severity, tilt_severity)
            std_scale = 1.0 + severity * (disturbance_std_scale - 1.0)
            std = std * std_scale.unsqueeze(1)
            env.extras.setdefault("log", {})["Metrics/pose_disturbance_std_scale"] = torch.mean(std_scale)

        error_squared = torch.square(
            asset.data.joint_pos.torch[:, self._joint_ids] - asset.data.default_joint_pos.torch[:, self._joint_ids]
        )
        reward = torch.exp(-torch.mean(error_squared / (std**2), dim=1))

        if pose_weight_velocity_threshold is not None:
            pose_weight = torch.where(
                linear_speed < pose_weight_velocity_threshold,
                torch.full_like(linear_speed, pose_weight_slow),
                torch.full_like(linear_speed, pose_weight_fast),
            )
            reward = reward * pose_weight

        return reward
