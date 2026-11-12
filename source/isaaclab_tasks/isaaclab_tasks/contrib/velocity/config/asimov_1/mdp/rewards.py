# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 reward terms that have no counterpart in the shared velocity MDP.

Foot heights are measured from the height of the same foot site at its last ground contact, so they do not
depend on the terrain elevation. A foot is in contact when its net contact force exceeds 1 N.
"""

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

CONTACT_FORCE_THRESHOLD = 1.0
"""Net contact force [N] above which a body counts as in contact."""


def _contact_flags(sensor: ContactSensor, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    return forces.norm(dim=-1) > CONTACT_FORCE_THRESHOLD


def _command_active(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    """Flag environments whose planar speed command [m/s] plus yaw-rate command [rad/s] exceeds the threshold."""
    command = env.command_manager.get_command(command_name)
    cmd_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
    return (cmd_norm > threshold).float()


def _selected_names(names: list[str], ids) -> list[str]:
    return list(names) if isinstance(ids, slice) else [names[i] for i in ids]


class _StanceHeight:
    """Height [m] of each foot site at its last ground contact."""

    def __init__(self, num_envs: int, num_feet: int, device: str):
        self.height = torch.zeros((num_envs, num_feet), device=device)
        self._stale = torch.ones(num_envs, dtype=torch.bool, device=device)

    def update(self, foot_z: torch.Tensor, in_contact: torch.Tensor) -> torch.Tensor:
        """Record the height of the feet in contact and return the height of every foot above its record.

        The first update after a reset records every foot, so a new episode never reads the previous one.
        """
        refresh = in_contact | self._stale.unsqueeze(1)
        self.height = torch.where(refresh, foot_z, self.height)
        self._stale[:] = False
        return foot_z - self.height

    def reset(self, env_ids: torch.Tensor | slice | None):
        self._stale[slice(None) if env_ids is None else env_ids] = True


class _FootHeightTerm(ManagerTermBase):
    """Base for the terms that track the foot-site height above the last ground contact."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        sensor_cfg: SceneEntityCfg = cfg.params["sensor_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset_feet = _selected_names(asset.body_names, asset_cfg.body_ids)
        sensor_feet = _selected_names(sensor.body_names, sensor_cfg.body_ids)
        if asset_feet != sensor_feet:
            raise ValueError(
                f"The feet of 'asset_cfg' {asset_feet} and 'sensor_cfg' {sensor_feet} must match in order."
            )
        self._stance = _StanceHeight(env.num_envs, len(asset_feet), env.device)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        self._stance.reset(env_ids)


class feet_clearance(_FootHeightTerm):
    """Penalize moving feet that are not at the target clearance.

    The cost is the sum over the feet of ``|height - target_height| * speed``, where the height [m] is that of
    the foot site above its last ground contact and the speed [m/s] is the horizontal world-frame speed of the
    site. It is zero unless the command is active.
    """

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        offset: tuple[float, float, float],
        command_name: str,
        command_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        height = self._stance.update(foot_pos_w(env, asset_cfg, offset)[:, :, 2], _contact_flags(sensor, sensor_cfg))
        speed = torch.norm(foot_vel_w(env, asset_cfg, offset)[:, :, :2], dim=-1)
        cost = torch.sum(torch.abs(height - target_height) * speed, dim=1)
        return cost * _command_active(env, command_name, command_threshold)


class feet_swing_height(_FootHeightTerm):
    """Penalize, at touchdown, swings whose peak height missed the target.

    Each foot records the peak height [m] of its site above its last ground contact while it is in the air.
    On the step a foot touches down, the cost adds ``(peak / target_height - 1)^2`` for that foot and its
    peak is cleared. The cost is summed over the feet and is zero unless the command is active.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._peak_heights = torch.zeros_like(self._stance.height)

    def reset(self, env_ids: torch.Tensor | slice | None = None):
        super().reset(env_ids)
        self._peak_heights[slice(None) if env_ids is None else env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        target_height: float,
        offset: tuple[float, float, float],
        command_name: str,
        command_threshold: float,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_contact = _contact_flags(sensor, sensor_cfg)
        height = self._stance.update(foot_pos_w(env, asset_cfg, offset)[:, :, 2], in_contact)
        self._peak_heights = torch.where(in_contact, self._peak_heights, torch.maximum(self._peak_heights, height))
        first_contact = sensor.compute_first_contact(dt=env.step_dt).torch[:, sensor_cfg.body_ids].bool()
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact, dim=1)
        num_landings = torch.clamp(torch.sum(first_contact), min=1)
        env.extras.setdefault("log", {})["Metrics/peak_height_mean"] = (
            torch.sum(self._peak_heights * first_contact) / num_landings
        )
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost * _command_active(env, command_name, command_threshold)


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the impact force of touchdowns.

    The cost is the sum over the feet of the net contact force magnitude [N] on the step a foot touches down.
    It is zero unless the command is active.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_magnitude = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids].norm(dim=-1)
    first_contact = sensor.compute_first_contact(dt=env.step_dt).torch[:, sensor_cfg.body_ids].bool()
    landing_impact = force_magnitude * first_contact
    num_landings = torch.clamp(torch.sum(first_contact), min=1)
    env.extras.setdefault("log", {})["Metrics/landing_force_mean"] = torch.sum(landing_impact) / num_landings
    return torch.sum(landing_impact, dim=1) * _command_active(env, command_name, command_threshold)


def feet_orientation_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet that are not flat.

    The cost is the sum over the selected bodies of the norm of the horizontal components of the unit gravity
    vector expressed in the body frame.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
    gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    gravity_b = math_utils.quat_apply_inverse(body_quat, gravity_dir.unsqueeze(1).expand(-1, body_quat.shape[1], -1))
    return torch.sum(torch.norm(gravity_b[:, :, :2], dim=-1), dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, ratio_threshold: float = 4.0) -> torch.Tensor:
    """Penalize feet pushing against a vertical surface.

    The cost is 1 for an environment if any selected foot is in contact and its horizontal world-frame contact
    force exceeds ``ratio_threshold`` times its vertical contact force.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids]
    stumble = torch.norm(forces[:, :, :2], dim=-1) > ratio_threshold * torch.abs(forces[:, :, 2])
    stumble &= _contact_flags(sensor, sensor_cfg)
    any_stumble = torch.any(stumble, dim=1).float()
    env.extras.setdefault("log", {})["Metrics/stumble_rate"] = torch.mean(any_stumble)
    return any_stumble


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 10.0
) -> torch.Tensor:
    """Penalize contacts between the sensor bodies and the bodies in the sensor's contact filter.

    The cost is the number of (sensor body, filtered body) pairs whose contact force exceeds
    ``force_threshold`` [N].
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    hit = sensor.data.force_matrix_w.torch.norm(dim=-1) > force_threshold
    return hit.sum(dim=(1, 2)).float()


def whole_body_angular_momentum(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """World-frame angular momentum [kg m^2/s] of the articulation about its center of mass.

    This is the sum over the bodies of the momentum of the body's center of mass about the system's center of
    mass and the momentum of the body's rotation about its own center of mass. It reads the simulated masses
    and inertias, so it follows mass randomization.
    """
    data = env.scene[asset_cfg.name].data
    mass = data.body_mass.torch.unsqueeze(-1)
    total_mass = mass.sum(dim=1)
    pos = data.body_com_pos_w.torch
    vel = data.body_com_lin_vel_w.torch
    com_pos = (mass * pos).sum(dim=1, keepdim=True) / total_mass.unsqueeze(1)
    com_vel = (mass * vel).sum(dim=1, keepdim=True) / total_mass.unsqueeze(1)
    orbital = mass * torch.cross(pos - com_pos, vel - com_vel, dim=-1)
    # the inertia is taken about the body's center of mass, with the axes of the link frame
    quat = data.body_link_quat_w.torch
    inertia = data.body_inertia.torch.reshape(*quat.shape[:2], 3, 3)
    ang_vel_b = math_utils.quat_apply_inverse(quat, data.body_com_ang_vel_w.torch)
    spin = math_utils.quat_apply(quat, torch.matmul(inertia, ang_vel_b.unsqueeze(-1)).squeeze(-1))
    return (orbital + spin).sum(dim=1)


def angular_momentum_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the squared norm of the whole-body angular momentum about the center of mass."""
    angmom_sq = torch.sum(torch.square(whole_body_angular_momentum(env, asset_cfg)), dim=-1)
    env.extras.setdefault("log", {})["Metrics/angular_momentum_mean"] = torch.mean(torch.sqrt(angmom_sq))
    return angmom_sq


class variable_posture(ManagerTermBase):
    """Reward staying near the default joint pose, with a tolerance that depends on the command.

    The reward is ``exp(-mean((q - q_default)^2 / std^2))`` over the selected joints, with joint positions in
    [rad]. The per-joint ``std`` [rad] is ``std_standing``, ``std_walking`` or ``std_running``, selected by the
    planar speed command [m/s] plus the yaw-rate command [rad/s] against ``walking_threshold`` and
    ``running_threshold``.

    When ``disturbance_std_scale`` is above 1, ``std`` grows by up to that factor while the robot is
    disturbed. The disturbance is the larger of the base linear-velocity tracking error [m/s] and the base
    tilt (norm of the horizontal projected gravity), each mapped linearly from its threshold to its full value.

    When ``pose_weight_velocity_threshold`` is set, the reward is multiplied by ``pose_weight_slow`` below
    that planar speed command and by ``pose_weight_fast`` above it.
    """

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
