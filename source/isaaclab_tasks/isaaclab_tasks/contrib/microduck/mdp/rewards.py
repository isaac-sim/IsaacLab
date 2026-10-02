# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck flat-walking rewards."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.string import resolve_matching_names_values

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def _required_entity_cfg(cfg: RewardTermCfg, key: str, term_name: str) -> SceneEntityCfg:
    """Fetch a scene entity configuration that a stateful term needs at construction time."""
    entity_cfg = cfg.params.get(key)
    if not isinstance(entity_cfg, SceneEntityCfg):
        raise ValueError(f"{term_name} requires a SceneEntityCfg under {key!r}, received {entity_cfg!r}.")
    return entity_cfg


def _required_joint_ids(entity_cfg: SceneEntityCfg, key: str, term_name: str) -> list[int]:
    """Fetch an explicit joint selection from a resolved scene entity configuration."""
    if isinstance(entity_cfg.joint_ids, slice):
        raise ValueError(f"{term_name} requires {key!r} to select joint names with preserve_order=True.")
    return entity_cfg.joint_ids


def _command_magnitude(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Upstream's scalar command magnitude ``|v_xy| + |w_z|``."""
    command = env.command_manager.get_command(command_name)
    return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


def _feet_height_above_ground(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Height of the selected foot bodies above the ground [m], shape (num_envs, num_feet)."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_z = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    return foot_pos_z - env.scene.env_origins[:, 2].unsqueeze(1)


def track_linear_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track linear velocity, including vertical drift, with tolerance ``std`` [m/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_lin_vel_b.torch
    xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
    z_error = torch.square(actual[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track yaw rate, including roll/pitch drift, with tolerance ``std`` [rad/s]."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    actual = asset.data.root_link_ang_vel_b.torch
    z_error = torch.square(command[:, 2] - actual[:, 2])
    xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


def upright(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping a body's frame aligned with gravity, using a Gaussian kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None:
        projected_gravity_b = asset.data.projected_gravity_b.torch
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(f"upright requires one body, received {asset_cfg.body_names}.")
        body_quat_w = asset.data.body_link_quat_w.torch[:, asset_cfg.body_ids]
        gravity_dir_w = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1).unsqueeze(1)
        projected_gravity_b = math_utils.quat_apply_inverse(body_quat_w, gravity_dir_w).squeeze(1)
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
    xy_squared = torch.nan_to_num(xy_squared, nan=1.0, posinf=1.0, neginf=1.0)
    return torch.exp(-xy_squared / std**2)


class pose_mode_switch(ManagerTermBase):
    """Reward holding the stand pose, with a per-joint tolerance that widens once the robot moves."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Resolve the three per-joint standard-deviation vectors."""
        super().__init__(cfg, env)
        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        asset: Articulation = env.scene[asset_cfg.name]
        joint_names = [asset.joint_names[index] for index in joint_ids]
        self._std = {}
        for key in ("std_standing", "std_walking", "std_running"):
            if key not in cfg.params:
                raise ValueError(f"{self.__name__} requires {key!r} to map joint names to standard deviations.")
            _, _, values = resolve_matching_names_values(cfg.params[key], joint_names)
            self._std[key] = torch.tensor(values, dtype=torch.float32, device=env.device).unsqueeze(0)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float,
        running_threshold: float,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the posture reward."""
        asset: Articulation = env.scene[asset_cfg.name]
        total_speed = _command_magnitude(env, command_name)
        standing_mask = (total_speed < walking_threshold).float().unsqueeze(1)
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float().unsqueeze(1)
        running_mask = (total_speed >= running_threshold).float().unsqueeze(1)
        std = (
            self._std["std_standing"] * standing_mask
            + self._std["std_walking"] * walking_mask
            + self._std["std_running"] * running_mask
        )
        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        default_joint_pos = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        error_squared = torch.square(joint_pos - default_joint_pos)
        return torch.exp(-torch.mean(error_squared / std**2, dim=1))


def head_pose_tracking(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track head joint offsets with tolerance ``std`` [rad], averaged over the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    error = measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids] - command
    return torch.exp(-((error / std) ** 2)).mean(dim=-1)


class head_pose_bias_penalty(ManagerTermBase):
    """Penalize a head-pose error that persists, as opposed to one that averages out."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-environment moving average."""
        super().__init__(cfg, env)
        asset_cfg = _required_entity_cfg(cfg, "asset_cfg", self.__name__)
        joint_ids = _required_joint_ids(asset_cfg, "asset_cfg", self.__name__)
        self._error_ema = torch.zeros(env.num_envs, len(joint_ids), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the average for the environments that restarted."""
        if env_ids is None:
            self._error_ema[:] = 0.0
        else:
            self._error_ema[env_ids] = 0.0

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, tau_s: float, asset_cfg: SceneEntityCfg
    ) -> torch.Tensor:
        """Return the negative absolute error averaged with time constant ``tau_s`` [s]."""
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        measured = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        error = measured - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids] - command
        alpha = min(1.0, float(env.step_dt) / max(tau_s, 1e-06))
        self._error_ema = (1.0 - alpha) * self._error_ema + alpha * error
        penalty = -self._error_ema.abs().mean(dim=-1)
        return penalty


def feet_air_time_windowed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    threshold_min: float,
    threshold_max: float,
    command_threshold: float,
) -> torch.Tensor:
    """Reward air time strictly between ``threshold_min`` and ``threshold_max`` [s] while moving."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
    reward = torch.sum(in_range.float(), dim=1)
    reward = reward * (_command_magnitude(env, command_name) > command_threshold).float()
    return reward


def foot_clearance(
    env: ManagerBasedRLEnv, target_height: float, command_name: str, asset_cfg: SceneEntityCfg, command_threshold: float
) -> torch.Tensor:
    """Penalize moving feet that deviate from ``target_height`` [m] above the ground."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_height = _feet_height_above_ground(env, asset_cfg)
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.abs(foot_height - target_height) * torch.norm(foot_vel_xy, dim=-1), dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


class foot_swing_height(ManagerTermBase):
    """Penalize, on touchdown, how far the peak of the swing was from a target height."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Allocate the per-foot peak heights."""
        super().__init__(cfg, env)
        sensor_cfg = _required_entity_cfg(cfg, "sensor_cfg", self.__name__)
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        num_feet = len(contact_sensor.data.current_contact_time.torch[0, sensor_cfg.body_ids])
        self._peak_heights = torch.zeros(env.num_envs, num_feet, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Forget the tracked peaks for the environments that restarted."""
        if env_ids is None:
            self._peak_heights[:] = 0.0
        else:
            self._peak_heights[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
    ) -> torch.Tensor:
        """Penalize touchdown error relative to ``target_height`` [m]."""
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        in_air = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] == 0.0
        foot_height = _feet_height_above_ground(env, asset_cfg)
        self._peak_heights = torch.where(in_air, torch.maximum(self._peak_heights, foot_height), self._peak_heights)
        first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids] > 0.5
        error = self._peak_heights / target_height - 1.0
        cost = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        cost = cost * (_command_magnitude(env, command_name) > command_threshold).float()
        self._peak_heights = torch.where(first_contact, torch.zeros_like(self._peak_heights), self._peak_heights)
        return cost


def foot_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize the squared planar speed of feet that are on the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids] > 0.0).float()
    foot_vel_xy = asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.square(torch.norm(foot_vel_xy, dim=-1)) * in_contact, dim=1)
    return cost * (_command_magnitude(env, command_name) > command_threshold).float()


def body_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the roll and pitch rate of selected bodies, in the world frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = asset.data.body_link_ang_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(torch.square(ang_vel_xy), dim=(1, 2))


def angular_momentum_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the squared magnitude of the articulation's angular momentum about its own centre of mass."""
    asset: Articulation = env.scene[asset_cfg.name]
    mass = asset.data.body_mass.torch[:, asset_cfg.body_ids]
    pos_w = asset.data.body_com_pos_w.torch[:, asset_cfg.body_ids]
    lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, asset_cfg.body_ids]
    ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, asset_cfg.body_ids]
    quat_w = asset.data.body_com_quat_w.torch[:, asset_cfg.body_ids]
    inertia_b = asset.data.body_inertia.torch[:, asset_cfg.body_ids].view(*mass.shape, 3, 3)
    total_mass = mass.sum(dim=1, keepdim=True).unsqueeze(-1)
    com_pos_w = (mass.unsqueeze(-1) * pos_w).sum(dim=1, keepdim=True) / total_mass
    com_lin_vel_w = (mass.unsqueeze(-1) * lin_vel_w).sum(dim=1, keepdim=True) / total_mass
    rot = math_utils.matrix_from_quat(quat_w)
    inertia_w = torch.matmul(rot, torch.matmul(inertia_b, rot.transpose(-2, -1)))
    spin = torch.matmul(inertia_w, ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbit = mass.unsqueeze(-1) * torch.cross(pos_w - com_pos_w, lin_vel_w - com_lin_vel_w, dim=-1)
    momentum = (spin + orbit).sum(dim=1)
    return torch.sum(torch.square(momentum), dim=-1)


def self_collision_cost(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, force_threshold: float = 1.0
) -> torch.Tensor:
    """Penalize parts of the robot that are touching the robot itself."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_matrix_w = contact_sensor.data.force_matrix_w
    if force_matrix_w is None:
        raise RuntimeError(f"{sensor_cfg.name} needs a filtered contact sensor to measure self-contacts.")
    forces = force_matrix_w.torch[:, sensor_cfg.body_ids]
    in_contact = torch.linalg.norm(forces, dim=-1) > force_threshold
    touching = in_contact.any(dim=-1)
    return touching.sum(dim=-1).float()
