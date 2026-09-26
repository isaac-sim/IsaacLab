# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the velocity-tracking locomotion environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.sensors import ContactSensor


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt).torch[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


class feet_air_time_variance(ManagerTermBase):
    """Penalize variance across feet of completed swing and stance durations [s²].

    Contact phases are measured at the control rate: a sensor's last completed timer
    can be overwritten by short contacts within a control step. Initial partial
    phases are ignored. Durations are clipped at ``max_time`` [s].
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        sensor_cfg = cfg.params["sensor_cfg"]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._elapsed = torch.zeros_like(sensor.data.current_air_time.torch[:, sensor_cfg.body_ids])
        self._air = torch.zeros_like(self._elapsed)
        self._stance = torch.zeros_like(self._elapsed)
        self._contact = torch.zeros_like(self._elapsed, dtype=torch.bool)
        self._known = torch.zeros_like(self._contact)
        self._initialized = torch.zeros(env.num_envs, 1, device=env.device, dtype=torch.bool)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        for buffer in (self._elapsed, self._air, self._stance, self._contact, self._known, self._initialized):
            buffer[env_ids] = 0

    def __call__(
        self, env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, max_time: float = 0.5
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        contact = sensor.data.current_air_time.torch[:, sensor_cfg.body_ids] <= 0.0
        changed = (contact != self._contact) & self._initialized
        complete = changed & self._known
        self._air.copy_(torch.where(complete & contact, self._elapsed, self._air))
        self._stance.copy_(torch.where(complete & ~contact, self._elapsed, self._stance))
        self._elapsed.copy_(torch.where(changed, 0.0, self._elapsed) + env.step_dt)
        self._known |= changed
        self._contact.copy_(contact)
        self._initialized.fill_(True)
        penalty = torch.zeros(env.num_envs, device=env.device)
        for duration in (self._air, self._stance):
            penalty += duration.clamp(max=max_time).var(dim=1, correction=0) * (duration > 0.0).all(dim=1)
        penalty *= torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
        return penalty


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contacts = (
        contact_sensor.data.net_normal_forces_w_history.torch[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > 1.0
    )
    body_vel = asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity-aligned robot frame with an exp kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w.torch), asset.data.root_lin_vel_w.torch[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(
        env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w.torch[:, 2]
    )
    return torch.exp(-ang_vel_error / std**2)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.linalg.norm(command[:, :2], dim=1) < command_threshold)


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize squared displacement of selected joints from their default angles [rad²]."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
            - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def _pelvis_clearance(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return root height above the median terrain ray hit [m]."""
    hits = env.scene[sensor_cfg.name].data.ray_hits_w.torch[..., 2]
    ground = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0).median(dim=1).values
    return env.scene[asset_cfg.name].data.root_pos_w.torch[:, 2] - ground


def pelvis_height_deficit_l2(
    env: ManagerBasedRLEnv, target_height: float, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize squared scan-relative height shortfall below ``target_height`` [m].

    Height is relative to the median of the entire terrain scan, not the supporting
    foot surface. Narrow raised treads can be outvoted by lower surrounding terrain,
    so this posture heuristic does not detect every crouch. Heights above the target
    incur no penalty.
    """
    return torch.clamp(target_height - _pelvis_clearance(env, asset_cfg, sensor_cfg), min=0.0).square()


def feet_flight(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return one when both selected feet are airborne, otherwise zero.

    Single-foot swings and double support incur no penalty.
    """
    air_time = env.scene.sensors[sensor_cfg.name].data.current_air_time.torch[:, sensor_cfg.body_ids]
    return torch.all(air_time > 0.0, dim=-1).float()
