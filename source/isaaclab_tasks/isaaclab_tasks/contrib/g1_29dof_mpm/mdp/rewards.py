# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gait rewards driven by the foot contact against the granular bed.

Contact comes from the reaction wrench that the coupler feeds back from the MPM entry to the
proxy feet, so these are the granular counterparts of the hybrid soft-contact rewards of
``g1_29dof_soft``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names_values

from ..env_cfg.scene_cfg import SAND_BED_SIZE

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..g1_mpm_env import G1MPMEnv


def base_height_above_testbed(env: G1MPMEnv, target_height: float) -> torch.Tensor:
    """Penalize root clearance from the tray's nominal sand surface along the tray normal."""
    tray = env.scene["testbed"]
    tray_id = env.testbed_tray_body_id
    root_in_tray = quat_apply_inverse(
        tray.data.body_quat_w.torch[:, tray_id],
        env.scene["robot"].data.root_pos_w.torch - tray.data.body_pos_w.torch[:, tray_id],
    )
    return (root_in_tray[:, 2] - SAND_BED_SIZE[2] - target_height).square()


def feet_air_time_positive_biped(
    env: G1MPMEnv,
    command_name: str,
    threshold: float,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Reward long single-stance phases while a motion command is active.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        threshold: Upper bound on the rewarded stance or swing duration [s].
        velocity_threshold: Command magnitude above which the reward applies [m/s or rad/s].

    Returns:
        The reward, shape ``(num_envs,)``.
    """
    air_time = env.foot_air_time
    contact_time = env.foot_contact_time
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no gait is asked of a standing robot
    command = env.command_manager.get_command(command_name)
    commanded = (torch.norm(command[:, :2], dim=1) > velocity_threshold) | (
        torch.abs(command[:, 2]) > velocity_threshold
    )
    return reward * commanded


def no_fly(
    env: G1MPMEnv,
    command_name: str,
    velocity_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize flight phases below a command speed at which running is acceptable.

    Args:
        env: Environment instance.
        command_name: Name of the velocity command term.
        velocity_threshold: Command speed above which flight is no longer penalized [m/s].

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    airborne = torch.sum(env.foot_contact, dim=-1) < 0.5
    command_speed = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    return airborne.float() * (command_speed < velocity_threshold).float()


def feet_pitch_contact(
    env: G1MPMEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize foot pitch at touchdown only, to encourage flat landings.

    Args:
        env: Environment instance.
        asset_cfg: Configuration selecting the foot bodies, in the environment's foot order.

    Returns:
        The penalty, shape ``(num_envs,)``.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    _, pitch, _ = euler_xyz_from_quat(body_quat.reshape(-1, 4))
    pitch = pitch.reshape(body_quat.shape[0], body_quat.shape[1])
    return torch.sum(torch.square(pitch) * env.foot_first_contact, dim=-1)


def reward_soft_landing(
    env: G1MPMEnv,
    command_name: str | None = "base_velocity",
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize foot force magnitude at touchdown using the combined terrain contact forces.

    The cost is in newtons, not impulse. Persistent contacts incur no cost.
    A zero motion command disables the penalty, matching the soft-contact task.
    """
    first_contact = env.foot_first_contact
    landing_force = torch.linalg.vector_norm(env.foot_contact_force, dim=-1) * first_contact
    cost = landing_force.sum(dim=-1)
    env.extras["log"]["Metrics/landing_force_mean"] = landing_force.sum() / first_contact.sum().clamp(min=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        active = torch.linalg.vector_norm(command[:, :2], dim=-1) + command[:, 2].abs() > command_threshold
        cost = cost * active
    return cost


class foot_touch_down_angle_penalty(ManagerTermBase):
    """Penalize foot pitch at first contact relative to the flat initial sand/approach surface."""

    def __init__(self, cfg: RewardTermCfg, env: G1MPMEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        tracked_ids, _ = self._asset.find_bodies(env.cfg.foot_body_expr, preserve_order=True)
        selected_ids = cfg.params["asset_cfg"].body_ids
        if isinstance(selected_ids, slice):
            selected_ids = list(range(self._asset.num_bodies))[selected_ids]
        self._contact_ids = torch.tensor([tracked_ids.index(i) for i in selected_ids], device=env.device)
        self._forward = torch.tensor([1.0, 0.0, 0.0], device=env.device)
        if cfg.params["angle_tolerance"] < 0.0:
            raise ValueError("angle_tolerance must be non-negative.")

    def __call__(self, env: G1MPMEnv, asset_cfg: SceneEntityCfg, angle_tolerance: float) -> torch.Tensor:
        foot_quat = self._asset.data.body_quat_w.torch[:, asset_cfg.body_ids]
        forward = quat_apply(foot_quat, self._forward.expand_as(foot_quat[..., :3]))
        angle = torch.atan2(forward[..., 2], torch.linalg.vector_norm(forward[..., :2], dim=-1))
        touchdown = env.foot_first_contact[:, self._contact_ids] > 0.0
        penalty = (angle.abs() - angle_tolerance).clamp_min(0.0).square()
        return torch.where(torch.isfinite(angle) & touchdown, penalty, 0.0).sum(dim=-1)


def reward_foot_lateral_symmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ref_dist: float,
) -> torch.Tensor:
    """L2 penalty for lateral foot separation deviating from reference."""
    asset: RigidObject = env.scene[asset_cfg.name]
    body_pos_in_base = torch.zeros_like(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :3])
    for i in range(len(asset_cfg.body_ids)):
        body_pos_in_base[:, i, :] = quat_apply_inverse(
            asset.data.root_quat_w.torch,
            asset.data.body_pos_w.torch[:, asset_cfg.body_ids, :3][:, i, :] - asset.data.root_pos_w.torch,
        )
    foot_lat_dist = torch.abs(body_pos_in_base[:, 0, 1] - body_pos_in_base[:, 1, 1])
    return torch.square(foot_lat_dist - ref_dist)


def track_heading_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    heading_error = math_utils.quat_error_magnitude(
        math_utils.yaw_quat(asset.data.root_quat_w.torch), env.command_manager.get_command(command_name)[:, 3:]
    )
    return torch.exp(-torch.square(heading_error) / std**2)


class variable_posture_l1(ManagerTermBase):
    """
    compute gaussian kernel reward to regularize robot's whole body posture for each gait.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        asset = env.scene[cfg.params["asset_cfg"].name]
        self.default_joint_pos = asset.data.default_joint_pos.torch

        _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

        _, _, weight_standing = resolve_matching_names_values(
            data=cfg.params["weight_standing"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_standing = torch.tensor(weight_standing, device=env.device, dtype=torch.float32)

        _, _, weight_walking = resolve_matching_names_values(
            data=cfg.params["weight_walking"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_walking = torch.tensor(weight_walking, device=env.device, dtype=torch.float32)

        _, _, weight_running = resolve_matching_names_values(
            data=cfg.params["weight_running"],  # type: ignore
            list_of_strings=joint_names,
        )
        self.weight_running = torch.tensor(weight_running, device=env.device, dtype=torch.float32)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        weight_standing: dict,
        weight_walking: dict,
        weight_running: dict,
        walking_threshold: float = 0.5,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        asset = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)

        linear_speed = torch.norm(command[:, :2], dim=-1)
        angular_speed = torch.abs(command[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < walking_threshold).float()
        walking_mask = ((total_speed >= walking_threshold) & (total_speed < running_threshold)).float()
        running_mask = (total_speed >= running_threshold).float()

        weight = (
            self.weight_standing * standing_mask.unsqueeze(1)
            + self.weight_walking * walking_mask.unsqueeze(1)
            + self.weight_running * running_mask.unsqueeze(1)
        )

        current_joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
        error = torch.abs(current_joint_pos - desired_joint_pos)
        return (weight * error).sum(dim=1)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque.torch * asset.data.joint_vel.torch), dim=-1)
    return reward


def action_rate_l2(env: ManagerBasedRLEnv, joint_idx: list[int]) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(
        torch.square(env.action_manager.action[:, joint_idx] - env.action_manager.prev_action[:, joint_idx]), dim=1
    )


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    tanh_mult: float,
    asset_cfg: SceneEntityCfg,
    standing_position_foot_z: float = 0.039,
    height_sensor_cfg: SceneEntityCfg | None = None,
    ground_height_offset: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Reward foot clearance relative to the mean scanned ground height.

    Args:
        env: Environment instance.
        target_height: Desired clearance above the standing foot height [m].
        std: Exponential denominator [m²], preserving the existing clearance reward.
        tanh_mult: Horizontal foot-speed multiplier [s/m].
        asset_cfg: Robot and foot bodies to measure.
        standing_position_foot_z: Standing foot body-origin height above the ground [m].
        height_sensor_cfg: Ray caster providing world-space ground hits. Without a sensor,
            the ground reference is world Z = 0, matching the existing reward.
        ground_height_offset: Offset added to the scanned ground height [m], as a scalar or a tensor
            of shape ``(num_envs, 1)``. For example, the sand depth above a scanned supporting floor.

    Returns:
        Velocity-weighted clearance reward, shape ``(num_envs,)``.
    """
    asset = env.scene[asset_cfg.name]
    ground_height = 0.0
    if height_sensor_cfg is not None:
        sensor = env.scene[height_sensor_cfg.name]
        ground_height = sensor.data.ray_hits_w.torch[..., 2].mean(dim=1, keepdim=True) + ground_height_offset
    height_error = torch.square(
        asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
        - ground_height
        - (target_height + standing_position_foot_z)
    )
    velocity_weight = torch.tanh(
        tanh_mult * torch.linalg.vector_norm(asset.data.body_lin_vel_w.torch[:, asset_cfg.body_ids, :2], dim=-1)
    )
    return torch.exp(-torch.sum(height_error * velocity_weight, dim=1) / std)
