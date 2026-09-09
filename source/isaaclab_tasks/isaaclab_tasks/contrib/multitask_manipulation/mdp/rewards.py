# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Selection-aware rewards for heterogeneous manipulation tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, RewardTermCfg
from isaaclab.utils import math as math_utils

from ..selection_utils import SceneEntitySelectionCfg
from .utils import _offset_body_pose

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def _action_slice(env: ManagerBasedRLEnv, term_names: tuple[str, ...]) -> slice:
    """Return the cached contiguous global action slice occupied by action terms."""
    cache = env.__dict__.setdefault("_multitask_action_slices", {})
    if term_names not in cache:
        starts = {}
        start = 0
        for name, dim in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
            starts[name] = (start, start + dim)
            start += dim
        cache[term_names] = slice(starts[term_names[0]][0], starts[term_names[-1]][1])
    return cache[term_names]


def _lift_goal_error(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    object_cfg: SceneEntitySelectionCfg,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return lift global IDs and object-to-goal position error [m]."""
    robot: Articulation = env.scene[robot_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    robot_rows = robot_cfg.instance_ids[object_cfg.env_ids]
    command = env.command_manager.get_command(command_name)[object_cfg.env_ids]
    goal_pos_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_pos_w.torch[robot_rows], robot.data.root_quat_w.torch[robot_rows], command[:, :3]
    )
    error = torch.linalg.norm(object_asset.data.root_pos_w.torch - goal_pos_w, dim=-1)
    return object_cfg.env_ids, error


def lift_ee_object_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    object_cfg: SceneEntitySelectionCfg,
    std: float,
) -> torch.Tensor:
    """Reward the lift OpenArm for bringing its TCP close to the object."""
    env_ids, tcp_pos_w, _ = _offset_body_pose(env, robot_cfg, (0.0, 0.0, 0.0))
    object_asset: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(
        object_asset.data.root_pos_w.torch[object_cfg.instance_ids[env_ids]] - tcp_pos_w, dim=-1
    )
    return robot_cfg.scatter_to_envs(1.0 - torch.tanh(distance / std))


def lift_object_height(
    env: ManagerBasedRLEnv, object_cfg: SceneEntitySelectionCfg, minimum_height: float
) -> torch.Tensor:
    """Reward lift environments whose object rises above an environment-frame height [m]."""
    object_asset: RigidObject = env.scene[object_cfg.name]
    height = object_asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[object_cfg.env_ids, 2]
    return object_cfg.scatter_to_envs((height > minimum_height).float())


def lift_goal_tracking(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    object_cfg: SceneEntitySelectionCfg,
    command_name: str,
    std: float,
    minimum_height: float,
) -> torch.Tensor:
    """Reward lift-object target tracking after the object clears the table."""
    env_ids, error = _lift_goal_error(env, robot_cfg, object_cfg, command_name)
    object_asset: RigidObject = env.scene[object_cfg.name]
    height = object_asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[env_ids, 2]
    reward = (height > minimum_height) * (1.0 - torch.tanh(error / std))
    return object_cfg.scatter_to_envs(reward)


class _LiftSuccessTerm(ManagerTermBase):
    """Maintain and log sticky success state for lift reward terms."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Log and clear success for reset lift environments."""
        object_cfg: SceneEntitySelectionCfg = self.cfg.params["object_cfg"]
        selected = env_ids[object_cfg.instance_ids[env_ids] >= 0]
        if selected.numel() == 0:
            return
        log = self._env.extras.setdefault("log", {})
        log["Metrics/lift_success_rate"] = self.succeeded[selected].float().mean().item()
        self.succeeded[selected] = False


class LiftGoalTracking(_LiftSuccessTerm):
    """Reward OpenArm goal tracking and log sticky per-episode success."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntitySelectionCfg,
        object_cfg: SceneEntitySelectionCfg,
        command_name: str,
        std: float,
        minimum_height: float,
        success_threshold: float,
    ) -> torch.Tensor:
        """Return coarse goal tracking and update sticky success state."""
        env_ids, error = _lift_goal_error(env, robot_cfg, object_cfg, command_name)
        object_asset: RigidObject = env.scene[object_cfg.name]
        height = object_asset.data.root_pos_w.torch[:, 2] - env.scene.env_origins[env_ids, 2]
        is_lifted = height > minimum_height
        self.succeeded[env_ids] |= is_lifted & (error < success_threshold)
        reward = is_lifted * (1.0 - torch.tanh(error / std))
        return object_cfg.scatter_to_envs(reward)


class LiftSuccess(_LiftSuccessTerm):
    """Reward lift-goal completion and log sticky episode success."""

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntitySelectionCfg,
        object_cfg: SceneEntitySelectionCfg,
        command_name: str,
        threshold: float,
    ) -> torch.Tensor:
        """Reward lift environments whose object reaches the sampled goal."""
        env_ids, error = _lift_goal_error(env, robot_cfg, object_cfg, command_name)
        succeeded = error < threshold
        self.succeeded[env_ids] |= succeeded
        return object_cfg.scatter_to_envs(succeeded.float())


def _cabinet_frames(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> tuple[torch.Tensor, ...]:
    """Return aligned cabinet-task TCP, handle, and fingertip poses."""
    env_ids, ee_pos, ee_quat = _offset_body_pose(env, robot_cfg, (0.0, 0.0, 0.1034))
    _, handle_pos, handle_quat = _offset_body_pose(env, cabinet_cfg, (0.305, 0.0, 0.01), (0.5, -0.5, -0.5, 0.5))
    _, left_pos, _ = _offset_body_pose(env, robot_cfg, (0.0, 0.0, 0.046), body_index=1)
    _, right_pos, _ = _offset_body_pose(env, robot_cfg, (0.0, 0.0, 0.046), body_index=2)
    cabinet_rows = cabinet_cfg.instance_ids[env_ids]
    return env_ids, ee_pos, ee_quat, handle_pos[cabinet_rows], handle_quat[cabinet_rows], left_pos, right_pos


def cabinet_approach_ee_handle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward cabinet environments for approaching the drawer handle."""
    env_ids, ee_pos, _, handle_pos, _, _, _ = _cabinet_frames(env, robot_cfg, cabinet_cfg)
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1)
    reward = torch.square(1.0 / (1.0 + distance**2))
    reward = torch.where(distance <= threshold, 2.0 * reward, reward)
    return robot_cfg.scatter_to_envs(torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))


def cabinet_align_ee_handle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Reward cabinet environments for aligning the TCP with the drawer handle."""
    env_ids, _, ee_quat, _, handle_quat, _, _ = _cabinet_frames(env, robot_cfg, cabinet_cfg)
    ee_rot = math_utils.matrix_from_quat(ee_quat)
    handle_rot = math_utils.matrix_from_quat(handle_quat)
    align_z = torch.sum(ee_rot[..., 2] * -handle_rot[..., 0], dim=-1)
    align_x = torch.sum(ee_rot[..., 0] * -handle_rot[..., 1], dim=-1)
    reward = 0.5 * (torch.sign(align_z) * align_z**2 + torch.sign(align_x) * align_x**2)
    return robot_cfg.scatter_to_envs(torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))


def _cabinet_grasp_alignment(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return cabinet IDs, grasp mask, handle position, and fingertip positions."""
    env_ids, _, _, handle_pos, _, left_pos, right_pos = _cabinet_frames(env, robot_cfg, cabinet_cfg)
    graspable = (right_pos[:, 2] < handle_pos[:, 2]) & (left_pos[:, 2] > handle_pos[:, 2])
    return env_ids, graspable, handle_pos, torch.stack((left_pos, right_pos), dim=1)


def cabinet_align_grasp(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Reward cabinet environments when the fingers straddle the drawer handle."""
    env_ids, graspable, _, _ = _cabinet_grasp_alignment(env, robot_cfg, cabinet_cfg)
    return robot_cfg.scatter_to_envs(graspable.float())


def cabinet_approach_gripper(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
    offset: float,
) -> torch.Tensor:
    """Reward cabinet environments for placing both fingers around the handle."""
    env_ids, graspable, handle_pos, fingertips = _cabinet_grasp_alignment(env, robot_cfg, cabinet_cfg)
    distances = torch.abs(fingertips[..., 2] - handle_pos[:, None, 2])
    reward = graspable * torch.sum(offset - distances, dim=-1)
    return robot_cfg.scatter_to_envs(torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))


def cabinet_grasp_handle(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
    threshold: float,
    open_joint_pos: float,
) -> torch.Tensor:
    """Reward closing the cabinet Franka fingers near the drawer handle."""
    env_ids, ee_pos, _, handle_pos, _, _, _ = _cabinet_frames(env, robot_cfg, cabinet_cfg)
    robot: Articulation = env.scene[robot_cfg.name]
    distance = torch.linalg.norm(handle_pos - ee_pos, dim=-1)
    reward = (distance <= threshold) * torch.sum(
        open_joint_pos - robot.data.joint_pos.torch[:, robot_cfg.joint_ids], dim=-1
    )
    return robot_cfg.scatter_to_envs(torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0))


class CabinetOpenDrawerBonus(ManagerTermBase):
    """Reward drawer opening and log cabinet-task success statistics."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        self.succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.best_drawer_pos = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: torch.Tensor) -> None:
        """Log and clear episode statistics for reset cabinet environments."""
        cabinet_cfg: SceneEntitySelectionCfg = self.cfg.params["cabinet_cfg"]
        selected = env_ids[cabinet_cfg.instance_ids[env_ids] >= 0]
        if selected.numel() == 0:
            return
        log = self._env.extras.setdefault("log", {})
        log["Metrics/cabinet_success_rate"] = self.succeeded[selected].float().mean().item()
        log["Metrics/cabinet_drawer_pos"] = self.best_drawer_pos[selected].mean().item()
        self.succeeded[selected] = False
        self.best_drawer_pos[selected] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        robot_cfg: SceneEntitySelectionCfg,
        cabinet_cfg: SceneEntitySelectionCfg,
        success_threshold: float,
    ) -> torch.Tensor:
        """Compute drawer displacement with a grasp-alignment multiplier."""
        cabinet: Articulation = env.scene[cabinet_cfg.name]
        drawer_pos = cabinet.data.joint_pos.torch[:, cabinet_cfg.joint_ids[0]]
        drawer_limits = cabinet.data.soft_joint_pos_limits.torch[:, cabinet_cfg.joint_ids[0]]
        drawer_pos = torch.nan_to_num(drawer_pos, nan=0.0, posinf=0.0, neginf=0.0)
        drawer_pos = torch.maximum(torch.minimum(drawer_pos, drawer_limits[:, 1]), drawer_limits[:, 0])
        env_ids, graspable, _, _ = _cabinet_grasp_alignment(env, robot_cfg, cabinet_cfg)
        self.succeeded[env_ids] |= drawer_pos > success_threshold
        self.best_drawer_pos[env_ids] = torch.maximum(self.best_drawer_pos[env_ids], drawer_pos)
        return robot_cfg.scatter_to_envs((graspable.float() + 1.0) * drawer_pos)


def cabinet_multi_stage_open(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntitySelectionCfg,
    cabinet_cfg: SceneEntitySelectionCfg,
) -> torch.Tensor:
    """Reward easy, medium, and hard drawer-opening milestones."""
    cabinet: Articulation = env.scene[cabinet_cfg.name]
    drawer_pos = cabinet.data.joint_pos.torch[:, cabinet_cfg.joint_ids[0]]
    _, graspable, _, _ = _cabinet_grasp_alignment(env, robot_cfg, cabinet_cfg)
    reward = (drawer_pos > 0.01) * 0.5
    reward += (drawer_pos > 0.2) * graspable
    reward += (drawer_pos > 0.3) * graspable
    return cabinet_cfg.scatter_to_envs(reward)


def reach_position_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntitySelectionCfg, command_name: str) -> torch.Tensor:
    """Return UR10 end-effector position error [m] in reach environments."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)[robot_cfg.env_ids]
    goal_pos_w, _ = math_utils.combine_frame_transforms(
        robot.data.root_pos_w.torch, robot.data.root_quat_w.torch, command[:, :3]
    )
    error = torch.linalg.norm(robot.data.body_pos_w.torch[:, robot_cfg.body_ids[0]] - goal_pos_w, dim=-1)
    return robot_cfg.scatter_to_envs(error)


def reach_orientation_error(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntitySelectionCfg, command_name: str
) -> torch.Tensor:
    """Return UR10 end-effector orientation error [rad] in reach environments."""
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)[robot_cfg.env_ids]
    goal_quat_w = math_utils.quat_mul(robot.data.root_quat_w.torch, command[:, 3:7])
    error = math_utils.quat_error_magnitude(robot.data.body_quat_w.torch[:, robot_cfg.body_ids[0]], goal_quat_w)
    return robot_cfg.scatter_to_envs(error)


def selected_action_l2(
    env: ManagerBasedRLEnv,
    task_asset_cfg: SceneEntitySelectionCfg,
    action_term_names: tuple[str, ...],
) -> torch.Tensor:
    """Penalize one task action head in environments containing that task asset."""
    selected_slice = _action_slice(env, action_term_names)
    penalty = torch.sum(torch.square(env.action_manager.action[:, selected_slice]), dim=-1)
    return penalty * (task_asset_cfg.instance_ids >= 0)


def selected_action_rate_l2(
    env: ManagerBasedRLEnv,
    task_asset_cfg: SceneEntitySelectionCfg,
    action_term_names: tuple[str, ...],
) -> torch.Tensor:
    """Penalize changes in one task action head where that task asset exists."""
    selected_slice = _action_slice(env, action_term_names)
    delta = env.action_manager.action[:, selected_slice] - env.action_manager.prev_action[:, selected_slice]
    return torch.sum(torch.square(delta), dim=-1) * (task_asset_cfg.instance_ids >= 0)


def selected_joint_vel_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntitySelectionCfg,
    max_velocity: float | None = None,
) -> torch.Tensor:
    """Penalize articulation joint velocities in environments containing it.

    Args:
        env: Manager-based RL environment.
        asset_cfg: Selection-aware articulation and joints.
        max_velocity: Optional absolute velocity bound [m/s or rad/s, depending on joint type].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = torch.nan_to_num(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
    if max_velocity is not None:
        joint_vel = joint_vel.clamp(-max_velocity, max_velocity)
    values = torch.sum(torch.square(joint_vel), dim=-1)
    return asset_cfg.scatter_to_envs(values)
