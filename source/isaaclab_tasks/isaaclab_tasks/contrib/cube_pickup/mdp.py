# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State observations and the sparse grasped-cube goal condition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_mul

from ...core.lift.mdp import object_reached_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube center positions [m] relative to environment origins in world axes, shape (num_envs, 3)."""
    return env.scene["object"].data.root_pos_w.torch - env.scene.env_origins


def hand_pose(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand")
) -> torch.Tensor:
    """Grasp-center poses with shape (num_envs, 7).

    Positions [m] are relative to environment origins in world axes; orientations are
    world-frame quaternions in xyzw order.
    """
    orientation = env.scene[asset_cfg.name].data.body_link_quat_w.torch[:, asset_cfg.body_ids[0]]
    if env._tcp_offset_rot is not None:
        orientation = quat_mul(orientation, env._tcp_offset_rot)
    position = _hand_position(env, asset_cfg)
    return torch.cat((position - env.scene.env_origins, orientation), dim=-1)


def object_to_hand(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand")
) -> torch.Tensor:
    """Cube-minus-grasp-center displacements [m] in world axes, shape (num_envs, 3)."""
    return env.scene["object"].data.root_pos_w.torch - _hand_position(env, asset_cfg)


def goal_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    hand_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return whether the grasped cube reached the commanded point.

    The cube must be within 0.035 m of the grasp center, with both finger positions
    between 0.01 m and 0.025 m, consistent with grasping this 40 mm cube.

    Args:
        env: Environment containing the robot and cube.
        command_name: Object pose command expressed in the robot root frame.
        threshold: Maximum cube-to-goal position error [m].
        hand_cfg: Robot hand used to measure the grasp center.
        gripper_cfg: Robot finger joints used to measure the grasp opening.

    Returns:
        Whether each environment's grasped cube reached the goal.
    """
    finger_positions = env.scene[gripper_cfg.name].data.joint_pos.torch[:, gripper_cfg.joint_ids]
    grasped = (torch.linalg.vector_norm(object_to_hand(env, hand_cfg), dim=-1) < 0.035) & (
        (finger_positions > 0.01) & (finger_positions < 0.025)
    ).all(dim=-1)
    return object_reached_goal(env, command_name=command_name, threshold=threshold) & grasped


def success_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return goal success per second [1/s], shape (num_envs,), for a unit terminal reward."""
    return env.termination_manager.get_term("success").float() / env.step_dt


def object_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return workspace-exit flags with shape (num_envs,), using positions in environment-frame [m]."""
    position = object_position(env)
    return (position[:, 2] < -0.02) | (position[:, 0] < 0.15) | (position[:, 0] > 0.85) | (position[:, 1].abs() > 0.45)


def _hand_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Compute the configured grasp-center position in the world frame."""
    robot = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    position = robot.data.body_link_pos_w.torch[:, body_id]
    if env._tcp_offset_pos is not None:
        position = position + quat_apply(robot.data.body_link_quat_w.torch[:, body_id], env._tcp_offset_pos)
    return position
