# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State observations and the sparse grasped-cube goal condition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from isaaclab_tasks.core.lift.mdp import object_reached_goal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Cube center position relative to the environment origin [m]."""
    return env.scene["object"].data.root_pos_w.torch - env.scene.env_origins


def hand_pose(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand")
) -> torch.Tensor:
    """Grasp-center position [m] and orientation [xyzw] relative to the environment origin."""
    robot = env.scene[asset_cfg.name]
    body_id = asset_cfg.body_ids[0]
    orientation = robot.data.body_link_quat_w.torch[:, body_id]
    offset = orientation.new_tensor((0.0, 0.0, 0.107)).expand(env.num_envs, -1)
    position = robot.data.body_link_pos_w.torch[:, body_id] + quat_apply(orientation, offset)
    return torch.cat((position - env.scene.env_origins, orientation), dim=-1)


def object_to_hand(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="panda_hand")
) -> torch.Tensor:
    """Cube displacement from the grasp center [m]."""
    return object_position(env) - hand_pose(env, asset_cfg)[:, :3]


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
    """Return the binary goal success for the terminal transition."""
    return env.termination_manager.get_term("success").float()


def object_dropped(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Terminate when the cube leaves the table workspace [m]."""
    position = object_position(env)
    return (position[:, 2] < -0.02) | (position[:, 0] < 0.15) | (position[:, 0] > 0.85) | (position[:, 1].abs() > 0.45)
