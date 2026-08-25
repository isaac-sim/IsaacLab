# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the manager-based handover task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def handover_reward(goal_distance: torch.Tensor, distance_scale: float) -> torch.Tensor:
    """Return one hand's Direct reward for the current object-goal distance."""
    return 2.0 * torch.exp(-distance_scale * goal_distance)


@torch.jit.script
def evaluate_handover_success(
    object_position: torch.Tensor, target_position: torch.Tensor, success_distance_threshold: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate handover success while exposing its physical error.

    Args:
        object_position: Object positions [m].
        target_position: Goal positions [m].
        success_distance_threshold: Exclusive successful goal-distance threshold [m].

    Returns:
        Per-environment success flags and object-to-goal distances [m].
    """
    goal_distance = torch.linalg.norm(object_position - target_position, ord=2, dim=-1)
    return goal_distance < success_distance_threshold, goal_distance


def handover_goal_distance_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    distance_scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward both hands for holding the object near its goal.

    The Direct environment sums one identical reward per hand, so this returns twice the
    single-hand value. The command term owns the episode success bookkeeping behind
    ``Metrics/success_rate``.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        distance_scale: Exponential decay rate of the distance reward [1/m].
        object_cfg: The configuration for the scene entity. Default is "object".
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    goal_pos = env.command_manager.get_command(command_name)[:, :3]
    goal_distance = torch.linalg.norm(object_pos - goal_pos, ord=2, dim=-1)
    return 2.0 * handover_reward(goal_distance, distance_scale)


def object_lin_vel_l2(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalize the object's squared linear speed so it settles instead of drifting.

    Args:
        env: The environment object.
        object_cfg: The configuration for the scene entity. Default is "object".

    Returns:
        Squared object linear speed [m^2/s^2].
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(object_asset.data.root_lin_vel_w.torch), dim=-1)


def hold_at_goal(
    env: ManagerBasedRLEnv,
    command_name: str,
    success_distance_threshold: float,
    hold_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward keeping the object inside the goal threshold at low speed.

    The distance term alone is maximised by touching the goal, so a policy that swings the
    object through it scores the same as one that parks it. This pays only while it stays.

    Args:
        env: The environment object.
        command_name: The command term to be used for extracting the goal.
        success_distance_threshold: Goal distance below which the object counts as placed [m].
        hold_speed: Object speed below which the object counts as settled [m/s].
        object_cfg: The configuration for the scene entity. Default is "object".

    Returns:
        One where the object is both placed and settled, zero elsewhere.
    """
    object_asset: RigidObject = env.scene[object_cfg.name]
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    goal_pos = env.command_manager.get_command(command_name)[:, :3]
    goal_distance = torch.linalg.norm(object_pos - goal_pos, ord=2, dim=-1)
    speed = torch.linalg.norm(object_asset.data.root_lin_vel_w.torch, ord=2, dim=-1)
    return ((goal_distance < success_distance_threshold) & (speed < hold_speed)).float()


def joint_deviation_when_released(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    release_distance: float = 0.08,
) -> torch.Tensor:
    """Penalize one hand's joint deviation from its default pose only while it is not holding.

    The hand's default pose is every joint at zero -- a flat, open hand -- so an unconditional
    deviation penalty also charges the curl required to hold the object, and works against the
    task. Gating on the nearest fingertip charges only the hand that has let go.

    Args:
        env: The environment object.
        asset_cfg: The hand to charge. Must name the fingertip bodies used for the gate.
        object_cfg: The configuration for the scene entity. Default is "object".
        release_distance: Nearest-fingertip distance beyond which the hand counts as released [m].

    Returns:
        Summed absolute joint deviation [rad] where the hand has released, zero elsewhere.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    tips = asset.data.body_pos_w.torch[:, asset_cfg.body_ids] - env.scene.env_origins.unsqueeze(1)
    object_pos = object_asset.data.root_pos_w.torch - env.scene.env_origins
    nearest = torch.linalg.norm(tips - object_pos.unsqueeze(1), ord=2, dim=-1).min(dim=1).values
    deviation = torch.sum(torch.abs(asset.data.joint_pos.torch - asset.data.default_joint_pos.torch), dim=-1)
    return deviation * (nearest > release_distance).float()
