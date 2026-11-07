# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sparse completion rewards and finite action penalties for conveyor transfer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def transfer_success_reward(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Return one on each stable transfer-completion transition."""
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    return command.new_success.float()


def terminal_failure(
    env: ManagerBasedRLEnv,
    command_name: str = "transfer",
) -> torch.Tensor:
    """Return one for non-timeout terminal failures."""
    command = env.command_manager.get_term(command_name)
    command.evaluate()
    return (env.reset_terminated & ~command.pending_success).float()


def action_term_l2(env: ManagerBasedRLEnv, action_name: str) -> torch.Tensor:
    """Penalize one named raw action term."""
    action = env.action_manager.get_term(action_name).raw_actions
    return torch.sum(torch.square(action), dim=1)


def finite_action_rate_l2(
    env: ManagerBasedRLEnv,
    action_names: tuple[str, ...] = ("arm_action", "gripper_action"),
) -> torch.Tensor:
    """Penalize changes between the finite policy commands accepted by action terms."""
    if not action_names:
        raise ValueError("At least one action term is required for the action-rate reward.")
    terms = tuple(env.action_manager.get_term(name) for name in action_names)
    action = torch.cat(tuple(term.raw_actions for term in terms), dim=1)
    previous_action = torch.cat(tuple(term.previous_actions for term in terms), dim=1)
    return torch.sum(torch.square(action - previous_action), dim=1)


def finite_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["panda_joint[1-7]"]),
    maximum_velocity: float = 3.0,
) -> torch.Tensor:
    """Penalize bounded arm velocity while sanitizing divergent states."""
    if maximum_velocity <= 0.0:
        raise ValueError("maximum_velocity must be positive.")
    robot: Articulation = env.scene[asset_cfg.name]
    velocity = robot.data.joint_vel.torch[:, asset_cfg.joint_ids]
    velocity = torch.nan_to_num(velocity, nan=0.0, posinf=maximum_velocity, neginf=-maximum_velocity)
    return torch.sum(torch.square(torch.clamp(velocity, -maximum_velocity, maximum_velocity)), dim=1)
