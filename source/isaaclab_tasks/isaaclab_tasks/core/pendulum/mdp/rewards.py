# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the manager-based Pendulum MARL task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .shared import links_upright, normalize_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _joint_data(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the selected Pendulum joint data used by reward terms."""
    asset = env.scene[cart_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    joint_vel = asset.data.joint_vel.torch
    return (
        joint_vel[:, cart_cfg.joint_ids],
        normalize_angle(joint_pos[:, pole_cfg.joint_ids]),
        joint_vel[:, pole_cfg.joint_ids],
        normalize_angle(joint_pos[:, pendulum_cfg.joint_ids]),
        joint_vel[:, pendulum_cfg.joint_ids],
    )


def alive(env: ManagerBasedEnv) -> torch.Tensor:
    """Return one for environments that did not terminate this step."""
    return 1.0 - env.termination_manager.terminated.float()


def terminated(env: ManagerBasedEnv) -> torch.Tensor:
    """Return one for environments that terminated this step."""
    return env.termination_manager.terminated.float()


def cart_velocity_l1(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return absolute cart velocity [m/s]."""
    cart_vel, _, _, _, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return torch.abs(cart_vel).sum(dim=1)


def pole_position(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the upper-link uprightness cosine."""
    _, pole_pos, _, _, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return torch.cos(pole_pos).sum(dim=1)


def pole_velocity_l1(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return absolute upper-link angular velocity [rad/s]."""
    _, _, pole_vel, _, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return torch.abs(pole_vel).sum(dim=1)


def lower_link_position(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return the lower physical-link uprightness cosine."""
    _, pole_pos, _, pendulum_pos, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return torch.cos(normalize_angle(pole_pos + pendulum_pos)).sum(dim=1)


def lower_link_velocity_l1(
    env: ManagerBasedEnv, cart_cfg: SceneEntityCfg, pole_cfg: SceneEntityCfg, pendulum_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Return absolute lower physical-link angular velocity [rad/s]."""
    _, _, pole_vel, _, pendulum_vel = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return torch.abs(pole_vel + pendulum_vel).sum(dim=1)


def upright(
    env: ManagerBasedEnv,
    cart_cfg: SceneEntityCfg,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
    success_upright_angle: float,
) -> torch.Tensor:
    """Return whether both physical links are upright."""
    _, pole_pos, _, pendulum_pos, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return links_upright(pole_pos.squeeze(-1), pendulum_pos.squeeze(-1), success_upright_angle).float()


def cart_action_l2(env: ManagerBasedEnv) -> torch.Tensor:
    """Return the joint squared action cost shared by both agents."""
    parent = env.parent
    cart_action = parent.get_agent("cart").action_manager.action
    pendulum_action = parent.get_agent("pendulum").action_manager.action
    return torch.sum(torch.square(cart_action), dim=1) + torch.sum(torch.square(pendulum_action), dim=1)
