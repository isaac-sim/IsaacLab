# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the Pendulum MARL task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .shared import compute_cart_observation, compute_pendulum_observation, normalize_angle

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _joint_data(
    env: ManagerBasedEnv,
    cart_cfg: SceneEntityCfg,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the selected Pendulum joint positions and velocities."""
    asset = env.scene[cart_cfg.name]
    joint_pos = asset.data.joint_pos.torch
    joint_vel = asset.data.joint_vel.torch
    return (
        joint_pos[:, cart_cfg.joint_ids],
        joint_vel[:, cart_cfg.joint_ids],
        normalize_angle(joint_pos[:, pole_cfg.joint_ids]),
        joint_vel[:, pole_cfg.joint_ids],
        normalize_angle(joint_pos[:, pendulum_cfg.joint_ids]),
        joint_vel[:, pendulum_cfg.joint_ids],
    )


def cart_observation(
    env: ManagerBasedEnv,
    cart_cfg: SceneEntityCfg,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the cart agent observation in final direct-task order."""
    cart_pos, cart_vel, pole_pos, pole_vel, _, _ = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return compute_cart_observation(cart_pos, cart_vel, pole_pos, pole_vel)


def pendulum_observation(
    env: ManagerBasedEnv,
    cart_cfg: SceneEntityCfg,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the pendulum agent observation in final direct-task order."""
    _, _, pole_pos, _, pendulum_pos, pendulum_vel = _joint_data(env, cart_cfg, pole_cfg, pendulum_cfg)
    return compute_pendulum_observation(pole_pos, pendulum_pos, pendulum_vel)


def state(
    env: ManagerBasedEnv,
    cart_cfg: SceneEntityCfg,
    pole_cfg: SceneEntityCfg,
    pendulum_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return the seven-value centralized state in final direct-task order."""
    return torch.cat(
        (
            cart_observation(env, cart_cfg, pole_cfg, pendulum_cfg),
            pendulum_observation(env, cart_cfg, pole_cfg, pendulum_cfg),
        ),
        dim=-1,
    )
