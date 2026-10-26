# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms of the MicroDuck on a ball."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import ball_pos_b

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_off_feet(
    env: ManagerBasedRLEnv, max_distance: float, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the ball center is more than ``max_distance`` [m] (horizontally) from the middle of the feet."""
    return torch.linalg.norm(ball_pos_b(env, robot_cfg, ball_cfg), dim=1) > max_distance


def base_too_low(
    env: ManagerBasedRLEnv, minimum_height: float, robot_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Terminate when the base is less than ``minimum_height`` [m] above the top of the ball."""
    ball = env.scene[ball_cfg.name]
    ball_top = ball.data.root_pos_w.torch[:, 2] + ball.cfg.spawn.radius
    return env.scene[robot_cfg.name].data.root_pos_w.torch[:, 2] - ball_top < minimum_height


def ball_outside_zone(env: ManagerBasedRLEnv, radius: float, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the ball is further than ``radius`` [m] (horizontally) from the origin of its environment."""
    offset = env.scene[ball_cfg.name].data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    return torch.linalg.norm(offset, dim=1) > radius
