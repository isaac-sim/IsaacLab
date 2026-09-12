# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination conditions for the H2 pick-and-place apple task."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import get_task_stage

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def apple_on_plate_and_released(
    env: ManagerBasedRLEnv,
    apple_cfg: SceneEntityCfg = SceneEntityCfg("apple"),
    plate_cfg: SceneEntityCfg = SceneEntityCfg("plate"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    xy_radius: float = 0.10,
    z_above: float = 0.025,
    z_window: float = 0.075,
    release_wrist_distance: float = 0.10,
) -> torch.Tensor:
    """Return whether the apple is on the plate and clear of the right wrist."""
    apple = env.scene[apple_cfg.name]
    plate = env.scene[plate_cfg.name]
    robot = env.scene[robot_cfg.name]

    apple_pos = apple.data.root_pos_w.torch
    plate_pos = plate.data.root_pos_w.torch

    dx = apple_pos[:, 0] - plate_pos[:, 0]
    dy = apple_pos[:, 1] - plate_pos[:, 1]
    horiz = torch.sqrt(dx * dx + dy * dy)
    in_xy = horiz < xy_radius

    dz = apple_pos[:, 2] - plate_pos[:, 2]
    in_z = (dz > z_above) & (dz < (z_above + z_window))

    body_names = list(robot.body_names)
    wrist_bid = body_names.index("right_wrist_yaw_link")
    wrist_pos = robot.data.body_pos_w.torch[:, wrist_bid]
    wrist_dist = torch.norm(apple_pos - wrist_pos, dim=-1)
    released = wrist_dist > release_wrist_distance

    return in_xy & in_z & released


def task_success_termination(
    env: ManagerBasedRLEnv,
    success_stage: int = 4,
    print_log: bool = False,
) -> torch.Tensor:
    """Terminate when the stage machine reaches the success stage."""
    stage = get_task_stage(env)
    task_complete = stage >= success_stage

    if print_log and task_complete.any():
        logger.info("Task completed in %d environment(s)!", task_complete.sum().item())

    return task_complete
