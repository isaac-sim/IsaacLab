# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .rewards import get_task_stage

if TYPE_CHECKING:
    from isaaclab.assets import RigidObject
    from isaaclab.envs import ManagerBasedRLEnv

logger = logging.getLogger(__name__)


def object_drop_termination(
    env: ManagerBasedRLEnv,
    drop_height_threshold: float = 0.5,
    asset_cfg1: SceneEntityCfg = SceneEntityCfg("trocar_1"),
    asset_cfg2: SceneEntityCfg = SceneEntityCfg("trocar_2"),
    print_log: bool = False,
) -> torch.Tensor:
    """Termination function that triggers when objects drop below threshold.

    This can be used as an alternative to auto-reset, marking the episode as terminated
    so the training framework handles the reset.

    Args:
        env: The environment instance
        drop_height_threshold: Height below which objects are considered dropped
        asset_cfg1: Configuration for first trocar
        asset_cfg2: Configuration for second trocar
        print_log: If True, print debug information.
    Returns:
        Boolean tensor indicating which environments should terminate due to drops
    """
    # Get rigid objects
    obj1: RigidObject = env.scene[asset_cfg1.name]
    obj2: RigidObject = env.scene[asset_cfg2.name]

    # Get positions
    pos1 = obj1.data.root_pos_w.torch
    pos2 = obj2.data.root_pos_w.torch
    # Check if either object has dropped
    dropped_1 = pos1[:, 2] < drop_height_threshold
    dropped_2 = pos2[:, 2] < drop_height_threshold

    dropped = dropped_1 | dropped_2

    if print_log and dropped.any():
        logger.debug("Drop termination triggered for %d environment(s)", dropped.sum().item())

    return dropped


def task_success_termination(
    env: ManagerBasedRLEnv,
    success_stage: int = 4,
    print_log: bool = False,
) -> torch.Tensor:
    """Termination condition: task is complete when stage reaches 4.

    Returns:
        torch.Tensor: Boolean tensor indicating which environments should terminate (num_envs,)
    """
    stage = get_task_stage(env)
    task_complete = stage >= success_stage

    if print_log and task_complete.any():
        logger.info("Task completed in %d environment(s)!", task_complete.sum().item())

    return task_complete
