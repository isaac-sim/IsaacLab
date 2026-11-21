# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Backend-independent end-effector state helpers for stack tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _end_effector_cache_entry(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    body_name: str,
    body_offset: tuple[float, float, float],
) -> tuple[Articulation, int, torch.Tensor]:
    """Resolve and cache one articulation body and its tool-frame offset."""
    robot: Articulation = env.scene[robot_cfg.name]
    cache = getattr(env, "_stack_end_effector_cache", None)
    if cache is None:
        cache = {}
        env._stack_end_effector_cache = cache

    cache_key = (robot_cfg.name, body_name, body_offset)
    entry = cache.get(cache_key)
    if entry is None:
        body_ids, _ = robot.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(f"Expected one end-effector body matching '{body_name}', found {len(body_ids)}.")
        entry = (
            body_ids[0],
            torch.tensor(body_offset, dtype=torch.float32, device=env.device),
        )
        cache[cache_key] = entry
    return robot, entry[0], entry[1]


def end_effector_pose(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
    *,
    env_ids: Sequence[int] | torch.Tensor | slice = slice(None),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a configured tool-center position [m] and orientation."""
    robot, body_id, tool_offset = _end_effector_cache_entry(env, robot_cfg, body_name, body_offset)

    body_position = robot.data.body_pos_w.torch[env_ids, body_id]
    body_orientation = robot.data.body_quat_w.torch[env_ids, body_id]
    tool_position = body_position + math_utils.quat_apply(
        body_orientation, tool_offset.expand(body_orientation.shape[0], -1)
    )
    return tool_position, body_orientation


def end_effector_velocity(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str = "panda_hand",
    body_offset: tuple[float, float, float] = (0.0, 0.0, 0.1034),
) -> torch.Tensor:
    """Return the tool-center linear/angular velocity in world coordinates."""
    robot, body_id, tool_offset = _end_effector_cache_entry(env, robot_cfg, body_name, body_offset)

    body_orientation = robot.data.body_quat_w.torch[:, body_id]
    body_velocity = robot.data.body_vel_w.torch[:, body_id]
    tool_offset_world = math_utils.quat_apply(body_orientation, tool_offset.expand(env.num_envs, -1))
    tool_linear_velocity = body_velocity[:, :3] + torch.linalg.cross(
        body_velocity[:, 3:],
        tool_offset_world,
    )
    return torch.cat((tool_linear_velocity, body_velocity[:, 3:]), dim=1)
