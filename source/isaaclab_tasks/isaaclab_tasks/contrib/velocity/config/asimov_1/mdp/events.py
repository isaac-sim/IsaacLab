# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    ranges: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str | None = "joint_pos",
):
    """Randomize default joint positions and keep position-action offsets synchronized."""
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    default_joint_pos = asset.data.default_joint_pos.torch
    if not hasattr(asset.data, "nominal_default_joint_pos"):
        asset.data.nominal_default_joint_pos = default_joint_pos.clone()

    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, slice):
        num_joints = asset.num_joints
        joint_ids = torch.arange(num_joints, device=asset.device)
    else:
        joint_ids = torch.tensor(joint_ids, device=asset.device)

    offsets = torch.empty((len(env_ids), len(joint_ids)), device=asset.device).uniform_(ranges[0], ranges[1])
    default_joint_pos[env_ids[:, None], joint_ids] += offsets

    if action_name is not None and hasattr(env, "action_manager"):
        try:
            term = env.action_manager.get_term(action_name)
        except (KeyError, ValueError):
            return
        offset = getattr(term, "_offset", None)
        if isinstance(offset, torch.Tensor):
            term._offset = default_joint_pos[:, term._joint_ids].clone()
