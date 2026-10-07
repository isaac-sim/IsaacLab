# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from ._metrics import cloth_pull_distance, tableware_assets

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def cloth_pull_progress(
    env: ManagerBasedRLEnv,
    target_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Reward cloth travel toward the pull goal."""
    return torch.clamp(cloth_pull_distance(env, asset_cfg) / target_distance, min=0.0, max=1.0)


def tableware_displacement(
    env: ManagerBasedRLEnv,
    asset_names: tuple[str, ...],
    distance_scale: float,
) -> torch.Tensor:
    """Reward keeping tableware near its initial horizontal position."""
    squared_distance = torch.zeros(env.num_envs, device=env.device)
    for asset in tableware_assets(env, asset_names):
        initial_position = asset.data.default_root_pose.torch[:, :3] + env.scene.env_origins
        squared_distance += torch.sum(torch.square(asset.data.root_pos_w.torch[:, :2] - initial_position[:, :2]), dim=1)
    return torch.exp(-squared_distance / distance_scale**2)


def tableware_upright(env: ManagerBasedRLEnv, asset_names: tuple[str, ...]) -> torch.Tensor:
    """Reward keeping the tableware's local z-axes upright."""
    upright = torch.zeros(env.num_envs, device=env.device)
    for asset in tableware_assets(env, asset_names):
        upright += torch.clamp(-asset.data.projected_gravity_b.torch[:, 2], min=0.0, max=1.0)
    return upright / len(asset_names)
