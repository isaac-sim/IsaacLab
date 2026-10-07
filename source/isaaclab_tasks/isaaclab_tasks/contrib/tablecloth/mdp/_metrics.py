# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import DeformableObject, RigidObject
    from isaaclab.envs import ManagerBasedRLEnv


def cloth_pull_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cloth"),
) -> torch.Tensor:
    """Return cloth center-of-mass travel along the negative x-axis [m]."""
    cloth: DeformableObject = env.scene[asset_cfg.name]
    initial_x = cloth.data.default_nodal_state_w.torch[..., 0].mean(dim=1)
    return initial_x - cloth.data.root_pos_w.torch[:, 0]


def tableware_assets(env: ManagerBasedRLEnv, asset_names: tuple[str, ...]) -> list[RigidObject]:
    """Resolve the configured tableware rigid objects."""
    return [env.scene[name] for name in asset_names]
