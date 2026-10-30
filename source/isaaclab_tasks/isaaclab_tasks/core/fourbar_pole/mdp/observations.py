# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the fourbar-pole swing-up environment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_cos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Cosine of the selected joint positions.

    Encodes the angle without the wrap-around discontinuity at ``+-pi`` so the policy sees a smooth
    signal as the pole swings through the bottom.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.cos(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])


def joint_pos_sin(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Sine of the selected joint positions (companion to :func:`joint_pos_cos`)."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sin(asset.data.joint_pos.torch[:, asset_cfg.joint_ids])
