# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Private tensor helpers for heterogeneous manipulation MDP terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils import math as math_utils

from ..selection_utils import SceneEntitySelectionCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def _offset_body_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntitySelectionCfg,
    offset_pos: tuple[float, float, float],
    offset_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    body_index: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return selected global IDs and an offset body pose in the world frame.

    Args:
        env: Manager-based RL environment.
        asset_cfg: Selection-aware entity containing exactly one body.
        offset_pos: Child-frame translation [m].
        offset_quat: Child-frame quaternion in ``(x, y, z, w)`` order.
        body_index: Index into the resolved body selection.

    Returns:
        Global environment IDs, positions [m], and quaternions for the selected view rows.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    pose = asset.data.body_link_pose_w.torch[:, asset_cfg.body_ids[body_index]]
    pos = pose.new_tensor(offset_pos).expand(len(pose), -1)
    quat = math_utils.convert_quat(pose.new_tensor(offset_quat).expand(len(pose), -1), to="wxyz")
    pos_w, quat_w = math_utils.combine_frame_transforms(pose[:, :3], pose[:, 3:7], pos, quat)
    return asset_cfg.env_ids, pos_w, quat_w
