# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset events for the manager-based handover task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv


def reset_object_at_hold_point(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    hand_cfg: SceneEntityCfg,
    position_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Reset the object at rest at a hand's hold point, with uniform position noise and a uniform random orientation.

    Args:
        env: The environment.
        env_ids: Environments to reset.
        pose_range: Position noise range per axis ``x``, ``y``, ``z`` [m]; a missing axis gets no noise.
        hand_cfg: The hand holding the object.
        position_offset: Hold point in the hand's local root frame [m].
        asset_cfg: The object.
    """
    hand: Articulation = env.scene[hand_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    hand_pose = hand.data.default_root_pose.torch[env_ids]
    offset = torch.tensor(position_offset, device=asset.device).expand(hand_pose.shape[0], 3)
    noise = math_utils.sample_uniform_from_ranges(pose_range, ("x", "y", "z"), hand_pose.shape[0], device=asset.device)
    positions = hand_pose[:, :3] + math_utils.quat_apply(hand_pose[:, 3:7], offset) + env.scene.env_origins[env_ids]
    orientations = math_utils.random_orientation(hand_pose.shape[0], device=asset.device)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat((positions + noise, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(
        root_velocity=torch.zeros_like(asset.data.default_root_vel.torch[env_ids]), env_ids=env_ids
    )
