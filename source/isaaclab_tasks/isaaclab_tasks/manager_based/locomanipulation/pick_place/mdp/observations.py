# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import warp as wp

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.utils.io_descriptors import generic_io_descriptor, record_shape
from isaaclab.managers import SceneEntityCfg


def upper_body_last_action(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract the last action of the upper body."""
    asset = env.scene[asset_cfg.name]
    joint_pos_target = wp.to_torch(asset.data.joint_pos_target)

    # Use joint_names from asset_cfg to find indices
    joint_names = asset_cfg.joint_names if hasattr(asset_cfg, "joint_names") else None
    if joint_names is None:
        raise ValueError("asset_cfg must have 'joint_names' attribute for upper_body_last_action.")

    # Find joint indices matching the provided joint_names (supports regex)
    # find_joints returns (joint_mask: wp.array, joint_names: list[str], joint_indices: list[int])
    _, _, joint_indices = asset.find_joints(joint_names)

    # Get upper body joint positions for all environments
    upper_body_joint_pos_target = joint_pos_target[:, joint_indices]

    return upper_body_joint_pos_target


@generic_io_descriptor(dtype=torch.float32, observation_type="Action", on_inspect=[record_shape])
def last_action_with_remap(
    env: ManagerBasedRLEnv,
    action_name: str | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Get the last raw action from an action term with reordering based on joint_ids.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their actions returned.

    Args:
        env: The manager-based RL environment.
        action_name: The name of the action term to get raw actions from.
        asset_cfg: The SceneEntity associated with this observation. The joint_ids are used to index/reorder.

    Returns:
        The raw actions tensor reordered by joint_ids.
        Shape: (num_envs, len(joint_ids))
    """
    if action_name is None:
        return env.action_manager.action[:, asset_cfg.joint_ids]
    else:
        return env.action_manager.get_term(action_name).raw_actions[:, asset_cfg.joint_ids]
