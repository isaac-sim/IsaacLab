# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination functions specific to the gear assembly manipulation environments."""

from __future__ import annotations

import carb
import torch
from typing import TYPE_CHECKING, Optional

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


__all__ = ["reset_when_gear_dropped"]


def reset_when_gear_dropped(
    env: ManagerBasedEnv,
    distance_threshold: float = 0.1,
    height_threshold: Optional[float] = None,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if the gear has fallen out of the gripper and return reset flags.
    
    Robot-specific parameters are retrieved from env.cfg (all required):
    - end_effector_body_name: Name of the end effector body
    - rot_offset: Rotation offset to apply to gear orientation (quaternion [w, x, y, z])

    Args:
        env: The environment containing the assets
        distance_threshold: Maximum allowed distance between gear grasp point and gripper (in meters)
        height_threshold: Optional minimum height for the gear (in meters, world frame)
        robot_asset_cfg: Configuration for the robot asset

    Returns:
        Boolean tensor indicating which environments should be reset
    """
    if not hasattr(env, '_current_gear_type'):
        raise ValueError("Environment does not have '_current_gear_type' attribute. Ensure randomize_gear_type event is configured.")

    if not hasattr(env.cfg, 'gear_offsets_grasp'):
        raise ValueError("Environment config does not have 'gear_offsets_grasp' attribute.")

    # Get robot-specific parameters from environment config (all required - no defaults)
    if not hasattr(env.cfg, 'end_effector_body_name'):
        raise ValueError(
            "Robot-specific parameter 'end_effector_body_name' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.end_effector_body_name = 'wrist_3_link'"
        )
    if not hasattr(env.cfg, 'rot_offset'):
        raise ValueError(
            "Robot-specific parameter 'rot_offset' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.rot_offset = [0.0, 0.707, 0.707, 0.0]"
        )
    
    end_effector_body_name = env.cfg.end_effector_body_name
    rot_offset = env.cfg.rot_offset

    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    device = env.device
    num_envs = env.num_envs

    # Initialize reset flags
    reset_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Get the end effector position using robot-specific body name
    try:
        eef_indices, _ = robot_asset.find_bodies([end_effector_body_name])
        if len(eef_indices) == 0:
            carb.log_warn(f"{end_effector_body_name} not found in robot body names. Cannot check gear drop condition.")
            return reset_flags

        eef_idx = eef_indices[0]
        eef_pos_world = robot_asset.data.body_link_pos_w[:, eef_idx]  # Shape: (num_envs, 3)

    except Exception as e:
        carb.log_warn(f"Could not get end effector pose: {e}")
        return reset_flags

    # OPTIMIZED: Fully vectorized gear drop detection
    # Stack all gear positions and quaternions
    all_gear_pos = torch.stack([
        env.scene["factory_gear_small"].data.root_link_pos_w,
        env.scene["factory_gear_medium"].data.root_link_pos_w,
        env.scene["factory_gear_large"].data.root_link_pos_w,
    ], dim=1)  # (num_envs, 3, 3)
    
    all_gear_quat = torch.stack([
        env.scene["factory_gear_small"].data.root_link_quat_w,
        env.scene["factory_gear_medium"].data.root_link_quat_w,
        env.scene["factory_gear_large"].data.root_link_quat_w,
    ], dim=1)  # (num_envs, 3, 4)
    
    # Convert gear types to indices
    gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
    gear_type_indices = torch.tensor(
        [gear_type_map[env._current_gear_type[i]] for i in range(num_envs)],
        device=device,
        dtype=torch.long
    )
    
    # Select gear data using advanced indexing
    env_indices = torch.arange(num_envs, device=device)
    gear_pos_world = all_gear_pos[env_indices, gear_type_indices]  # (num_envs, 3)
    gear_quat_world = all_gear_quat[env_indices, gear_type_indices]  # (num_envs, 4)
    
    # Apply rotation offset to all gears if provided
    if rot_offset is not None:
        rot_offset_tensor = torch.tensor(rot_offset, device=device).unsqueeze(0).expand(num_envs, -1)
        gear_quat_world = math_utils.quat_mul(gear_quat_world, rot_offset_tensor)
    
    # Get grasp offsets for all environments (vectorized)
    gear_grasp_offsets = torch.stack([
        torch.tensor(env.cfg.gear_offsets_grasp[env._current_gear_type[i]], 
                     device=device, dtype=torch.float32)
        for i in range(num_envs)
    ])  # (num_envs, 3)
    
    # Transform grasp offsets to world frame (vectorized)
    gear_grasp_pos_world = gear_pos_world + math_utils.quat_apply(
        gear_quat_world, gear_grasp_offsets
    )
    
    # Compute distances for all environments (vectorized)
    distances = torch.norm(gear_grasp_pos_world - eef_pos_world, dim=-1)  # (num_envs,)
    
    # Check distance threshold (vectorized)
    reset_flags = distances > distance_threshold
    
    # Check height threshold if provided (vectorized)
    if height_threshold is not None:
        gear_heights = gear_pos_world[:, 2]  # Z coordinates
        reset_flags |= gear_heights < height_threshold
    
    return reset_flags

