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
    rot_offset: Optional[list[float]] = None,
) -> torch.Tensor:
    """Check if the gear has fallen out of the gripper and return reset flags.

    Args:
        env: The environment containing the assets
        distance_threshold: Maximum allowed distance between gear grasp point and gripper (in meters)
        height_threshold: Optional minimum height for the gear (in meters, world frame)
        robot_asset_cfg: Configuration for the robot asset
        rot_offset: Rotation offset to apply to gear orientation (quaternion [w, x, y, z])

    Returns:
        Boolean tensor indicating which environments should be reset
    """
    if not hasattr(env, '_current_gear_type'):
        raise ValueError("Environment does not have '_current_gear_type' attribute. Ensure randomize_gear_type event is configured.")

    if not hasattr(env.cfg, 'gear_offsets_grasp'):
        raise ValueError("Environment config does not have 'gear_offsets_grasp' attribute.")

    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    device = env.device
    num_envs = env.num_envs

    # Initialize reset flags
    reset_flags = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # Get the wrist_3_link (end effector) position
    try:
        wrist_3_indices, _ = robot_asset.find_bodies(["wrist_3_link"])
        if len(wrist_3_indices) == 0:
            carb.log_warn("wrist_3_link not found in robot body names. Cannot check gear drop condition.")
            return reset_flags

        wrist_3_idx = wrist_3_indices[0]
        eef_pos_world = robot_asset.data.body_link_pos_w[:, wrist_3_idx]  # Shape: (num_envs, 3)

    except Exception as e:
        carb.log_warn(f"Could not get end effector pose: {e}")
        return reset_flags

    # Convert rotation offset to tensor if provided
    if rot_offset is not None:
        rot_offset_tensor = torch.tensor(rot_offset, device=device)
    else:
        rot_offset_tensor = None

    # Check each environment's gear position against gripper position
    for env_id in range(num_envs):
        gear_key = env._current_gear_type[env_id]
        selected_asset_name = f"factory_{gear_key}"

        # Get the gear asset for this environment
        gear_asset: RigidObject = env.scene[selected_asset_name]
        gear_pos_world = gear_asset.data.root_link_pos_w[env_id]
        gear_quat_world = gear_asset.data.root_link_quat_w[env_id]

        # Apply rotation offset to gear orientation if provided
        if rot_offset_tensor is not None:
            gear_quat_world = math_utils.quat_mul(gear_quat_world, rot_offset_tensor)

        # Get the grasp offset for this gear type from config
        gear_grasp_offset = env.cfg.gear_offsets_grasp[gear_key]
        grasp_offset_tensor = torch.tensor(gear_grasp_offset, device=device, dtype=gear_pos_world.dtype)

        # Transform grasp offset from gear frame to world frame
        gear_grasp_pos_world = gear_pos_world + math_utils.quat_apply(
            gear_quat_world.unsqueeze(0), grasp_offset_tensor.unsqueeze(0)
        )[0]

        # Compute distance between gear grasp point and gripper
        distance = torch.norm(gear_grasp_pos_world - eef_pos_world[env_id])

        # # Print distance for debugging
        # if env_id == 0:  # Only print for first environment to avoid spam
        # print(f"[Env {env_id}] Gear-Gripper Distance: {distance:.4f}m (threshold: {distance_threshold:.4f}m)")

        # Check distance threshold
        if distance > distance_threshold:
            # print(f"[Env {env_id}] RESET: Gear dropped! Distance {distance:.4f}m > threshold {distance_threshold:.4f}m")
            reset_flags[env_id] = True
            continue

        # Check height threshold if provided
        if height_threshold is not None:
            gear_height = gear_pos_world[2]  # Z coordinate in world frame
            if gear_height < height_threshold:
                # print(f"[Env {env_id}] RESET: Gear too low! Height {gear_height:.4f}m < threshold {height_threshold:.4f}m")
                reset_flags[env_id] = True

    return reset_flags

