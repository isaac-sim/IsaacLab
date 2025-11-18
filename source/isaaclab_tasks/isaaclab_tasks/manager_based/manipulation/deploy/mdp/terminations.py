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


__all__ = ["reset_when_gear_dropped", "reset_when_gear_orientation_exceeds_threshold"]


def reset_when_gear_dropped(
    env: ManagerBasedEnv,
    distance_threshold: float = 0.1,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if the gear has fallen out of the gripper and return reset flags.
    
    Robot-specific parameters are retrieved from env.cfg (all required):
    - end_effector_body_name: Name of the end effector body
    - grasp_rot_offset: Rotation offset to apply to gear orientation (quaternion [w, x, y, z])

    Args:
        env: The environment containing the assets
        distance_threshold: Maximum allowed distance between gear grasp point and gripper (in meters)
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
    if not hasattr(env.cfg, 'grasp_rot_offset'):
        raise ValueError(
            "Robot-specific parameter 'grasp_rot_offset' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.grasp_rot_offset = [0.0, 0.707, 0.707, 0.0]"
        )
    
    end_effector_body_name = env.cfg.end_effector_body_name
    grasp_rot_offset = env.cfg.grasp_rot_offset

    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    device = env.device
    num_envs = env.num_envs

    # Use shared cache (must be initialized by initialize_shared_gear_cache event)
    if not hasattr(env, '_shared_gear_cache'):
        raise RuntimeError(
            "Shared gear cache not initialized. Ensure 'initialize_shared_gear_cache' is called "
            "during startup or reset events before this termination function."
        )

    # Reuse cached tensors from shared cache
    cache = env._shared_gear_cache
    env_indices = cache['env_indices']
    gear_type_map = cache['gear_type_map']
    reset_flags = cache['reset_flags']
    reset_flags.fill_(False)  # Reset to False for this iteration
    
    # Get grasp rotation offset tensor if available
    grasp_rot_offset_tensor = cache.get('grasp_rot_offset_tensor', None)

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

    # Use cached buffers for gear positions and quaternions (avoid torch.stack allocation)
    all_gear_pos = cache['all_gear_pos_buffer']
    all_gear_pos[:, 0, :] = env.scene["factory_gear_small"].data.root_link_pos_w
    all_gear_pos[:, 1, :] = env.scene["factory_gear_medium"].data.root_link_pos_w
    all_gear_pos[:, 2, :] = env.scene["factory_gear_large"].data.root_link_pos_w
    
    all_gear_quat = cache['all_gear_quat_buffer']
    all_gear_quat[:, 0, :] = env.scene["factory_gear_small"].data.root_link_quat_w
    all_gear_quat[:, 1, :] = env.scene["factory_gear_medium"].data.root_link_quat_w
    all_gear_quat[:, 2, :] = env.scene["factory_gear_large"].data.root_link_quat_w
    
    # Convert gear types to indices - use shared cache buffer
    gear_type_indices = cache['gear_type_indices']
    for i in range(num_envs):
        gear_type_indices[i] = gear_type_map[env._current_gear_type[i]]
    
    # Select gear data using advanced indexing
    gear_pos_world = all_gear_pos[env_indices, gear_type_indices]  # (num_envs, 3)
    gear_quat_world = all_gear_quat[env_indices, gear_type_indices]  # (num_envs, 4)
    
    # Apply rotation offset to all gears if provided
    if grasp_rot_offset_tensor is not None:
        gear_quat_world = math_utils.quat_mul(gear_quat_world, grasp_rot_offset_tensor)
    
    # Get grasp offsets for all environments using cached tensors from shared cache
    # Reuse the gear_grasp_offsets_buffer from shared cache
    gear_grasp_offsets = cache['gear_grasp_offsets_buffer']
    for i in range(num_envs):
        gear_grasp_offsets[i] = cache['gear_grasp_offset_tensors'][env._current_gear_type[i]]
    
    # Transform grasp offsets to world frame (vectorized)
    gear_grasp_pos_world = gear_pos_world + math_utils.quat_apply(
        gear_quat_world, gear_grasp_offsets
    )
    
    # Compute distances for all environments (vectorized)
    distances = torch.norm(gear_grasp_pos_world - eef_pos_world, dim=-1)  # (num_envs,)
    
    # Check distance threshold (vectorized)
    reset_flags[:] = distances > distance_threshold

    return reset_flags


def reset_when_gear_orientation_exceeds_threshold(
    env: ManagerBasedEnv,
    roll_threshold_deg: float = 30.0,
    pitch_threshold_deg: float = 30.0,
    yaw_threshold_deg: float = 180.0,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Check if the gear's orientation relative to the gripper exceeds thresholds and return reset flags.

    This function computes the relative orientation between the gear and the end effector (gripper),
    converts it to Euler angles (roll, pitch, yaw), and checks if any angle exceeds the configured thresholds.

    Robot-specific parameters are retrieved from env.cfg (all required):
    - end_effector_body_name: Name of the end effector body
    - grasp_rot_offset: Rotation offset to apply to gear orientation (quaternion [w, x, y, z])

    Args:
        env: The environment containing the assets
        roll_threshold_deg: Maximum allowed roll angle deviation in degrees
        pitch_threshold_deg: Maximum allowed pitch angle deviation in degrees
        yaw_threshold_deg: Maximum allowed yaw angle deviation in degrees
        robot_asset_cfg: Configuration for the robot asset

    Returns:
        Boolean tensor indicating which environments should be reset
    """
    if not hasattr(env, '_current_gear_type'):
        raise ValueError("Environment does not have '_current_gear_type' attribute. Ensure randomize_gear_type event is configured.")

    # Get robot-specific parameters from environment config (all required - no defaults)
    if not hasattr(env.cfg, 'end_effector_body_name'):
        raise ValueError(
            "Robot-specific parameter 'end_effector_body_name' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.end_effector_body_name = 'wrist_3_link'"
        )
    if not hasattr(env.cfg, 'grasp_rot_offset'):
        raise ValueError(
            "Robot-specific parameter 'grasp_rot_offset' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.grasp_rot_offset = [0.0, 0.707, 0.707, 0.0]"
        )

    end_effector_body_name = env.cfg.end_effector_body_name
    grasp_rot_offset = env.cfg.grasp_rot_offset

    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    device = env.device
    num_envs = env.num_envs

    # Use shared cache (must be initialized by initialize_shared_gear_cache event)
    if not hasattr(env, '_shared_gear_cache'):
        raise RuntimeError(
            "Shared gear cache not initialized. Ensure 'initialize_shared_gear_cache' is called "
            "during startup or reset events before this termination function."
        )

    # Reuse cached tensors from shared cache
    cache = env._shared_gear_cache
    
    # Initialize orientation-specific cached values on first call
    if not hasattr(env, '_gear_orientation_thresholds'):
        env._gear_orientation_thresholds = {
            'roll_threshold_rad': torch.deg2rad(torch.tensor(roll_threshold_deg, device=device)),
            'pitch_threshold_rad': torch.deg2rad(torch.tensor(pitch_threshold_deg, device=device)),
            'yaw_threshold_rad': torch.deg2rad(torch.tensor(yaw_threshold_deg, device=device)),
        }
    
    thresholds = env._gear_orientation_thresholds
    env_indices = cache['env_indices']
    gear_type_map = cache['gear_type_map']
    reset_flags = cache['reset_flags']
    reset_flags.fill_(False)  # Reset to False for this iteration
    roll_threshold_rad = thresholds['roll_threshold_rad']
    pitch_threshold_rad = thresholds['pitch_threshold_rad']
    yaw_threshold_rad = thresholds['yaw_threshold_rad']
    
    # Get grasp rotation offset tensor if available
    grasp_rot_offset_tensor = cache.get('grasp_rot_offset_tensor', None)

    # Get the end effector orientation using robot-specific body name
    try:
        eef_indices, _ = robot_asset.find_bodies([end_effector_body_name])
        if len(eef_indices) == 0:
            carb.log_warn(f"{end_effector_body_name} not found in robot body names. Cannot check gear orientation condition.")
            return reset_flags

        eef_idx = eef_indices[0]
        eef_quat_world = robot_asset.data.body_link_quat_w[:, eef_idx]  # Shape: (num_envs, 4)

    except Exception as e:
        carb.log_warn(f"Could not get end effector orientation: {e}")
        return reset_flags

    # Use cached buffer for gear quaternions (avoid torch.stack allocation)
    all_gear_quat = cache['all_gear_quat_buffer']
    all_gear_quat[:, 0, :] = env.scene["factory_gear_small"].data.root_link_quat_w
    all_gear_quat[:, 1, :] = env.scene["factory_gear_medium"].data.root_link_quat_w
    all_gear_quat[:, 2, :] = env.scene["factory_gear_large"].data.root_link_quat_w

    # Convert gear types to indices - use shared cache buffer
    gear_type_indices = cache['gear_type_indices']
    for i in range(num_envs):
        gear_type_indices[i] = gear_type_map[env._current_gear_type[i]]

    # Select gear data using advanced indexing
    gear_quat_world = all_gear_quat[env_indices, gear_type_indices]  # (num_envs, 4)

    # Apply rotation offset to all gears if provided
    if grasp_rot_offset_tensor is not None:
        gear_quat_world = math_utils.quat_mul(gear_quat_world, grasp_rot_offset_tensor)

    # Compute relative orientation: q_rel = q_gear * q_eef^-1
    eef_quat_inv = math_utils.quat_conjugate(eef_quat_world)
    relative_quat = math_utils.quat_mul(gear_quat_world, eef_quat_inv)

    # Convert relative quaternion to Euler angles (roll, pitch, yaw)
    # Using XYZ extrinsic convention (returns tuple of tensors)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(relative_quat)


    # Check if any angle exceeds its threshold
    reset_flags[:] = (torch.abs(roll) > roll_threshold_rad) | \
                     (torch.abs(pitch) > pitch_threshold_rad) | \
                     (torch.abs(yaw) > yaw_threshold_rad)

    # # Print values that crossed the threshold for environments being reset
    # if reset_flags.any():
    #     reset_env_indices = torch.where(reset_flags)[0]
    #     for idx in reset_env_indices:
    #         roll_deg = torch.rad2deg(roll[idx]).item()
    #         pitch_deg = torch.rad2deg(pitch[idx]).item()
    #         yaw_deg = torch.rad2deg(yaw[idx]).item()
    #         exceeded = []
    #         if torch.abs(roll[idx]) > roll_threshold_rad:
    #             exceeded.append(f"roll={roll_deg:.2f}° (threshold={roll_threshold_deg}°)")
    #         if torch.abs(pitch[idx]) > pitch_threshold_rad:
    #             exceeded.append(f"pitch={pitch_deg:.2f}° (threshold={pitch_threshold_deg}°)")
    #         if torch.abs(yaw[idx]) > yaw_threshold_rad:
    #             exceeded.append(f"yaw={yaw_deg:.2f}° (threshold={yaw_threshold_deg}°)")
    #         print(f"Env {idx.item()} reset due to orientation threshold exceeded: {', '.join(exceeded)}")
    #         input("Press Enter to continue...")

    return reset_flags

