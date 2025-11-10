# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaacsim.core.utils.torch.transformations import tf_combine

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils.math import quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_keypoint_offsets_full_6d(add_cube_center_kp: bool = False, device: torch.device | None = None) -> torch.Tensor:
    """Get keypoints for pose alignment comparison. Pose is aligned if all axis are aligned.

    Args:
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)
        device: Device to create the tensor on

    Returns:
        Keypoint offsets tensor of shape (num_keypoints, 3)
    """
    if add_cube_center_kp:
        keypoint_corners = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    else:
        keypoint_corners = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    keypoint_corners = torch.tensor(keypoint_corners, device=device, dtype=torch.float32)
    keypoint_corners = torch.cat((keypoint_corners, -keypoint_corners[-3:]), dim=0)  # use both negative and positive

    return keypoint_corners


def compute_keypoint_distance(
    current_pos: torch.Tensor,
    current_quat: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat: torch.Tensor,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Compute keypoint distance between current and target poses.

    This function creates keypoints from the current and target poses and calculates
    the L2 norm distance between corresponding keypoints. The keypoints are created
    by applying offsets to the poses and transforming them to world coordinates.

    Args:
        current_pos: Current position tensor of shape (num_envs, 3)
        current_quat: Current quaternion tensor of shape (num_envs, 4)
        target_pos: Target position tensor of shape (num_envs, 3)
        target_quat: Target quaternion tensor of shape (num_envs, 4)
        keypoint_scale: Scale factor for keypoint offsets
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)
        device: Device to create tensors on

    Returns:
        Keypoint distance tensor of shape (num_envs, num_keypoints) where each element
        is the L2 norm distance between corresponding keypoints
    """
    if device is None:
        device = current_pos.device

    num_envs = current_pos.shape[0]

    # Get keypoint offsets
    keypoint_offsets = get_keypoint_offsets_full_6d(add_cube_center_kp, device)
    keypoint_offsets = keypoint_offsets * keypoint_scale
    num_keypoints = keypoint_offsets.shape[0]

    # Expand keypoints for all environments: (num_envs, num_keypoints, 3)
    keypoint_offsets_batch = keypoint_offsets.unsqueeze(0).expand(num_envs, -1, -1)
    
    # Flatten for batch processing: (num_envs * num_keypoints, 3)
    keypoint_offsets_flat = keypoint_offsets_batch.reshape(-1, 3)
    
    # Expand quaternions and positions for all keypoints
    current_quat_expanded = current_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4)
    current_pos_expanded = current_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3)
    target_quat_expanded = target_quat.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 4)
    target_pos_expanded = target_pos.unsqueeze(1).expand(-1, num_keypoints, -1).reshape(-1, 3)
    
    # Create identity quaternion for all transforms
    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).unsqueeze(0).expand(num_envs * num_keypoints, -1)
    
    # Transform all keypoints at once
    _, keypoints_current_flat = tf_combine(current_quat_expanded, current_pos_expanded, identity_quat, keypoint_offsets_flat)
    _, keypoints_target_flat = tf_combine(target_quat_expanded, target_pos_expanded, identity_quat, keypoint_offsets_flat)
    
    # Reshape back to (num_envs, num_keypoints, 3)
    keypoints_current = keypoints_current_flat.reshape(num_envs, num_keypoints, 3)
    keypoints_target = keypoints_target_flat.reshape(num_envs, num_keypoints, 3)
    
    # Calculate L2 norm distance between corresponding keypoints
    keypoint_dist_sep = torch.norm(keypoints_target - keypoints_current, p=2, dim=-1)

    return keypoint_dist_sep


def keypoint_command_error(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Compute keypoint distance between current and desired poses from command.

    The function computes the keypoint distance between the current pose of the end effector from
    the frame transformer sensor and the desired pose from the command. Keypoints are created by
    applying offsets to both poses and the distance is computed as the L2-norm between corresponding keypoints.

    Args:
        env: The environment containing the asset
        command_name: Name of the command containing desired pose
        asset_cfg: Configuration of the asset to track (not used, kept for compatibility)
        keypoint_scale: Scale factor for keypoint offsets
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)

    Returns:
        Keypoint distance tensor of shape (num_envs, num_keypoints) where each element
        is the L2 norm distance between corresponding keypoints
    """
    # extract the frame transformer sensor
    asset: FrameTransformer = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired pose from command (position and orientation)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]

    # transform desired pose to world frame using source frame from frame transformer
    des_pos_w = des_pos_b
    des_quat_w = des_quat_b

    # get current pose in world frame from frame transformer (end effector pose)
    curr_pos_w = asset.data.target_pos_source[:, 0]  # First target frame is end_effector
    curr_quat_w = asset.data.target_quat_source[:, 0]  # First target frame is end_effector

    # compute keypoint distance
    keypoint_dist_sep = compute_keypoint_distance(
        current_pos=curr_pos_w,
        current_quat=curr_quat_w,
        target_pos=des_pos_w,
        target_quat=des_quat_w,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        device=curr_pos_w.device,
    )

    # Return mean distance across keypoints to match expected reward shape (num_envs,)
    return keypoint_dist_sep.mean(-1)


def keypoint_command_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
    kp_use_sum_of_exps: bool = True,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Compute exponential keypoint reward between current and desired poses from command.

    The function computes the keypoint distance between the current pose of the end effector from
    the frame transformer sensor and the desired pose from the command, then applies an exponential
    reward function. The reward is computed using the formula: 1 / (exp(a * distance) + b + exp(-a * distance))
    where a and b are coefficients.

    Args:
        env: The environment containing the asset
        command_name: Name of the command containing desired pose
        asset_cfg: Configuration of the asset to track (not used, kept for compatibility)
        kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
        kp_use_sum_of_exps: Whether to use sum of exponentials (True) or single exponential (False)
        keypoint_scale: Scale factor for keypoint offsets
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)

    Returns:
        Exponential keypoint reward tensor of shape (num_envs,) where each element
        is the exponential reward value
    """
    # extract the frame transformer sensor
    asset: FrameTransformer = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # obtain the desired pose from command (position and orientation)
    des_pos_b = command[:, :3]
    des_quat_b = command[:, 3:7]

    # transform desired pose to world frame using source frame from frame transformer
    des_pos_w = des_pos_b
    des_quat_w = des_quat_b

    # get current pose in world frame from frame transformer (end effector pose)
    curr_pos_w = asset.data.target_pos_source[:, 0]  # First target frame is end_effector
    curr_quat_w = asset.data.target_quat_source[:, 0]  # First target frame is end_effector

    # compute keypoint distance
    keypoint_dist_sep = compute_keypoint_distance(
        current_pos=curr_pos_w,
        current_quat=curr_quat_w,
        target_pos=des_pos_w,
        target_quat=des_quat_w,
        keypoint_scale=keypoint_scale,
        add_cube_center_kp=add_cube_center_kp,
        device=curr_pos_w.device,
    )

    # compute exponential reward
    keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])  # shape: (num_envs,)

    if kp_use_sum_of_exps:
        # Use sum of exponentials: average across keypoints for each coefficient
        for coeff in kp_exp_coeffs:
            a, b = coeff
            keypoint_reward_exp += (
                1.0 / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))
            ).mean(-1)
    else:
        # Use single exponential: average keypoint distance first, then apply exponential
        keypoint_dist = keypoint_dist_sep.mean(-1)  # shape: (num_envs,)
        for coeff in kp_exp_coeffs:
            a, b = coeff
            keypoint_reward_exp += 1.0 / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))

    return keypoint_reward_exp

def keypoint_entity_error(
    env: ManagerBasedRLEnv, 
    asset_cfg_1: SceneEntityCfg,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Compute keypoint distance between a RigidObject and the dynamically selected gear.

    The function computes the keypoint distance between the poses of two entities.
    Keypoints are created by applying offsets to both poses and the distance is 
    computed as the L2-norm between corresponding keypoints.

    Args:
        env: The environment containing the assets
        asset_cfg_1: Configuration of the first asset (RigidObject)
        keypoint_scale: Scale factor for keypoint offsets
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)

    Returns:
        Keypoint distance tensor of shape (num_envs,) where each element
        is the mean L2 norm distance between corresponding keypoints
    """
    # extract the assets (to enable type hinting)
    asset_1 = env.scene[asset_cfg_1.name]  # RigidObject
    
    # get current poses in world frame for asset_1
    # asset_1 is RigidObject - use body_pos_w and body_quat_w
    curr_pos_1 = asset_1.data.body_pos_w[:, 0]  # type: ignore
    curr_quat_1 = asset_1.data.body_quat_w[:, 0]  # type: ignore
    
    if hasattr(env, '_current_gear_type'):
        current_gear_type = env._current_gear_type  # type: ignore
        
        # Stack all gear positions and quaternions
        all_gear_pos = torch.stack([
            env.scene["factory_gear_small"].data.body_pos_w[:, 0],
            env.scene["factory_gear_medium"].data.body_pos_w[:, 0],
            env.scene["factory_gear_large"].data.body_pos_w[:, 0],
        ], dim=1)  # (num_envs, 3, 3)
        
        all_gear_quat = torch.stack([
            env.scene["factory_gear_small"].data.body_quat_w[:, 0],
            env.scene["factory_gear_medium"].data.body_quat_w[:, 0],
            env.scene["factory_gear_large"].data.body_quat_w[:, 0],
        ], dim=1)  # (num_envs, 3, 4)
        
        # Convert gear types to indices
        gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        gear_type_indices = torch.tensor(
            [gear_type_map.get(current_gear_type[i], 1) for i in range(env.num_envs)],
            device=curr_pos_1.device,
            dtype=torch.long
        )
        
        # Select positions and quaternions using advanced indexing
        env_indices = torch.arange(env.num_envs, device=curr_pos_1.device)
        curr_pos_2_all = all_gear_pos[env_indices, gear_type_indices]
        curr_quat_2_all = all_gear_quat[env_indices, gear_type_indices]
        
        # Compute keypoint distance for all environments at once
        keypoint_dist_sep = compute_keypoint_distance(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2_all,
            target_quat=curr_quat_2_all,
            keypoint_scale=keypoint_scale,
            add_cube_center_kp=add_cube_center_kp,
            device=curr_pos_1.device
        )
    else:
        # Fallback to factory_gear_medium if _current_gear_type is not available
        print("WARNING: env._current_gear_type is not available, using factory_gear_medium as fallback")
        asset_2 = env.scene['factory_gear_medium']
        
        # get current poses in world frame for asset_2
        curr_pos_2 = asset_2.data.body_pos_w[:, 0]  # type: ignore
        curr_quat_2 = asset_2.data.body_quat_w[:, 0]  # type: ignore
        
        # compute keypoint distance
        keypoint_dist_sep = compute_keypoint_distance(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
            add_cube_center_kp=add_cube_center_kp,
            device=curr_pos_1.device
        )
    
    # Return mean distance across keypoints to match expected reward shape (num_envs,)
    return keypoint_dist_sep.mean(-1)


def keypoint_entity_error_exp(
    env: ManagerBasedRLEnv, 
    asset_cfg_1: SceneEntityCfg,
    kp_exp_coeffs: list[tuple[float, float]] = [(1.0, 0.1)],
    kp_use_sum_of_exps: bool = True,
    keypoint_scale: float = 1.0,
    add_cube_center_kp: bool = True,
) -> torch.Tensor:
    """Compute exponential keypoint reward between a RigidObject and the dynamically selected gear.

    The function computes the keypoint distance between the poses of two entities,
    then applies an exponential reward function. The reward is computed using the 
    formula: 1 / (exp(a * distance) + b + exp(-a * distance)) where a and b are coefficients.

    Args:
        env: The environment containing the assets
        asset_cfg_1: Configuration of the first asset (RigidObject)
        kp_exp_coeffs: List of (a, b) coefficient pairs for exponential reward
        kp_use_sum_of_exps: Whether to use sum of exponentials (True) or single exponential (False)
        keypoint_scale: Scale factor for keypoint offsets
        add_cube_center_kp: Whether to include the center keypoint (0, 0, 0)

    Returns:
        Exponential keypoint reward tensor of shape (num_envs,) where each element
        is the exponential reward value
    """
    # extract the assets (to enable type hinting)
    asset_1 = env.scene[asset_cfg_1.name]  # RigidObject
    
    # get current poses in world frame for asset_1
    # asset_1 is RigidObject - use body_pos_w and body_quat_w
    curr_pos_1 = asset_1.data.body_pos_w[:, 0]  # type: ignore
    curr_quat_1 = asset_1.data.body_quat_w[:, 0]  # type: ignore

    if hasattr(env, '_current_gear_type'):
        current_gear_type = env._current_gear_type  # type: ignore
        
        # Stack all gear positions and quaternions
        all_gear_pos = torch.stack([
            env.scene["factory_gear_small"].data.body_pos_w[:, 0],
            env.scene["factory_gear_medium"].data.body_pos_w[:, 0],
            env.scene["factory_gear_large"].data.body_pos_w[:, 0],
        ], dim=1)  # (num_envs, 3, 3)
        
        all_gear_quat = torch.stack([
            env.scene["factory_gear_small"].data.body_quat_w[:, 0],
            env.scene["factory_gear_medium"].data.body_quat_w[:, 0],
            env.scene["factory_gear_large"].data.body_quat_w[:, 0],
        ], dim=1)  # (num_envs, 3, 4)
        
        # Convert gear types to indices
        gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        gear_type_indices = torch.tensor(
            [gear_type_map.get(current_gear_type[i], 1) for i in range(env.num_envs)],
            device=curr_pos_1.device,
            dtype=torch.long
        )
        
        # Select positions and quaternions using advanced indexing
        env_indices = torch.arange(env.num_envs, device=curr_pos_1.device)
        curr_pos_2_all = all_gear_pos[env_indices, gear_type_indices]
        curr_quat_2_all = all_gear_quat[env_indices, gear_type_indices]
        
        # Compute keypoint distance for all environments at once
        keypoint_dist_sep = compute_keypoint_distance(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2_all,
            target_quat=curr_quat_2_all,
            keypoint_scale=keypoint_scale,
            add_cube_center_kp=add_cube_center_kp,
            device=curr_pos_1.device
        )
        
        # compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])
        
        if kp_use_sum_of_exps:
            # Use sum of exponentials: average across keypoints for each coefficient
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (1. / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))).mean(-1)
        else:
            # Use single exponential: average keypoint distance first, then apply exponential
            keypoint_dist = keypoint_dist_sep.mean(-1)  # shape: (num_envs,)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1. / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))
    else:
        # Fallback to factory_gear_medium if _current_gear_type is not available
        print("WARNING: env._current_gear_type is not available, using factory_gear_medium as fallback")
        asset_2 = env.scene['factory_gear_medium']
        
        # get current poses in world frame for asset_2
        curr_pos_2 = asset_2.data.body_pos_w[:, 0]  # type: ignore
        curr_quat_2 = asset_2.data.body_quat_w[:, 0]  # type: ignore
        
        # compute keypoint distance
        keypoint_dist_sep = compute_keypoint_distance(
            current_pos=curr_pos_1,
            current_quat=curr_quat_1,
            target_pos=curr_pos_2,
            target_quat=curr_quat_2,
            keypoint_scale=keypoint_scale,
            add_cube_center_kp=add_cube_center_kp,
            device=curr_pos_1.device
        )
        
        # compute exponential reward
        keypoint_reward_exp = torch.zeros_like(keypoint_dist_sep[:, 0])
        
        if kp_use_sum_of_exps:
            # Use sum of exponentials: average across keypoints for each coefficient
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += (1. / (torch.exp(a * keypoint_dist_sep) + b + torch.exp(-a * keypoint_dist_sep))).mean(-1)
        else:
            # Use single exponential: average keypoint distance first, then apply exponential
            keypoint_dist = keypoint_dist_sep.mean(-1)  # shape: (num_envs,)
            for coeff in kp_exp_coeffs:
                a, b = coeff
                keypoint_reward_exp += 1. / (torch.exp(a * keypoint_dist) + b + torch.exp(-a * keypoint_dist))
    
    return keypoint_reward_exp
