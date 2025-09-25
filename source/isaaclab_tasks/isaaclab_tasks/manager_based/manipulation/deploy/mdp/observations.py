# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaacsim.core.utils.torch.transformations import tf_combine

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gear_shaft_pos_w(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base")
) -> torch.Tensor:
    """Gear shaft position in world frame with offset applied.
    
    Args:
        env: The environment containing the assets
        asset_cfg: Configuration of the gear base asset
        
    Returns:
        Gear shaft position tensor of shape (num_envs, 3)
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    
    # get base gear position and orientation
    base_pos = asset.data.root_pos_w
    base_quat = asset.data.root_quat_w
    

    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    # get offset for the specific gear type
    gear_offsets = getattr(env.cfg, 'gear_offsets', {})
    
    # Initialize shaft positions
    shaft_pos = torch.zeros_like(base_pos)
    
    # Apply offsets for each environment based on their gear type
    for i in range(env.num_envs):
        # gear_type = current_gear_type[i] if i < len(current_gear_type) else "gear_medium"
        gear_type = current_gear_type[i]
        
        offset = torch.tensor(gear_offsets[gear_type], device=base_pos.device, dtype=base_pos.dtype)
        offset_pos = offset.unsqueeze(0)
        shaft_pos[i] = base_pos[i] + offset_pos[0]
        # Apply position and rotation offsets if provided
        # create identity quaternion
        rot_offset_tensor = torch.tensor([1.0, 0.0, 0.0, 0.0], device=base_pos.device, dtype=base_pos.dtype).unsqueeze(0)
        _, shaft_pos[i] = tf_combine(
            base_quat[i:i+1], base_pos[i:i+1],
            rot_offset_tensor, offset_pos
        )

    return shaft_pos - env.scene.env_origins


def gear_shaft_quat_w(
    env: ManagerBasedRLEnv, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("factory_gear_base"),
) -> torch.Tensor:
    """Gear shaft orientation in world frame.
    
    Args:
        env: The environment containing the assets
        asset_cfg: Configuration of the gear base asset
        gear_type: Type of gear ('gear_small', 'gear_medium', 'gear_large')
        
    Returns:
        Gear shaft orientation tensor of shape (num_envs, 4)
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    
    # get base quaternion
    base_quat = asset.data.root_quat_w
    
    # ensure w component is positive for each environment
    # if w is negative, negate the entire quaternion to maintain same orientation
    w_negative = base_quat[:, 0] < 0
    positive_quat = base_quat.clone()
    positive_quat[w_negative] = -base_quat[w_negative]
    
    return positive_quat

def gear_pos_w(
    env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    """Gear position in world frame.
    
    Args:
        env: The environment containing the assets

    Returns:
        Gear position tensor of shape (num_envs, 3)
    """
    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    # Create a mapping from gear type to asset name
    gear_to_asset_mapping = {
        'gear_small': 'factory_gear_small',
        'gear_medium': 'factory_gear_medium', 
        'gear_large': 'factory_gear_large'
    }
    
    # Initialize positions array
    gear_positions = torch.zeros((env.num_envs, 3), device=env.device)
    
    # Get positions for each environment based on their gear type
    for i in range(env.num_envs):
        gear_type = current_gear_type[i]
        asset_name = gear_to_asset_mapping.get(gear_type)
        
        asset: RigidObject = env.scene[asset_name]
        
        # Get position for this specific environment
        gear_positions[i] = asset.data.root_pos_w[i]
    
    return gear_positions

def gear_quat_w(
    env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    """Gear orientation in world frame.
    
    Args:
        env: The environment containing the assets
        
    Returns:
        Gear orientation tensor of shape (num_envs, 4)
    """
    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    # Create a mapping from gear type to asset name
    gear_to_asset_mapping = {
        'gear_small': 'factory_gear_small',
        'gear_medium': 'factory_gear_medium', 
        'gear_large': 'factory_gear_large'
    }
    
    # Initialize quaternions array
    gear_positive_quat = torch.zeros((env.num_envs, 4), device=env.device)
    
    # Get quaternions for each environment based on their gear type
    for i in range(env.num_envs):
        gear_type = current_gear_type[i]
        asset_name = gear_to_asset_mapping.get(gear_type)
        
        # Get the asset for this specific gear type
        asset: RigidObject = env.scene[asset_name]
        
        # Get quaternion for this specific environment
        gear_quat = asset.data.root_quat_w[i]
        
        # ensure w component is positive for each environment
        # if w is negative, negate the entire quaternion to maintain same orientation
        if gear_quat[0] < 0:
            gear_quat = -gear_quat
        
        gear_positive_quat[i] = gear_quat
    
    return gear_positive_quat