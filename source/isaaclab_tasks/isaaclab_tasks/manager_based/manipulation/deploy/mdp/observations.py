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
    
    # Access shared cache (initialized during prestartup events)
    cache = env._shared_gear_cache
    
    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    # Update offsets using cached gear offset tensors from shared cache
    offsets = cache['offsets_buffer']
    for i in range(env.num_envs):
        offsets[i] = cache['gear_offset_tensors'][current_gear_type[i]]
    
    # Use cached identity_quat
    identity_quat = cache['identity_quat']
    
    _, shaft_pos = tf_combine(base_quat, base_pos, identity_quat, offsets)

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
    # Use shared cache (must be initialized by initialize_shared_gear_cache event)
    if not hasattr(env, '_shared_gear_cache'):
        raise RuntimeError(
            "Shared gear cache not initialized. Ensure 'initialize_shared_gear_cache' is called "
            "during startup or reset events before this observation function."
        )
    
    cache = env._shared_gear_cache
    
    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    all_gear_positions = torch.stack([
        env.scene["factory_gear_small"].data.root_pos_w,
        env.scene["factory_gear_medium"].data.root_pos_w,
        env.scene["factory_gear_large"].data.root_pos_w,
    ], dim=1)  # Shape: (num_envs, 3, 3)
    
    # Update gear_type_indices using cached tensor
    gear_type_indices = cache['gear_type_indices']
    gear_type_map = cache['gear_type_map']
    for i in range(env.num_envs):
        gear_type_indices[i] = gear_type_map[current_gear_type[i]]
    
    gear_positions = all_gear_positions[cache['env_indices'], gear_type_indices]
    
    return gear_positions - env.scene.env_origins

def gear_quat_w(
    env: ManagerBasedRLEnv, 
) -> torch.Tensor:
    """Gear orientation in world frame.
    
    Args:
        env: The environment containing the assets
        
    Returns:
        Gear orientation tensor of shape (num_envs, 4)
    """
    # Use shared cache (must be initialized by initialize_shared_gear_cache event)
    if not hasattr(env, '_shared_gear_cache'):
        raise RuntimeError(
            "Shared gear cache not initialized. Ensure 'initialize_shared_gear_cache' is called "
            "during startup or reset events before this observation function."
        )
    
    cache = env._shared_gear_cache
    
    # get current gear type from environment, use default if not set
    if not hasattr(env, '_current_gear_type'):
        print("Environment does not have attribute '_current_gear_type'. Using default_gear_type from configuration.")
        default_gear_type = getattr(env.cfg, 'default_gear_type', 'gear_medium')
        current_gear_type = [default_gear_type] * env.num_envs
    else:
        current_gear_type = env._current_gear_type  # type: ignore
    
    all_gear_quat = torch.stack([
        env.scene["factory_gear_small"].data.root_quat_w,
        env.scene["factory_gear_medium"].data.root_quat_w,
        env.scene["factory_gear_large"].data.root_quat_w,
    ], dim=1)  # Shape: (num_envs, 3, 4)
    
    # Update gear_type_indices using cached tensor
    gear_type_indices = cache['gear_type_indices']
    gear_type_map = cache['gear_type_map']
    for i in range(env.num_envs):
        gear_type_indices[i] = gear_type_map[current_gear_type[i]]
    
    gear_quat = all_gear_quat[cache['env_indices'], gear_type_indices]
    
    w_negative = gear_quat[:, 0] < 0
    gear_positive_quat = gear_quat.clone()
    gear_positive_quat[w_negative] = -gear_quat[w_negative]
    
    return gear_positive_quat