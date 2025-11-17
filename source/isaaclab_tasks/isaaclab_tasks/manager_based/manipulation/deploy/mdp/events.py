# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""


from __future__ import annotations

import carb
import torch
from typing import TYPE_CHECKING, Literal, Optional, List

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import sample_uniform
import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils

from isaaclab_tasks.direct.automate import factory_control as fc
import random

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def set_default_joint_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    default_pose: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    # Set the default pose for robots in all envs
    asset = env.scene[asset_cfg.name]
    asset.data.default_joint_pos = torch.tensor(default_pose, device=env.device).repeat(env.num_envs, 1)

def randomize_gear_type(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    gear_types: List[str] = ["gear_small", "gear_medium", "gear_large"]
):
    """Randomize the gear type being used for the specified environments.
    
    Args:
        env: The environment containing the assets
        env_ids: Environment IDs to randomize
        gear_types: List of available gear types to choose from
    """
    # Randomly select gear type for each environment
    selected_gears = [random.choice(gear_types) for _ in range(len(env_ids))]
    
    # Store the selected gear type in the environment instance
    # This will be used by the observation functions
    if not hasattr(env, '_current_gear_type'):
        env._current_gear_type = ["gear_medium"] * env.num_envs
    
    for i, env_id in enumerate(env_ids):
        env._current_gear_type[env_id] = selected_gears[i]

def set_robot_to_grasp_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_threshold: float = 1e-6,
    rot_threshold: float = 1e-6,
    max_iterations: int = 10,
    pos_randomization_range: Optional[dict] = None,
):
    """Set robot to grasp pose using robot-specific parameters from env config.
    
    Robot-specific parameters are retrieved from env.cfg (all required):
    - end_effector_body_name: Name of the end effector body for IK
    - num_arm_joints: Number of arm joints (excluding gripper)
    - grasp_rot_offset: Rotation offset for grasp pose (quaternion [w, x, y, z])
    - gripper_joint_setter_func: Function to set gripper joint positions
    """
    # Get robot-specific parameters from environment config (all required - no defaults)
    if not hasattr(env.cfg, 'end_effector_body_name'):
        raise ValueError(
            "Robot-specific parameter 'end_effector_body_name' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.end_effector_body_name = 'wrist_3_link'"
        )
    if not hasattr(env.cfg, 'num_arm_joints'):
        raise ValueError(
            "Robot-specific parameter 'num_arm_joints' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.num_arm_joints = 6"
        )
    if not hasattr(env.cfg, 'grasp_rot_offset'):
        raise ValueError(
            "Robot-specific parameter 'grasp_rot_offset' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.grasp_rot_offset = [0.0, 0.707, 0.707, 0.0]"
        )
    if not hasattr(env.cfg, 'gripper_joint_setter_func'):
        raise ValueError(
            "Robot-specific parameter 'gripper_joint_setter_func' not found in env.cfg. "
            "Please define this parameter in your robot-specific configuration file. "
            "Example: self.gripper_joint_setter_func = set_finger_joint_pos_custom"
        )

    end_effector_body_name = env.cfg.end_effector_body_name
    num_arm_joints = env.cfg.num_arm_joints
    grasp_rot_offset = env.cfg.grasp_rot_offset
    gripper_joint_setter_func = env.cfg.gripper_joint_setter_func

    # Convert rotation offset to tensor if provided
    if grasp_rot_offset is not None:
        grasp_rot_offset_tensor = torch.tensor(grasp_rot_offset, device=env.device).unsqueeze(0).expand(len(env_ids), -1)
    else:
        grasp_rot_offset_tensor = None
    
    # Check if environment has gear type selection
    if not hasattr(env, '_current_gear_type'):
        raise ValueError("Environment does not have '_current_gear_type' attribute. Ensure randomize_gear_type event is configured.")

    i = 0
    robot_asset: Articulation = env.scene[robot_asset_cfg.name]
    joint_pos = robot_asset.data.joint_pos[env_ids].clone()
    joint_vel = robot_asset.data.joint_vel[env_ids].clone()

    while i < max_iterations:
        robot_asset: Articulation = env.scene[robot_asset_cfg.name]
        # Add gaussian noise to joint states
        joint_pos = robot_asset.data.joint_pos[env_ids].clone()
        joint_vel = robot_asset.data.joint_vel[env_ids].clone()

        # OPTIMIZED: Vectorized gear data fetching
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        
        # Stack all gear positions and quaternions
        all_gear_pos = torch.stack([
            env.scene["factory_gear_small"].data.root_link_pos_w,
            env.scene["factory_gear_medium"].data.root_link_pos_w,
            env.scene["factory_gear_large"].data.root_link_pos_w,
        ], dim=1)[env_ids]  # (len(env_ids), 3, 3)
        
        all_gear_quat = torch.stack([
            env.scene["factory_gear_small"].data.root_link_quat_w,
            env.scene["factory_gear_medium"].data.root_link_quat_w,
            env.scene["factory_gear_large"].data.root_link_quat_w,
        ], dim=1)[env_ids]  # (len(env_ids), 3, 4)
        
        # Convert gear types to indices for the resetting environments
        gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        gear_type_indices = torch.tensor(
            [gear_type_map[env._current_gear_type[env_id]] for env_id in env_ids_list],
            device=env.device,
            dtype=torch.long
        )
        
        # Select gear data using advanced indexing
        local_env_indices = torch.arange(len(env_ids), device=env.device)
        grasp_object_pos_world = all_gear_pos[local_env_indices, gear_type_indices]
        grasp_object_quat = all_gear_quat[local_env_indices, gear_type_indices]
        
        # First apply rotation offset to get the object's orientation
        if grasp_rot_offset_tensor is not None:
            # Apply rotation offset by quaternion multiplication
            # grasp_rot_offset is assumed to be in quaternion format (w, x, y, z)
            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, grasp_rot_offset_tensor)
        
        # OPTIMIZED: Vectorized grasp offsets application
        # Get grasp offsets for all environments
        gear_grasp_offsets = torch.stack([
            torch.tensor(env.cfg.gear_offsets_grasp[env._current_gear_type[env_id]], 
                         device=env.device, dtype=grasp_object_pos_world.dtype)
            for env_id in env_ids_list
        ])  # (len(env_ids), 3)
        
        if pos_randomization_range is not None:
            pos_keys = ["x", "y", "z"]
            range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
            ranges_pos = torch.tensor(range_list_pos, device=env.device)
            rand_pos_offsets = math_utils.sample_uniform(ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device)
            gear_grasp_offsets += rand_pos_offsets
        
        # Transform all offsets from gear frame to world frame (vectorized)
        grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(
            grasp_object_quat, gear_grasp_offsets
        )
        
        # Convert to environment-relative coordinates by subtracting environment origins
        grasp_object_pos = grasp_object_pos_world
        
        # Get end effector pose of the robot using robot-specific body name
        try:
            # Find the index of the end effector body
            eef_indices, eef_names = robot_asset.find_bodies([end_effector_body_name])
            eef_idx = eef_indices[0] if len(eef_indices) > 0 else None
            
            if len(eef_indices) == 1:
                eef_idx = eef_indices[0]  # Get the first (and should be only) index
                
                # Get the specific end effector body pose
                eef_pos_world = robot_asset.data.body_link_pos_w[env_ids, eef_idx]  # Shape: (len(env_ids), 3)
                eef_quat = robot_asset.data.body_link_quat_w[env_ids, eef_idx]  # Shape: (len(env_ids), 4)
                
                # Convert to environment-relative coordinates
                eef_pos = eef_pos_world

                # You can also get the full pose as [pos, quat] (7-dimensional)
                eef_pos = robot_asset.data.body_pos_w[env_ids, eef_idx]
                eef_quat = robot_asset.data.body_quat_w[env_ids, eef_idx]

            elif len(eef_indices) > 1:
                print(f"{end_effector_body_name} found multiple times in robot body names")
                print(f"Available body names: {robot_asset.body_names}")
            else:
                print(f"{end_effector_body_name} not found in robot body names")
                print(f"Available body names: {robot_asset.body_names}")
                
        except Exception as e:
            print(f"Could not get end effector pose: {e}")
            print("You may need to adjust the body name or access method based on your robot configuration")

        # Compute error to target using end effector as current and grasp_object as target
        if len(eef_indices) > 0:
            # Get current end effector pose
            current_eef_pos = eef_pos  # end effector position
            current_eef_quat = eef_quat  # end effector orientation
            
            # Get target pose (grasp object)
            target_eef_pos = grasp_object_pos  # grasp object position
            target_eef_quat = grasp_object_quat  # grasp object orientation

            # Compute pose error
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=current_eef_pos,
                fingertip_midpoint_quat=current_eef_quat,
                ctrl_target_fingertip_midpoint_pos=target_eef_pos,
                ctrl_target_fingertip_midpoint_quat=target_eef_quat,
                jacobian_type='geometric',
                rot_error_type='axis_angle')
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
            # Check if pose error is within threshold
            pos_error_norm = torch.norm(pos_error, dim=-1)
            rot_error_norm = torch.norm(axis_angle_error, dim=-1)
            
            
            # break from IK loop if all environments have converged
            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                break

            # Solve DLS problem for inverse kinematics
            # Get jacobian for the end effector body using the same approach as task_space_actions.py
            try:
                # Get all jacobians from the robot
                jacobians = robot_asset.root_physx_view.get_jacobians().clone()

                # Select the jacobian for the end effector body
                # For fixed-base robots, Jacobian excludes the base body (index 0)
                # So if eef_idx is 6, the Jacobian index should be 5
                jacobi_body_idx = eef_idx - 1
                
                
                jacobian = jacobians[env_ids, jacobi_body_idx, :, :]  # Only first 6 joints (arm, not gripper)
                
                delta_dof_pos = fc._get_delta_dof_pos(
                    delta_pose=delta_hand_pose,
                    ik_method="dls",
                    jacobian=jacobian,
                    device=env.device,
                )
                
                # Update joint positions - only update arm joints (first 6)
                joint_pos += delta_dof_pos
                joint_vel = torch.zeros_like(joint_pos)

                # Set into the physics simulation

                
            except Exception as e:
                print(f"Error in IK computation: {e}")
                print("Note: You may need to implement proper jacobian computation for your robot")

        # Set into the physics simulation
        robot_asset.set_joint_position_target(joint_pos, env_ids=env_ids)
        robot_asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)
        robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        i += 1

    # Close the gripper by setting the finger_joint based on gear type
    all_joints, all_joints_names = robot_asset.find_joints([".*"])
    
    joint_pos = robot_asset.data.joint_pos[env_ids].clone()
    
    # Get finger joints (all joints after arm joints)
    finger_joints = all_joints[num_arm_joints:]
    finger_joint_names = all_joints_names[num_arm_joints:]



    # Set gripper joint positions using robot-specific setter function
    for row_idx, env_id in enumerate(env_ids_list):
        gear_key = env._current_gear_type[env_id]
        hand_grasp_width = env.cfg.hand_grasp_width[gear_key]
        gripper_joint_setter_func(joint_pos, [row_idx], finger_joints, hand_grasp_width)

    robot_asset.set_joint_position_target(joint_pos, joint_ids=all_joints, env_ids=env_ids)
    robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    
    # for i in range(5):
    #     env.sim.render()
    #     input("Press Enter to continue...")

    # Set finger joints to closed position based on gear type for each environment
    for row_idx, env_id in enumerate(env_ids_list):
        gear_key = env._current_gear_type[env_id]
        hand_close_width = env.cfg.hand_close_width[gear_key]
        gripper_joint_setter_func(joint_pos, [row_idx], finger_joints, hand_close_width)

    robot_asset.set_joint_position_target(joint_pos, joint_ids=all_joints, env_ids=env_ids)
    # for i in range(5):
    #     env.sim.render()
    #     input("Press Enter to continue...")

def randomize_gears_and_base_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict = {},
    velocity_range: dict = {},
    gear_pos_range: dict = {},
    rot_randomization_range: dict = {},

):
    """Randomize both the gear base pose and the poses of all gear types with the same value,
    then apply the per-env `gear_pos_range` only to the gear selected by `randomize_gear_type`.
    
    OPTIMIZED: Vectorized approach using masks instead of Python loops over environments.
    """
    if not hasattr(env, '_current_gear_type'):
        raise ValueError("Environment does not have '_current_gear_type' attribute. Ensure randomize_gear_type event is configured.")

    device = env.device

    # Shared pose samples for all assets (base and all gears)
    pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
    range_list_pose = [pose_range.get(key, (0.0, 0.0)) for key in pose_keys]
    ranges_pose = torch.tensor(range_list_pose, device=device)
    rand_pose_samples = math_utils.sample_uniform(ranges_pose[:, 0], ranges_pose[:, 1], (len(env_ids), 6), device=device)

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
    )

    # Shared velocity samples for all assets (base and all gears)
    range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in pose_keys]
    ranges_vel = torch.tensor(range_list_vel, device=device)
    rand_vel_samples = math_utils.sample_uniform(ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=device)

    # Prepare assets: base + all possible gears (only those present in scene will be processed)
    base_asset_name = "factory_gear_base"
    possible_gear_assets = [
        "factory_gear_small",
        "factory_gear_medium",
        "factory_gear_large",
        # "table",
    ]


    positions_by_asset = {}
    orientations_by_asset = {}
    velocities_by_asset = {}

    # Combine base and gear assets into one pass to avoid duplication
    asset_names_to_process = [base_asset_name] + possible_gear_assets
    for asset_name in asset_names_to_process:
        asset: RigidObject | Articulation = env.scene[asset_name]
        root_states = asset.data.default_root_state[env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        velocities = root_states[:, 7:13] + rand_vel_samples
        positions_by_asset[asset_name] = positions
        orientations_by_asset[asset_name] = orientations
        velocities_by_asset[asset_name] = velocities

    # Per-env gear offset (gear_pos_range) applied only to the selected gear
    range_list_gear = [gear_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges_gear = torch.tensor(range_list_gear, device=device)
    rand_gear_offsets = math_utils.sample_uniform(ranges_gear[:, 0], ranges_gear[:, 1], (len(env_ids), 3), device=device)

    # Per-env gear orientation offset (rot_randomization_range) applied only to the selected gear
    range_list_rot = [rot_randomization_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges_rot = torch.tensor(range_list_rot, device=device)
    rand_rot_offsets = math_utils.sample_uniform(ranges_rot[:, 0], ranges_rot[:, 1], (len(env_ids), 3), device=device)
    rand_rot_quats = math_utils.quat_from_euler_xyz(rand_rot_offsets[:, 0], rand_rot_offsets[:, 1], rand_rot_offsets[:, 2])

    # OPTIMIZED: Create masks for each gear type instead of Python loop
    env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)

    # Create gear type mapping for vectorization
    gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
    gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]

    # Convert current gear types to indices (single list comprehension)
    gear_type_indices = torch.tensor(
        [gear_type_map[env._current_gear_type[env_id]] for env_id in env_ids_list],
        device=device,
        dtype=torch.long
    )

    # Apply offsets using vectorized operations with masks (no Python loop over envs)
    for gear_idx, asset_name in enumerate(gear_asset_names):
        if asset_name in positions_by_asset:
            # Create mask for environments using this gear type
            mask = gear_type_indices == gear_idx  # Shape: (len(env_ids),)

            # Apply position offsets only to selected environments (vectorized)
            positions_by_asset[asset_name][mask] = (
                positions_by_asset[asset_name][mask] + rand_gear_offsets[mask]
            )

            # Apply orientation offsets only to selected environments (vectorized)
            orientations_by_asset[asset_name][mask] = math_utils.quat_mul(
                orientations_by_asset[asset_name][mask], rand_rot_quats[mask]
            )

    # Write back to sim for all prepared assets (fully vectorized)
    for asset_name in positions_by_asset.keys():
        asset = env.scene[asset_name]
        positions = positions_by_asset[asset_name]
        orientations = orientations_by_asset[asset_name]
        velocities = velocities_by_asset[asset_name]
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)