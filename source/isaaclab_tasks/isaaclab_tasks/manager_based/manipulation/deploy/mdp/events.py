# Copyright (c) 2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""


from __future__ import annotations

import carb
import time
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
    # print(f"env._current_gear_type: {env._current_gear_type}")

def set_robot_to_grasp_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_threshold: float = 1e-6,
    rot_threshold: float = 1e-6,
    max_iterations: int = 10,
    rot_offset: Optional[list[float]] = None,
    pos_randomization_range: Optional[dict] = None,
    num_arm_joints: int = 6
):
    # Convert rotation offset to tensor if provided
    if rot_offset is not None:
        rot_offset_tensor = torch.tensor(rot_offset, device=env.device).unsqueeze(0).expand(len(env_ids), -1)
    else:
        rot_offset_tensor = None
    
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

        # Get gear positions and orientations for each environment based on selected gear type
        env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
        grasp_object_pos_world = torch.zeros(len(env_ids), 3, device=env.device)
        grasp_object_quat = torch.zeros(len(env_ids), 4, device=env.device)
        
        for row_idx, env_id in enumerate(env_ids_list):
            # Get the gear type for this environment
            gear_key = env._current_gear_type[env_id]
            selected_asset_name = f"factory_{gear_key}"
            
            # Get the gear asset for this environment
            gear_asset: RigidObject = env.scene[selected_asset_name]
            grasp_object_pos_world[row_idx] = gear_asset.data.root_link_pos_w[env_id].clone()
            grasp_object_quat[row_idx] = gear_asset.data.root_link_quat_w[env_id].clone()

        
        # First apply rotation offset to get the object's orientation
        if rot_offset_tensor is not None:
            # Apply rotation offset by quaternion multiplication
            # rot_offset is assumed to be in quaternion format (w, x, y, z)
            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, rot_offset_tensor)
            # print(f"Applied rot_offset: {rot_offset}")
            # print(f"grasp_object_quat after offset: {grasp_object_quat}")
        
        if pos_randomization_range is not None:
            pos_keys = ["x", "y", "z"]
            range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
            ranges_pos = torch.tensor(range_list_pos, device=env.device)
            rand_pos_offsets = math_utils.sample_uniform(ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device)
        # Apply gear-specific grasp offsets to each environment
        for row_idx, env_id in enumerate(env_ids_list):
            gear_key = env._current_gear_type[env_id]
            gear_grasp_offset = env.cfg.gear_offsets_grasp[gear_key]
            grasp_offset_tensor = torch.tensor(gear_grasp_offset, device=env.device, dtype=grasp_object_pos_world.dtype)

            if pos_randomization_range is not None:
                grasp_offset_tensor += rand_pos_offsets[row_idx]
            
            # Transform position offset from rotated object frame to world frame
            grasp_object_pos_world[row_idx] = grasp_object_pos_world[row_idx] + math_utils.quat_apply(
                grasp_object_quat[row_idx:row_idx+1], grasp_offset_tensor.unsqueeze(0)
            )[0]
        
        # Convert to environment-relative coordinates by subtracting environment origins
        grasp_object_pos = grasp_object_pos_world
        
        # Get end effector pose of the robot
        # Get the specific wrist_3_link pose
        try:
            # Find the index of the wrist_3_link body
            wrist_3_indices, wrist_3_names = robot_asset.find_bodies(["wrist_3_link"])
            wrist_3_idx = wrist_3_indices[0]
            
            if len(wrist_3_indices) == 1:
                wrist_3_idx = wrist_3_indices[0]  # Get the first (and should be only) index
                
                # Get the specific wrist_3_link pose
                eef_pos_world = robot_asset.data.body_link_pos_w[env_ids, wrist_3_idx]  # Shape: (len(env_ids), 3)
                eef_quat = robot_asset.data.body_link_quat_w[env_ids, wrist_3_idx]  # Shape: (len(env_ids), 4)
                
                # Convert to environment-relative coordinates
                eef_pos = eef_pos_world


                # You can also get the full pose as [pos, quat] (7-dimensional)
                eef_pos = robot_asset.data.body_pos_w[env_ids, wrist_3_idx]
                eef_quat = robot_asset.data.body_quat_w[env_ids, wrist_3_idx]
                # print(f"eef_pos: {eef_pos}")
                # print(f"eef_quat: {eef_quat}")
            elif len(wrist_3_indices) > 1:
                print("wrist_3_link found multiple times in robot body names")
                print(f"Available body names: {robot_asset.body_names}")
            else:
                print("wrist_3_link not found in robot body names")
                print(f"Available body names: {robot_asset.body_names}")
                
        except Exception as e:
            print(f"Could not get end effector pose: {e}")
            print("You may need to adjust the body name or access method based on your robot configuration")
        

        # Compute error to target using wrist_3_link as current and grasp_object as target
        if len(wrist_3_indices) > 0:
            # Get current end effector pose (wrist_3_link)
            current_eef_pos = eef_pos  # wrist_3_link position
            current_eef_quat = eef_quat  # wrist_3_link orientation
            
            # Get target pose (grasp object)
            target_eef_pos = grasp_object_pos  # grasp object position
            target_eef_quat = grasp_object_quat  # grasp object orientation

            # print(f"current_eef_pos: {current_eef_pos[0]}")
            # print(f"current_eef_quat: {current_eef_quat[0]}")
            # print(f"target_eef_pos: {target_eef_pos[0]}")
            # print(f"target_eef_quat: {target_eef_quat[0]}")

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
            
            # print(f"Pose error - position: {pos_error[0]}")
            # print(f"Pose error - orientation: {axis_angle_error[0]}")
            # print(f"Position error norm: {pos_error_norm[0]}")
            # print(f"Rotation error norm: {rot_error_norm[0]}")
            
            # Check if all environments have converged
            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                # print(f"Converged after {i} iterations!")
                # print(f"pos_error_norm: {pos_error_norm}")
                # print(f"rot_error_norm: {rot_error_norm}")
                # print(f"pos_error: {pos_error}")
                # print(f"axis_angle_error: {axis_angle_error}")
                break

            # Solve DLS problem for inverse kinematics
            # Get jacobian for the wrist_3_link using the same approach as task_space_actions.py
            try:
                # Get all jacobians from the robot
                jacobians = robot_asset.root_physx_view.get_jacobians().clone()
				
                # Select the jacobian for the wrist_3_link body
                # For fixed-base robots, Jacobian excludes the base body (index 0)
                # So if wrist_3_idx is 6, the Jacobian index should be 5
                jacobi_body_idx = wrist_3_idx - 1
                
                
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


        # print(f"delta_dof_pos.abs().max(): {delta_dof_pos.abs().max()}")

        # if delta_dof_pos.abs().max() < 1e-5:
        #     break

        # print(f"Step {i}")
        
        i += 1
    # if i >= max_iterations:
    #     print(f"IK did not converge after {max_iterations} iterations")
    #     # print(f"pos_error_norm: {pos_error_norm}")
    #     # print(f"rot_error_norm: {rot_error_norm}")
    #     # print(f"pos_error: {pos_error}")
    #     # print(f"axis_angle_error: {axis_angle_error}")
    #     # raise RuntimeError(f"IK did not converge after {max_iterations} iterations")

    # # print(f"delta_dof_pos.abs().max(): {delta_dof_pos.abs().max()}")
    # for i in range(10):
    #     env.sim.render()
    #     input("Going to set object position...")
    
    # TODO: Resttting the gear based on teh IK solution is not working as expected. The gear is not being placed in the correct position.
    # # Set the grasp object's pose to the current end-effector pose in world coordinates (with optional offsets)
    # # The offsets should be applied in reverse to get from wrist_3_link to gear pose
    # gear_pos_world = current_eef_pos
    # gear_quat_world = current_eef_quat
    
    # # Apply position offset (subtract to get from wrist_3_link to gear position)
    # if pos_offset_tensor is not None:
    #     gear_pos_world = gear_pos_world - pos_offset_tensor
    
    # # Apply rotation offset (inverse to get from wrist_3_link to gear orientation)
    # if rot_offset_tensor is not None:
    #     # Apply inverse rotation offset
    #     gear_quat_world = math_utils.quat_mul(current_eef_quat, math_utils.quat_conjugate(rot_offset_tensor))
    
    # for i in range(10):
    #     env.sim.render()
    #     input("Press Enter to continue 1...")
    
    # # Set the grasp object's pose for each environment based on selected gear type
    # for row_idx, env_id in enumerate(env_ids_list):
    #     gear_key = env._current_gear_type[env_id]
    #     selected_asset_name = f"factory_{gear_key}"
    #     gear_asset: RigidObject = env.scene[selected_asset_name]
    #     gear_asset.write_root_pose_to_sim(
    #         torch.cat([gear_pos_world[row_idx:row_idx+1], gear_quat_world[row_idx:row_idx+1]], dim=-1), 
    #         env_ids=torch.tensor([env_id], device=env.device)
    #     )

    # Close the gripper by setting the finger_joint based on gear type
    all_joints, all_joints_names = robot_asset.find_joints([".*"])
    
    joint_pos = robot_asset.data.joint_pos[env_ids].clone()
    
    # @ireti: We need to change this so it workes with teh public 2f140 gripper that uses mimic joints
    finger_joints = all_joints[num_arm_joints:]
    finger_joint_names = all_joints_names[num_arm_joints:]

    # for i in range(10):
    #     env.sim.render()
    #     input("Press Enter to continue 2...")

    hand_grasp_pos = env.cfg.hand_grasp_pos[gear_key]
    set_finger_joint_pos_2f_140(joint_pos, env_ids_list, finger_joints, hand_grasp_pos)

    robot_asset.set_joint_position_target(joint_pos, joint_ids=all_joints, env_ids=env_ids)
    robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # for i in range(10):
    #     env.sim.render()
    #     input("Press Enter to continue 3...")

    # Set finger joints based on gear type for each environment
    hand_close_pos = env.cfg.hand_close_pos
    set_finger_joint_pos_2f_140(joint_pos, env_ids_list, finger_joints, hand_close_pos)

    robot_asset.set_joint_position_target(joint_pos, joint_ids=all_joints, env_ids=env_ids)

    # for i in range(10):
    #     env.sim.render()
    #     input("Press Enter to continue 4...")

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
        # "factory_gear_small",
        "factory_gear_medium",
        # "factory_gear_large",
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

    env_ids_list = env_ids.tolist() if isinstance(env_ids, torch.Tensor) else list(env_ids)
    for row_idx, env_id in enumerate(env_ids_list):
        # env._current_gear_type[env_id] is expected to be one of: "gear_small", "gear_medium", "gear_large"
        gear_key = env._current_gear_type[env_id]
        selected_asset_name = f"factory_{gear_key}"
        if selected_asset_name in positions_by_asset:
            positions_by_asset[selected_asset_name][row_idx] = (
                positions_by_asset[selected_asset_name][row_idx] + rand_gear_offsets[row_idx]
            )
            # Apply additional orientation randomization to the selected gear
            orientations_by_asset[selected_asset_name][row_idx] = math_utils.quat_mul(
                orientations_by_asset[selected_asset_name][row_idx], rand_rot_quats[row_idx]
            )

    # Write back to sim for all prepared assets
    for asset_name in positions_by_asset.keys():
        asset = env.scene[asset_name]
        positions = positions_by_asset[asset_name]
        orientations = orientations_by_asset[asset_name]
        velocities = velocities_by_asset[asset_name]
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

def set_finger_joint_pos_2f_140(
    joint_pos: torch.Tensor,
    env_ids_list: list[int],
    finger_joints: list[int],
    finger_joint_position: float,
):
    # Get hand close positions for each environment based on gear type
    for row_idx, env_id in enumerate(env_ids_list):
        
        joint_pos[row_idx, finger_joints[0]] = finger_joint_position
        joint_pos[row_idx, finger_joints[1]] = finger_joint_position
        
        # outer finger joints
        joint_pos[row_idx, finger_joints[2]] = 0
        joint_pos[row_idx, finger_joints[3]] = 0

        joint_pos[row_idx, finger_joints[4]] = -finger_joint_position
        joint_pos[row_idx, finger_joints[5]] = -finger_joint_position

        joint_pos[row_idx, finger_joints[6]] = finger_joint_position
        joint_pos[row_idx, finger_joints[7]] = finger_joint_position