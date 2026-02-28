# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based event terms specific to the gear assembly manipulation environments."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch
import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.direct.automate import factory_control as fc

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class randomize_gear_type(ManagerTermBase):
    """Randomize and manage the gear type being used for each environment.

    This class stores the current gear type for each environment and provides a mapping
    from gear type names to indices. It serves as the central manager for gear type state
    that other MDP terms depend on.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the gear type randomization term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Extract gear types from config (required parameter)
        if "gear_types" not in cfg.params:
            raise ValueError("'gear_types' parameter is required in randomize_gear_type configuration")
        self.gear_types: list[str] = cfg.params["gear_types"]

        # Create gear type mapping (shared across all terms)
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}

        # Store current gear type for each environment (as list for easy access)
        # Initialize all to first gear type in the list
        self._current_gear_type = [self.gear_types[0]] * env.num_envs

        # Store current gear type indices as tensor for efficient vectorized access
        # Initialize all to first gear type index
        first_gear_idx = self.gear_type_map[self.gear_types[0]]
        self._current_gear_type_indices = torch.full(
            (env.num_envs,), first_gear_idx, device=env.device, dtype=torch.long
        )

        # Store reference on environment for other terms to access
        env._gear_type_manager = self

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        gear_types: list[str] = ["gear_small", "gear_medium", "gear_large"],
    ):
        """Randomize the gear type for specified environments.

        Args:
            env: The environment containing the assets
            env_ids: Environment IDs to randomize
            gear_types: List of available gear types to choose from
        """
        # Randomly select gear type for each environment
        # Use the parameter passed to __call__ (not self.gear_types) to allow runtime overrides
        for env_id in env_ids.tolist():
            chosen_gear = random.choice(gear_types)
            self._current_gear_type[env_id] = chosen_gear
            self._current_gear_type_indices[env_id] = self.gear_type_map[chosen_gear]

    def get_gear_type(self, env_id: int) -> str:
        """Get the current gear type for a specific environment."""
        return self._current_gear_type[env_id]

    def get_all_gear_types(self) -> list[str]:
        """Get current gear types for all environments."""
        return self._current_gear_type

    def get_all_gear_type_indices(self) -> torch.Tensor:
        """Get current gear type indices for all environments as a tensor.

        Returns:
            Tensor of shape (num_envs,) with gear type indices (0=small, 1=medium, 2=large)
        """
        return self._current_gear_type_indices


class set_robot_to_grasp_pose(ManagerTermBase):
    """Set robot to grasp pose using IK with pre-cached tensors.

    This class-based term caches all required tensors and gear offsets during initialization,
    avoiding repeated allocations and lookups during execution.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the set robot to grasp pose term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Get robot asset configuration
        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        # Get robot-specific parameters from environment config (all required)
        # Validate required parameters
        if "end_effector_body_name" not in cfg.params:
            raise ValueError(
                "'end_effector_body_name' parameter is required in set_robot_to_grasp_pose configuration. "
                "Example: 'wrist_3_link'"
            )
        if "num_arm_joints" not in cfg.params:
            raise ValueError(
                "'num_arm_joints' parameter is required in set_robot_to_grasp_pose configuration. Example: 6 for UR10e"
            )
        if "grasp_rot_offset" not in cfg.params:
            raise ValueError(
                "'grasp_rot_offset' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a quaternion [x, y, z, w]. Example: [0.707, 0.707, 0.0, 0.0]"
            )
        if "gripper_joint_setter_func" not in cfg.params:
            raise ValueError(
                "'gripper_joint_setter_func' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a function to set gripper joint positions."
            )

        self.end_effector_body_name = cfg.params["end_effector_body_name"]
        self.num_arm_joints = cfg.params["num_arm_joints"]
        self.gripper_joint_setter_func = cfg.params["gripper_joint_setter_func"]

        # Pre-cache gear grasp offsets as tensors (required parameter)
        if "gear_offsets_grasp" not in cfg.params:
            raise ValueError(
                "'gear_offsets_grasp' parameter is required in set_robot_to_grasp_pose configuration. "
                "It should be a dict with keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )
        gear_offsets_grasp = cfg.params["gear_offsets_grasp"]
        if not isinstance(gear_offsets_grasp, dict):
            raise TypeError(
                f"'gear_offsets_grasp' parameter must be a dict, got {type(gear_offsets_grasp).__name__}. "
                "It should have keys 'gear_small', 'gear_medium', 'gear_large' mapping to [x, y, z] offsets."
            )

        self.gear_grasp_offset_tensors = {}
        for gear_type in ["gear_small", "gear_medium", "gear_large"]:
            if gear_type not in gear_offsets_grasp:
                raise ValueError(
                    f"'{gear_type}' offset is required in 'gear_offsets_grasp' parameter. "
                    f"Found keys: {list(gear_offsets_grasp.keys())}"
                )
            self.gear_grasp_offset_tensors[gear_type] = torch.tensor(
                gear_offsets_grasp[gear_type], device=env.device, dtype=torch.float32
            )

        # Stack grasp offset tensors for vectorized indexing (shape: 3, 3)
        # Index 0=small, 1=medium, 2=large
        self.gear_grasp_offsets_stacked = torch.stack(
            [
                self.gear_grasp_offset_tensors["gear_small"],
                self.gear_grasp_offset_tensors["gear_medium"],
                self.gear_grasp_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-cache grasp rotation offset tensor
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        # Pre-allocate buffers for batch operations
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self.local_env_indices = torch.arange(env.num_envs, device=env.device)
        self.gear_grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

        # Cache hand grasp/close widths
        self.hand_grasp_width = env.cfg.hand_grasp_width
        self.hand_close_width = env.cfg.hand_close_width

        # Find end effector index once
        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End effector body '{self.end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]

        # Find jacobian body index (for fixed-base robots, subtract 1)
        self.jacobi_body_idx = self.eef_idx - 1

        # Find all joints once
        all_joints, all_joints_names = self.robot_asset.find_joints([".*"])
        self.all_joints = all_joints
        self.finger_joints = all_joints[self.num_arm_joints :]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-6,
        max_iterations: int = 10,
        pos_randomization_range: dict | None = None,
        gear_offsets_grasp: dict | None = None,
        end_effector_body_name: str | None = None,
        num_arm_joints: int | None = None,
        grasp_rot_offset: list | None = None,
        gripper_joint_setter_func: callable | None = None,
    ):
        """Set robot to grasp pose using IK.

        Args:
            env: Environment instance
            env_ids: Environment IDs to reset
            robot_asset_cfg: Robot asset configuration (unused, kept for compatibility)
            pos_threshold: Position convergence threshold
            rot_threshold: Rotation convergence threshold
            max_iterations: Maximum IK iterations
            pos_randomization_range: Optional position randomization range
        """
        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager

        # Slice buffers for current batch size
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        local_env_indices = self.local_env_indices[:num_reset_envs]
        gear_grasp_offsets = self.gear_grasp_offsets_buffer[:num_reset_envs]
        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]

        # IK loop
        for i in range(max_iterations):
            # Get current joint state
            joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(self.robot_asset.data.joint_vel)[env_ids].clone()

            # Stack all gear positions and quaternions
            all_gear_pos = torch.stack(
                [
                    wp.to_torch(env.scene["factory_gear_small"].data.root_link_pos_w),
                    wp.to_torch(env.scene["factory_gear_medium"].data.root_link_pos_w),
                    wp.to_torch(env.scene["factory_gear_large"].data.root_link_pos_w),
                ],
                dim=1,
            )[env_ids]

            all_gear_quat = torch.stack(
                [
                    wp.to_torch(env.scene["factory_gear_small"].data.root_link_quat_w),
                    wp.to_torch(env.scene["factory_gear_medium"].data.root_link_quat_w),
                    wp.to_torch(env.scene["factory_gear_large"].data.root_link_quat_w),
                ],
                dim=1,
            )[env_ids]

            # Get gear type indices directly as tensor
            all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()
            gear_type_indices[:] = all_gear_type_indices[env_ids]

            # Select gear data using advanced indexing
            grasp_object_pos_world = all_gear_pos[local_env_indices, gear_type_indices]
            grasp_object_quat = all_gear_quat[local_env_indices, gear_type_indices]

            # Apply rotation offset
            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, grasp_rot_offset_tensor)

            # Get grasp offsets (vectorized)
            gear_grasp_offsets[:] = self.gear_grasp_offsets_stacked[gear_type_indices]

            # Add position randomization if specified
            if pos_randomization_range is not None:
                pos_keys = ["x", "y", "z"]
                range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
                ranges_pos = torch.tensor(range_list_pos, device=env.device)
                rand_pos_offsets = math_utils.sample_uniform(
                    ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device
                )
                gear_grasp_offsets = gear_grasp_offsets + rand_pos_offsets

            # Transform offsets from gear frame to world frame
            grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(
                grasp_object_quat, gear_grasp_offsets
            )

            # Get end effector pose
            eef_pos = wp.to_torch(self.robot_asset.data.body_pos_w)[env_ids, self.eef_idx]
            eef_quat = wp.to_torch(self.robot_asset.data.body_quat_w)[env_ids, self.eef_idx]

            # Compute pose error
            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=eef_pos,
                fingertip_midpoint_quat=eef_quat,
                ctrl_target_fingertip_midpoint_pos=grasp_object_pos_world,
                ctrl_target_fingertip_midpoint_quat=grasp_object_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            # Check convergence
            pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
            rot_error_norm = torch.linalg.norm(axis_angle_error, dim=-1)

            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                break

            # Solve IK using jacobian
            jacobians = wp.to_torch(self.robot_asset.root_view.get_jacobians()).clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, :]

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=env.device,
            )

            # Update joint positions
            joint_pos = joint_pos + delta_dof_pos
            joint_vel = torch.zeros_like(joint_pos)

            # Write to sim
            self.robot_asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot_asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
            self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Set gripper to grasp position
        joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()

        # Get gear types for all environments
        all_gear_types = gear_type_manager.get_all_gear_types()
        for row_idx, env_id in enumerate(env_ids.tolist()):
            gear_key = all_gear_types[env_id]
            hand_grasp_width = self.hand_grasp_width[gear_key]
            self.gripper_joint_setter_func(joint_pos, [row_idx], self.finger_joints, hand_grasp_width)

        self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)
        self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Set gripper to closed position
        for row_idx, env_id in enumerate(env_ids.tolist()):
            gear_key = all_gear_types[env_id]
            hand_close_width = self.hand_close_width[gear_key]
            self.gripper_joint_setter_func(joint_pos, [row_idx], self.finger_joints, hand_close_width)

        self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)


class randomize_gears_and_base_pose(ManagerTermBase):
    """Randomize both the gear base pose and individual gear poses.

    This class-based term pre-caches all tensors needed for randomization.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomize gears and base pose term.

        Args:
            cfg: Event term configuration
            env: Environment instance
        """
        super().__init__(cfg, env)

        # Pre-allocate gear type mapping and indices
        self.gear_type_map = {"gear_small": 0, "gear_medium": 1, "gear_large": 2}
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        # Cache asset names
        self.gear_asset_names = ["factory_gear_small", "factory_gear_medium", "factory_gear_large"]
        self.base_asset_name = "factory_gear_base"

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict = {},
        velocity_range: dict = {},
        gear_pos_range: dict = {},
    ):
        """Randomize gear base and gear poses.

        Args:
            env: Environment instance
            env_ids: Environment IDs to randomize
            pose_range: Pose randomization range for base and all gears
            velocity_range: Velocity randomization range
            gear_pos_range: Additional position randomization for selected gear only
        """
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager
        device = env.device

        # Shared pose samples for all assets
        pose_keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        range_list_pose = [pose_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_pose = torch.tensor(range_list_pose, device=device)
        rand_pose_samples = math_utils.sample_uniform(
            ranges_pose[:, 0], ranges_pose[:, 1], (len(env_ids), 6), device=device
        )

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )

        # Shared velocity samples
        range_list_vel = [velocity_range.get(key, (0.0, 0.0)) for key in pose_keys]
        ranges_vel = torch.tensor(range_list_vel, device=device)
        rand_vel_samples = math_utils.sample_uniform(
            ranges_vel[:, 0], ranges_vel[:, 1], (len(env_ids), 6), device=device
        )

        # Prepare poses for all assets
        positions_by_asset = {}
        orientations_by_asset = {}
        velocities_by_asset = {}

        asset_names_to_process = [self.base_asset_name] + self.gear_asset_names
        for asset_name in asset_names_to_process:
            asset: RigidObject | Articulation = env.scene[asset_name]
            default_root_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
            default_root_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()
            positions = default_root_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]
            orientations = math_utils.quat_mul(default_root_pose[:, 3:7], orientations_delta)
            velocities = default_root_vel + rand_vel_samples
            positions_by_asset[asset_name] = positions
            orientations_by_asset[asset_name] = orientations
            velocities_by_asset[asset_name] = velocities

        # Per-env gear offset (gear_pos_range) applied only to selected gear
        range_list_gear = [gear_pos_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
        ranges_gear = torch.tensor(range_list_gear, device=device)
        rand_gear_offsets = math_utils.sample_uniform(
            ranges_gear[:, 0], ranges_gear[:, 1], (len(env_ids), 3), device=device
        )

        # Get gear type indices directly as tensor
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]
        all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()
        gear_type_indices[:] = all_gear_type_indices[env_ids]

        # Apply offsets using vectorized operations with masks
        for gear_idx, asset_name in enumerate(self.gear_asset_names):
            if asset_name in positions_by_asset:
                mask = gear_type_indices == gear_idx
                positions_by_asset[asset_name][mask] = positions_by_asset[asset_name][mask] + rand_gear_offsets[mask]

        # Write to sim
        for asset_name in positions_by_asset.keys():
            asset = env.scene[asset_name]
            positions = positions_by_asset[asset_name]
            orientations = orientations_by_asset[asset_name]
            velocities = velocities_by_asset[asset_name]
            asset.write_root_pose_to_sim_index(root_pose=torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim_index(root_velocity=velocities, env_ids=env_ids)
