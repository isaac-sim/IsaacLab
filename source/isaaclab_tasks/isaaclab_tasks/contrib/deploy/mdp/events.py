# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based event terms for manipulation deployment environments."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from isaaclab_tasks.contrib.automate import factory_control as fc

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
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
        max_iterations: int = 50,
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
            joint_pos = self.robot_asset.data.joint_pos.torch[env_ids].clone()
            joint_vel = self.robot_asset.data.joint_vel.torch[env_ids].clone()

            # Stack all gear positions and quaternions
            all_gear_pos = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_pos_w.torch,
                    env.scene["factory_gear_large"].data.root_link_pos_w.torch,
                ],
                dim=1,
            )[env_ids]

            all_gear_quat = torch.stack(
                [
                    env.scene["factory_gear_small"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_medium"].data.root_link_quat_w.torch,
                    env.scene["factory_gear_large"].data.root_link_quat_w.torch,
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
            eef_pos = self.robot_asset.data.body_pos_w.torch[env_ids, self.eef_idx]
            eef_quat = self.robot_asset.data.body_quat_w.torch[env_ids, self.eef_idx]

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

            # Solve IK using jacobian. ``body_link_jacobian_w`` prepends ``num_base_dofs``
            # floating-base columns on the DoF axis (0 for fixed-base, 6 for floating-base);
            # slice past them so the column axis aligns with the actuated-joint state.
            jacobians = self.robot_asset.data.body_link_jacobian_w.torch.clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, self.robot_asset.num_base_dofs :]

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=env.device,
            )

            # Update joint positions
            joint_pos = joint_pos + delta_dof_pos

            # Wrap arm joint positions to fall within robot's actual joint limits
            joint_pos_limits = self.robot_asset.data.joint_pos_limits.torch[env_ids, : self.num_arm_joints, :]
            joint_min = joint_pos_limits[:, :, 0]
            joint_max = joint_pos_limits[:, :, 1]
            joint_range = joint_max - joint_min

            # Wrap only the arm joint positions (not gripper joints)
            arm_joint_pos = joint_pos[:, : self.num_arm_joints]
            arm_joint_pos = torch.where(
                joint_range > 0,
                joint_min + torch.remainder(arm_joint_pos - joint_min, joint_range),
                arm_joint_pos,
            )
            joint_pos[:, : self.num_arm_joints] = arm_joint_pos

            joint_vel = torch.zeros_like(joint_pos)

            # Write to sim
            self.robot_asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot_asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
            self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Reset joint velocities to zero after IK convergence
        joint_vel = torch.zeros_like(self.robot_asset.data.joint_vel.torch[env_ids])

        # Set gripper to grasp position
        joint_pos = self.robot_asset.data.joint_pos.torch[env_ids].clone()

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
            default_root_pose = asset.data.default_root_pose.torch[env_ids].clone()
            default_root_vel = asset.data.default_root_vel.torch[env_ids].clone()
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


class set_robot_to_object_grasp_pose(ManagerTermBase):
    """Set robot to a grasp pose over a single named target object using IK.

    Generic single-object counterpart of :class:`set_robot_to_grasp_pose` (which
    is keyed on the gear-type manager). This term targets a single named
    :class:`~isaaclab.assets.RigidObject` with a fixed grasp offset, suitable
    for cable insertion and other single-object manipulation tasks.

    Args:
        target_object_name: Name of the rigid object in the scene to grasp.
        end_effector_body_name: Name of the end-effector body on the robot.
        num_arm_joints: Number of arm joints (the remaining joints are
            assumed to be gripper/finger joints).
        grasp_offset: Position offset ``[x, y, z]`` [m] applied in the
            (rotated) object frame to define the IK target. Defaults to zero.
        grasp_rot_offset: Quaternion offset ``(x, y, z, w)`` applied to the
            object orientation to define the IK target.
        gripper_joint_setter_func: Callable used to set finger joint positions
            for the configured grasp/close widths.
        robot_asset_cfg: Robot asset configuration. Defaults to
            ``SceneEntityCfg("robot")``.
        pos_threshold: IK position-error tolerance [m].
        rot_threshold: IK rotation-error tolerance [rad].
        max_iterations: Maximum IK iterations per env reset.
        pos_randomization_range: Optional dict with keys ``"x"``, ``"y"``,
            ``"z"`` mapping to ``(low, high)`` tuples [m] for per-reset
            randomization of the grasp offset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot_asset_cfg: SceneEntityCfg = cfg.params.get("robot_asset_cfg", SceneEntityCfg("robot"))
        self.robot_asset: Articulation = env.scene[self.robot_asset_cfg.name]

        for required in (
            "end_effector_body_name",
            "num_arm_joints",
            "grasp_rot_offset",
            "gripper_joint_setter_func",
            "target_object_name",
        ):
            if required not in cfg.params:
                raise ValueError(f"'{required}' is required in set_robot_to_object_grasp_pose configuration.")

        self.end_effector_body_name: str = cfg.params["end_effector_body_name"]
        self.num_arm_joints: int = cfg.params["num_arm_joints"]
        self.gripper_joint_setter_func = cfg.params["gripper_joint_setter_func"]
        self.target_object_name: str = cfg.params["target_object_name"]

        grasp_offset = cfg.params.get("grasp_offset", [0.0, 0.0, 0.0])
        self.grasp_offset_tensor = torch.tensor(grasp_offset, device=env.device, dtype=torch.float32)

        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        self.grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)

        self.hand_grasp_width = env.cfg.hand_grasp_width
        self.hand_close_width = env.cfg.hand_close_width
        # hand_hold_width: joint angle where fingers just touch the held object
        # surface.  Written as the physical STATE so there is no mesh overlap.
        # Falls back to hand_close_width when not set (original behaviour).
        self.hand_hold_width = getattr(env.cfg, "hand_hold_width", self.hand_close_width)

        eef_indices, _ = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End effector body '{self.end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]
        self.jacobi_body_idx = self.eef_idx - 1

        all_joints, _ = self.robot_asset.find_joints([".*"])
        self.all_joints = all_joints
        self.finger_joints = all_joints[self.num_arm_joints :]

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        pos_threshold: float = 1e-6,
        rot_threshold: float = 1e-6,
        max_iterations: int = 50,
        pos_randomization_range: dict | None = None,
        target_object_name: str | None = None,
        grasp_offset: list | None = None,
        end_effector_body_name: str | None = None,
        num_arm_joints: int | None = None,
        grasp_rot_offset: list | None = None,
        gripper_joint_setter_func: callable | None = None,
    ):
        num_reset_envs = len(env_ids)
        grasp_offsets = self.grasp_offsets_buffer[:num_reset_envs]
        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]

        # One-shot debug log to confirm the event fires and report IK convergence.
        # Remove or guard once the grasp wiring is verified.
        debug_first_call = not getattr(self, "_debug_printed", False)
        if debug_first_call:
            self._debug_printed = True
            target_object_dbg: RigidObject = env.scene[self.target_object_name]
            init_obj_pos = wp.to_torch(target_object_dbg.data.root_link_pos_w)[env_ids][0].tolist()
            init_obj_quat = wp.to_torch(target_object_dbg.data.root_link_quat_w)[env_ids][0].tolist()
            init_eef_pos = wp.to_torch(self.robot_asset.data.body_pos_w)[env_ids, self.eef_idx][0].tolist()
            init_eef_quat = wp.to_torch(self.robot_asset.data.body_quat_w)[env_ids, self.eef_idx][0].tolist()
            print(
                f"[GRASP-DBG] set_robot_to_object_grasp_pose fired:"
                f" target={self.target_object_name!r} num_reset_envs={num_reset_envs}"
                f" eef_idx={self.eef_idx} num_arm_joints={self.num_arm_joints}\n"
                f"           grasp_offset={self.grasp_offset_tensor.tolist()}"
                f" grasp_rot_offset(xyzw)={self.grasp_rot_offset_tensor[0].tolist()}"
                f" hand_close_width={self.hand_close_width}\n"
                f"           init_obj_pos_w={init_obj_pos}"
                f" init_obj_quat(xyzw)={init_obj_quat}\n"
                f"           init_eef_pos_w={init_eef_pos}"
                f" init_eef_quat(xyzw)={init_eef_quat}"
            )

        last_pos_err = None
        last_rot_err = None
        converged_at = -1
        last_target_pos = None
        last_target_quat = None

        for _iter in range(max_iterations):
            joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()
            joint_vel = wp.to_torch(self.robot_asset.data.joint_vel)[env_ids].clone()

            target_object: RigidObject = env.scene[self.target_object_name]
            grasp_object_pos_world = wp.to_torch(target_object.data.root_link_pos_w)[env_ids]
            grasp_object_quat = wp.to_torch(target_object.data.root_link_quat_w)[env_ids]

            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, grasp_rot_offset_tensor)

            grasp_offsets[:] = self.grasp_offset_tensor

            if pos_randomization_range is not None:
                pos_keys = ["x", "y", "z"]
                range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
                ranges_pos = torch.tensor(range_list_pos, device=env.device)
                rand_pos_offsets = math_utils.sample_uniform(
                    ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device
                )
                grasp_offsets = grasp_offsets + rand_pos_offsets

            grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(grasp_object_quat, grasp_offsets)

            eef_pos = wp.to_torch(self.robot_asset.data.body_pos_w)[env_ids, self.eef_idx]
            eef_quat = wp.to_torch(self.robot_asset.data.body_quat_w)[env_ids, self.eef_idx]

            last_target_pos = grasp_object_pos_world.clone()
            last_target_quat = grasp_object_quat.clone()

            pos_error, axis_angle_error = fc.get_pose_error(
                fingertip_midpoint_pos=eef_pos,
                fingertip_midpoint_quat=eef_quat,
                ctrl_target_fingertip_midpoint_pos=grasp_object_pos_world,
                ctrl_target_fingertip_midpoint_quat=grasp_object_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)

            pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
            rot_error_norm = torch.linalg.norm(axis_angle_error, dim=-1)
            last_pos_err = pos_error_norm
            last_rot_err = rot_error_norm

            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                converged_at = _iter
                break

            jacobians = wp.to_torch(self.robot_asset.root_view.get_jacobians()).clone()
            jacobian = jacobians[env_ids, self.jacobi_body_idx, :, :]

            delta_dof_pos = fc._get_delta_dof_pos(
                delta_pose=delta_hand_pose,
                ik_method="dls",
                jacobian=jacobian,
                device=env.device,
            )

            joint_pos = joint_pos + delta_dof_pos

            joint_pos_limits = wp.to_torch(self.robot_asset.data.joint_pos_limits)[env_ids, : self.num_arm_joints, :]
            joint_min = joint_pos_limits[:, :, 0]
            joint_max = joint_pos_limits[:, :, 1]
            joint_range = joint_max - joint_min

            arm_joint_pos = joint_pos[:, : self.num_arm_joints]
            arm_joint_pos = torch.where(
                joint_range > 0,
                joint_min + torch.remainder(arm_joint_pos - joint_min, joint_range),
                arm_joint_pos,
            )
            joint_pos[:, : self.num_arm_joints] = arm_joint_pos

            joint_vel = torch.zeros_like(joint_pos)

            self.robot_asset.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
            self.robot_asset.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
            self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        # Snap the held object to the achieved gripper pose so the gripper actually
        # holds it after closing. Without this, any IK residual error or USD
        # geometry offset leaves the object outside the finger gap and gravity drops
        # Snap the held object to the achieved gripper pose so the gripper
        # actually holds it after closing.
        held_object = env.scene[self.target_object_name]
        achieved_hand_pos = wp.to_torch(self.robot_asset.data.body_pos_w)[env_ids, self.eef_idx].clone()
        achieved_hand_quat = wp.to_torch(self.robot_asset.data.body_quat_w)[env_ids, self.eef_idx].clone()

        # Object orientation: inverse of grasp_rot_offset applied to the achieved hand quat,
        # because IK target was ``hand = obj * grasp_rot_offset`` => ``obj = hand * grasp_rot_offset^{-1}``.
        inv_grasp_rot_offset = math_utils.quat_conjugate(grasp_rot_offset_tensor)
        target_obj_quat = math_utils.quat_mul(achieved_hand_quat, inv_grasp_rot_offset)

        # Object position: IK target was
        #   ``hand_pos = obj_pos + R(obj * grasp_rot_offset) * grasp_offset``
        #             = ``obj_pos + R(hand_quat) * grasp_offset``
        # so ``obj_pos = hand_pos - R(hand_quat) * grasp_offset``.
        grasp_offset_in_world = math_utils.quat_apply(achieved_hand_quat, grasp_offsets)
        target_obj_pos = achieved_hand_pos - grasp_offset_in_world

        new_root_pose = torch.cat([target_obj_pos, target_obj_quat], dim=-1)
        zero_velocity = torch.zeros((len(env_ids), 6), device=env.device, dtype=torch.float32)
        held_object.write_root_pose_to_sim(new_root_pose, env_ids=env_ids)
        held_object.write_root_velocity_to_sim(zero_velocity, env_ids=env_ids)

        if debug_first_call:
            pos_err_max = float(last_pos_err.max().item()) if last_pos_err is not None else float("nan")
            rot_err_max = float(last_rot_err.max().item()) if last_rot_err is not None else float("nan")
            tgt_pos0 = target_obj_pos[0].tolist()
            tgt_quat0 = target_obj_quat[0].tolist()
            eef_pos0 = achieved_hand_pos[0].tolist()
            eef_quat0 = achieved_hand_quat[0].tolist()
            ik_target_pos0 = last_target_pos[0].tolist() if last_target_pos is not None else None
            ik_target_quat0 = last_target_quat[0].tolist() if last_target_quat is not None else None
            print(
                f"[GRASP-DBG] IK done: converged_at_iter={converged_at}\n"
                f"           max_pos_err={pos_err_max:.6f} m, max_rot_err={rot_err_max:.6f} rad\n"
                f"           IK_target_pos_w[0]={ik_target_pos0}\n"
                f"           IK_target_quat(xyzw)[0]={ik_target_quat0}\n"
                f"           achieved_eef_pos_w[0]={eef_pos0}\n"
                f"           achieved_eef_quat(xyzw)[0]={eef_quat0}\n"
                f"           snapped_obj_pos_w[0]={tgt_pos0}\n"
                f"           snapped_obj_quat(xyzw)[0]={tgt_quat0}"
            )

        joint_vel = torch.zeros_like(wp.to_torch(self.robot_asset.data.joint_vel)[env_ids])
        joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()

        # Write gripper STATE at ``hand_hold_width`` (fingers just touching the
        # plug, no mesh overlap) and set the TARGET to ``hand_close_width``
        # (fully closed) so the actuator drive squeezes around the plug.
        self.gripper_joint_setter_func(
            joint_pos, list(range(num_reset_envs)), self.finger_joints, self.hand_hold_width
        )
        self.robot_asset.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot_asset.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self.gripper_joint_setter_func(
            joint_pos, list(range(num_reset_envs)), self.finger_joints, self.hand_close_width
        )
        self.robot_asset.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)


class reset_plug_at_goal_curriculum(ManagerTermBase):
    """Reset a fraction of plugs at the goal position (at-goal curriculum).

    For each reset batch, a fraction ``at_goal_prob`` of environments have the
    plug placed along the insertion axis at a random depth (from socket opening
    to full insertion) with goal orientation. The remaining environments get
    normal pose randomization.

    This replaces the simple ``reset_root_state_uniform`` for the plug when
    curriculum-based training is desired.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.plug: RigidObject = env.scene[cfg.params["plug_cfg"].name]
        self.socket: RigidObject = env.scene[cfg.params["socket_cfg"].name]

        self.at_goal_prob: float = cfg.params.get("at_goal_prob", 0.8)

        # Optional linear annealing of at_goal_prob over training iterations.
        # `at_goal_prob` is the starting value; it decays linearly to
        # `at_goal_prob_final` between `anneal_start_iter` and `anneal_end_iter`.
        # Annealing is active only when both `at_goal_prob_final` and
        # `anneal_end_iter` are provided (otherwise the probability is constant).
        # Iterations are derived from the env step counter via `num_steps_per_env`
        # (one RL iteration == `num_steps_per_env` env steps).
        self.at_goal_prob_final = cfg.params.get("at_goal_prob_final", None)
        self.anneal_start_iter: float = cfg.params.get("anneal_start_iter", 0.0)
        self.anneal_end_iter = cfg.params.get("anneal_end_iter", None)
        self.num_steps_per_env = cfg.params.get("num_steps_per_env", None)

        insertion_axis = cfg.params.get("insertion_axis", [0.0, 0.0, 1.0])
        self.insertion_axis = torch.tensor(insertion_axis, device=env.device, dtype=torch.float32)
        self.insertion_axis = self.insertion_axis / self.insertion_axis.norm()

        self.insertion_length: float = cfg.params.get("insertion_length", 0.02)

        socket_offset = cfg.params.get("socket_insertion_offset", [0.0, 0.0, 0.0])
        self.socket_insertion_offset = torch.tensor(socket_offset, device=env.device, dtype=torch.float32)

        plug_offset = cfg.params.get("plug_insertion_offset", [0.0, 0.0, 0.0])
        self.plug_insertion_offset = torch.tensor(plug_offset, device=env.device, dtype=torch.float32)

        goal_rot = cfg.params.get("goal_rot", [0.0, 0.0, 0.0, 1.0])
        self.goal_rot = torch.tensor(goal_rot, device=env.device, dtype=torch.float32)

        self.normal_pose_range: dict = cfg.params.get("normal_pose_range", {})

        self.identity_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0], device=env.device, dtype=torch.float32
        )

    def _current_at_goal_prob(self, env: ManagerBasedEnv) -> float:
        """Return the at-goal probability for the current training progress.

        Linearly interpolates from ``at_goal_prob`` to ``at_goal_prob_final``
        between ``anneal_start_iter`` and ``anneal_end_iter``. Returns the
        constant ``at_goal_prob`` when annealing is not fully configured.
        """
        if (
            self.at_goal_prob_final is None
            or self.anneal_end_iter is None
            or not self.num_steps_per_env
        ):
            return self.at_goal_prob

        current_iter = env.common_step_counter / float(self.num_steps_per_env)
        span = max(float(self.anneal_end_iter) - float(self.anneal_start_iter), 1e-9)
        frac = (current_iter - float(self.anneal_start_iter)) / span
        frac = min(max(frac, 0.0), 1.0)
        return self.at_goal_prob + frac * (float(self.at_goal_prob_final) - self.at_goal_prob)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        plug_cfg: SceneEntityCfg | None = None,
        socket_cfg: SceneEntityCfg | None = None,
        at_goal_prob: float = 0.8,
        insertion_axis: list | None = None,
        insertion_length: float = 0.02,
        socket_insertion_offset: list | None = None,
        plug_insertion_offset: list | None = None,
        goal_rot: list | None = None,
        normal_pose_range: dict | None = None,
        at_goal_prob_final: float | None = None,
        anneal_start_iter: float = 0.0,
        anneal_end_iter: float | None = None,
        num_steps_per_env: int | None = None,
    ):
        num_envs = len(env_ids)

        socket_pos = wp.to_torch(self.socket.data.root_pos_w)[env_ids]
        socket_quat = wp.to_torch(self.socket.data.root_quat_w)[env_ids]

        # Compute socket keypoint origin in world frame
        socket_offset_batch = self.socket_insertion_offset.unsqueeze(0).expand(num_envs, -1)
        id_quat_batch = self.identity_quat.unsqueeze(0).expand(num_envs, -1)
        kp_origin_w, _ = math_utils.combine_frame_transforms(
            socket_pos, socket_quat, socket_offset_batch, id_quat_batch,
        )

        # Insertion axis in world frame (rotated by socket orientation)
        insertion_axis_w = math_utils.quat_apply(socket_quat, self.insertion_axis.unsqueeze(0).expand(num_envs, -1))

        # Goal plug orientation in world frame
        goal_quat_w = math_utils.quat_mul(socket_quat, self.goal_rot.unsqueeze(0).expand(num_envs, -1))

        # Plug keypoint offset rotated into world frame (for converting kp pos -> root pos)
        plug_offset_batch = self.plug_insertion_offset.unsqueeze(0).expand(num_envs, -1)
        plug_kp_in_world = math_utils.quat_apply(goal_quat_w, plug_offset_batch)

        # Default (normal) reset: small random perturbation around default plug pose
        pose_range = self.normal_pose_range
        rand_pos = torch.zeros(num_envs, 3, device=env.device)
        for i, key in enumerate(["x", "y", "z"]):
            rng = pose_range.get(key, [0.0, 0.0])
            rand_pos[:, i] = torch.empty(num_envs, device=env.device).uniform_(rng[0], rng[1])

        default_plug_pos = wp.to_torch(self.plug.data.default_root_state)[env_ids, :3] + env.scene.env_origins[env_ids]
        normal_plug_pos = default_plug_pos + rand_pos
        normal_plug_quat = wp.to_torch(self.plug.data.default_root_state)[env_ids, 3:7]

        plug_pos = normal_plug_pos.clone()
        plug_quat = normal_plug_quat.clone()

        # At-goal curriculum: place fraction of envs along insertion axis.
        # The probability may be annealed over training iterations.
        current_at_goal_prob = self._current_at_goal_prob(env)
        if current_at_goal_prob > 0.0 and num_envs > 0:
            # Per-env Bernoulli draw: each resetting env independently has
            # probability `current_at_goal_prob` of being seeded at goal. This is
            # robust to the reset batch size, including single-env resets
            # (manager-based envs reset per-termination, so batches are often
            # size 1 -- a fixed `int(num_envs * prob)` count would round down to
            # 0 there).
            at_goal_mask = torch.rand(num_envs, device=env.device) < current_at_goal_prob
            at_goal_local = at_goal_mask.nonzero(as_tuple=False).squeeze(-1)
            num_at_goal = int(at_goal_local.numel())
            if num_at_goal > 0:
                depth_rand = torch.rand(num_at_goal, 1, device=env.device)
                goal_kp_pos = kp_origin_w[at_goal_local] + depth_rand * insertion_axis_w[at_goal_local] * self.insertion_length

                # Convert keypoint position to plug root position
                plug_pos[at_goal_local] = goal_kp_pos - plug_kp_in_world[at_goal_local]
                plug_quat[at_goal_local] = goal_quat_w[at_goal_local]

        new_root_pose = torch.cat([plug_pos, plug_quat], dim=-1)
        zero_vel = torch.zeros(num_envs, 6, device=env.device, dtype=torch.float32)
        self.plug.write_root_pose_to_sim(new_root_pose, env_ids=env_ids)
        self.plug.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
