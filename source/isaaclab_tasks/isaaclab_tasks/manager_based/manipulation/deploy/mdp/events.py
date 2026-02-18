# Copyright (c) 2025-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Class-based event terms specific to the gear assembly manipulation environments.

Migrated from PhysX to Newton. Key changes:
- Replaced ``factory_control`` dependency with ``ik_utils``
- Data access returns Warp arrays -> ``wp.to_torch()`` for torch operations
- ``default_root_state`` -> ``default_root_pose`` + ``default_root_vel`` (split on Newton)
- All quaternion operations use XYZW convention (isaaclab.utils.math)
"""

from __future__ import annotations

import os
import random
from typing import IO, TYPE_CHECKING

import torch

import warp as wp

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sim._impl.newton_manager import NewtonManager

from . import ik_utils

_ik_log_file: IO | None = None


def _get_ik_log() -> IO:
    """Get or create the IK log file handle (raw file I/O, no logging module)."""
    global _ik_log_file
    if _ik_log_file is None:
        log_path = os.path.join(os.getcwd(), "ik_grasp_pose_newton.log")
        print(f"[IK LOG] Writing IK log to: {log_path}", flush=True)
        _ik_log_file = open(log_path, "w")
    return _ik_log_file

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

    Newton migration: Uses ``ik_utils`` instead of ``factory_control`` for IK computation.
    Uses ``compute_numerical_jacobian`` instead of ``root_physx_view.get_jacobians()``.
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
                "It should be a quaternion [x, y, z, w] (XYZW convention)."
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
            )  # (3,)

        # Stack grasp offset tensors for vectorized indexing
        # Index 0=small, 1=medium, 2=large
        self.gear_grasp_offsets_stacked = torch.stack(  # (3, 3) — [num_gear_types, xyz]
            [
                self.gear_grasp_offset_tensors["gear_small"],
                self.gear_grasp_offset_tensors["gear_medium"],
                self.gear_grasp_offset_tensors["gear_large"],
            ],
            dim=0,
        )

        # Pre-cache grasp rotation offset tensor (XYZW convention)
        grasp_rot_offset = cfg.params["grasp_rot_offset"]
        self.grasp_rot_offset_tensor = (  # (N, 4) — [num_envs, xyzw]
            torch.tensor(grasp_rot_offset, device=env.device, dtype=torch.float32).unsqueeze(0).repeat(env.num_envs, 1)
        )

        # Pre-allocate buffers for batch operations
        self.gear_type_indices = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)  # (N,)
        self.local_env_indices = torch.arange(env.num_envs, device=env.device)  # (N,)
        self.gear_grasp_offsets_buffer = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)  # (N, 3)

        # Cache hand grasp/close widths
        self.hand_grasp_width = env.cfg.hand_grasp_width
        self.hand_close_width = env.cfg.hand_close_width

        # Find end effector index once
        _, _, eef_indices = self.robot_asset.find_bodies([self.end_effector_body_name])
        if len(eef_indices) == 0:
            raise ValueError(f"End effector body '{self.end_effector_body_name}' not found in robot")
        self.eef_idx = eef_indices[0]

        # Build list of arm joint IDs for numerical Jacobian
        joint_names, _, all_joints = self.robot_asset.find_joints([".*"])
        self.arm_joint_ids = list(all_joints[: self.num_arm_joints])  # list of length A (num_arm_joints)
        self.all_joints = all_joints  # list of length J (num_all_joints = A + F)
        self.finger_joints = all_joints[self.num_arm_joints :]  # list of length F (num_finger_joints)
        print(f"[GRASP INIT] all_joints ({len(all_joints)}): {all_joints}", flush=True)
        print(f"[GRASP INIT] joint_names: {joint_names}", flush=True)
        print(f"[GRASP INIT] arm_joint_ids ({len(self.arm_joint_ids)}): {self.arm_joint_ids}", flush=True)
        print(f"[GRASP INIT] finger_joints ({len(self.finger_joints)}): {self.finger_joints}", flush=True)
        print(f"[GRASP INIT] finger_joints type: {type(self.finger_joints)}", flush=True)

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
        log = _get_ik_log()
        log.write(f"[EVENT] set_robot_to_grasp_pose() called for env_ids: {env_ids.tolist()}\n")

        # Check if gear type manager exists
        if not hasattr(env, "_gear_type_manager"):
            raise RuntimeError(
                "Gear type manager not initialized. Ensure randomize_gear_type event is configured "
                "in your environment's event configuration before this event term is used."
            )

        gear_type_manager: randomize_gear_type = env._gear_type_manager

        # Notation: R = len(env_ids) (reset batch), J = num_all_joints, A = num_arm_joints

        # Slice buffers for current batch size
        num_reset_envs = len(env_ids)
        gear_type_indices = self.gear_type_indices[:num_reset_envs]  # (R,)
        local_env_indices = self.local_env_indices[:num_reset_envs]  # (R,)
        gear_grasp_offsets = self.gear_grasp_offsets_buffer[:num_reset_envs]  # (R, 3)
        grasp_rot_offset_tensor = self.grasp_rot_offset_tensor[env_ids]  # (R, 4)

        # IK loop
        log.write(f"{'='*80}\n")
        log.write(f"[IK LOOP] Starting IK with max_iterations={max_iterations}\n")
        log.write(f"{'='*80}\n")
        for i in range(max_iterations):
            log.write(f"--- Iteration {i+1}/{max_iterations} ---\n")
            # Get current joint state (convert from Warp to torch)
            joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()  # (R, J)
            joint_vel = wp.to_torch(self.robot_asset.data.joint_vel)[env_ids].clone()  # (R, J)

            if i == 0:
                log.write(f"  initial_joint_pos: {joint_pos[0].cpu().numpy()}\n")

            # Stack all gear positions and quaternions (convert from Warp)
            # Each root_link_pos_w is (N, 3); stack on dim=1 -> (N, 3, 3); then index -> (R, 3, 3)
            all_gear_pos = torch.stack(
                [
                    wp.to_torch(env.scene["factory_gear_small"].data.root_link_pos_w),  # (N, 3)
                    wp.to_torch(env.scene["factory_gear_medium"].data.root_link_pos_w),  # (N, 3)
                    wp.to_torch(env.scene["factory_gear_large"].data.root_link_pos_w),  # (N, 3)
                ],
                dim=1,
            )[env_ids]  # (R, 3, 3) — [batch, gear_type, xyz]

            # Each root_link_quat_w is (N, 4); stack on dim=1 -> (N, 3, 4); then index -> (R, 3, 4)
            all_gear_quat = torch.stack(
                [
                    wp.to_torch(env.scene["factory_gear_small"].data.root_link_quat_w),  # (N, 4)
                    wp.to_torch(env.scene["factory_gear_medium"].data.root_link_quat_w),  # (N, 4)
                    wp.to_torch(env.scene["factory_gear_large"].data.root_link_quat_w),  # (N, 4)
                ],
                dim=1,
            )[env_ids]  # (R, 3, 4) — [batch, gear_type, xyzw]

            # Get gear type indices directly as tensor
            all_gear_type_indices = gear_type_manager.get_all_gear_type_indices()  # (N,)
            gear_type_indices[:] = all_gear_type_indices[env_ids]  # (R,)

            # Select gear data using advanced indexing — picks one gear per env
            grasp_object_pos_world = all_gear_pos[local_env_indices, gear_type_indices]  # (R, 3)
            grasp_object_quat = all_gear_quat[local_env_indices, gear_type_indices]  # (R, 4)

            if i == 0:
                log.write(f"  gear_type_idx:    {gear_type_indices[0].item()}\n")
                log.write(f"  gear_pos_raw:     {grasp_object_pos_world[0].cpu().numpy()}\n")
                log.write(f"  gear_quat_raw:    {grasp_object_quat[0].cpu().numpy()}\n")
                log.write(f"  grasp_rot_offset: {grasp_rot_offset_tensor[0].cpu().numpy()}\n")

            # Apply rotation offset (XYZW convention)
            grasp_object_quat = math_utils.quat_mul(grasp_object_quat, grasp_rot_offset_tensor)  # (R, 4)

            # Get grasp offsets (vectorized)
            gear_grasp_offsets[:] = self.gear_grasp_offsets_stacked[gear_type_indices]  # (R, 3)

            # Add position randomization if specified
            if pos_randomization_range is not None:
                pos_keys = ["x", "y", "z"]
                range_list_pos = [pos_randomization_range.get(key, (0.0, 0.0)) for key in pos_keys]
                ranges_pos = torch.tensor(range_list_pos, device=env.device)  # (3, 2)
                rand_pos_offsets = math_utils.sample_uniform(
                    ranges_pos[:, 0], ranges_pos[:, 1], (len(env_ids), 3), device=env.device
                )  # (R, 3)
                gear_grasp_offsets = gear_grasp_offsets + rand_pos_offsets  # (R, 3)

            # Transform offsets from gear frame to world frame
            grasp_object_pos_world = grasp_object_pos_world + math_utils.quat_apply(  # (R, 3)
                grasp_object_quat, gear_grasp_offsets  # (R, 4), (R, 3)
            )

            # Get end effector pose (convert from Warp)
            # body_link_pos_w is (N, num_bodies, 3); index by [env_ids, eef_idx] -> (R, 3)
            eef_pos = wp.to_torch(self.robot_asset.data.body_link_pos_w)[env_ids, self.eef_idx]  # (R, 3)
            eef_quat = wp.to_torch(self.robot_asset.data.body_link_quat_w)[env_ids, self.eef_idx]  # (R, 4)

            # Compute pose error using ik_utils (XYZW convention)
            pos_error, axis_angle_error = ik_utils.get_pose_error(
                fingertip_midpoint_pos=eef_pos,  # (R, 3)
                fingertip_midpoint_quat=eef_quat,  # (R, 4)
                ctrl_target_fingertip_midpoint_pos=grasp_object_pos_world,  # (R, 3)
                ctrl_target_fingertip_midpoint_quat=grasp_object_quat,  # (R, 4)
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )  # pos_error: (R, 3), axis_angle_error: (R, 3)
            delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)  # (R, 6)

            # Check convergence
            pos_error_norm = torch.norm(pos_error, dim=-1)  # (R,)
            rot_error_norm = torch.norm(axis_angle_error, dim=-1)  # (R,)
            log.write(f"  pos_error_norm: {pos_error_norm[0].item():.10f}\n")
            log.write(f"  rot_error_norm: {rot_error_norm[0].item():.10f}\n")
            log.write(f"  pos_error:      {pos_error[0].cpu().numpy()}\n")
            log.write(f"  rot_error:      {axis_angle_error[0].cpu().numpy()}\n")
            log.write(f"  eef_pos:        {eef_pos[0].cpu().numpy()}\n")
            log.write(f"  eef_quat:       {eef_quat[0].cpu().numpy()}\n")
            log.write(f"  target_pos:     {grasp_object_pos_world[0].cpu().numpy()}\n")
            log.write(f"  target_quat:    {grasp_object_quat[0].cpu().numpy()}\n")

            if torch.all(pos_error_norm < pos_threshold) and torch.all(rot_error_norm < rot_threshold):
                log.write(f"  CONVERGED at iteration {i+1}\n")
                break

            # Compute numerical Jacobian (replaces root_physx_view.get_jacobians())
            jacobian = ik_utils.compute_numerical_jacobian(
                robot=self.robot_asset,
                arm_joint_ids=self.arm_joint_ids,
                eef_body_idx=self.eef_idx,
                env_ids=env_ids,
            )  # (R, 6, A)

            # Solve IK using damped least squares
            delta_dof_pos = ik_utils.solve_ik_dls(
                jacobian=jacobian,  # (R, 6, A)
                delta_pose=delta_hand_pose,  # (R, 6)
                lambda_val=0.1,
            )  # (R, A)

            log.write(f"  delta_dof_pos:  {delta_dof_pos[0].cpu().numpy()}\n")

            # Update joint positions (only arm joints)
            joint_pos[:, : self.num_arm_joints] = joint_pos[:, : self.num_arm_joints] + delta_dof_pos  # (R, A) += (R, A)
            # joint_vel = torch.zeros_like(joint_pos)  # (R, J) — zeroed ALL joints including gripper
            joint_vel[:, : self.num_arm_joints] = 0  # only zero arm velocities, preserve gripper

            log.write(f"  joint_pos_new:  {joint_pos[0].cpu().numpy()}\n")

            # Write to sim (arm + gripper state; gripper qpos is read-back unchanged,
            # gripper qvel is preserved from sim read on line 292)
            # self.robot_asset.set_joint_position_target(joint_pos, env_ids=env_ids)  # (R, J)
            # self.robot_asset.set_joint_velocity_target(joint_vel, env_ids=env_ids)  # (R, J)
            self.robot_asset.set_joint_position_target(joint_pos[:, :self.num_arm_joints], joint_ids=self.arm_joint_ids, env_ids=env_ids)
            self.robot_asset.set_joint_velocity_target(joint_vel[:, :self.num_arm_joints], joint_ids=self.arm_joint_ids, env_ids=env_ids)
            self.robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            NewtonManager.forward_kinematics()

        log.write(f"[IK LOOP] FINISHED after {i+1} iterations\n")
        log.write(f"  final_joint_pos: {joint_pos[0].cpu().numpy()}\n")

        # Write arm joint state to sim (gripper joints left at current state)
        joint_pos = wp.to_torch(self.robot_asset.data.joint_pos)[env_ids].clone()  # (R, J)
        joint_vel = torch.zeros_like(joint_pos)  # (R, J)

        self.robot_asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        NewtonManager.forward_kinematics()

        # Set arm actuator targets to hold current position
        self.robot_asset.set_joint_position_target(
            joint_pos[:, :self.num_arm_joints], joint_ids=self.arm_joint_ids, env_ids=env_ids
        )

        # Drive gripper via tendon actuator only (control.mujoco.ctrl).
        # MJCF ctrlrange [0, 255]: 0 = fully open, 255 = fully closed.
        # Map hand_grasp_width (driver joint range [0, 0.8]) to ctrl range [0, 255].
        all_gear_types = gear_type_manager.get_all_gear_types()
        grasp_ctrl = torch.zeros(len(env_ids), 1, device=env.device)
        for row_idx, env_id in enumerate(env_ids.tolist()):
            gear_key = all_gear_types[env_id]
            hand_grasp_width = self.hand_grasp_width[gear_key]
            grasp_ctrl[row_idx, 0] = (hand_grasp_width / 0.8) * 255.0
            log.write(f"  env={env_id}, gear={gear_key}, grasp_width={hand_grasp_width}, ctrl={grasp_ctrl[row_idx, 0].item():.1f}\n")

        log.write(f"  Setting tendon grasp ctrl: {grasp_ctrl[0].item():.1f}\n")
        self.robot_asset.set_tendon_actuator_target(grasp_ctrl, tendon_names=None)

        log.write(f"{'='*80}\n")
        log.flush()


class randomize_gears_and_base_pose(ManagerTermBase):
    """Randomize both the gear base pose and individual gear poses.

    This class-based term pre-caches all tensors needed for randomization.

    Newton migration: Uses ``default_root_pose`` + ``default_root_vel`` instead of
    ``default_root_state``. Data access returns Warp arrays -> ``wp.to_torch()``.
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

            # Newton: default_root_pose (pos+quat) and default_root_vel are separate
            default_pose = wp.to_torch(asset.data.default_root_pose)[env_ids].clone()
            default_vel = wp.to_torch(asset.data.default_root_vel)[env_ids].clone()

            positions = default_pose[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]
            orientations = math_utils.quat_mul(default_pose[:, 3:7], orientations_delta)
            velocities = default_vel + rand_vel_samples

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
            asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
