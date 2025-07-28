# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from pink.tasks import FrameTask
from isaaclab.controllers.local_frame_task import LocalFrameTask



import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import pink_actions_cfg


class PinkInverseKinematicsAction(ActionTerm):
    r"""Pink Inverse Kinematics action term.

    This action term processes the action tensor and sets these setpoints in the pink IK framework.
    The action tensor is ordered in the order of the tasks defined in PinkIKControllerCfg.
    """

    cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg
    """Configuration for the Pink Inverse Kinematics action term."""

    _asset: Articulation
    """The articulation asset to which the action term is applied."""

    def __init__(self, cfg: pink_actions_cfg.PinkInverseKinematicsActionCfg, env: ManagerBasedEnv):
        """Initialize the Pink Inverse Kinematics action term.

        Args:
            cfg: The configuration for this action term.
            env: The environment in which the action term will be applied.
        """
        super().__init__(cfg, env)
        
        self._env = env
        self._sim_dt = env.sim.get_physics_dt()
        
        # Initialize joint information
        self._initialize_joint_info()
        
        # Initialize IK controllers
        self._initialize_ik_controllers()
        
        # Initialize action tensors
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        
        # Initialize hand pose transformations
        self._initialize_hand_pose_transforms()

    def _initialize_joint_info(self) -> None:
        """Initialize joint IDs and names based on configuration."""
        # Resolve pink controlled joints
        self._isaaclab_controlled_joint_ids, self._isaaclab_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.controlled_joint_names = self._isaaclab_controlled_joint_names
        self.cfg.controller.controlled_joint_indices = self._isaaclab_controlled_joint_ids
        self._isaaclab_all_joint_ids = list(range(len(self._asset.data.joint_names)))
        self.cfg.controller.all_joint_names = self._asset.data.joint_names

        # Resolve hand joints
        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(self.cfg.hand_joint_names)
        
        # Combine all joint information
        self._controlled_joint_ids = self._isaaclab_controlled_joint_ids + self._hand_joint_ids
        self._controlled_joint_names = self._isaaclab_controlled_joint_names + self._hand_joint_names

    def _initialize_ik_controllers(self) -> None:
        """Initialize Pink IK controllers for all environments."""
        assert self._env.num_envs > 0, "Number of environments specified are less than 1."
        
        self._ik_controllers = []
        for _ in range(self._env.num_envs):
            self._ik_controllers.append(
                PinkIKController(
                    cfg=copy.deepcopy(self.cfg.controller), robot_cfg=self._env.scene.cfg.robot, device=self.device
                )
            )

    def _initialize_hand_pose_transforms(self) -> None:
        """Initialize hand pose transformation matrices."""
        # Create nominal hand pose rotation matrix
        hand_rot_offset = torch.tensor(
            self.cfg.controller.hand_rotational_offset, 
            device=self.device
        )
        self.nominal_hand_pose_rotmat = math_utils.matrix_from_quat(hand_rot_offset)

    # ==================== Properties ====================

    @property
    def hand_joint_dim(self) -> int:
        """Dimension for hand joint positions."""
        return self.cfg.controller.num_hand_joints

    @property
    def position_dim(self) -> int:
        """Dimension for position (x, y, z)."""
        return 3

    @property
    def orientation_dim(self) -> int:
        """Dimension for orientation (w, x, y, z)."""
        return 4

    @property
    def pose_dim(self) -> int:
        """Total pose dimension (position + orientation)."""
        return self.position_dim + self.orientation_dim

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (based on number of tasks and pose dimension)."""
        # Count only FrameTask instances in variable_input_tasks
        frame_tasks_count = sum(
            1 for task in self._ik_controllers[0].cfg.variable_input_tasks if isinstance(task, FrameTask)
        )
        return frame_tasks_count * self.pose_dim + self.hand_joint_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    # ==================== Action Processing ====================

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """
        # Store raw actions
        self._raw_actions[:] = actions

        # Extract and process action components
        actions_clone = actions.clone()
        self._target_hand_joint_positions = actions_clone[:, -self.hand_joint_dim:]
        
        # Get base link frame transformation
        self.base_link_frame_in_world_rf = self._get_base_link_frame_transform()
        
        # Process controlled frame poses
        controlled_frame_poses = self._extract_controlled_frame_poses(actions_clone)
        transformed_poses = self._transform_poses_to_base_link_frame(controlled_frame_poses)
        
        # Set targets for all tasks
        self._set_task_targets(transformed_poses)

    def _get_base_link_frame_transform(self) -> torch.Tensor:
        """Get the base link frame transformation matrix.
        
        Returns:
            Base link frame transformation matrix.
        """
        # Get base link frame pose in world origin
        articulation_data = self._env.scene[self.cfg.controller.articulation_name].data
        base_link_idx = articulation_data.body_names.index(self.cfg.controller.base_link_name)
        base_link_frame_in_world_origin = articulation_data.body_link_state_w[:, base_link_idx, :7]

        # Transform to environment origin frame
        base_link_frame_in_world_rf = base_link_frame_in_world_origin.clone()
        base_link_frame_in_world_rf[:, :3] -= self._env.scene.env_origins

        # Create transformation matrix
        return math_utils.make_pose(
            base_link_frame_in_world_rf[:, :3],
            math_utils.matrix_from_quat(base_link_frame_in_world_rf[:, 3:7])
        )

    def _extract_controlled_frame_poses(self, actions: torch.Tensor) -> torch.Tensor:
        """Extract controlled frame poses from action tensor.
        
        Args:
            actions: The action tensor.
            
        Returns:
            Stacked controlled frame poses tensor.
        """
        num_tasks = len(self._ik_controllers[0].cfg.variable_input_tasks)
        controlled_frames = []
        
        for task_index in range(num_tasks):
            # Extract position and orientation for this task
            pos_start = task_index * self.pose_dim
            pos_end = pos_start + self.position_dim
            quat_start = pos_end
            quat_end = (task_index + 1) * self.pose_dim
            
            position = actions[:, pos_start:pos_end]
            quaternion = actions[:, quat_start:quat_end]
            
            # Create pose matrix
            pose = math_utils.make_pose(
                position, 
                math_utils.matrix_from_quat(quaternion)
            )
            controlled_frames.append(pose)
        
        return torch.stack(controlled_frames)

    def _transform_poses_to_base_link_frame(self, poses: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform poses from world frame to base link frame.
        
        Args:
            poses: Poses in world frame.
            
        Returns:
            Tuple of (positions, rotation_matrices) in base link frame.
        """
        # Transform poses to base link frame
        base_link_inv = math_utils.pose_inv(self.base_link_frame_in_world_rf)
        transformed_poses = math_utils.pose_in_A_to_pose_in_B(poses, base_link_inv)
        
        # Extract position and rotation
        positions, rotation_matrices = math_utils.unmake_pose(transformed_poses)
        
        # Apply hand rotational offset
        rotation_matrices = rotation_matrices @ self.nominal_hand_pose_rotmat
        
        return positions, rotation_matrices

    def _set_task_targets(self, transformed_poses: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Set targets for all tasks across all environments.
        
        Args:
            transformed_poses: Tuple of (positions, rotation_matrices) in base link frame.
        """
        positions, rotation_matrices = transformed_poses
        
        for env_index, ik_controller in enumerate(self._ik_controllers):
            for task_index, task in enumerate(ik_controller.cfg.variable_input_tasks):
                if isinstance(task, LocalFrameTask):
                    target = task.transform_target_to_base
                elif isinstance(task, FrameTask):
                    target = task.transform_target_to_world
                else:
                    continue
                
                # Set position and rotation targets
                target.translation = positions[task_index, env_index, :].cpu().numpy()
                target.rotation = rotation_matrices[task_index, env_index, :].cpu().numpy()
                
                task.set_target(target)

    # ==================== Action Application ====================

    def apply_actions(self) -> None:
        """Apply the computed joint positions based on the inverse kinematics solution."""
        # Compute IK solutions for all environments
        ik_joint_positions = self._compute_ik_solutions()
        
        # Combine IK and hand joint positions
        all_joint_positions = torch.cat((ik_joint_positions, self._target_hand_joint_positions), dim=1)
        self._processed_actions = all_joint_positions
        
        # Apply joint position targets
        self._asset.set_joint_position_target(self._processed_actions, self._controlled_joint_ids)

    def _compute_ik_solutions(self) -> torch.Tensor:
        """Compute IK solutions for all environments.
        
        Returns:
            IK joint positions tensor for all environments.
        """
        ik_solutions = []
        
        for env_index, ik_controller in enumerate(self._ik_controllers):
            # Get current joint positions for this environment
            current_joint_pos = self._asset.data.joint_pos.cpu().numpy()[env_index]
            
            # Compute IK solution
            joint_pos_des = ik_controller.compute(current_joint_pos, self._sim_dt)
            ik_solutions.append(joint_pos_des)
        
        return torch.stack(ik_solutions)

    # ==================== Reset ====================

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.

        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
