# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.utils.io_descriptors import GenericActionIODescriptor

    from . import pink_actions_cfg


class PinkInverseKinematicsAction(ActionTerm):
    r"""Pink Inverse Kinematics action term.

    This action term processes the action tensor and sets these setpoints in the pink IK framework
    The action tensor is ordered in the order of the tasks defined in PinkIKControllerCfg
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

        # Resolve joint IDs and names based on the configuration
        self._pink_controlled_joint_ids, self._pink_controlled_joint_names = self._asset.find_joints(
            self.cfg.pink_controlled_joint_names
        )
        self.cfg.controller.joint_names = self._pink_controlled_joint_names
        self._hand_joint_ids, self._hand_joint_names = self._asset.find_joints(self.cfg.hand_joint_names)
        self._joint_ids = self._pink_controlled_joint_ids + self._hand_joint_ids
        self._joint_names = self._pink_controlled_joint_names + self._hand_joint_names

        # Initialize the Pink IK controller
        assert env.num_envs > 0, "Number of environments specified are less than 1."
        self._ik_controllers = []
        for _ in range(env.num_envs):
            self._ik_controllers.append(PinkIKController(cfg=copy.deepcopy(self.cfg.controller), device=self.device))

        # Create tensors to store raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self.raw_actions)

        # Get the simulation time step
        self._sim_dt = env.sim.get_physics_dt()

        self.total_time = 0  # Variable to accumulate the total time
        self.num_runs = 0  # Counter for the number of runs

        # Save the base_link_frame pose in the world frame as a transformation matrix in
        # order to transform the desired pose of the controlled_frame to be with respect to the base_link_frame
        # Shape of env.scene[self.cfg.articulation_name].data.body_link_state_w is (num_instances, num_bodies, 13)
        base_link_frame_in_world_origin = env.scene[self.cfg.controller.articulation_name].data.body_link_state_w[
            :,
            env.scene[self.cfg.controller.articulation_name].data.body_names.index(self.cfg.controller.base_link_name),
            :7,
        ]

        # Get robot base link frame in env origin frame
        base_link_frame_in_env_origin = copy.deepcopy(base_link_frame_in_world_origin)
        base_link_frame_in_env_origin[:, :3] -= self._env.scene.env_origins

        self.base_link_frame_in_env_origin = math_utils.make_pose(
            base_link_frame_in_env_origin[:, :3], math_utils.matrix_from_quat(base_link_frame_in_env_origin[:, 3:7])
        )

    # """
    # Properties.
    # """

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
        # The tasks for all the controllers are the same, hence just using the first one to calculate the action_dim
        return len(self._ik_controllers[0].cfg.variable_input_tasks) * self.pose_dim + self.hand_joint_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """Get the raw actions tensor."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """Get the processed actions tensor."""
        return self._processed_actions

    @property
    def IO_descriptor(self) -> GenericActionIODescriptor:
        """The IO descriptor of the action term.

        This descriptor is used to describe the action term of the pink inverse kinematics action.
        It adds the following information to the base descriptor:
        - scale: The scale of the action term.
        - offset: The offset of the action term.
        - clip: The clip of the action term.
        - pink_controller_joint_names: The names of the pink controller joints.
        - hand_joint_names: The names of the hand joints.
        - controller_cfg: The configuration of the pink controller.

        Returns:
            The IO descriptor of the action term.
        """
        super().IO_descriptor
        self._IO_descriptor.shape = (self.action_dim,)
        self._IO_descriptor.dtype = str(self.raw_actions.dtype)
        self._IO_descriptor.action_type = "PinkInverseKinematicsAction"
        self._IO_descriptor.pink_controller_joint_names = self._pink_controlled_joint_names
        self._IO_descriptor.hand_joint_names = self._hand_joint_names
        self._IO_descriptor.extras["controller_cfg"] = self.cfg.controller.__dict__
        return self._IO_descriptor

    # """
    # Operations.
    # """

    def process_actions(self, actions: torch.Tensor):
        """Process the input actions and set targets for each task.

        Args:
            actions: The input actions tensor.
        """
        # Store the raw actions
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions

        # Make a copy of actions before modifying so that raw actions are not modified
        actions_clone = actions.clone()

        # Extract hand joint positions (last 22 values)
        self._target_hand_joint_positions = actions_clone[:, -self.hand_joint_dim :]

        # The action tensor provides the desired pose of the controlled_frame with respect to the env origin frame
        # But the pink IK controller expects the desired pose of the controlled_frame with respect to the base_link_frame
        # So we need to transform the desired pose of the controlled_frame to be with respect to the base_link_frame

        # Get the controlled_frame pose wrt to the env origin frame
        all_controlled_frames_in_env_origin = []
        # The contrllers for all envs are the same, hence just using the first one to get the number of variable_input_tasks
        for task_index in range(len(self._ik_controllers[0].cfg.variable_input_tasks)):
            controlled_frame_in_env_origin_pos = actions_clone[
                :, task_index * self.pose_dim : task_index * self.pose_dim + self.position_dim
            ]
            controlled_frame_in_env_origin_quat = actions_clone[
                :, task_index * self.pose_dim + self.position_dim : (task_index + 1) * self.pose_dim
            ]
            controlled_frame_in_env_origin = math_utils.make_pose(
                controlled_frame_in_env_origin_pos, math_utils.matrix_from_quat(controlled_frame_in_env_origin_quat)
            )
            all_controlled_frames_in_env_origin.append(controlled_frame_in_env_origin)
        # Stack all the controlled_frame poses in the env origin frame. Shape is (num_tasks, num_envs , 4, 4)
        all_controlled_frames_in_env_origin = torch.stack(all_controlled_frames_in_env_origin)

        # Transform the controlled_frame to be with respect to the base_link_frame using batched matrix multiplication
        controlled_frame_in_base_link_frame = math_utils.pose_in_A_to_pose_in_B(
            all_controlled_frames_in_env_origin, math_utils.pose_inv(self.base_link_frame_in_env_origin)
        )

        controlled_frame_in_base_link_frame_pos, controlled_frame_in_base_link_frame_mat = math_utils.unmake_pose(
            controlled_frame_in_base_link_frame
        )

        # Loop through each task and set the target
        for env_index, ik_controller in enumerate(self._ik_controllers):
            for task_index, task in enumerate(ik_controller.cfg.variable_input_tasks):
                target = task.transform_target_to_world
                target.translation = controlled_frame_in_base_link_frame_pos[task_index, env_index, :].cpu().numpy()
                target.rotation = controlled_frame_in_base_link_frame_mat[task_index, env_index, :].cpu().numpy()
                task.set_target(target)

    def apply_actions(self):
        # start_time = time.time()  # Capture the time before the step
        """Apply the computed joint positions based on the inverse kinematics solution."""
        all_envs_joint_pos_des = []
        for env_index, ik_controller in enumerate(self._ik_controllers):
            curr_joint_pos = self._asset.data.joint_pos[:, self._pink_controlled_joint_ids].cpu().numpy()[env_index]
            joint_pos_des = ik_controller.compute(curr_joint_pos, self._sim_dt)
            all_envs_joint_pos_des.append(joint_pos_des)
        all_envs_joint_pos_des = torch.stack(all_envs_joint_pos_des)

        # Combine IK joint positions with hand joint positions
        all_envs_joint_pos_des = torch.cat((all_envs_joint_pos_des, self._target_hand_joint_positions), dim=1)
        self._processed_actions = all_envs_joint_pos_des

        self._asset.set_joint_position_target(self._processed_actions, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        """Reset the action term for specified environments.

        Args:
            env_ids: A list of environment IDs to reset. If None, all environments are reset.
        """
        self._raw_actions[env_ids] = torch.zeros(self.action_dim, device=self.device)
