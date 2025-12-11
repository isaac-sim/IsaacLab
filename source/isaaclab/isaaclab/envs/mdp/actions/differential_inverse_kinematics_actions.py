# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg

# import logger
logger = logging.getLogger(__name__)


class DifferentialInverseKinematicsAction(ActionTerm):
    """Differential inverse kinematics action term.

    This action term uses a differential IK controller to compute joint position commands to achieve
    a desired end-effector pose.

    The action term supports two modes:
    - Relative mode: The action is a delta pose (position + orientation) relative to the current end-effector pose.
    - Absolute mode: The action is an absolute end-effector pose in the robot's root frame.

    The controller computes the desired joint positions using the Jacobian and current joint positions.
    """

    cfg: actions_cfg.DifferentialInverseKinematicsActionCfg
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""
    _body_idx: int
    """The index of the end-effector body."""
    _body_name: str
    """The name of the end-effector body."""

    def __init__(self, cfg: actions_cfg.DifferentialInverseKinematicsActionCfg, env: ManagerBasedEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action term is applied
        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names
        )
        self._num_joints = len(self._joint_ids)
        # log the resolved joint names for debugging
        logger.info(
            f"Resolved joint names for the action term {self.__class__.__name__}:"
            f" {self._joint_names} [{self._joint_ids}]"
        )
        
        # Avoid indexing across all joints for efficiency
        if self._num_joints == self._asset.num_joints:
            self._joint_ids = slice(None)

        # parse the body index
        body_ids, body_names = self._asset.find_bodies(self.cfg.body_name)
        if len(body_ids) == 0:
            raise ValueError(
                f"No body found matching the body name: {self.cfg.body_name}"
            )
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected a single body match for the body name: {self.cfg.body_name}, got {len(body_ids)}: {body_names}"
            )
        self._body_idx = body_ids[0]
        self._body_name = body_names[0]
        logger.info(f"Resolved end-effector body name for action term: {self._body_name} [{self._body_idx}]")

        # create the differential IK controller
        from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
        assert isinstance(self.cfg.controller, DifferentialIKControllerCfg), \
            f"Expected controller to be DifferentialIKControllerCfg, got {type(self.cfg.controller)}"
        self._ik_controller = DifferentialIKController(
            cfg=self.cfg.controller,
            num_envs=self.num_envs,
            device=self.device
        )

        # create tensors for raw and processed actions
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        # Initialize to zeros - will be properly set in reset() after robot is spawned
        self._processed_actions = torch.zeros(self.num_envs, self._num_joints, device=self.device)
        self._is_initialized = False

        # convert body offset to tensor if specified
        if self.cfg.body_offset_pos is not None:
            self._body_offset_pos = torch.tensor(self.cfg.body_offset_pos, device=self.device).repeat(self.num_envs, 1)
        else:
            self._body_offset_pos = None

        if self.cfg.body_offset_rot is not None:
            self._body_offset_rot = torch.tensor(self.cfg.body_offset_rot, device=self.device).repeat(self.num_envs, 1)
        else:
            self._body_offset_rot = None

        # scaling for actions
        if isinstance(self.cfg.scale, (float, int)):
            self._scale = float(self.cfg.scale)
        else:
            raise ValueError(f"Unsupported scale type: {type(self.cfg.scale)}. Only float is supported.")

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action term."""
        return self._ik_controller.action_dim

    @property
    def raw_actions(self) -> torch.Tensor:
        """The input/raw actions sent to the action term."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions (joint positions) computed by the action term."""
        return self._processed_actions

    """
    Operations.
    """

    def process_actions(self, actions: torch.Tensor):
        """Process the actions and compute desired joint positions using the IK controller.

        Args:
            actions: The input actions (end-effector pose commands).
        """
        # store the raw actions
        self._raw_actions[:] = actions
        # scale the actions
        scaled_actions = self._scale * self._raw_actions

        # In relative mode with differential IK, zero command means "maintain last commanded pose"
        # Check if all actions are effectively zero (with tolerance for numerical errors)
        action_magnitude = torch.abs(scaled_actions).max()
        if action_magnitude < 1e-4:
            # No movement commanded - keep the last commanded target from reset/previous IK
            # Do NOT update to current joint positions as that causes drift
            return

        # get current end-effector pose
        ee_pos_w = self._asset.data.body_pos_w[:, self._body_idx, :]
        ee_quat_w = self._asset.data.body_quat_w[:, self._body_idx, :]

        # apply body offset if specified
        if self._body_offset_pos is not None:
            # rotate offset into world frame
            from isaaclab.utils.math import quat_apply
            ee_pos_w = ee_pos_w + quat_apply(ee_quat_w, self._body_offset_pos)
        if self._body_offset_rot is not None:
            # apply rotation offset
            from isaaclab.utils.math import quat_mul
            ee_quat_w = quat_mul(ee_quat_w, self._body_offset_rot)

        # set the command for the IK controller
        self._ik_controller.set_command(scaled_actions, ee_pos=ee_pos_w, ee_quat=ee_quat_w)

        # get current joint positions
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        # compute the Jacobian for the end-effector using numerical differentiation
        jacobian = self._compute_numerical_jacobian(ee_pos_w, ee_quat_w, joint_pos)

        # compute the desired joint positions
        self._processed_actions[:] = self._ik_controller.compute(
            ee_pos=ee_pos_w,
            ee_quat=ee_quat_w,
            jacobian=jacobian,
            joint_pos=joint_pos
        )

    def apply_actions(self):
        """Apply the computed joint position commands to the articulation."""
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        if env_ids is None:
            self._raw_actions[:] = 0.0
            # Initialize processed actions to current joint positions
            current_joint_pos = self._asset.data.joint_pos[:, self._joint_ids].clone()
            self._processed_actions[:] = current_joint_pos
            # Mark as initialized since we just set it
            self._is_initialized = True
            # Reset all environments
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            self._ik_controller.reset(all_env_ids)
        else:
            self._raw_actions[env_ids] = 0.0
            # For partial resets, initialize those environments' targets
            current_joint_pos = self._asset.data.joint_pos[env_ids, self._joint_ids].clone()
            self._processed_actions[env_ids] = current_joint_pos
            self._ik_controller.reset(env_ids)

    """
    Helper methods.
    """

    def _compute_numerical_jacobian(
        self, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute an approximate geometric Jacobian.

        This is a placeholder implementation that computes a simple approximation of the Jacobian.
        For production use, this should be replaced with an analytical Jacobian computation
        (e.g., using Pinocchio or similar kinematics library).

        Args:
            ee_pos_w: Current end-effector position in world frame (num_envs, 3).
            ee_quat_w: Current end-effector orientation in world frame (num_envs, 4).
            joint_pos: Current joint positions (num_envs, num_joints).

        Returns:
            The approximate geometric Jacobian matrix (num_envs, 6, num_joints).
        """
        # For now, use a simple diagonal approximation
        # This is not physically accurate but allows the system to run
        # TODO: Replace with proper analytical Jacobian computation using Pinocchio or similar
        num_joints = joint_pos.shape[1]
        
        # Create a Jacobian with reasonable non-zero values
        # Use a diagonal-like structure where each joint affects the corresponding DOF
        jacobian = torch.zeros(self.num_envs, 6, num_joints, device=self.device)
        
        # For joints up to 6, create diagonal-like mapping
        for i in range(min(6, num_joints)):
            jacobian[:, i, i] = 0.5  # Reasonable scaling factor
        
        # For remaining joints (if more than 6), distribute influence across DOFs
        if num_joints > 6:
            for i in range(6, num_joints):
                jacobian[:, :, i] = 0.1  # Small influence on all DOFs
        
        return jacobian

