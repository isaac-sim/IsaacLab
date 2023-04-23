# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING
from typing import Dict, Optional, Tuple

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    quat_apply,
    quat_inv,
)


@configclass
class DifferentialInverseKinematicsCfg:
    """Configuration for inverse differential kinematics controller."""

    command_type: str = MISSING
    """Type of command: "position_abs", "position_rel", "pose_abs", "pose_rel"."""

    ik_method: str = MISSING
    """Method for computing inverse of Jacobian: "pinv", "svd", "trans", "dls"."""

    ik_params: Optional[Dict[str, float]] = None
    """Parameters for the inverse-kinematics method. (default: obj:`None`).

    - Moore-Penrose pseudo-inverse ("pinv"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
    - Adaptive Singular Value Decomposition ("svd"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
        - "min_singular_value": Single values less than this are suppressed to zero (default: 1e-5).
    - Jacobian transpose ("trans"):
        - "k_val": Scaling of computed delta-dof positions (default: 1.0).
    - Damped Moore-Penrose pseudo-inverse ("dls"):
        - "lambda_val": Damping coefficient (default: 0.1).
    """

    position_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Position offset from parent body to end-effector frame in parent body frame."""
    rotation_offset: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Rotational offset from parent body to end-effector frame in parent body frame."""

    position_command_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the position command received. Used only in relative mode."""
    rotation_command_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the rotation command received. Used only in relative mode."""


class DifferentialInverseKinematics:
    """Inverse differential kinematics controller.

    This controller uses the Jacobian mapping from joint-space velocities to end-effector velocities
    to compute the delta-change in the joint-space that moves the robot closer to a desired end-effector
    position.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:
        - "pinv": Moore-Penrose pseudo-inverse
        - "svd": Adaptive singular-value decomposition (SVD)
        - "trans": Transpose of matrix
        - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)

    Note: We use the quaternions in the convention: [w, x, y, z].

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
        [2] https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
    """

    _DEFAULT_IK_PARAMS = {
        "pinv": {"k_val": 1.0},
        "svd": {"k_val": 1.0, "min_singular_value": 1e-5},
        "trans": {"k_val": 1.0},
        "dls": {"lambda_val": 0.1},
    }
    """Default parameters for different inverse kinematics approaches."""

    def __init__(self, cfg: DifferentialInverseKinematicsCfg, num_robots: int, device: str):
        """Initialize the controller.

        Args:
            cfg (DifferentialInverseKinematicsCfg): The configuration for the controller.
            num_robots (int): The number of robots to control.
            device (str): The device to use for computations.

        Raises:
            ValueError: When configured IK-method is not supported.
            ValueError: When configured command type is not supported.
        """
        # store inputs
        self.cfg = cfg
        self.num_robots = num_robots
        self._device = device
        # check valid input
        if self.cfg.ik_method not in ["pinv", "svd", "trans", "dls"]:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}.")
        if self.cfg.command_type not in ["position_abs", "position_rel", "pose_abs", "pose_rel"]:
            raise ValueError(f"Unsupported inverse-kinematics command: {self.cfg.command_type}.")

        # update parameters for IK-method
        self._ik_params = self._DEFAULT_IK_PARAMS[self.cfg.ik_method].copy()
        if self.cfg.ik_params is not None:
            self._ik_params.update(self.cfg.ik_params)
        # end-effector offsets
        # -- position
        tool_child_link_pos = torch.tensor(self.cfg.position_offset, device=self._device)
        self._tool_child_link_pos = tool_child_link_pos.repeat(self.num_robots, 1)
        # -- orientation
        tool_child_link_rot = torch.tensor(self.cfg.rotation_offset, device=self._device)
        self._tool_child_link_rot = tool_child_link_rot.repeat(self.num_robots, 1)
        # transform from tool -> parent frame
        self._tool_parent_link_rot = quat_inv(self._tool_child_link_rot)
        self._tool_parent_link_pos = -quat_apply(self._tool_parent_link_rot, self._tool_child_link_pos)
        # scaling of command
        self._position_command_scale = torch.diag(torch.tensor(self.cfg.position_command_scale, device=self._device))
        self._rotation_command_scale = torch.diag(torch.tensor(self.cfg.rotation_command_scale, device=self._device))

        # create buffers
        self.desired_ee_pos = torch.zeros(self.num_robots, 3, device=self._device)
        self.desired_ee_rot = torch.zeros(self.num_robots, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_robots, self.num_actions, device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        if "position" in self.cfg.command_type:
            return 3
        elif self.cfg.command_type == "pose_rel":
            return 6
        elif self.cfg.command_type == "pose_abs":
            return 7
        else:
            raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

    """
    Operations.
    """

    def initialize(self):
        """Initialize the internals."""
        pass

    def reset_idx(self, robot_ids: torch.Tensor = None):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor):
        """Set target end-effector pose command."""
        # check input size
        if command.shape != (self.num_robots, self.num_actions):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_robots, self.num_actions)}'."
            )
        # store command
        self._command[:] = command

    def compute(
        self,
        current_ee_pos: torch.Tensor,
        current_ee_rot: torch.Tensor,
        jacobian: torch.Tensor,
        joint_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Returns:
            torch.Tensor: The target joint positions commands.
        """
        # compute the desired end-effector pose
        if "position_rel" in self.cfg.command_type:
            # scale command
            self._command @= self._position_command_scale
            # compute targets
            self.desired_ee_pos = current_ee_pos + self._command
            self.desired_ee_rot = current_ee_rot
        elif "position_abs" in self.cfg.command_type:
            # compute targets
            self.desired_ee_pos = self._command
            self.desired_ee_rot = current_ee_rot
        elif "pose_rel" in self.cfg.command_type:
            # scale command
            self._command[:, 0:3] @= self._position_command_scale
            self._command[:, 3:6] @= self._rotation_command_scale
            # compute targets
            self.desired_ee_pos, self.desired_ee_rot = apply_delta_pose(current_ee_pos, current_ee_rot, self._command)
        elif "pose_abs" in self.cfg.command_type:
            # compute targets
            self.desired_ee_pos = self._command[:, 0:3]
            self.desired_ee_rot = self._command[:, 3:7]
        else:
            raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

        # transform from ee -> parent
        # TODO: Make this optional to reduce overhead?
        desired_parent_pos, desired_parent_rot = combine_frame_transforms(
            self.desired_ee_pos, self.desired_ee_rot, self._tool_parent_link_pos, self._tool_parent_link_rot
        )
        # transform from ee -> parent
        # TODO: Make this optional to reduce overhead?
        current_parent_pos, current_parent_rot = combine_frame_transforms(
            current_ee_pos, current_ee_rot, self._tool_parent_link_pos, self._tool_parent_link_rot
        )
        # compute pose error between current and desired
        position_error, axis_angle_error = compute_pose_error(
            current_parent_pos, current_parent_rot, desired_parent_pos, desired_parent_rot, rot_error_type="axis_angle"
        )
        # compute the delta in joint-space
        if "position" in self.cfg.command_type:
            jacobian_pos = jacobian[:, 0:3]
            delta_joint_positions = self._compute_delta_dof_pos(delta_pose=position_error, jacobian=jacobian_pos)
        else:
            pose_error = torch.cat((position_error, axis_angle_error), dim=1)
            delta_joint_positions = self._compute_delta_dof_pos(delta_pose=pose_error, jacobian=jacobian)
        # return the desired joint positions
        return joint_positions + delta_joint_positions

    """
    Helper functions.
    """

    def _compute_delta_dof_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in dos-position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired end-effector
        position.

        Args:
            delta_pose (torch.Tensor): The desired delta pose in shape [N, 3 or 6].
            jacobian (torch.Tensor): The geometric jacobian matrix in shape [N, 3 or 6, num-dof]

        Returns:
            torch.Tensor: The desired delta in joint space.
        """
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self._ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self._ik_params["k_val"]
            min_singular_value = self._ik_params["min_singular_value"]
            # computation
            # U: 6xd, S: dxd, V: d x num-dof
            U, S, Vh = torch.linalg.svd(jacobian)
            S_inv = 1.0 / S
            S_inv = torch.where(S > min_singular_value, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2)[:, :, :6]
                @ torch.diag_embed(S_inv)
                @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_dof_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self._ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_dof_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self._ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_dof_pos = jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            delta_dof_pos = delta_dof_pos.squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_dof_pos
