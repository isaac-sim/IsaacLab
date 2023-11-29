# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import Sequence

from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.math import apply_delta_pose, compute_pose_error


@configclass
class OperationSpaceControllerCfg:
    """Configuration for operation-space controller."""

    command_types: Sequence[str] = MISSING
    """Type of command.

    It has two sub-strings joined by underscore:
        - type of command mode: "position", "pose", "force"
        - type of command resolving: "abs" (absolute), "rel" (relative)
    """

    impedance_mode: str = MISSING
    """Type of gains for motion control: "fixed", "variable", "variable_kp"."""

    uncouple_motion_wrench: bool = False
    """Whether to decouple the wrench computation from task-space pose (motion) error."""

    motion_control_axes: Sequence[int] = (1, 1, 1, 1, 1, 1)
    """Motion direction to control. Mark as 0/1 for each axis."""
    force_control_axes: Sequence[int] = (0, 0, 0, 0, 0, 0)
    """Force direction to control. Mark as 0/1 for each axis."""

    inertial_compensation: bool = False
    """Whether to perform inertial compensation for motion control (inverse dynamics)."""

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""

    stiffness: float | Sequence[float] = MISSING
    """The positional gain for determining wrenches based on task-space pose error."""

    damping_ratio: float | Sequence[float] | None = None
    """The damping ratio is used in-conjunction with positional gain to compute wrenches
    based on task-space velocity error.

    The following math operation is performed for computing velocity gains:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`.
    """

    stiffness_limits: tuple[float, float] = (0, 300)
    """Minimum and maximum values for positional gains.

    Note: Used only when :obj:`impedance_mode` is "variable" or "variable_kp".
    """

    damping_ratio_limits: tuple[float, float] = (0, 100)
    """Minimum and maximum values for damping ratios used to compute velocity gains.

    Note: Used only when :obj:`impedance_mode` is "variable".
    """

    force_stiffness: float | Sequence[float] = None
    """The positional gain for determining wrenches for closed-loop force control.

    If obj:`None`, then open-loop control of desired forces is performed.
    """

    position_command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the position command received. Used only in relative mode."""
    rotation_command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the rotation command received. Used only in relative mode."""


class OperationSpaceController:
    """Operation-space controller.

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
    """

    def __init__(self, cfg: OperationSpaceControllerCfg, num_robots: int, num_dof: int, device: str):
        """Initialize operation-space controller.

        Args:
            cfg: The configuration for operation-space controller.
            num_robots: The number of robots to control.
            num_dof: The number of degrees of freedom of the robot.
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control command is provided.
        """
        # store inputs
        self.cfg = cfg
        self.num_robots = num_robots
        self.num_dof = num_dof
        self._device = device

        # resolve tasks-pace target dimensions
        self.target_list = list()
        for command_type in self.cfg.command_types:
            if "position" in command_type:
                self.target_list.append(3)
            elif command_type == "pose_rel":
                self.target_list.append(6)
            elif command_type == "pose_abs":
                self.target_list.append(7)
            elif command_type == "force_abs":
                self.target_list.append(6)
            else:
                raise ValueError(f"Invalid control command: {command_type}.")
        self.target_dim = sum(self.target_list)

        # create buffers
        # -- selection matrices
        self._selection_matrix_motion = torch.diag(
            torch.tensor(self.cfg.motion_control_axes, dtype=torch.float, device=self._device)
        )
        self._selection_matrix_force = torch.diag(
            torch.tensor(self.cfg.force_control_axes, dtype=torch.float, device=self._device)
        )
        # -- commands
        self._task_space_target = torch.zeros(self.num_robots, self.target_dim, device=self._device)
        # -- scaling of command
        self._position_command_scale = torch.diag(torch.tensor(self.cfg.position_command_scale, device=self._device))
        self._rotation_command_scale = torch.diag(torch.tensor(self.cfg.rotation_command_scale, device=self._device))
        # -- motion control gains
        self._p_gains = torch.zeros(self.num_robots, 6, device=self._device)
        self._p_gains[:] = torch.tensor(self.cfg.stiffness, device=self._device)
        self._d_gains = 2 * torch.sqrt(self._p_gains) * torch.tensor(self.cfg.damping_ratio, device=self._device)
        # -- force control gains
        if self.cfg.force_stiffness is not None:
            self._p_wrench_gains = torch.zeros(self.num_robots, 6, device=self._device)
            self._p_wrench_gains[:] = torch.tensor(self.cfg.force_stiffness, device=self._device)
        else:
            self._p_wrench_gains = None
        # -- position gain limits
        self._p_gains_limits = torch.zeros(self.num_robots, 6, device=self._device)
        self._p_gains_limits[..., 0], self._p_gains_limits[..., 1] = (
            self.cfg.stiffness_limits[0],
            self.cfg.stiffness_limits[1],
        )
        # -- damping ratio limits
        self._damping_ratio_limits = torch.zeros_like(self._p_gains_limits)
        self._damping_ratio_limits[..., 0], self._damping_ratio_limits[..., 1] = (
            self.cfg.damping_ratio_limits[0],
            self.cfg.damping_ratio_limits[1],
        )
        # -- storing outputs
        self._desired_torques = torch.zeros(self.num_robots, self.num_dof, self.num_dof, device=self._device)

    """
    Properties.
    """

    @property
    def num_actions(self) -> int:
        """Dimension of the action space of controller."""
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # task-space pose
            return self.target_dim
        elif self.cfg.impedance_mode == "variable_kp":
            # task-space pose + stiffness
            return self.target_dim + 6
        elif self.cfg.impedance_mode == "variable":
            # task-space pose + stiffness + damping
            return self.target_dim + 6 + 6
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

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
        """Set target end-effector pose or force command.

        Args:
            command: The target end-effector pose or force command.
        """
        # check input size
        if command.shape != (self.num_robots, self.num_actions):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_robots, self.num_actions)}'."
            )
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # joint positions
            self._task_space_target[:] = command
        elif self.cfg.impedance_mode == "variable_kp":
            # split input command
            task_space_command, stiffness = torch.tensor_split(command, (self.target_dim, 6), dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[0], max=self._p_gains_limits[1])
            # joint positions + stiffness
            self._task_space_target[:] = task_space_command.squeeze(dim=-1)
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains)  # critically damped
        elif self.cfg.impedance_mode == "variable":
            # split input command
            task_space_command, stiffness, damping_ratio = torch.tensor_split(command, 3, dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[0], max=self._p_gains_limits[1])
            damping_ratio = damping_ratio.clip_(min=self._damping_ratio_limits[0], max=self._damping_ratio_limits[1])
            # joint positions + stiffness + damping
            self._task_space_target[:] = task_space_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * damping_ratio
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

    def compute(
        self,
        jacobian: torch.Tensor,
        ee_pose: torch.Tensor | None = None,
        ee_vel: torch.Tensor | None = None,
        ee_force: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian: The Jacobian matrix of the end-effector.
            ee_pose: The current end-effector pose. It is a tensor of shape
                (num_robots, 7), which contains the position and quaternion (w, x, y, z). Defaults to None.
            ee_vel: The current end-effector velocity. It is a tensor of shape
                (num_robots, 6), which contains the linear and angular velocities. Defaults to None.
            ee_force: The current external force on the end-effector.
                It is a tensor of shape (num_robots, 3), which contains the linear force. Defaults to None.
            mass_matrix: The joint-space inertial matrix. Defaults to None.
            gravity: The joint-space gravity vector. Defaults to None.

        Raises:
            ValueError: When the end-effector pose is not provided for the 'position_rel' command.
            ValueError: When the end-effector pose is not provided for the 'position_abs' command.
            ValueError: When the end-effector pose is not provided for the 'pose_rel' command.
            ValueError: When an invalid command type is provided.
            ValueError: When motion-control is enabled but the end-effector pose or velocity is not provided.
            ValueError: When force-control is enabled but the end-effector force is not provided.
            ValueError: When inertial compensation is enabled but the mass matrix  is not provided.
            ValueError: When gravity compensation is enabled but the gravity vector is not provided.

        Returns:
            The target joint torques commands.
        """
        # buffers for motion/force control
        desired_ee_pos = None
        desired_ee_rot = None
        desired_ee_force = None
        # resolve the commands
        target_groups = torch.split(self._task_space_target, self.target_list)
        for command_type, target in zip(self.cfg.command_types, target_groups):
            if command_type == "position_rel":
                # check input is provided
                if ee_pose is None:
                    raise ValueError("End-effector pose is required for 'position_rel' command.")
                # scale command
                target @= self._position_command_scale
                # compute targets
                desired_ee_pos = ee_pose[:, :3] + target
                desired_ee_rot = ee_pose[:, 3:]
            elif command_type == "position_abs":
                # check input is provided
                if ee_pose is None:
                    raise ValueError("End-effector pose is required for 'position_abs' command.")
                # compute targets
                desired_ee_pos = target
                desired_ee_rot = ee_pose[:, 3:]
            elif command_type == "pose_rel":
                # check input is provided
                if ee_pose is None:
                    raise ValueError("End-effector pose is required for 'pose_rel' command.")
                # scale command
                target[:, 0:3] @= self._position_command_scale
                target[:, 3:6] @= self._rotation_command_scale
                # compute targets
                desired_ee_pos, desired_ee_rot = apply_delta_pose(ee_pose[:, :3], ee_pose[:, 3:], target)
            elif command_type == "pose_abs":
                # compute targets
                desired_ee_pos = target[:, 0:3]
                desired_ee_rot = target[:, 3:7]
            elif command_type == "force_abs":
                # compute targets
                desired_ee_force = target
            else:
                raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

        # reset desired joint torques
        self._desired_torques[:] = 0
        # compute for motion-control
        if desired_ee_pos is not None:
            # check input is provided
            if ee_pose is None or ee_vel is None:
                raise ValueError("End-effector pose and velocity are required for motion control.")
            # -- end-effector tracking error
            pose_error = compute_pose_error(
                ee_pose[:, :3], ee_pose[:, 3:], desired_ee_pos, desired_ee_rot, rot_error_type="axis_angle"
            )
            velocity_error = -ee_vel  # zero target velocity
            # -- desired end-effector acceleration (spring damped system)
            des_ee_acc = self._p_gains * pose_error + self._d_gains * velocity_error
            # -- inertial compensation
            if self.cfg.inertial_compensation:
                # check input is provided
                if mass_matrix is None:
                    raise ValueError("Mass matrix is required for inertial compensation.")
                # compute task-space dynamics quantities
                # wrench = (J M^(-1) J^T)^(-1) * \ddot(x_des)
                mass_matrix_inv = torch.inverse(mass_matrix)
                if self.cfg.uncouple_motion_wrench:
                    # decoupled-mass matrices
                    lambda_pos = torch.inverse(jacobian[:, 0:3] @ mass_matrix_inv * jacobian[:, 0:3].T)
                    lambda_ori = torch.inverse(jacobian[:, 3:6] @ mass_matrix_inv * jacobian[:, 3:6].T)
                    # desired end-effector wrench (from pseudo-dynamics)
                    decoupled_force = lambda_pos @ des_ee_acc[:, 0:3]
                    decoupled_torque = lambda_ori @ des_ee_acc[:, 3:6]
                    des_motion_wrench = torch.cat(decoupled_force, decoupled_torque)
                else:
                    # coupled dynamics
                    lambda_full = torch.inverse(jacobian @ mass_matrix_inv * jacobian.T)
                    # desired end-effector wrench (from pseudo-dynamics)
                    des_motion_wrench = lambda_full @ des_ee_acc
            else:
                # task-space impedance control
                # wrench = \ddot(x_des)
                des_motion_wrench = des_ee_acc
            # -- joint-space wrench
            self._desired_torques += jacobian.T @ self._selection_matrix_motion @ des_motion_wrench

        # compute for force control
        if desired_ee_force is not None:
            # -- task-space wrench
            if self.cfg.stiffness is not None:
                # check input is provided
                if ee_force is None:
                    raise ValueError("End-effector force is required for closed-loop force control.")
                # closed-loop control
                des_force_wrench = desired_ee_force + self._p_wrench_gains * (desired_ee_force - ee_force)
            else:
                # open-loop control
                des_force_wrench = desired_ee_force
            # -- joint-space wrench
            self._desired_torques += jacobian.T @ self._selection_matrix_force @ des_force_wrench

        # add gravity compensation (bias correction)
        if self.cfg.gravity_compensation:
            # check input is provided
            if gravity is None:
                raise ValueError("Gravity vector is required for gravity compensation.")
            # add gravity compensation
            self._desired_torques += gravity

        return self._desired_torques
