# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .operational_space_cfg import OperationSpaceControllerCfg


class OperationSpaceController:
    """Operation-space controller.

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
    """

    def __init__(self, cfg: OperationSpaceControllerCfg, num_envs: int, device: str):
        """Initialize operation-space controller.

        Args:
            cfg: The configuration for operation-space controller.
            num_envs: The number of environments.
            device: The device to use for computations.

        Raises:
            ValueError: When invalid control command is provided.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        # resolve tasks-pace target dimensions
        self.target_list = list()
        for command_type in self.cfg.target_types:
            if command_type == "pose_rel":
                self.target_list.append(6)
            elif command_type == "pose_abs":
                self.target_list.append(7)
            elif command_type == "wrench_abs":
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
            torch.tensor(self.cfg.wrench_control_axes, dtype=torch.float, device=self._device)
        )
        # -- commands
        self._task_space_target = torch.zeros(self.num_envs, self.target_dim, device=self._device)
        # -- buffers for motion/force control
        self.desired_ee_pos = None
        self.desired_ee_rot = None
        self.desired_ee_wrench = None
        # -- motion control gains
        self._p_gains = torch.zeros(self.num_envs, 6, device=self._device)
        self._p_gains[:] = torch.tensor(self.cfg.stiffness, device=self._device)
        self._d_gains = 2 * torch.sqrt(self._p_gains) * torch.tensor(self.cfg.damping_ratio, device=self._device)
        # -- force control gains
        if self.cfg.wrench_stiffness is not None:
            self._p_wrench_gains = torch.zeros(self.num_envs, 6, device=self._device)
            self._p_wrench_gains[:] = torch.tensor(self.cfg.wrench_stiffness, device=self._device)
        else:
            self._p_wrench_gains = None
        # -- position gain limits
        self._p_gains_limits = torch.zeros(self.num_envs, 6, 2, device=self._device)
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
        # -- end-effector contact wrench
        self._ee_contact_wrench = torch.zeros(self.num_envs, 6, device=self._device)

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the action space of controller."""
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # task-space targets
            return self.target_dim
        elif self.cfg.impedance_mode == "variable_kp":
            # task-space targets + stiffness
            return self.target_dim + 6
        elif self.cfg.impedance_mode == "variable":
            # task-space targets + stiffness + damping
            return self.target_dim + 6 + 6
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

    """
    Operations.
    """

    def reset(self):
        """Reset the internals.
        """
        self.desired_ee_pos = None
        self.desired_ee_rot = None
        self.desired_ee_wrench = None

    def initialize(self):
        """Initialize the internals."""
        pass

    def reset_idx(self):
        """Reset the internals."""
        pass

    def set_command(self, command: torch.Tensor, current_ee_pose: torch.Tensor | None = None):
        """Set the task-space targets and impedance parameters.

        Args:
            command (torch.Tensor): A concatenated tensor of shape ('num_envs', 'action_dim') containing task-space
                targets (i.e., pose/wrench) and impedance parameters.
            current_ee_pose (torch.Tensor, optional): Current end-effector pose of shape ('num_envs', 7), containing position
                and quaternion (w, x, y, z). Required for relative commands. Defaults to None.

        Format:
            Task-space targets, ordered according to 'command_types':

                Absolute pose: shape ('num_envs', 7), containing position and quaternion (w, x, y, z).
                Relative pose: shape ('num_envs', 6), containing delta position and rotation in axis-angle form.
                Absolute wrench: shape ('num_envs', 6), containing force and torque.

            Impedance parameters: stiffness for 'variable_kp', or stiffness, followed by damping ratio for 'variable':

                Stiffness: shape ('num_envs', 6)
                Damping ratio: shape ('num_envs', 6)

        """
        # check input size
        if command.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_envs, self.action_dim)}'."
            )
        # impedance mode
        if self.cfg.impedance_mode == "fixed":
            # task space targets (i.e., pose/wrench)
            self._task_space_target[:] = command
        elif self.cfg.impedance_mode == "variable_kp":
            # split input command
            task_space_command, stiffness = torch.split(command, (self.target_dim, 6), dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[..., 0], max=self._p_gains_limits[..., 1])
            # task space targets + stiffness
            self._task_space_target[:] = task_space_command.squeeze(dim=-1)
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * torch.tensor(self.cfg.damping_ratio, device=self._device)
        elif self.cfg.impedance_mode == "variable":
            # split input command
            task_space_command, stiffness, damping_ratio = torch.split(command, (self.target_dim, 6, 6), dim=-1)
            # format command
            stiffness = stiffness.clip_(min=self._p_gains_limits[..., 0], max=self._p_gains_limits[..., 1])
            damping_ratio = damping_ratio.clip_(
                min=self._damping_ratio_limits[..., 0], max=self._damping_ratio_limits[..., 1]
            )
            # task space targets + stiffness + damping
            self._task_space_target[:] = task_space_command
            self._p_gains[:] = stiffness
            self._d_gains[:] = 2 * torch.sqrt(self._p_gains) * damping_ratio
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

        # resolve the target commands
        target_groups = torch.split(self._task_space_target, self.target_list, dim=1)
        for command_type, target in zip(self.cfg.target_types, target_groups):
            if command_type == "pose_rel":
                # check input is provided
                if current_ee_pose is None:
                    raise ValueError("Current pose is required for 'pose_rel' command.")
                # compute targets
                self.desired_ee_pos, self.desired_ee_rot = apply_delta_pose(current_ee_pose[:, :3], current_ee_pose[:, 3:], target)
            elif command_type == "pose_abs":
                # compute targets
                self.desired_ee_pos = target[:, 0:3]
                self.desired_ee_rot = target[:, 3:7]
            elif command_type == "wrench_abs":
                # compute targets
                self.desired_ee_wrench = target
            else:
                raise ValueError(f"Invalid control command: {self.cfg.command_type}.")

    def compute(
        self,
        jacobian: torch.Tensor,
        current_ee_pose: torch.Tensor | None = None,
        current_ee_vel: torch.Tensor | None = None,
        current_ee_force: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian: The Jacobian matrix of the end-effector.
            current_ee_pose: The current end-effector pose. It is a tensor of shape
                (num_robots, 7), which contains the position and quaternion (w, x, y, z). Defaults to None.
            current_ee_vel: The current end-effector velocity. It is a tensor of shape
                (num_robots, 6), which contains the linear and angular velocities. Defaults to None.
            current_ee_force: The current external force on the end-effector.
                It is a tensor of shape (num_robots, 3), which contains the linear force. Defaults to None.
            mass_matrix: The joint-space inertial matrix. Defaults to None.
            gravity: The joint-space gravity vector. Defaults to None.

        Raises:
            ValueError: When the current end-effector pose is not provided for the 'pose_rel' command.
            ValueError: When an invalid task-space target type is provided.
            ValueError: When motion-control is enabled but the current end-effector pose or velocity is not provided.
            ValueError: When force-control is enabled but the current end-effector force is not provided.
            ValueError: When inertial compensation is enabled but the mass matrix  is not provided.
            ValueError: When gravity compensation is enabled but the gravity vector is not provided.

        Returns:
            The target joint torques commands.
        """

        # deduce number of DoF
        nDoF = jacobian.shape[2]
        # create joint effort vector
        joint_efforts = torch.zeros(self.num_envs, nDoF, device=self._device)

        # compute for motion-control
        if self.desired_ee_pos is not None:
            # check input is provided
            if current_ee_pose is None or current_ee_vel is None:
                raise ValueError("Current end-effector pose and velocity are required for motion control.")
            # -- end-effector tracking error
            pose_error = torch.cat(
                compute_pose_error(
                    current_ee_pose[:, :3],
                    current_ee_pose[:, 3:],
                    self.desired_ee_pos,
                    self.desired_ee_rot,
                    rot_error_type="axis_angle",
                ),
                dim=-1,
            )
            velocity_error = -current_ee_vel  # zero target velocity
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
                    lambda_pos = torch.inverse(jacobian[:, 0:3] @ mass_matrix_inv @ jacobian[:, 0:3].mT)
                    lambda_ori = torch.inverse(jacobian[:, 3:6] @ mass_matrix_inv @ jacobian[:, 3:6].mT)
                    # desired end-effector wrench (from pseudo-dynamics)
                    decoupled_force = (lambda_pos @ des_ee_acc[:, 0:3].unsqueeze(-1)).squeeze(-1)
                    decoupled_torque = (lambda_ori @ des_ee_acc[:, 3:6].unsqueeze(-1)).squeeze(-1)
                    des_motion_wrench = torch.cat([decoupled_force, decoupled_torque], dim=-1)
                else:
                    # coupled dynamics
                    lambda_full = torch.inverse(jacobian @ mass_matrix_inv @ jacobian.mT)
                    des_motion_wrench = (lambda_full @ des_ee_acc.unsqueeze(-1)).squeeze(-1)
            else:
                # task-space impedance control
                # wrench = \ddot(x_des)
                des_motion_wrench = des_ee_acc
            # -- joint-space commands
            joint_efforts += (jacobian.mT @ self._selection_matrix_motion @ des_motion_wrench.unsqueeze(-1)).squeeze(-1)

        # compute for force control
        if self.desired_ee_wrench is not None:
            # -- task-space wrench
            if self.cfg.wrench_stiffness is not None:
                # check input is provided
                if current_ee_force is None:
                    raise ValueError("Current end-effector force is required for closed-loop force control.")
                # We can only measure the force component at the contact, so only apply the feedback for only the force
                # component, keep the torque component open loop
                self._ee_contact_wrench[:, 0:3] = current_ee_force
                self._ee_contact_wrench[:, 3:6] = self.desired_ee_wrench[:, 3:6]
                # closed-loop control
                des_force_wrench = self.desired_ee_wrench + self._p_wrench_gains * (
                    self.desired_ee_wrench - self._ee_contact_wrench
                )
            else:
                # open-loop control
                des_force_wrench = self.desired_ee_wrench
            # -- joint-space commands
            joint_efforts += (jacobian.mT @ self._selection_matrix_force @ des_force_wrench.unsqueeze(-1)).squeeze(-1)

        # add gravity compensation (bias correction)
        if self.cfg.gravity_compensation:
            # check input is provided
            if gravity is None:
                raise ValueError("Gravity vector is required for gravity compensation.")
            # add gravity compensation
            joint_efforts += gravity

        return joint_efforts
