# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from omni.isaac.lab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from .operational_space_cfg import OperationalSpaceControllerCfg


class OperationalSpaceController:
    """Operational-space controller.

    Reference:
        [1] https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf
    """

    def __init__(self, cfg: OperationalSpaceControllerCfg, num_envs: int, device: str):
        """Initialize operational-space controller.

        Args:
            cfg: The configuration for operational-space controller.
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
        # -- selection matrices, which might be defined in the task reference frame different from the root frame
        self._selection_matrix_motion_task = torch.diag_embed(
            torch.tensor(self.cfg.motion_control_axes_task, dtype=torch.float, device=self._device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        self._selection_matrix_force_task = torch.diag_embed(
            torch.tensor(self.cfg.contact_wrench_control_axes_task, dtype=torch.float, device=self._device)
            .unsqueeze(0)
            .repeat(self.num_envs, 1)
        )
        # -- selection matrices in root frame
        self._selection_matrix_motion_b = torch.zeros_like(self._selection_matrix_motion_task)
        self._selection_matrix_force_b = torch.zeros_like(self._selection_matrix_force_task)
        # -- commands
        self._task_space_target_task = torch.zeros(self.num_envs, self.target_dim, device=self._device)
        # -- buffers for motion/force control
        self.desired_ee_pose_task = None
        self.desired_ee_pose_b = None
        self.desired_ee_wrench_task = None
        self.desired_ee_wrench_b = None
        # -- motion control gains
        self._motion_p_gains_task = torch.diag_embed(
            torch.ones(self.num_envs, 6, device=self._device)
            * torch.tensor(self.cfg.motion_stiffness_task, dtype=torch.float, device=self._device)
        )
        # -- -- zero out the axes that are not motion controlled, as keeping them non-zero will cause other axes to act due to coupling
        self._motion_p_gains_task[:] = self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
        self._motion_d_gains_task = torch.diag_embed(
            2
            * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
            * torch.as_tensor(self.cfg.motion_damping_ratio_task, dtype=torch.float, device=self._device).reshape(1, -1)
        )
        # -- -- motion control gains in root frame
        self._motion_p_gains_b = torch.zeros_like(self._motion_p_gains_task)
        self._motion_d_gains_b = torch.zeros_like(self._motion_d_gains_task)
        # -- force control gains
        if self.cfg.contact_wrench_stiffness_task is not None:
            self._contact_wrench_p_gains_task = torch.diag_embed(
                torch.ones(self.num_envs, 6, device=self._device)
                * torch.tensor(self.cfg.contact_wrench_stiffness_task, dtype=torch.float, device=self._device)
            )
            self._contact_wrench_p_gains_task[:] = (
                self._selection_matrix_force_task @ self._contact_wrench_p_gains_task[:]
            )
            # -- -- force control gains in root frame
            self._contact_wrench_p_gains_b = torch.zeros_like(self._contact_wrench_p_gains_task)
        else:
            self._contact_wrench_p_gains_task = None
            self._contact_wrench_p_gains_b = None
        # -- position gain limits
        self._motion_p_gains_limits = torch.zeros(self.num_envs, 6, 2, device=self._device)
        self._motion_p_gains_limits[..., 0], self._motion_p_gains_limits[..., 1] = (
            self.cfg.motion_stiffness_limits_task[0],
            self.cfg.motion_stiffness_limits_task[1],
        )
        # -- damping ratio limits
        self._motion_damping_ratio_limits = torch.zeros_like(self._motion_p_gains_limits)
        self._motion_damping_ratio_limits[..., 0], self._motion_damping_ratio_limits[..., 1] = (
            self.cfg.motion_damping_ratio_limits_task[0],
            self.cfg.motion_damping_ratio_limits_task[1],
        )
        # -- end-effector contact wrench
        self._ee_contact_wrench_b = torch.zeros(self.num_envs, 6, device=self._device)

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
        """Reset the internals."""
        self.desired_ee_pose_b = None
        self.desired_ee_pose_task = None
        self.desired_ee_wrench_b = None
        self.desired_ee_wrench_task = None

    def initialize(self):
        """Initialize the internals."""
        pass

    def reset_idx(self):
        """Reset the internals."""
        pass

    def set_command(
        self,
        command: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_task_frame_pose_b: torch.Tensor | None = None,
    ):
        """Set the task-space targets and impedance parameters.

        Args:
            command (torch.Tensor): A concatenated tensor of shape ('num_envs', 'action_dim') containing task-space
                targets (i.e., pose/wrench) and impedance parameters.
            current_ee_pose_b (torch.Tensor, optional): Current end-effector pose, in root frame, of shape
                ('num_envs', 7), containing position and quaternion (w, x, y, z). Required for relative
                commands. Defaults to None.
            current_task_frame_pose_b: Current pose of the task frame, in root frame, in which the targets and the
                (motion/wrench) control axes are defined. It is a tensor of shape (num_envs, 7),
                containing position and the quaternion (w, x, y, z). Defaults to None.

        Format:
            Task-space targets, ordered according to 'command_types':

                Absolute pose: shape ('num_envs', 7), containing position and quaternion (w, x, y, z).
                Relative pose: shape ('num_envs', 6), containing delta position and rotation in axis-angle form.
                Absolute wrench: shape ('num_envs', 6), containing force and torque.

            Impedance parameters: stiffness for 'variable_kp', or stiffness, followed by damping ratio for 'variable':

                Stiffness: shape ('num_envs', 6)
                Damping ratio: shape ('num_envs', 6)

        """
        # Check the input dimensions
        if command.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Invalid command shape '{command.shape}'. Expected: '{(self.num_envs, self.action_dim)}'."
            )

        # Resolve the impedance parameters
        if self.cfg.impedance_mode == "fixed":
            # task space targets (i.e., pose/wrench)
            self._task_space_target_task[:] = command
        elif self.cfg.impedance_mode == "variable_kp":
            # split input command
            task_space_command, stiffness = torch.split(command, (self.target_dim, 6), dim=-1)
            # format command
            stiffness = stiffness.clip_(
                min=self._motion_p_gains_limits[..., 0], max=self._motion_p_gains_limits[..., 1]
            )
            # task space targets + stiffness
            self._task_space_target_task[:] = task_space_command.squeeze(dim=-1)
            self._motion_p_gains_task[:] = torch.diag_embed(stiffness)
            self._motion_p_gains_task[:] = self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
            self._motion_d_gains_task = torch.diag_embed(
                2
                * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
                * torch.as_tensor(self.cfg.motion_damping_ratio_task, dtype=torch.float, device=self._device).reshape(
                    1, -1
                )
            )
        elif self.cfg.impedance_mode == "variable":
            # split input command
            task_space_command, stiffness, damping_ratio = torch.split(command, (self.target_dim, 6, 6), dim=-1)
            # format command
            stiffness = stiffness.clip_(
                min=self._motion_p_gains_limits[..., 0], max=self._motion_p_gains_limits[..., 1]
            )
            damping_ratio = damping_ratio.clip_(
                min=self._motion_damping_ratio_limits[..., 0], max=self._motion_damping_ratio_limits[..., 1]
            )
            # task space targets + stiffness + damping
            self._task_space_target_task[:] = task_space_command
            self._motion_p_gains_task[:] = torch.diag_embed(stiffness)
            self._motion_p_gains_task[:] = self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
            self._motion_d_gains_task[:] = torch.diag_embed(
                2 * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt() * damping_ratio
            )
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

        if current_task_frame_pose_b is None:
            current_task_frame_pose_b = torch.tensor(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * self.num_envs, device=self._device
            )

        # Resolve the target commands
        target_groups = torch.split(self._task_space_target_task, self.target_list, dim=1)
        for command_type, target in zip(self.cfg.target_types, target_groups):
            if command_type == "pose_rel":
                # check input is provided
                if current_ee_pose_b is None:
                    raise ValueError("Current pose is required for 'pose_rel' command.")
                # Transform the current pose from base/root frame to task frame
                current_ee_pos_task, current_ee_rot_task = subtract_frame_transforms(
                    current_task_frame_pose_b[:, :3],
                    current_task_frame_pose_b[:, 3:],
                    current_ee_pose_b[:, :3],
                    current_ee_pose_b[:, 3:],
                )
                # compute targets in task frame
                desired_ee_pos_task, desired_ee_rot_task = apply_delta_pose(
                    current_ee_pos_task, current_ee_rot_task, target
                )
                self.desired_ee_pose_task = torch.cat([desired_ee_pos_task, desired_ee_rot_task], dim=-1)
            elif command_type == "pose_abs":
                # compute targets
                self.desired_ee_pose_task = target.clone()
            elif command_type == "wrench_abs":
                # compute targets
                self.desired_ee_wrench_task = target.clone()
            else:
                raise ValueError(f"Invalid control command: {command_type}.")

        # Rotation of task frame wrt root frame, converts a coordinate from task frame to root frame.
        R_task_b = matrix_from_quat(current_task_frame_pose_b[:, 3:])
        # Rotation of root frame wrt task frame, converts a coordinate from root frame to task frame.
        R_b_task = R_task_b.mT

        # Transform motion control stiffness gains from task frame to root frame
        self._motion_p_gains_b[:, 0:3, 0:3] = R_task_b @ self._motion_p_gains_task[:, 0:3, 0:3] @ R_b_task
        self._motion_p_gains_b[:, 3:6, 3:6] = R_task_b @ self._motion_p_gains_task[:, 3:6, 3:6] @ R_b_task

        # Transform motion control damping gains from task frame to root frame
        self._motion_d_gains_b[:, 0:3, 0:3] = R_task_b @ self._motion_d_gains_task[:, 0:3, 0:3] @ R_b_task
        self._motion_d_gains_b[:, 3:6, 3:6] = R_task_b @ self._motion_d_gains_task[:, 3:6, 3:6] @ R_b_task

        # Transform contact wrench gains from task frame to root frame (if applicable)
        if self._contact_wrench_p_gains_task is not None and self._contact_wrench_p_gains_b is not None:
            self._contact_wrench_p_gains_b[:, 0:3, 0:3] = (
                R_task_b @ self._contact_wrench_p_gains_task[:, 0:3, 0:3] @ R_b_task
            )
            self._contact_wrench_p_gains_b[:, 3:6, 3:6] = (
                R_task_b @ self._contact_wrench_p_gains_task[:, 3:6, 3:6] @ R_b_task
            )

        # Transform selection matrices from target frame to base frame
        self._selection_matrix_motion_b[:, 0:3, 0:3] = (
            R_task_b @ self._selection_matrix_motion_task[:, 0:3, 0:3] @ R_b_task
        )
        self._selection_matrix_motion_b[:, 3:6, 3:6] = (
            R_task_b @ self._selection_matrix_motion_task[:, 3:6, 3:6] @ R_b_task
        )
        self._selection_matrix_force_b[:, 0:3, 0:3] = (
            R_task_b @ self._selection_matrix_force_task[:, 0:3, 0:3] @ R_b_task
        )
        self._selection_matrix_force_b[:, 3:6, 3:6] = (
            R_task_b @ self._selection_matrix_force_task[:, 3:6, 3:6] @ R_b_task
        )

        # Transform desired pose from task frame to root frame
        if self.desired_ee_pose_task is not None:
            self.desired_ee_pose_b = torch.zeros_like(self.desired_ee_pose_task)
            self.desired_ee_pose_b[:, :3], self.desired_ee_pose_b[:, 3:] = combine_frame_transforms(
                current_task_frame_pose_b[:, :3],
                current_task_frame_pose_b[:, 3:],
                self.desired_ee_pose_task[:, :3],
                self.desired_ee_pose_task[:, 3:],
            )

        # Transform desired wrenches to root frame
        if self.desired_ee_wrench_task is not None:
            self.desired_ee_wrench_b = torch.zeros_like(self.desired_ee_wrench_task)
            self.desired_ee_wrench_b[:, :3] = (R_task_b @ self.desired_ee_wrench_task[:, :3].unsqueeze(-1)).squeeze(-1)
            self.desired_ee_wrench_b[:, 3:] = (R_task_b @ self.desired_ee_wrench_task[:, 3:].unsqueeze(-1)).squeeze(
                -1
            ) + torch.cross(current_task_frame_pose_b[:, :3], self.desired_ee_wrench_b[:, :3], dim=-1)

    def compute(
        self,
        jacobian_b: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_ee_vel_b: torch.Tensor | None = None,
        current_ee_force_b: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian_b: The Jacobian matrix of the end-effector in root frame. It is a tensor of shape
                (num_envs, 6, num_DoF).
            current_ee_pose_b: The current end-effector pose in root frame. It is a tensor of shape
                (num_envs, 7), which contains the position and quaternion (w, x, y, z). Defaults to None.
            current_ee_vel_b: The current end-effector velocity in root frame. It is a tensor of shape
                (num_envs, 6), which contains the linear and angular velocities. Defaults to None.
            current_ee_force_b: The current external force on the end-effector in root frame.
                It is a tensor of shape (num_envs, 3), which contains the linear force. Defaults to None.
            mass_matrix: The joint-space mass/inertia matrix. It is a tensor of shape (num_envs, num_DoF, num_DoF).
                Defaults to None.
            gravity: The joint-space gravity vector. It is a tensor of shape (num_envs, num_DoF). Defaults to None.

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
        num_DoF = jacobian_b.shape[2]
        # create joint effort vector
        joint_efforts = torch.zeros(self.num_envs, num_DoF, device=self._device)

        # compute joint efforts for motion-control
        if self.desired_ee_pose_b is not None:
            # check input is provided
            if current_ee_pose_b is None or current_ee_vel_b is None:
                raise ValueError("Current end-effector pose and velocity are required for motion control.")
            # -- end-effector tracking error
            pose_error_b = torch.cat(
                compute_pose_error(
                    current_ee_pose_b[:, :3],
                    current_ee_pose_b[:, 3:],
                    self.desired_ee_pose_b[:, :3],
                    self.desired_ee_pose_b[:, 3:],
                    rot_error_type="axis_angle",
                ),
                dim=-1,
            )
            velocity_error_b = -current_ee_vel_b  # zero target velocity
            # -- desired end-effector acceleration (spring damped system)
            des_ee_acc_b = self._motion_p_gains_b @ pose_error_b.unsqueeze(
                -1
            ) + self._motion_d_gains_b @ velocity_error_b.unsqueeze(-1)
            # -- inertial compensation
            if self.cfg.inertial_compensation:
                # check input is provided
                if mass_matrix is None:
                    raise ValueError("Mass matrix is required for inertial compensation.")
                # compute task-space dynamics quantities
                # operational space command force = (J M^(-1) J^T)^(-1) * \ddot(x_des)
                mass_matrix_inv = torch.inverse(mass_matrix)
                if self.cfg.decoupled_motion_calculations:
                    # decoupled-mass matrices
                    os_mass_matrix_pos = torch.inverse(jacobian_b[:, 0:3] @ mass_matrix_inv @ jacobian_b[:, 0:3].mT)
                    os_mass_matrix_ori = torch.inverse(jacobian_b[:, 3:6] @ mass_matrix_inv @ jacobian_b[:, 3:6].mT)
                    # (Generalized) operational space command forces (from pseudo-dynamics)
                    decoupled_command_force_b = os_mass_matrix_pos @ des_ee_acc_b[:, 0:3]
                    decoupled_command_torque_b = os_mass_matrix_ori @ des_ee_acc_b[:, 3:6]
                    os_command_forces_b = torch.cat([decoupled_command_force_b, decoupled_command_torque_b], dim=1)
                else:
                    # coupled dynamics
                    os_mass_matrix_full = torch.inverse(jacobian_b @ mass_matrix_inv @ jacobian_b.mT)
                    # (Generalized) operational space command forces
                    os_command_forces_b = os_mass_matrix_full @ des_ee_acc_b
            else:
                # task-space impedance control: command forces = \ddot(x_des)
                os_command_forces_b = des_ee_acc_b
            # -- joint-space commands
            joint_efforts += (jacobian_b.mT @ self._selection_matrix_motion_b @ os_command_forces_b).squeeze(-1)

        # compute joint efforts for contact wrench/force control
        if self.desired_ee_wrench_b is not None:
            # -- task-space contact wrench
            if self.cfg.contact_wrench_stiffness_task is not None:
                # check input is provided
                if current_ee_force_b is None:
                    raise ValueError("Current end-effector force is required for closed-loop force control.")
                # We can only measure the force component at the contact, so only apply the feedback for only the force
                # component, keep the control of moment components open loop
                self._ee_contact_wrench_b[:, 0:3] = current_ee_force_b
                self._ee_contact_wrench_b[:, 3:6] = self.desired_ee_wrench_b[:, 3:6]
                # closed-loop control with feedforward term
                os_contact_wrench_command_b = self.desired_ee_wrench_b.unsqueeze(
                    -1
                ) + self._contact_wrench_p_gains_b @ (self.desired_ee_wrench_b - self._ee_contact_wrench_b).unsqueeze(
                    -1
                )
            else:
                # open-loop control
                os_contact_wrench_command_b = self.desired_ee_wrench_b.unsqueeze(-1)
            # -- joint-space commands
            joint_efforts += (jacobian_b.mT @ self._selection_matrix_force_b @ os_contact_wrench_command_b).squeeze(-1)

        # add gravity compensation (bias correction)
        if self.cfg.gravity_compensation:
            # check input is provided
            if gravity is None:
                raise ValueError("Gravity vector is required for gravity compensation.")
            # add gravity compensation
            joint_efforts += gravity

        return joint_efforts
