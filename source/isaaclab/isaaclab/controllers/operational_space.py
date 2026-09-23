# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import (
    apply_delta_pose,
    combine_frame_transforms,
    compute_pose_error,
    matrix_from_quat,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from .operational_space_cfg import OperationalSpaceControllerCfg


def _compute_task_space_forces(
    inverse_task_space_mass_matrix: torch.Tensor,
    accelerations: torch.Tensor,
    conditioning_thresholds: tuple[float, float],
) -> torch.Tensor:
    """Solve for task forces, selectively damping poorly conditioned inertia directions."""
    inverse_task_space_mass_matrix = 0.5 * (inverse_task_space_mass_matrix + inverse_task_space_mass_matrix.mT)
    lower, upper = conditioning_thresholds
    num_matrices = inverse_task_space_mass_matrix.shape[0]
    # Positive definiteness of A - upper * ||A||_F * I certifies lambda_min(A) / lambda_max(A) > upper.
    # Only matrices that fail this sufficient condition need an eigendecomposition.
    shifted = inverse_task_space_mass_matrix.clone()
    shifted.diagonal(dim1=-2, dim2=-1).sub_(
        upper * torch.linalg.vector_norm(inverse_task_space_mass_matrix, dim=(-2, -1)).unsqueeze(-1)
    )
    _, info = torch.linalg.cholesky_ex(shifted)
    damping_ids = torch.nonzero(info).flatten()
    if damping_ids.numel() < num_matrices:
        forces = torch.linalg.solve_ex(inverse_task_space_mass_matrix, accelerations)[0]
        if damping_ids.numel() == 0:
            return forces
        inverse_task_space_mass_matrix = inverse_task_space_mass_matrix[damping_ids]
        accelerations = accelerations[damping_ids]

    eigenvalues, eigenvectors = torch.linalg.eigh(inverse_task_space_mass_matrix)
    tiny = torch.finfo(eigenvalues.dtype).tiny
    eigenvalues = eigenvalues.clamp_min(0.0)
    scale = eigenvalues[:, -1:].clamp_min(tiny)
    relative_eigenvalues = eigenvalues / scale
    weights = ((relative_eigenvalues - lower) / (upper - lower)).clamp(0.0, 1.0)
    weights = weights.square() * (3.0 - 2.0 * weights)
    damping = lower * scale * (1.0 - weights)
    inverse_eigenvalues = (eigenvalues + damping).clamp_min(tiny).reciprocal()
    inverse_eigenvalues = torch.where(eigenvalues[:, -1:] > 0.0, inverse_eigenvalues, 0.0)
    damped_forces = eigenvectors @ (inverse_eigenvalues.unsqueeze(-1) * (eigenvectors.mT @ accelerations))
    if damping_ids.numel() == num_matrices:
        return damped_forces
    forces[damping_ids] = damped_forces
    return forces


class OperationalSpaceController:
    """Operational-space controller.

    Reference:

    1. `A unified approach for motion and force control of robot manipulators: The operational space formulation <http://dx.doi.org/10.1109/JRA.1987.1087068>`_
       by Oussama Khatib (Stanford University)
    2. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    """

    def __init__(self, cfg: OperationalSpaceControllerCfg, num_envs: int, device: str):
        """Initialize operational-space controller.

        Args:
            cfg: The configuration for operational-space controller.
            num_envs: The number of environments.
            device: The device to use for computations.

        Raises:
            ValueError: When an invalid control command or inertia conditioning thresholds are provided.
        """
        self._controller = None
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device

        lower, upper = self.cfg.inertia_conditioning_thresholds
        if not 0.0 < lower < upper <= 1.0:
            raise ValueError("Inertia conditioning thresholds must satisfy 0 < lower < upper <= 1.")

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
        self._task_frame_pose_b = torch.zeros(self.num_envs, 7, device=self._device)
        self._task_frame_pose_b[:, 6] = 1.0
        # -- Placeholders for motion/force control
        self.desired_ee_pose_task = None
        self.desired_ee_pose_b = None
        self.desired_ee_wrench_task = None
        self.desired_ee_wrench_b = None
        # -- motion control gains
        self._motion_p_gains_task = torch.diag_embed(
            torch.ones(self.num_envs, 6, device=self._device)
            * torch.tensor(self.cfg.motion_stiffness_task, dtype=torch.float, device=self._device)
        )
        # -- -- zero out the axes that are not motion controlled, as keeping them non-zero will cause other axes
        # -- -- to act due to coupling
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

        # -- buffers for null-space control gains
        self._nullspace_p_gain = torch.tensor(self.cfg.nullspace_stiffness, dtype=torch.float, device=self._device)
        self._nullspace_d_gain = (
            2
            * torch.sqrt(self._nullspace_p_gain)
            * torch.tensor(self.cfg.nullspace_damping_ratio, dtype=torch.float, device=self._device)
        )

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

    def set_command(
        self,
        command: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None = None,
        current_task_frame_pose_b: torch.Tensor | None = None,
    ):
        """Set the task-space targets and impedance parameters.

        Args:
            command (torch.Tensor): A concatenated tensor of shape (``num_envs``, ``action_dim``) containing task-space
                targets (i.e., pose/wrench) and impedance parameters.
            current_ee_pose_b (torch.Tensor, optional): Current end-effector pose, in root frame, of shape
                (``num_envs``, 7), containing position and quaternion ``(x, y, z, w)``. Required for relative
                commands and used as the orientation fallback for degenerate absolute pose quaternions.
                Defaults to None.
            current_task_frame_pose_b: Current pose of the task frame, in root frame, in which the targets and the
                (motion/wrench) control axes are defined. It is a tensor of shape (``num_envs``, 7),
                containing position and the quaternion ``(x, y, z, w)``. Defaults to None.

        Format:
            Task-space targets, ordered according to 'command_types':

                Absolute pose: shape (``num_envs``, 7), containing position and quaternion ``(x, y, z, w)``.
                The quaternion is normalized before use. Entries that cannot be normalized (e.g. all zeros or
                non-finite) fall back to the current end-effector orientation when ``current_ee_pose_b`` is
                provided, and to the identity orientation otherwise.
                Relative pose: shape (``num_envs``, 6), containing delta position and rotation in axis-angle form.
                Absolute wrench: shape (``num_envs``, 6), containing force and torque.

            Impedance parameters: stiffness for ``variable_kp``, or stiffness, followed by damping ratio for
            ``variable``:

                Stiffness: shape (``num_envs``, 6)
                Damping ratio: shape (``num_envs``, 6)

        Raises:
            ValueError: When the command dimensions are invalid.
            ValueError: When an invalid impedance mode is provided.
            ValueError: When the current end-effector pose is not provided for the ``pose_rel`` command.
            ValueError: When an invalid control command is provided.
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
            task_space_command, stiffness = torch.split(command, [self.target_dim, 6], dim=-1)
            # format command
            stiffness = stiffness.clip_(
                min=self._motion_p_gains_limits[..., 0], max=self._motion_p_gains_limits[..., 1]
            )
            # task space targets + stiffness
            self._task_space_target_task[:] = task_space_command.squeeze(dim=-1)
            self._motion_p_gains_task[:] = torch.diag_embed(stiffness)
            self._motion_p_gains_task[:] = self._selection_matrix_motion_task @ self._motion_p_gains_task[:]
            self._motion_d_gains_task[:] = torch.diag_embed(
                2
                * torch.diagonal(self._motion_p_gains_task, dim1=-2, dim2=-1).sqrt()
                * torch.as_tensor(self.cfg.motion_damping_ratio_task, dtype=torch.float, device=self._device).reshape(
                    1, -1
                )
            )
        elif self.cfg.impedance_mode == "variable":
            # split input command
            task_space_command, stiffness, damping_ratio = torch.split(command, [self.target_dim, 6, 6], dim=-1)
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
            # xyzw format: identity quat is [0, 0, 0, 1]
            current_task_frame_pose_b = torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * self.num_envs, device=self._device
            )

        self._task_frame_pose_b.copy_(current_task_frame_pose_b)

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
                desired_ee_pose_task = torch.cat([desired_ee_pos_task, desired_ee_rot_task], dim=-1)
            elif command_type == "pose_abs":
                # normalize the target orientation so that unnormalized policy outputs do not scale the
                # orientation error; degenerate quaternions fall back to the current end-effector orientation
                desired_ee_pose_task = target.clone()
                desired_ee_quat_task = desired_ee_pose_task[:, 3:7]
                normalized_quat = desired_ee_quat_task / torch.linalg.norm(desired_ee_quat_task, dim=-1, keepdim=True)
                is_valid = torch.isfinite(normalized_quat).all(dim=-1, keepdim=True)
                if current_ee_pose_b is not None:
                    _, fallback_quat = subtract_frame_transforms(
                        current_task_frame_pose_b[:, :3],
                        current_task_frame_pose_b[:, 3:],
                        current_ee_pose_b[:, :3],
                        current_ee_pose_b[:, 3:],
                    )
                else:
                    fallback_quat = current_task_frame_pose_b.new_tensor([0.0, 0.0, 0.0, 1.0]).expand(self.num_envs, 4)
                desired_ee_pose_task[:, 3:7] = torch.where(is_valid, normalized_quat, fallback_quat)
            elif command_type == "wrench_abs":
                # compute targets
                self.desired_ee_wrench_task = target.clone()
                continue
            else:
                raise ValueError(f"Invalid control command: {command_type}.")

            if self.desired_ee_pose_task is None:
                self.desired_ee_pose_task = desired_ee_pose_task
            else:
                self.desired_ee_pose_task.copy_(desired_ee_pose_task)

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
            if self.desired_ee_wrench_b is None:
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
        current_joint_pos: torch.Tensor | None = None,
        current_joint_vel: torch.Tensor | None = None,
        nullspace_joint_pos_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Performs inference with the controller.

        Args:
            jacobian_b: The Jacobian matrix of the end-effector in root frame. It is a tensor of shape
                (``num_envs``, 6, ``num_DoF``).
            current_ee_pose_b: The current end-effector pose in root frame. It is a tensor of shape
                (``num_envs``, 7), which contains the position and quaternion ``(x, y, z, w)``. Defaults to ``None``.
            current_ee_vel_b: The current end-effector velocity in root frame. It is a tensor of shape
                (``num_envs``, 6), which contains the linear and angular velocities. Defaults to None.
            current_ee_force_b: The current external force on the end-effector in root frame. It is a tensor of
                shape (``num_envs``, 3), which contains the linear force. Defaults to ``None``.
            mass_matrix: The joint-space mass/inertia matrix. It is a tensor of shape (``num_envs``, ``num_DoF``,
                ``num_DoF``). Defaults to ``None``.
            gravity: The joint-space gravity vector. It is a tensor of shape (``num_envs``, ``num_DoF``). Defaults
                to ``None``.
            current_joint_pos: The current joint positions. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            current_joint_vel: The current joint velocities. It is a tensor of shape (``num_envs``, ``num_DoF``).
                Defaults to ``None``.
            nullspace_joint_pos_target: The target joint positions the null space controller is trying to enforce, when
                possible. It is a tensor of shape (``num_envs``, ``num_DoF``).

        Raises:
            ValueError: When motion-control is enabled but the current end-effector pose or velocity is not provided.
            ValueError: When inertial dynamics decoupling is enabled but the mass matrix is not provided.
            ValueError: When the current end-effector pose is not provided for the ``pose_rel`` command.
            ValueError: When closed-loop force control is enabled but the current end-effector force is not provided.
            ValueError: When gravity compensation is enabled but the gravity vector is not provided.
            ValueError: When null-space control is enabled but the system is not redundant.
            ValueError: When dynamically consistent null-space control is enabled without motion targets or a
                mass matrix.
            ValueError: When null-space control is enabled but the current joint positions and velocities are not
                provided.
            ValueError: When target joint positions are provided for null-space control but their dimensions do not
                match the current joint positions.
            ValueError: When an invalid null-space control method is provided.

        Returns:
            Tensor: The joint efforts computed by the controller. It is a tensor of shape (``num_envs``, ``num_DoF``).
        """
        if self.cfg.implementation == "newton":
            return self._compute_newton(
                jacobian_b,
                current_ee_pose_b,
                current_ee_vel_b,
                current_ee_force_b,
                mass_matrix,
                gravity,
                current_joint_pos,
                current_joint_vel,
                nullspace_joint_pos_target,
            )

        # deduce number of DoF
        num_DoF = jacobian_b.shape[2]
        # create joint effort vector
        joint_efforts = torch.zeros(self.num_envs, num_DoF, device=self._device)

        # Compute posture acceleration before motion control so both can share the task-inertia solve.
        if self.cfg.nullspace_control != "none":
            if num_DoF <= 6:
                raise ValueError("Null-space control is only applicable for redundant manipulators.")
            if self.cfg.nullspace_control != "position":
                raise ValueError(f"Invalid null-space control method: {self.cfg.nullspace_control}.")
            if current_joint_pos is None or current_joint_vel is None:
                raise ValueError("Current joint positions and velocities are required for null-space control.")
            if nullspace_joint_pos_target is None:
                nullspace_joint_pos_target = torch.zeros_like(current_joint_pos)
            elif nullspace_joint_pos_target.shape != current_joint_pos.shape:
                raise ValueError(
                    f"The target nullspace joint positions shape '{nullspace_joint_pos_target.shape}' does not"
                    f"match the current joint positions shape '{current_joint_pos.shape}'."
                )
            joint_acc_nullspace = (
                self._nullspace_p_gain * (nullspace_joint_pos_target - current_joint_pos)
                - self._nullspace_d_gain * current_joint_vel
            ).unsqueeze(-1)
            if self.cfg.inertial_dynamics_decoupling and not self.cfg.partial_inertial_dynamics_decoupling:
                if self.desired_ee_pose_b is None or mass_matrix is None:
                    raise ValueError(
                        "Motion targets and a mass matrix are required for dynamically consistent control."
                    )

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
            velocity_error_b = -current_ee_vel_b  # zero target velocity. The target is assumed to be stationary.
            # -- desired end-effector acceleration (spring-damper system)
            des_ee_acc_b = self._motion_p_gains_b @ pose_error_b.unsqueeze(
                -1
            ) + self._motion_d_gains_b @ velocity_error_b.unsqueeze(-1)
            # -- Inertial dynamics decoupling
            if self.cfg.inertial_dynamics_decoupling:
                # check input is provided
                if mass_matrix is None:
                    raise ValueError("Mass matrix is required for inertial decoupling.")
                # Keep inertia products and their cancellation in double precision near weak task directions.
                jacobian_b_d = jacobian_b.double()
                mass_matrix_d = mass_matrix.double()
                inverse_task_mass = jacobian_b_d @ torch.linalg.solve(mass_matrix_d, jacobian_b_d.mT)
                accelerations = des_ee_acc_b.double()
                if self.cfg.partial_inertial_dynamics_decoupling:
                    # Filter both uncoupled blocks in one batch.
                    blocks = torch.stack((inverse_task_mass[:, :3, :3], inverse_task_mass[:, 3:, 3:]), dim=1)
                    os_command_forces_b = _compute_task_space_forces(
                        blocks.flatten(0, 1), accelerations.reshape(-1, 3, 1), self.cfg.inertia_conditioning_thresholds
                    ).reshape(self.num_envs, 6, 1)
                else:
                    if self.cfg.nullspace_control == "position":
                        joint_acc_nullspace = joint_acc_nullspace.double()
                        accelerations = torch.cat((accelerations, jacobian_b_d @ joint_acc_nullspace), dim=-1)
                    task_forces_b = _compute_task_space_forces(
                        inverse_task_mass, accelerations, self.cfg.inertia_conditioning_thresholds
                    )
                    os_command_forces_b = task_forces_b[:, :, :1]
            else:
                # Task-space impedance control: command forces = \ddot(x_des).
                # Please note that the definition of task-space impedance control varies in literature.
                # This implementation ignores the inertial term. For inertial decoupling,
                # use inertial_dynamics_decoupling=True.
                os_command_forces_b = des_ee_acc_b
            # -- joint-space commands
            joint_efforts += (
                (jacobian_b.mT @ self._selection_matrix_motion_b).to(os_command_forces_b.dtype) @ os_command_forces_b
            ).squeeze(-1)

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

        if self.cfg.nullspace_control == "position":
            if self.cfg.inertial_dynamics_decoupling and not self.cfg.partial_inertial_dynamics_decoupling:
                # (I - J^T Lambda J M^-1) M ddq = M ddq - J^T Lambda J ddq.
                # The second solved force shares the task damping, retaining decoupling in undamped directions.
                tau_null = mass_matrix_d @ joint_acc_nullspace - jacobian_b_d.mT @ task_forces_b[:, :, 1:]
            else:
                # Moore-Penrose projection without full inertial decoupling.
                nullspace_jacobian_transpose = (
                    torch.eye(n=num_DoF, device=self._device) - jacobian_b.mT @ torch.pinverse(jacobian_b).mT
                )
                tau_null = mass_matrix @ joint_acc_nullspace if mass_matrix is not None else joint_acc_nullspace
                tau_null = nullspace_jacobian_transpose @ tau_null
            joint_efforts += tau_null.squeeze(-1)

        return joint_efforts

    def _compute_newton(
        self,
        jacobian_b: torch.Tensor,
        current_ee_pose_b: torch.Tensor | None,
        current_ee_vel_b: torch.Tensor | None,
        current_ee_force_b: torch.Tensor | None,
        mass_matrix: torch.Tensor | None,
        gravity: torch.Tensor | None,
        current_joint_pos: torch.Tensor | None,
        current_joint_vel: torch.Tensor | None,
        nullspace_joint_pos_target: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate Newton using cached workspace inferred from the Jacobian."""
        if self._controller is None or jacobian_b.shape[2] != self._num_dof:
            self._num_dof = jacobian_b.shape[2]
            self._initialize_newton()
        self._operational_frame.copy_(self._task_frame_pose_b)
        # -- Newton input ports
        self._jacobian.copy_(jacobian_b)
        if self.cfg.gravity_compensation:
            if gravity is None:
                raise ValueError("Gravity vector is required for gravity compensation.")
            self._gravity.copy_(gravity)
        else:
            self._gravity.zero_()

        # -- motion state and target (zero gains suppress uncommanded motion)
        if self.desired_ee_pose_b is not None:
            if current_ee_pose_b is None or current_ee_vel_b is None:
                raise ValueError("Current end-effector pose and velocity are required for motion control.")
            if self.cfg.inertial_dynamics_decoupling and mass_matrix is None:
                raise ValueError("Mass matrix is required for inertial decoupling.")
        if current_ee_pose_b is not None:
            self._tool_pose.copy_(current_ee_pose_b)
        else:
            self._tool_pose.zero_()
            self._tool_pose[:, 6] = 1.0
        if current_ee_vel_b is not None:
            self._tool_twist.copy_(current_ee_vel_b)
        else:
            self._tool_twist.zero_()
        if self.desired_ee_pose_task is not None:
            self._desired_pose.copy_(self.desired_ee_pose_task)
            self._motion_stiffness.copy_(self._motion_p_gains_task.diagonal(dim1=-2, dim2=-1))
            self._motion_damping.copy_(self._motion_d_gains_task.diagonal(dim1=-2, dim2=-1))
        else:
            self._desired_pose.zero_()
            self._desired_pose[:, 6] = 1.0
            self._motion_stiffness.zero_()
            self._motion_damping.zero_()

        # -- wrench target and feedback, expressed in the root frame
        if self._wrench_control:
            if self.desired_ee_wrench_b is not None:
                self._desired_wrench.copy_(self.desired_ee_wrench_b)
                if self._wrench_feedback:
                    if current_ee_force_b is None:
                        raise ValueError("Current end-effector force is required for closed-loop force control.")
                    self._measured_wrench[:, :3] = current_ee_force_b
                    # Moments are unmeasured: matching the target keeps them open loop.
                    self._measured_wrench[:, 3:] = self.desired_ee_wrench_b[:, 3:]
            else:
                self._desired_wrench.zero_()
                if self._wrench_feedback:
                    self._measured_wrench.zero_()

        # -- null-space posture target (desired velocity remains zero)
        if self._nullspace_control:
            if (
                self.cfg.inertial_dynamics_decoupling
                and not self.cfg.partial_inertial_dynamics_decoupling
                and mass_matrix is None
            ):
                raise ValueError("Mass matrix inverse is required for dynamically consistent pseudo-inverse")
            if current_joint_pos is None or current_joint_vel is None:
                raise ValueError("Current joint positions and velocities are required for null-space control.")
            if nullspace_joint_pos_target is not None and nullspace_joint_pos_target.shape != current_joint_pos.shape:
                raise ValueError(
                    f"The target nullspace joint positions shape '{nullspace_joint_pos_target.shape}' does not"
                    f"match the current joint positions shape '{current_joint_pos.shape}'."
                )
            self._joint_pos.copy_(current_joint_pos)
            self._joint_vel.copy_(current_joint_vel)
            if nullspace_joint_pos_target is None:
                self._nullspace_target.zero_()
            else:
                self._nullspace_target.copy_(nullspace_joint_pos_target)

        if self.cfg.inertial_dynamics_decoupling:
            self._mass_matrix.copy_(mass_matrix)

        # -- solve and return an independent snapshot (dt is unused)
        self._controller.step(inputs=self._controller_input, outputs=self._controller_output, dt=0.0)
        return self._joint_efforts.clone()

    def _initialize_newton(self) -> None:
        """Construct Newton and bind the fixed input and output ports."""
        from newton.controllers import ControllerOperationalSpaceModelFree

        self._wrench_control = "wrench_abs" in self.cfg.target_types
        self._wrench_feedback = self._wrench_control and self.cfg.contact_wrench_stiffness_task is not None
        self._nullspace_control = self.cfg.nullspace_control == "position"
        if self._nullspace_control and self._num_dof <= 6:
            raise ValueError("Null-space control is only applicable for redundant manipulators.")

        # -- construct Newton controller and ports
        # Live gains support variable impedance; Newton accepts selection axes only with wrench control.
        self._controller = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.full(self.num_envs, self._num_dof, dtype=wp.int32, device=self._device),
            motion_stiffness=None,
            motion_damping=None,
            operational_frame_pose_world=None,
            use_inertia_decoupling=self.cfg.inertial_dynamics_decoupling,
            use_partial_inertia_decoupling=self.cfg.partial_inertial_dynamics_decoupling,
            use_gravity_compensation=True,
            use_wrench_feedforward=self._wrench_control,
            use_wrench_feedback=self._wrench_feedback,
            motion_selection_axes=(
                wp.spatial_vector(*self.cfg.motion_control_axes_task) if self._wrench_control else None
            ),
            wrench_selection_axes=(
                wp.spatial_vector(*self.cfg.contact_wrench_control_axes_task) if self._wrench_control else None
            ),
            wrench_stiffness=(
                wp.from_torch(
                    self._contact_wrench_p_gains_task.diagonal(dim1=-2, dim2=-1).contiguous(), dtype=wp.spatial_vector
                )
                if self._wrench_feedback
                else None
            ),
            use_null_space_control=self._nullspace_control,
            null_space_stiffness=float(self._nullspace_p_gain) if self._nullspace_control else None,
            null_space_damping=float(self._nullspace_d_gain) if self._nullspace_control else None,
            device=self._device,
        )
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()

        self._operational_frame = wp.to_torch(self._controller_input.operational_frame_pose_world)
        self._motion_stiffness = wp.to_torch(self._controller_input.motion_stiffness)
        self._motion_damping = wp.to_torch(self._controller_input.motion_damping)

        # Torch views for per-step state and output.
        self._jacobian = wp.to_torch(self._controller_input.jacobian_tool_world)
        if self.cfg.inertial_dynamics_decoupling:
            self._mass_matrix = wp.to_torch(self._controller_input.mass_matrix)
        self._gravity = wp.to_torch(self._controller_input.gravity_force).view(self.num_envs, self._num_dof)
        self._tool_pose = wp.to_torch(self._controller_input.tool_pose_world)
        self._tool_twist = wp.to_torch(self._controller_input.tool_twist_world)
        self._desired_pose = wp.to_torch(self._controller_input.desired_tool_pose_operational)
        if self._wrench_control:
            self._desired_wrench = wp.to_torch(self._controller_input.desired_wrench_world)
        if self._wrench_feedback:
            self._measured_wrench = wp.to_torch(self._controller_input.measured_wrench_world)
        if self._nullspace_control:
            self._joint_pos = wp.to_torch(self._controller_input.joint_q).view(self.num_envs, self._num_dof)
            self._joint_vel = wp.to_torch(self._controller_input.joint_qd).view(self.num_envs, self._num_dof)
            self._nullspace_target = wp.to_torch(self._controller_input.joint_q_des_null).view(
                self.num_envs, self._num_dof
            )
        self._joint_efforts = wp.to_torch(self._controller_output.joint_f).view(self.num_envs, self._num_dof)
