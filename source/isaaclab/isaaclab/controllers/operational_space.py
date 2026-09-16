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
    matrix_from_quat,
    subtract_frame_transforms,
)

if TYPE_CHECKING:
    from .operational_space_cfg import OperationalSpaceControllerCfg


class OperationalSpaceController:
    """Operational-space controller.

    The task-space impedance law, the contact-wrench law and null-space control are evaluated by
    Newton's model-free operational-space controller
    (:class:`newton.controllers.ControllerOperationalSpaceModelFree`). Target resolution, the task
    frame and the gain schedule remain in Isaac Lab. Motion-axis selection precedes inertia
    decoupling, which requires at least six controlled joints. The task frame is handed to Newton as
    the operational frame, so gains,
    selection axes and targets stay expressed in it rather than being rotated into the root frame
    here. Solves run through float32 internal buffers.

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

        # resolve which laws the Newton backend has to be built with; the target types are static
        # configuration, so the backend's feature set never changes over the controller's lifetime
        self._wrench_control = "wrench_abs" in self.cfg.target_types
        self._wrench_feedback = self._wrench_control and self.cfg.contact_wrench_stiffness_task is not None
        self._nullspace_control = self.cfg.nullspace_control == "position"

        # create buffers
        # -- selection axes, defined in the task reference frame, which might differ from the root frame
        self._selection_axes_motion_task = torch.tensor(
            self.cfg.motion_control_axes_task, dtype=torch.float, device=self._device
        )
        self._selection_axes_force_task = torch.tensor(
            self.cfg.contact_wrench_control_axes_task, dtype=torch.float, device=self._device
        )
        # -- commands
        self._task_space_target_task = torch.zeros(self.num_envs, self.target_dim, device=self._device)
        # -- task frame, in root frame, the targets and control axes are defined in
        self._task_frame_pose_b = torch.zeros(self.num_envs, 7, device=self._device)
        self._task_frame_pose_b[:, 6] = 1.0  # xyzw format: identity quat is [0, 0, 0, 1]
        # -- targets remain None until commanded
        self.desired_ee_pose_task = None
        self.desired_ee_pose_b = None
        self.desired_ee_wrench_task = None
        self.desired_ee_wrench_b = None
        self._desired_ee_pose_task_buf = torch.zeros(self.num_envs, 7, device=self._device)
        self._desired_ee_pose_b_buf = torch.zeros(self.num_envs, 7, device=self._device)
        self._desired_ee_wrench_task_buf = torch.zeros(self.num_envs, 6, device=self._device)
        self._desired_ee_wrench_b_buf = torch.zeros(self.num_envs, 6, device=self._device)
        # -- damping ratio, resolved once: it is static configuration the gain schedule re-reads
        self._motion_damping_ratio_task = torch.as_tensor(
            self.cfg.motion_damping_ratio_task, dtype=torch.float, device=self._device
        ).reshape(1, -1)
        # -- motion control gains, per task axis
        self._motion_p_gains_task = torch.zeros(self.num_envs, 6, device=self._device)
        self._motion_p_gains_task[:] = torch.tensor(
            self.cfg.motion_stiffness_task, dtype=torch.float, device=self._device
        )
        # -- -- zero out the axes that are not motion controlled, as keeping them non-zero will cause other axes
        # -- -- to act due to coupling
        self._motion_p_gains_task *= self._selection_axes_motion_task
        self._motion_d_gains_task = 2 * self._motion_p_gains_task.sqrt() * self._motion_damping_ratio_task
        # -- force control gains
        if self.cfg.contact_wrench_stiffness_task is not None:
            self._contact_wrench_p_gains_task = torch.zeros(self.num_envs, 6, device=self._device)
            self._contact_wrench_p_gains_task[:] = torch.tensor(
                self.cfg.contact_wrench_stiffness_task, dtype=torch.float, device=self._device
            )
            self._contact_wrench_p_gains_task *= self._selection_axes_force_task
        else:
            self._contact_wrench_p_gains_task = None
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

        # -- buffers for null-space control gains
        self._nullspace_p_gain = torch.tensor(self.cfg.nullspace_stiffness, dtype=torch.float, device=self._device)
        self._nullspace_d_gain = (
            2
            * torch.sqrt(self._nullspace_p_gain)
            * torch.tensor(self.cfg.nullspace_damping_ratio, dtype=torch.float, device=self._device)
        )

        # the Newton backend is built on the first ``compute`` call, once the Jacobian reveals the
        # number of controlled DOFs
        self._controller = None
        self._num_dof = None

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
            self._motion_p_gains_task[:] = stiffness * self._selection_axes_motion_task
            self._motion_d_gains_task[:] = 2 * self._motion_p_gains_task.sqrt() * self._motion_damping_ratio_task
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
            self._motion_p_gains_task[:] = stiffness * self._selection_axes_motion_task
            self._motion_d_gains_task[:] = 2 * self._motion_p_gains_task.sqrt() * damping_ratio
        else:
            raise ValueError(f"Invalid impedance mode: {self.cfg.impedance_mode}.")

        if current_task_frame_pose_b is None:
            # xyzw format: identity quat is [0, 0, 0, 1]
            self._task_frame_pose_b.zero_()
            self._task_frame_pose_b[:, 6] = 1.0
        else:
            self._task_frame_pose_b[:] = current_task_frame_pose_b
        current_task_frame_pose_b = self._task_frame_pose_b

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
                self._desired_ee_pose_task_buf[:, :3] = desired_ee_pos_task
                self._desired_ee_pose_task_buf[:, 3:] = desired_ee_rot_task
                self.desired_ee_pose_task = self._desired_ee_pose_task_buf
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
                self.desired_ee_pose_task = desired_ee_pose_task
            elif command_type == "wrench_abs":
                # compute targets
                self._desired_ee_wrench_task_buf[:] = target
                self.desired_ee_wrench_task = self._desired_ee_wrench_task_buf
            else:
                raise ValueError(f"Invalid control command: {command_type}.")

        # Transform desired pose from task frame to root frame
        if self.desired_ee_pose_task is not None:
            self.desired_ee_pose_b = self._desired_ee_pose_b_buf
            self.desired_ee_pose_b[:, :3], self.desired_ee_pose_b[:, 3:] = combine_frame_transforms(
                current_task_frame_pose_b[:, :3],
                current_task_frame_pose_b[:, 3:],
                self.desired_ee_pose_task[:, :3],
                self.desired_ee_pose_task[:, 3:],
            )

        # Transform desired wrenches to root frame
        if self.desired_ee_wrench_task is not None:
            # Rotation of task frame wrt root frame, converts a coordinate from task frame to root frame.
            R_task_b = matrix_from_quat(current_task_frame_pose_b[:, 3:])
            self.desired_ee_wrench_b = self._desired_ee_wrench_b_buf
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
            ValueError: When dynamically consistent pseudo-inverse is enabled but the mass matrix inverse is not
                provided.
            ValueError: When null-space control is enabled but the current joint positions and velocities are not
                provided.
            ValueError: When target joint positions are provided for null-space control but their dimensions do not
                match the current joint positions.
            ValueError: When an invalid null-space control method is provided.

        Returns:
            Tensor: The joint efforts computed by the controller. It is a tensor of shape (``num_envs``, ``num_DoF``).
        """

        # deduce number of DoF
        num_DoF = jacobian_b.shape[2]

        # check the inputs the requested laws need, before handing anything to the backend
        if self.desired_ee_pose_b is not None:
            if current_ee_pose_b is None or current_ee_vel_b is None:
                raise ValueError("Current end-effector pose and velocity are required for motion control.")
            if self.cfg.inertial_dynamics_decoupling and mass_matrix is None:
                raise ValueError("Mass matrix is required for inertial decoupling.")
        if self.desired_ee_wrench_b is not None:
            if self.cfg.contact_wrench_stiffness_task is not None and current_ee_force_b is None:
                raise ValueError("Current end-effector force is required for closed-loop force control.")
        if self.cfg.gravity_compensation and gravity is None:
            raise ValueError("Gravity vector is required for gravity compensation.")
        if self.cfg.nullspace_control != "none":
            if num_DoF <= 6:
                raise ValueError("Null-space control is only applicable for redundant manipulators.")
            if (
                self.cfg.inertial_dynamics_decoupling
                and not self.cfg.partial_inertial_dynamics_decoupling
                and mass_matrix is None
            ):
                raise ValueError("Mass matrix inverse is required for dynamically consistent pseudo-inverse")
            if self.cfg.nullspace_control == "position":
                if current_joint_pos is None or current_joint_vel is None:
                    raise ValueError("Current joint positions and velocities are required for null-space control.")
                if (
                    nullspace_joint_pos_target is not None
                    and nullspace_joint_pos_target.shape != current_joint_pos.shape
                ):
                    raise ValueError(
                        f"The target nullspace joint positions shape '{nullspace_joint_pos_target.shape}' does not"
                        f"match the current joint positions shape '{current_joint_pos.shape}'."
                    )
            else:
                raise ValueError(f"Invalid null-space control method: {self.cfg.nullspace_control}.")

        if self._num_dof != num_DoF:
            self._initialize_controller(num_DoF)

        # -- Newton input ports
        self._jacobian.copy_(jacobian_b)
        if self.cfg.inertial_dynamics_decoupling:
            self._mass_matrix.copy_(mass_matrix)
        if self.cfg.gravity_compensation:
            self._gravity.copy_(gravity)
        if current_ee_pose_b is not None:
            self._tool_pose.copy_(current_ee_pose_b)
        else:
            self._tool_pose.zero_()
            self._tool_pose[:, 6] = 1.0
        if current_ee_vel_b is not None:
            self._tool_twist.copy_(current_ee_vel_b)
        else:
            self._tool_twist.zero_()

        # -- motion target (zero gains suppress uncommanded motion)
        inputs = self._controller_input
        if self.desired_ee_pose_task is not None:
            self._desired_pose.copy_(self.desired_ee_pose_task)
            inputs.motion_stiffness = self._motion_stiffness_port
            inputs.motion_damping = self._motion_damping_port
        else:
            self._desired_pose.zero_()
            self._desired_pose[:, 6] = 1.0
            inputs.motion_stiffness = self._zero_spatial
            inputs.motion_damping = self._zero_spatial

        # -- wrench target and feedback, expressed in the root frame
        if self._wrench_control:
            if self.desired_ee_wrench_b is not None:
                self._desired_wrench.copy_(self.desired_ee_wrench_b)
                if self._wrench_feedback:
                    self._measured_wrench[:, :3] = current_ee_force_b
                    # Moments are unmeasured: matching the target keeps them open loop.
                    self._measured_wrench[:, 3:] = self.desired_ee_wrench_b[:, 3:]
            else:
                self._desired_wrench.zero_()
                if self._wrench_feedback:
                    self._measured_wrench.zero_()

        # -- null-space posture target (desired velocity remains zero)
        if self._nullspace_control:
            self._joint_pos.copy_(current_joint_pos)
            self._joint_vel.copy_(current_joint_vel)
            if nullspace_joint_pos_target is None:
                self._nullspace_target.zero_()
            else:
                self._nullspace_target.copy_(nullspace_joint_pos_target)

        # -- solve and return an independent snapshot (dt is unused)
        self._controller.step(inputs=inputs, outputs=self._controller_output, dt=0.0)
        return self._joint_efforts.clone()

    """
    Internal helpers.
    """

    def _initialize_controller(self, num_dof: int) -> None:
        """Construct Newton ports once the Jacobian reveals the controlled joint count."""
        from newton.controllers import ControllerOperationalSpaceModelFree

        num_envs = self.num_envs

        # -- construct Newton controller and ports
        controlled_dofs_per_robot = wp.full(num_envs, num_dof, dtype=wp.int32, device=self._device)

        # Live gains support variable impedance; Newton accepts selection axes only with wrench control.
        self._controller = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=controlled_dofs_per_robot,
            motion_stiffness=None,
            motion_damping=None,
            operational_frame_pose_world=None,
            use_inertia_decoupling=self.cfg.inertial_dynamics_decoupling,
            use_partial_inertia_decoupling=self.cfg.partial_inertial_dynamics_decoupling,
            use_gravity_compensation=self.cfg.gravity_compensation,
            use_wrench_feedforward=self._wrench_control,
            use_wrench_feedback=self._wrench_feedback,
            motion_selection_axes=(
                wp.spatial_vector(*self.cfg.motion_control_axes_task) if self._wrench_control else None
            ),
            wrench_selection_axes=(
                wp.spatial_vector(*self.cfg.contact_wrench_control_axes_task) if self._wrench_control else None
            ),
            wrench_stiffness=(
                wp.from_torch(self._contact_wrench_p_gains_task, dtype=wp.spatial_vector)
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
        self._num_dof = num_dof

        # -- Torch views of Newton-owned arrays
        inputs = self._controller_input
        self._jacobian = wp.to_torch(inputs.jacobian_tool_world)
        self._tool_pose = wp.to_torch(inputs.tool_pose_world)
        self._tool_twist = wp.to_torch(inputs.tool_twist_world)
        self._desired_pose = wp.to_torch(inputs.desired_tool_pose_operational)
        if self.cfg.inertial_dynamics_decoupling:
            self._mass_matrix = wp.to_torch(inputs.mass_matrix)
        if self.cfg.gravity_compensation:
            self._gravity = wp.to_torch(inputs.gravity_force).view(num_envs, num_dof)
        if self._wrench_control:
            self._desired_wrench = wp.to_torch(inputs.desired_wrench_world)
        if self._wrench_feedback:
            self._measured_wrench = wp.to_torch(inputs.measured_wrench_world)
        if self._nullspace_control:
            self._joint_pos = wp.to_torch(inputs.joint_q).view(num_envs, num_dof)
            self._joint_vel = wp.to_torch(inputs.joint_qd).view(num_envs, num_dof)
            self._nullspace_target = wp.to_torch(inputs.joint_q_des_null).view(num_envs, num_dof)
        self._joint_efforts = wp.to_torch(self._controller_output.joint_f).view(num_envs, num_dof)

        # -- transfer the command frame and gain schedule; set_command updates these views in place
        task_frame = wp.to_torch(inputs.operational_frame_pose_world)
        task_frame.copy_(self._task_frame_pose_b)
        self._task_frame_pose_b = task_frame
        self._motion_stiffness_port = inputs.motion_stiffness
        self._motion_damping_port = inputs.motion_damping
        p_gains = wp.to_torch(inputs.motion_stiffness)
        d_gains = wp.to_torch(inputs.motion_damping)
        p_gains.copy_(self._motion_p_gains_task)
        d_gains.copy_(self._motion_d_gains_task)
        self._motion_p_gains_task, self._motion_d_gains_task = p_gains, d_gains
        self._zero_spatial = wp.zeros(num_envs, dtype=wp.spatial_vector, device=self._device)
