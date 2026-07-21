# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton_controllers import ControllerDifferentialKinematicsModelFree

from isaaclab.utils.math import apply_delta_pose, compute_pose_error

if TYPE_CHECKING:
    from .differential_ik_cfg import DifferentialIKControllerCfg


class DifferentialIKController:
    r"""Differential inverse kinematics (IK) controller.

    This controller is based on the concept of differential inverse kinematics [1, 2] which is a method for computing
    the change in joint positions that yields the desired change in pose.

    .. math::

        \Delta \mathbf{q} &= \mathbf{J}^{\dagger} \Delta \mathbf{x} \\
        \mathbf{q}_{\text{desired}} &= \mathbf{q}_{\text{current}} + \Delta \mathbf{q}

    where :math:`\mathbf{J}^{\dagger}` is the pseudo-inverse of the Jacobian matrix :math:`\mathbf{J}`,
    :math:`\Delta \mathbf{x}` is the desired change in pose, and :math:`\mathbf{q}_{\text{current}}`
    is the current joint positions.

    To deal with singularity in Jacobian, the following methods are supported for computing inverse of the Jacobian:

    - "pinv": Moore-Penrose pseudo-inverse
    - "svd": Adaptive singular-value decomposition (SVD)
    - "trans": Transpose of matrix
    - "dls": Damped version of Moore-Penrose pseudo-inverse (also called Levenberg-Marquardt)


    .. caution::
        The controller does not assume anything about the frames of the current and desired end-effector pose,
        or the joint-space velocities. It is up to the user to ensure that these quantities are given
        in the correct format.

    Reference:

    1. `Robot Dynamics Lecture Notes <https://ethz.ch/content/dam/ethz/special-interest/mavt/robotics-n-intelligent-systems/rsl-dam/documents/RobotDynamics2017/RD_HS2017script.pdf>`_
       by Marco Hutter (ETH Zurich)
    2. `Introduction to Inverse Kinematics <https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf>`_
       by Samuel R. Buss (University of California, San Diego)

    """

    def __init__(self, cfg: DifferentialIKControllerCfg, num_envs: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The configuration for the controller.
            num_envs: The number of environments.
            device: The device to use for computations.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self._device)
        self._torch_device = self.ee_pos_des.device
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32, device=self._device)
        # -- optional per-axis orientation task weights (used for "pose" command types only)
        if self.cfg.orientation_weight is None:
            self._orientation_weight = None
        else:
            ori_weight = self.cfg.orientation_weight
            weight_tuple = (
                (float(ori_weight),) * 3
                if isinstance(ori_weight, (int, float))
                else tuple(float(value) for value in ori_weight)
            )
            self._orientation_weight = torch.tensor(weight_tuple, dtype=torch.float32, device=self._device)
        # -- optional joint position limits for null-space joint-limit avoidance (set externally)
        self._joint_pos_lower = None
        self._joint_pos_upper = None
        # -- Newton controller and its stable Torch/Warp bridge (the number of joints is known on first compute)
        self._controller: ControllerDifferentialKinematicsModelFree | None = None
        self._controller_input = None
        self._controller_output = None
        self._task_error = None
        self._task_jacobian = None
        self._joint_pos = None
        self._joint_pos_des = None
        self._time_step = None

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        """Dimension of the controller's input command."""
        if self.cfg.command_type == "position":
            return 3  # (x, y, z)
        elif self.cfg.command_type == "pose" and self.cfg.use_relative_mode:
            return 6  # (dx, dy, dz, droll, dpitch, dyaw)
        else:
            return 7  # (x, y, z, qx, qy, qz, qw)

    """
    Operations.
    """

    def reset(self, env_ids: torch.Tensor = None):
        """Reset the internals.

        Args:
            env_ids: The environment indices to reset. If None, then all environments are reset.
        """
        pass

    def set_command(
        self, command: torch.Tensor, ee_pos: torch.Tensor | None = None, ee_quat: torch.Tensor | None = None
    ):
        """Set target end-effector pose command.

        Based on the configured command type and relative mode, the method computes the desired end-effector pose.
        It is up to the user to ensure that the command is given in the correct frame. The method only
        applies the relative mode if the command type is ``position_rel`` or ``pose_rel``.

        Args:
            command: The input command in shape (N, 3), (N, 6), or (N, 7). Position components are [m],
                relative rotation components are [rad], and absolute quaternion components are dimensionless.
            ee_pos: The current end-effector position [m] in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current dimensionless end-effector orientation quaternion (x, y, z, w) in shape (N, 4).
                This is only needed if the command type is ``position_*`` or ``pose_rel``.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
            ValueError: If an input has an unexpected shape or device.
            TypeError: If an input is not a :class:`torch.Tensor`.
        """
        self._validate_tensor(command, "command", (self.num_envs, self.action_dim))
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            self._validate_tensor(ee_quat, "ee_quat", (self.num_envs, 4))
            ee_quat = ee_quat.to(dtype=torch.float32)
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
                self._validate_tensor(ee_pos, "ee_pos", (self.num_envs, 3))
                ee_pos = ee_pos.to(dtype=torch.float32)
                self.ee_pos_des[:] = ee_pos + self._command
                self.ee_quat_des[:] = ee_quat
            else:
                self.ee_pos_des[:] = self._command
                self.ee_quat_des[:] = ee_quat
        else:
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None or ee_quat is None:
                    raise ValueError(
                        "Neither end-effector position nor orientation can be None for `pose_rel` command type!"
                    )
                self._validate_tensor(ee_pos, "ee_pos", (self.num_envs, 3))
                self._validate_tensor(ee_quat, "ee_quat", (self.num_envs, 4))
                ee_pos = ee_pos.to(dtype=torch.float32)
                ee_quat = ee_quat.to(dtype=torch.float32)
                ee_pos_des, ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
                self.ee_pos_des[:] = ee_pos_des
                self.ee_quat_des[:] = ee_quat_des
            else:
                self.ee_pos_des[:] = self._command[:, 0:3]
                # renormalize the commanded quaternion (callers may pass a slightly non-unit quat)
                quat = self._command[:, 3:7]
                self.ee_quat_des[:] = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)

    def set_joint_pos_limits(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Provide the controlled joints' position limits for null-space joint-limit avoidance.

        Only used when
        :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_gain`
        is positive. The IK action term injects these automatically on its first step; call this
        manually only when using the controller standalone.

        Args:
            lower: Lower joint-position limits [m or rad, depending on joint type] in shape (num_joints,).
            upper: Upper joint-position limits [m or rad, depending on joint type] in shape (num_joints,).

        Raises:
            ValueError: If the limits have different lengths or an unexpected number of joints.
            TypeError: If either limit is not a :class:`torch.Tensor`.
        """
        self._validate_tensor(lower, "lower", (None,), check_device=False)
        self._validate_tensor(upper, "upper", (None,), check_device=False)
        if lower.shape != upper.shape:
            raise ValueError(
                f"Expected lower and upper limits to have the same shape, got {lower.shape} and {upper.shape}."
            )
        if self._controller is not None and lower.shape[0] != self._controller.num_dofs:
            raise ValueError(f"Expected limits for {self._controller.num_dofs} joints, got {lower.shape[0]}.")
        self._joint_pos_lower = lower.to(device=self._torch_device, dtype=torch.float32, copy=True)
        self._joint_pos_upper = upper.to(device=self._torch_device, dtype=torch.float32, copy=True)
        if self._controller is not None:
            self._controller.set_joint_pos_limits(self._joint_pos_lower, self._joint_pos_upper)

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position [m] in shape (N, 3).
            ee_quat: The current dimensionless end-effector orientation quaternion in shape (N, 4).
            jacobian: The geometric Jacobian in shape (N, 6, num_joints). Its linear rows map joint velocities
                to [m/s], and its angular rows map joint velocities to [rad/s].
            joint_pos: The current joint positions [m or rad, depending on joint type] in shape (N, num_joints).
            out: Optional joint-position output buffer [m or rad, depending on joint type] in shape
                (N, num_joints). When provided, the result is copied into and returned from this buffer. This
                avoids the allocation used for the default snapshot return.

        Returns:
            Target joint positions [m or rad, depending on joint type] in shape (N, num_joints). With ``out=None``,
            this is a newly allocated snapshot that remains unchanged by later calls. Otherwise, the supplied
            ``out`` buffer is returned.

        Raises:
            ValueError: If an input has an unexpected shape or device.
            TypeError: If an input is not a :class:`torch.Tensor`, or if ``out`` is not floating-point.
        """
        self._validate_tensor(joint_pos, "joint_pos", (self.num_envs, None))
        num_joints = joint_pos.shape[1]
        self._validate_tensor(ee_pos, "ee_pos", (self.num_envs, 3))
        self._validate_tensor(ee_quat, "ee_quat", (self.num_envs, 4))
        self._validate_tensor(jacobian, "jacobian", (self.num_envs, 6, num_joints))
        if out is not None:
            self._validate_tensor(out, "out", (self.num_envs, num_joints))
            if not out.is_floating_point():
                raise TypeError(f"Expected out to be a floating-point tensor, got {out.dtype}.")
        # The historical Torch implementation accepted integral tensors through implicit promotion. Preserve that
        # public behavior while normalizing all inputs to the Newton controller's float32 bridge.
        ee_pos_float = ee_pos.to(dtype=torch.float32)
        ee_quat_float = ee_quat.to(dtype=torch.float32)
        jacobian_float = jacobian.to(dtype=torch.float32)
        joint_pos_float = joint_pos.to(dtype=torch.float32)
        # Assemble the task Jacobian and error in Isaac Lab so subclasses can shape the task before
        # Newton solves it. In particular, SO-101 masks selected orientation Jacobian columns here.
        if "position" in self.cfg.command_type:
            task_jacobian = jacobian_float[:, 0:3]
            task_error = self.ee_pos_des - ee_pos_float
        else:
            task_jacobian, task_error = self._compute_pose_task(ee_pos_float, ee_quat_float, jacobian_float)
        # Lazily initialize because the public constructor predates and does not take the number of joints.
        self._initialize_controller(num_joints)
        # Copy into stable, controller-owned buffers. This avoids rebinding Warp views between calls and
        # makes graph capture/replay safe for the graphable Newton solver variants.
        self._task_error.copy_(task_error)
        self._task_jacobian.copy_(task_jacobian)
        self._joint_pos.copy_(joint_pos_float)
        # A unit time step preserves Isaac Lab's historical q_target = q + delta_q contract. Newton's
        # controller otherwise interprets its solver output as a velocity and integrates it by time_step.
        self._controller.compute(self._controller_input, self._controller_output, None, None, self._time_step)
        if out is None:
            output_dtype = joint_pos.dtype if joint_pos.is_floating_point() else torch.float32
            return self._joint_pos_des.to(dtype=output_dtype, copy=True)
        out.copy_(self._joint_pos_des)
        return out

    """
    Helper functions.
    """

    def _validate_tensor(
        self,
        tensor: torch.Tensor,
        name: str,
        expected_shape: tuple[int | None, ...],
        *,
        check_device: bool = True,
    ) -> None:
        """Validate a tensor before it crosses the stable Torch/Warp bridge.

        Args:
            tensor: Tensor to validate.
            name: Argument name used in error messages.
            expected_shape: Required shape, with ``None`` for a dimension whose size is unconstrained.
            check_device: Whether to require the controller device.

        Raises:
            ValueError: If the tensor has an unexpected shape or device.
            TypeError: If the value is not a :class:`torch.Tensor`.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}.")
        if check_device and tensor.device != self._torch_device:
            raise ValueError(f"Expected {name} on {self._torch_device}, got {tensor.device}.")
        if tensor.ndim != len(expected_shape) or any(
            expected is not None and actual != expected for actual, expected in zip(tensor.shape, expected_shape)
        ):
            raise ValueError(f"Expected {name} with shape {expected_shape}, got {tuple(tensor.shape)}.")

    def _initialize_controller(self, num_joints: int) -> None:
        """Initialize the Newton controller and stable Torch/Warp bridge.

        Args:
            num_joints: Number of controlled joints.

        Raises:
            ValueError: If a later call changes the number of controlled joints.
        """
        if self._controller is not None:
            if self._controller.num_dofs != num_joints:
                raise ValueError(
                    f"Expected {self._controller.num_dofs} controlled joints after initialization, got {num_joints}."
                )
            return
        if self._joint_pos_lower is not None and self._joint_pos_lower.shape[0] != num_joints:
            raise ValueError(f"Expected limits for {num_joints} joints, got {self._joint_pos_lower.shape[0]}.")

        method_map = {
            "pinv": ControllerDifferentialKinematicsModelFree.IkMethod.PSEUDO_INVERSE,
            "svd": ControllerDifferentialKinematicsModelFree.IkMethod.SVD,
            "trans": ControllerDifferentialKinematicsModelFree.IkMethod.TRANSPOSE,
            "dls": ControllerDifferentialKinematicsModelFree.IkMethod.DAMPED_LEAST_SQUARES,
            "adaptive_dls": ControllerDifferentialKinematicsModelFree.IkMethod.ADAPTIVE_DAMPED_LEAST_SQUARES,
        }
        command_type_map = {
            "position": ControllerDifferentialKinematicsModelFree.CommandType.POSITION,
            "pose": ControllerDifferentialKinematicsModelFree.CommandType.POSE,
        }
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")

        bandwidth = self.cfg.ik_params.get("k_val", 1.0)
        controller_kwargs = {
            "num_robots": self.num_envs,
            "num_dofs": num_joints,
            "bandwidth": wp.full(self.num_envs, bandwidth, dtype=wp.float32, device=self._device),
            "ik_method": method_map[self.cfg.ik_method],
            "command_type": command_type_map[self.cfg.command_type],
            "task_error_attr": "task_error",
            "joint_limit_avoidance_gain": self.cfg.joint_limit_avoidance_gain,
            "joint_limit_avoidance_margin": self.cfg.joint_limit_avoidance_margin,
            "device": self._device,
        }
        if self.cfg.ik_method == "dls":
            controller_kwargs["solver_damping"] = wp.full(
                self.num_envs, self.cfg.ik_params["lambda_val"], dtype=wp.float32, device=self._device
            )
        elif self.cfg.ik_method == "svd":
            controller_kwargs["min_singular_value"] = self.cfg.ik_params["min_singular_value"]
        elif self.cfg.ik_method == "adaptive_dls":
            controller_kwargs.update(
                lambda_min=self.cfg.ik_params["lambda_min"],
                lambda_max=self.cfg.ik_params["lambda_max"],
                sigma_thresh=self.cfg.ik_params["sigma_thresh"],
            )

        self._controller = ControllerDifferentialKinematicsModelFree(**controller_kwargs)
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()

        task_dim = 3 if self.cfg.command_type == "position" else 6
        self._task_error = torch.zeros(self.num_envs, task_dim, dtype=torch.float32, device=self._device)
        self._task_jacobian = torch.zeros(self.num_envs, task_dim, num_joints, dtype=torch.float32, device=self._device)
        self._joint_pos = torch.zeros(self.num_envs, num_joints, dtype=torch.float32, device=self._device)
        self._controller_input.task_error = wp.from_torch(self._task_error)
        self._controller_input.jacobian = wp.from_torch(self._task_jacobian)
        self._controller_input.joint_q = wp.from_torch(self._joint_pos)
        self._joint_pos_des = wp.to_torch(self._controller_output.joint_target_q)
        self._time_step = wp.ones(1, dtype=wp.float32, device=self._device)

        if self._joint_pos_lower is not None:
            self._controller.set_joint_pos_limits(self._joint_pos_lower, self._joint_pos_upper)

    def _compute_pose_task(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the (optionally orientation-weighted) pose task Jacobian and error.

        The orientation error is the axis-angle of ``q_des * q_cur^-1`` from
        :func:`~isaaclab.utils.math.compute_pose_error`. When
        :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.orientation_weight`
        is set, the 3 orientation rows of both the Jacobian and the error are scaled per
        base-frame axis (a weight of 0 drops that axis from the solve). Subclasses may override
        this to further shape the task (e.g. masking which joints serve orientation).

        Args:
            ee_pos: Current end-effector position in shape (N, 3).
            ee_quat: Current end-effector orientation (x, y, z, w) in shape (N, 4).
            jacobian: The geometric Jacobian in shape (N, 6, num_joints).

        Returns:
            A tuple ``(task_jacobian, task_error)`` with the (N, 6, num_joints) task Jacobian and
            the (N, 6) task-space error.
        """
        position_error, axis_angle_error = compute_pose_error(
            ee_pos, ee_quat, self.ee_pos_des, self.ee_quat_des, rot_error_type="axis_angle"
        )
        task_jacobian = jacobian
        if self._orientation_weight is not None:
            weight = self._orientation_weight
            task_jacobian = torch.cat([jacobian[:, 0:3, :], jacobian[:, 3:6, :] * weight.view(1, 3, 1)], dim=1)
            axis_angle_error = axis_angle_error * weight.view(1, 3)
        task_error = torch.cat((position_error, axis_angle_error), dim=1)
        return task_jacobian, task_error
