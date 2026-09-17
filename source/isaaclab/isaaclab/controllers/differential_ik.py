# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

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
        self._controller = None
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        # create buffers
        self.ee_pos_des = torch.zeros(self.num_envs, 3, device=self._device)
        self.ee_quat_des = torch.zeros(self.num_envs, 4, device=self._device)
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, device=self._device)
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
            self._orientation_weight = torch.tensor(weight_tuple, device=self._device)
        # -- optional joint position limits for null-space joint-limit avoidance (set externally)
        self._joint_pos_lower = None
        self._joint_pos_upper = None
        # -- identity quaternion (x, y, z, w), the last-resort fallback for a degenerate command
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).repeat(self.num_envs, 1)

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

        Absolute ``pose`` commands normalize finite quaternions; unnormalizable entries use
        :paramref:`ee_quat`, or identity when :paramref:`ee_quat` is omitted.

        Args:
            command: The input command in shape (N, 3) or (N, 6) or (N, 7).
            ee_pos: The current end-effector position in shape (N, 3).
                This is only needed if the command type is ``position_rel`` or ``pose_rel``.
            ee_quat: The current end-effector orientation (x, y, z, w) in shape (N, 4).
                This is needed if the command type is ``position_*`` or ``pose_rel``. For absolute
                ``pose`` commands it is optional and used only as the fallback orientation for an
                unnormalizable commanded quaternion.

        Raises:
            ValueError: If the command type is ``position_*`` and :attr:`ee_quat` is None.
            ValueError: If the command type is ``position_rel`` and :attr:`ee_pos` is None.
            ValueError: If the command type is ``pose_rel`` and either :attr:`ee_pos` or :attr:`ee_quat` is None.
        """
        # store command
        self._command[:] = command
        # compute the desired end-effector pose
        if self.cfg.command_type == "position":
            # we need end-effector orientation even though we are in position mode
            # this is only needed for display purposes
            if ee_quat is None:
                raise ValueError("End-effector orientation can not be None for `position_*` command type!")
            # compute targets
            if self.cfg.use_relative_mode:
                if ee_pos is None:
                    raise ValueError("End-effector position can not be None for `position_rel` command type!")
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
                ee_pos_des, ee_quat_des = apply_delta_pose(ee_pos, ee_quat, self._command)
            else:
                ee_pos_des = self._command[:, 0:3]
                # normalize valid quaternions and use the fallback for non-finite results
                quat = self._command[:, 3:7]
                normalized_quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
                is_valid = torch.isfinite(normalized_quat).all(dim=-1, keepdim=True)
                fallback_quat = self._identity_quat if ee_quat is None else ee_quat
                ee_quat_des = torch.where(is_valid, normalized_quat, fallback_quat)
            if self.cfg.use_newton:
                self.ee_pos_des[:] = ee_pos_des
                self.ee_quat_des[:] = ee_quat_des
            else:
                self.ee_pos_des, self.ee_quat_des = ee_pos_des, ee_quat_des

    def set_joint_pos_limits(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Provide the controlled joints' position limits for null-space joint-limit avoidance.

        Only used when
        :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_gain`
        is positive. With Newton, supply limits before the first compute call; this setter
        initializes the backend from the limit count if needed. Later updates retain its buffers.
        The Lab backend also permits supplying limits after computing has started.

        Args:
            lower: Lower joint-position limits [m or rad, depending on joint type] in shape (num_joints,).
            upper: Upper joint-position limits [m or rad, depending on joint type] in shape (num_joints,).

        Raises:
            ValueError: If the limits have different lengths or an unexpected number of joints.
        """
        self._joint_pos_lower = lower.to(self._device)
        self._joint_pos_upper = upper.to(self._device)
        if not self.cfg.use_newton or self.cfg.joint_limit_avoidance_gain <= 0.0:
            return
        if lower.ndim != 1 or lower.shape != upper.shape:
            raise ValueError(f"Expected matching one-dimensional limits, got {lower.shape} and {upper.shape}.")
        if self._controller is not None and lower.shape != (self._num_joints,):
            raise ValueError(f"Expected limits for {self._num_joints} joints, got {lower.shape}.")
        self._joint_pos_lower = self._joint_pos_lower.float()
        self._joint_pos_upper = self._joint_pos_upper.float()
        if self._controller is None:
            self._num_joints = lower.numel()
            self._initialize_newton()
        self._controller.set_joint_limits(
            joint_pos_lower=wp.from_torch(self._joint_pos_lower.repeat(self.num_envs)),
            joint_pos_upper=wp.from_torch(self._joint_pos_upper.repeat(self.num_envs)),
        )

    def compute(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position in shape (N, 3).
            ee_quat: The current end-effector orientation in shape (N, 4).
            jacobian: The geometric jacobian matrix in shape (N, 6, num_joints).
            joint_pos: The current joint positions in shape (N, num_joints).

        Returns:
            The target joint positions commands in shape (N, num_joints).
        """
        if self.cfg.use_newton:
            return self._compute_newton(ee_pos, ee_quat, jacobian, joint_pos)

        # assemble the task Jacobian and task-space error
        if "position" in self.cfg.command_type:
            task_jacobian = jacobian[:, 0:3]
            task_error = self.ee_pos_des - ee_pos
        else:
            task_jacobian, task_error = self._compute_pose_task(ee_pos, ee_quat, jacobian)
        # compute the delta in joint-space
        delta_joint_pos = self._compute_delta_joint_pos(delta_pose=task_error, jacobian=task_jacobian)
        # add an optional null-space joint-limit-avoidance bias (a no-op when joint_limit_avoidance_gain == 0)
        delta_joint_pos = delta_joint_pos + self._joint_limit_avoidance(joint_pos, task_jacobian)
        # return the desired joint positions
        return joint_pos + delta_joint_pos

    """
    Helper functions.
    """

    def _compute_delta_joint_pos(self, delta_pose: torch.Tensor, jacobian: torch.Tensor) -> torch.Tensor:
        """Computes the change in joint position that yields the desired change in pose.

        The method uses the Jacobian mapping from joint-space velocities to end-effector velocities
        to compute the delta-change in the joint-space that moves the robot closer to a desired
        end-effector position.

        Args:
            delta_pose: The desired delta pose in shape (N, 3) or (N, 6).
            jacobian: The geometric jacobian matrix in shape (N, 3, num_joints) or (N, 6, num_joints).

        Returns:
            The desired delta in joint space. Shape is (N, num-jointsß).
        """
        if self.cfg.ik_params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' is not defined!")
        # compute the delta in joint-space
        if self.cfg.ik_method == "pinv":  # Jacobian pseudo-inverse
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_pinv = torch.linalg.pinv(jacobian)
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "svd":  # adaptive SVD
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            min_singular_value = self.cfg.ik_params["min_singular_value"]
            # computation
            U, S, Vh = torch.linalg.svd(jacobian, full_matrices=False)
            S_inv = 1.0 / S
            S_inv = torch.where(min_singular_value < S, S_inv, torch.zeros_like(S_inv))
            jacobian_pinv = (
                torch.transpose(Vh, dim0=1, dim1=2) @ torch.diag_embed(S_inv) @ torch.transpose(U, dim0=1, dim1=2)
            )
            delta_joint_pos = k_val * jacobian_pinv @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "trans":  # Jacobian transpose
            # parameters
            k_val = self.cfg.ik_params["k_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            delta_joint_pos = k_val * jacobian_T @ delta_pose.unsqueeze(-1)
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "dls":  # damped least squares
            # parameters
            lambda_val = self.cfg.ik_params["lambda_val"]
            # computation
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = (lambda_val**2) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = (
                jacobian_T @ torch.inverse(jacobian @ jacobian_T + lambda_matrix) @ delta_pose.unsqueeze(-1)
            )
            delta_joint_pos = delta_joint_pos.squeeze(-1)
        elif self.cfg.ik_method == "adaptive_dls":  # manipulability-aware damped least squares
            # parameters
            lambda_min = self.cfg.ik_params["lambda_min"]
            lambda_max = self.cfg.ik_params["lambda_max"]
            sigma_thresh = self.cfg.ik_params["sigma_thresh"]
            # per-environment squared damping: lambda_min^2 away from singularities, ramping
            # quadratically up to lambda_max^2 as the smallest task-Jacobian singular value -> 0
            # (Maciejewski-Klein). Keying off the full task Jacobian damps both position and
            # orientation rank-loss configurations.
            sigma_min = torch.linalg.svdvals(jacobian)[:, -1]  # (N,)
            ratio = (sigma_min / sigma_thresh).clamp(max=1.0)
            lambda_sq = lambda_min**2 + (1.0 - ratio**2) * (lambda_max**2 - lambda_min**2)  # (N,)
            jacobian_T = torch.transpose(jacobian, dim0=1, dim1=2)
            lambda_matrix = lambda_sq.view(-1, 1, 1) * torch.eye(n=jacobian.shape[1], device=self._device)
            delta_joint_pos = torch.bmm(
                jacobian_T,
                torch.linalg.solve(torch.bmm(jacobian, jacobian_T) + lambda_matrix, delta_pose.unsqueeze(-1)),
            ).squeeze(-1)
        else:
            raise ValueError(f"Unsupported inverse-kinematics method: {self.cfg.ik_method}")

        return delta_joint_pos

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

    def _joint_limit_avoidance(self, joint_pos: torch.Tensor, task_jacobian: torch.Tensor) -> torch.Tensor:
        """Null-space joint-centering bias that keeps joints off their position limits.

        Projects a center-seeking joint velocity (active only within
        :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_margin`
        of a limit) into the null space of the position (linear) task rows, so it never perturbs
        the commanded end-effector position. Returns zeros when disabled (``joint_limit_avoidance_gain == 0``) or
        before joint limits are provided via :meth:`set_joint_pos_limits`.

        Args:
            joint_pos: Current joint positions in shape (N, num_joints).
            task_jacobian: The task Jacobian in shape (N, T, num_joints); rows 0-2 are the
                position (linear) rows.

        Returns:
            The joint-space correction in shape (N, num_joints).
        """
        if self.cfg.joint_limit_avoidance_gain <= 0.0 or self._joint_pos_lower is None:
            return torch.zeros_like(joint_pos)
        lower, upper = self._joint_pos_lower, self._joint_pos_upper
        q_mid = 0.5 * (lower + upper)
        dist = torch.minimum(joint_pos - lower, upper - joint_pos)  # margin to nearest limit
        activation = 1.0 - (dist / self.cfg.joint_limit_avoidance_margin).clamp(0.0, 1.0)  # 1 at the limit, 0 mid-range
        dq_center = -self.cfg.joint_limit_avoidance_gain * activation * (joint_pos - q_mid)  # toward the joint center
        j_pos = task_jacobian[:, :3, :]
        j_pos_pinv = torch.linalg.pinv(j_pos)
        num_joints = task_jacobian.shape[2]
        null_proj = torch.eye(num_joints, device=self._device) - torch.bmm(j_pos_pinv, j_pos)
        return torch.bmm(null_proj, dq_center.unsqueeze(-1)).squeeze(-1)

    def _compute_newton(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor, joint_pos: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Newton using cached workspace inferred from the joint positions."""
        if self._controller is None or joint_pos.shape[1] != self._num_joints:
            self._num_joints = joint_pos.shape[1]
            self._initialize_newton()
        self._desired_pose[:, :3].copy_(self.ee_pos_des)
        self._desired_pose[:, 3:].copy_(self.ee_quat_des)
        # -- Newton input ports
        self._task_jacobian.copy_(jacobian)
        self._joint_pos.copy_(joint_pos)
        self._tool_pose[:, :3] = ee_pos
        self._tool_pose[:, 3:] = ee_quat
        # -- solve and return
        # A unit time step preserves q_target = q + delta_q.
        self._controller.step(inputs=self._controller_input, outputs=self._controller_output, dt=1.0)
        output_dtype = joint_pos.dtype if joint_pos.is_floating_point() else torch.float32
        return self._joint_pos_des.to(dtype=output_dtype, copy=True)

    def _initialize_newton(self) -> None:
        """Construct Newton and allocate its input and output ports."""
        from newton.controllers import ControllerDifferentialIKModelFree, DifferentialIKMethod

        use_joint_limits = self.cfg.joint_limit_avoidance_gain > 0.0
        if use_joint_limits and self._joint_pos_lower is None:
            raise ValueError("Set joint position limits before computing with Newton joint-limit avoidance enabled.")
        # -- translate Isaac Lab solver configuration
        method_map = {
            "pinv": DifferentialIKMethod.PSEUDO_INVERSE,
            "svd": DifferentialIKMethod.TRUNCATED_SVD,
            "trans": DifferentialIKMethod.TRANSPOSE,
            "dls": DifferentialIKMethod.DAMPED_LEAST_SQUARES,
            "adaptive_dls": DifferentialIKMethod.ADAPTIVE_DAMPING,
        }
        params = self.cfg.ik_params
        if params is None:
            raise RuntimeError(f"Inverse-kinematics parameters for method '{self.cfg.ik_method}' are not defined!")
        axis_weight = [1.0] * 6
        if self.cfg.command_type == "position":
            axis_weight[3:] = [0.0] * 3
        elif self._orientation_weight is not None:
            axis_weight[3:] = self._orientation_weight.tolist()
        # The previous adaptive solver included zero-weight rows in its SVD. Newton drops them;
        # preserve maximum damping when those rows made the historical task rank-deficient.
        task_dim = 3 if self.cfg.command_type == "position" else 6
        rank_deficient_weights = sum(weight != 0.0 for weight in axis_weight) < min(task_dim, self._num_joints)
        fixed_adaptive = self.cfg.ik_method == "adaptive_dls" and (
            rank_deficient_weights or params["lambda_min"] == params["lambda_max"]
        )
        method = "dls" if fixed_adaptive else self.cfg.ik_method
        # -- construct Newton controller and ports
        self._controller = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=wp.full(self.num_envs, self._num_joints, dtype=wp.int32, device=self._device),
            axis_weight=wp.spatial_vector(*axis_weight),
            bandwidth=params.get("k_val", 1.0),
            damping=params["lambda_max"] if fixed_adaptive else params.get("lambda_val"),
            ik_method=method_map[method],
            adaptive_damping_min=params.get("lambda_min") if method == "adaptive_dls" else None,
            adaptive_damping_max=params.get("lambda_max") if method == "adaptive_dls" else None,
            adaptive_damping_threshold=params.get("sigma_thresh") if method == "adaptive_dls" else None,
            truncated_svd_threshold=params.get("min_singular_value"),
            use_joint_limit_avoidance=use_joint_limits,
            joint_limit_avoidance_gain=self.cfg.joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=self.cfg.joint_limit_avoidance_margin,
            joint_pos_lower=(wp.from_torch(self._joint_pos_lower.repeat(self.num_envs)) if use_joint_limits else None),
            joint_pos_upper=(wp.from_torch(self._joint_pos_upper.repeat(self.num_envs)) if use_joint_limits else None),
            # Preserve the position-only null-space projector used by Isaac Lab's limit avoidance.
            null_space_axes=wp.spatial_vector(1, 1, 1, 0, 0, 0) if use_joint_limits else None,
            null_space_damping=0.0 if use_joint_limits else None,
            device=self._device,
        )
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()
        self._task_jacobian = wp.to_torch(self._controller_input.jacobian_tool_world)
        self._joint_pos = wp.to_torch(self._controller_input.joint_q).view(self.num_envs, self._num_joints)
        self._tool_pose = wp.to_torch(self._controller_input.tool_pose_world)
        self._desired_pose = wp.to_torch(self._controller_input.desired_tool_pose_world)
        self._joint_pos_des = wp.to_torch(self._controller_output.joint_q_target).view(self.num_envs, self._num_joints)
