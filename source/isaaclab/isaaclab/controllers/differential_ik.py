# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.utils.math import apply_delta_pose

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

    Newton's model-free controller evaluates the IK solve in float32. Command resolution stays in Isaac Lab.

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

        Raises:
            ValueError: If ``cfg.num_joints`` is unset.
        """
        # store inputs
        self.cfg = cfg
        self.num_envs = num_envs
        self._device = device
        if cfg.num_joints is None:
            raise ValueError("cfg.num_joints must be set before constructing DifferentialIKController.")
        self._num_joints = cfg.num_joints
        self._use_joint_limits = False
        self._joint_pos_lower = None
        self._joint_pos_upper = None
        # Share persistent command storage with Newton, including after enabling limit avoidance.
        self._desired_pose = torch.zeros(self.num_envs, 7, dtype=torch.float32, device=self._device)
        self.ee_pos_des = self._desired_pose[:, :3]
        self.ee_quat_des = self._desired_pose[:, 3:]
        # -- input command
        self._command = torch.zeros(self.num_envs, self.action_dim, dtype=torch.float32, device=self._device)
        # -- optional per-axis orientation task weights (used for "pose" command types only)
        if self.cfg.orientation_weight is None:
            self._orientation_weight = None
        else:
            ori_weight = self.cfg.orientation_weight
            self._orientation_weight = (
                (float(ori_weight),) * 3
                if isinstance(ori_weight, (int, float))
                else tuple(float(value) for value in ori_weight)
            )
        # -- identity quaternion (x, y, z, w), the last-resort fallback for a degenerate command
        self._identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device).repeat(self.num_envs, 1)
        self._initialize_controller()

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
            command: The input command in shape (N, 3), (N, 6), or (N, 7). Position components are [m],
                relative rotation components are [rad], and absolute quaternion components are dimensionless.
            ee_pos: The current end-effector position [m] in shape (N, 3).
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
                self.ee_pos_des[:] = ee_pos_des
                self.ee_quat_des[:] = ee_quat_des
            else:
                self.ee_pos_des[:] = self._command[:, 0:3]
                # normalize valid quaternions and use the fallback for non-finite results
                quat = self._command[:, 3:7]
                normalized_quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True)
                is_valid = torch.isfinite(normalized_quat).all(dim=-1, keepdim=True)
                fallback_quat = self._identity_quat if ee_quat is None else ee_quat
                self.ee_quat_des[:] = torch.where(is_valid, normalized_quat, fallback_quat)

    def set_joint_pos_limits(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Provide the controlled joints' position limits for null-space joint-limit avoidance.

        Only used when
        :attr:`~isaaclab.controllers.differential_ik_cfg.DifferentialIKControllerCfg.joint_limit_avoidance_gain`
        is positive. Limits can be supplied before or after the first compute call. Supplying
        them for the first time enables Newton's fixed avoidance feature by rebuilding
        the backend; recapture CUDA graphs after that transition. Subsequent updates retain it.

        Args:
            lower: Lower joint-position limits [m or rad, depending on joint type] in shape (num_joints,).
            upper: Upper joint-position limits [m or rad, depending on joint type] in shape (num_joints,).

        Raises:
            ValueError: If the limits have different lengths or an unexpected number of joints.
        """
        if lower.shape != upper.shape:
            raise ValueError(f"Expected matching lower and upper limit shapes, got {lower.shape} and {upper.shape}.")
        if lower.shape != (self._num_joints,):
            raise ValueError(f"Expected limits for {self._num_joints} joints, got {lower.shape}.")
        self._joint_pos_lower = lower.to(device=self._device, dtype=torch.float32)
        self._joint_pos_upper = upper.to(device=self._device, dtype=torch.float32)
        if self.cfg.joint_limit_avoidance_gain > 0.0:
            if self._use_joint_limits:
                self._controller.set_joint_limits(
                    joint_pos_lower=wp.from_torch(self._joint_pos_lower.repeat(self.num_envs)),
                    joint_pos_upper=wp.from_torch(self._joint_pos_upper.repeat(self.num_envs)),
                )
            else:
                self._initialize_controller()

    def compute(
        self,
        ee_pos: torch.Tensor,
        ee_quat: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the target joint positions that will yield the desired end effector pose.

        Args:
            ee_pos: The current end-effector position [m] in shape (N, 3).
            ee_quat: The current dimensionless end-effector orientation quaternion in shape (N, 4).
            jacobian: The geometric Jacobian in shape (N, 6, num_joints). Its linear rows map joint velocities
                to [m/s], and its angular rows map joint velocities to [rad/s].
            joint_pos: The current joint positions [m or rad, depending on joint type] in shape (N, num_joints).

        Returns:
            Target joint positions [m or rad, depending on joint type] in shape (N, num_joints).
            The returned tensor is an independent snapshot that remains unchanged by later calls.
        """
        # -- fixed topology and output contract
        if joint_pos.shape[1] != self._num_joints:
            raise ValueError(f"Expected {self._num_joints} controlled joints, got {joint_pos.shape[1]}.")
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

    """
    Helper functions.
    """

    def _initialize_controller(self) -> None:
        """Construct Newton and allocate its input and output ports."""
        from newton.controllers import ControllerDifferentialIKModelFree, DifferentialIKMethod

        self._use_joint_limits = self.cfg.joint_limit_avoidance_gain > 0.0 and self._joint_pos_lower is not None
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
            axis_weight[3:] = self._orientation_weight
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
            use_joint_limit_avoidance=self._use_joint_limits,
            joint_limit_avoidance_gain=self.cfg.joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=self.cfg.joint_limit_avoidance_margin,
            joint_pos_lower=(
                wp.from_torch(self._joint_pos_lower.repeat(self.num_envs)) if self._use_joint_limits else None
            ),
            joint_pos_upper=(
                wp.from_torch(self._joint_pos_upper.repeat(self.num_envs)) if self._use_joint_limits else None
            ),
            # Preserve the position-only null-space projector used by Isaac Lab's limit avoidance.
            null_space_axes=wp.spatial_vector(1, 1, 1, 0, 0, 0) if self._use_joint_limits else None,
            null_space_damping=0.0 if self._use_joint_limits else None,
            device=self._device,
        )
        self._controller_input = self._controller.input()
        self._controller_output = self._controller.output()
        self._task_jacobian = wp.to_torch(self._controller_input.jacobian_tool_world)
        self._joint_pos = wp.to_torch(self._controller_input.joint_q).view(self.num_envs, self._num_joints)
        self._tool_pose = wp.to_torch(self._controller_input.tool_pose_world)
        self._controller_input.desired_tool_pose_world = wp.from_torch(self._desired_pose, dtype=wp.transform)
        self._joint_pos_des = wp.to_torch(self._controller_output.joint_q_target).view(self.num_envs, self._num_joints)
