# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.controllers import ControllerOperationalSpaceModelFree

from .utils import bind_inputs, per_joint, spatial_vector

if TYPE_CHECKING:
    from .operational_space_cfg import NewtonOperationalSpaceControllerCfg


class NewtonOperationalSpaceController:
    """Operational-space control through Newton's model-free controller.

    Wraps :class:`newton.controllers.ControllerOperationalSpaceModelFree` for batched torch inputs. The caller
    provides the end-effector state, Jacobian, and dynamics terms, so the controller works with any physics
    backend. Poses, twists, wrenches, and the Jacobian must share one frame, for example the robot root frame;
    desired poses, twists, and gains are expressed in the operational frame.

    Inputs are bound to Newton without copies when they are contiguous float32 tensors. The returned efforts
    are a view of Newton's output buffer and are overwritten by the next :meth:`compute`.
    """

    def __init__(self, cfg: NewtonOperationalSpaceControllerCfg, num_envs: int, num_joints: int, device: str):
        """Initialize the controller.

        Args:
            cfg: The controller configuration.
            num_envs: The number of environments.
            num_joints: The number of controlled joints per environment.
            device: The device to use for computations.
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_joints = num_joints

        frame = cfg.operational_frame_pose
        frame = None if frame is None else wp.transform(wp.vec3(*frame[:3]), wp.quat(*frame[3:]))
        linear_frame, angular_frame = cfg.linear_selection_frame, cfg.angular_selection_frame
        self._controller = ControllerOperationalSpaceModelFree(
            controlled_dofs_per_robot=wp.full(num_envs, num_joints, dtype=wp.int32, device=device),
            motion_stiffness=spatial_vector(cfg.motion_stiffness),
            motion_damping=spatial_vector(cfg.motion_damping),
            operational_frame_pose_world=frame,
            use_inertia_decoupling=cfg.use_inertia_decoupling,
            use_partial_inertia_decoupling=cfg.use_partial_inertia_decoupling,
            use_gravity_compensation=cfg.use_gravity_compensation,
            use_wrench_feedforward=cfg.use_wrench_feedforward,
            use_wrench_feedback=cfg.use_wrench_feedback,
            motion_selection_axes=spatial_vector(cfg.motion_selection_axes),
            wrench_selection_axes=spatial_vector(cfg.wrench_selection_axes),
            wrench_stiffness=spatial_vector(cfg.wrench_stiffness),
            linear_selection_frame_operational=None if linear_frame is None else wp.quat(*linear_frame),
            angular_selection_frame_operational=None if angular_frame is None else wp.quat(*angular_frame),
            use_null_space_control=cfg.use_null_space_control,
            null_space_stiffness=per_joint(cfg.null_space_stiffness, num_envs, device),
            null_space_damping=per_joint(cfg.null_space_damping, num_envs, device),
            device=device,
        )
        self._inputs = self._controller.input()
        self._outputs = self._controller.output()
        self._joint_efforts = wp.to_torch(self._outputs.joint_f).view(num_envs, num_joints)

    @property
    def newton_controller(self) -> ControllerOperationalSpaceModelFree:
        """The wrapped Newton controller."""
        return self._controller

    def compute(
        self,
        jacobian: torch.Tensor,
        ee_pose: torch.Tensor,
        ee_vel: torch.Tensor,
        ee_pose_des: torch.Tensor,
        *,
        ee_vel_des: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        operational_frame_pose: torch.Tensor | None = None,
        ee_wrench_des: torch.Tensor | None = None,
        ee_wrench: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
        null_space_joint_pos_target: torch.Tensor | None = None,
        null_space_joint_vel_target: torch.Tensor | None = None,
        motion_stiffness: torch.Tensor | None = None,
        motion_damping: torch.Tensor | None = None,
        wrench_stiffness: torch.Tensor | None = None,
        linear_selection_frame: torch.Tensor | None = None,
        angular_selection_frame: torch.Tensor | None = None,
        null_space_stiffness: torch.Tensor | None = None,
        null_space_damping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the joint efforts.

        Keyword inputs are only accepted when the configuration enables the matching feature or leaves the
        matching gain or frame live. ``None`` keeps the previously bound tensor, or zeros if none was bound.

        Args:
            jacobian: End-effector Jacobian, shape (num_envs, 6, num_joints).
            ee_pose: Current end-effector pose ``(x, y, z, qx, qy, qz, qw)`` [m, unitless], shape (num_envs, 7).
            ee_vel: Current end-effector twist ``(linear, angular)`` [m/s, rad/s], shape (num_envs, 6).
            ee_pose_des: Desired end-effector pose in the operational frame [m, unitless], shape (num_envs, 7).
            ee_vel_des: Desired end-effector twist in the operational frame [m/s, rad/s], shape (num_envs, 6).
            mass_matrix: Joint-space mass matrix, shape (num_envs, num_joints, num_joints).
            gravity: Gravity generalized forces [N or N·m, depending on joint type], shape (num_envs, num_joints).
            operational_frame_pose: Operational frame pose [m, unitless], shape (num_envs, 7).
            ee_wrench_des: Desired end-effector wrench ``(force, moment)`` [N, N·m], shape (num_envs, 6).
            ee_wrench: Measured end-effector wrench ``(force, moment)`` [N, N·m], shape (num_envs, 6).
            joint_pos: Current joint positions [m or rad, depending on joint type], shape (num_envs, num_joints).
            joint_vel: Current joint velocities [m/s or rad/s, depending on joint type],
                shape (num_envs, num_joints).
            null_space_joint_pos_target: Posture target [m or rad, depending on joint type],
                shape (num_envs, num_joints).
            null_space_joint_vel_target: Posture velocity target [m/s or rad/s, depending on joint type],
                shape (num_envs, num_joints).
            motion_stiffness: Task-space pose-error gain, shape (num_envs, 6).
            motion_damping: Task-space velocity-error gain, shape (num_envs, 6).
            wrench_stiffness: Wrench-error gain, shape (num_envs, 6).
            linear_selection_frame: Linear selection frame orientation ``(qx, qy, qz, qw)``, shape (num_envs, 4).
            angular_selection_frame: Angular selection frame orientation ``(qx, qy, qz, qw)``, shape (num_envs, 4).
            null_space_stiffness: Posture position-error gain, shape (num_envs, num_joints).
            null_space_damping: Posture velocity-error gain, shape (num_envs, num_joints).

        Returns:
            The joint efforts [N or N·m, depending on joint type], shape (num_envs, num_joints).
        """
        bind_inputs(
            self._inputs,
            jacobian_tool_world=jacobian,
            tool_pose_world=ee_pose,
            tool_twist_world=ee_vel,
            desired_tool_pose_operational=ee_pose_des,
            desired_twist_operational=ee_vel_des,
            mass_matrix=mass_matrix,
            gravity_force=gravity,
            operational_frame_pose_world=operational_frame_pose,
            desired_wrench_world=ee_wrench_des,
            measured_wrench_world=ee_wrench,
            joint_q=joint_pos,
            joint_qd=joint_vel,
            joint_q_des_null=null_space_joint_pos_target,
            joint_qd_des_null=null_space_joint_vel_target,
            motion_stiffness=motion_stiffness,
            motion_damping=motion_damping,
            wrench_stiffness=wrench_stiffness,
            linear_selection_frame_operational=linear_selection_frame,
            angular_selection_frame_operational=angular_selection_frame,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
        )
        # the operational-space law does not integrate, so dt is unused
        self._controller.step(inputs=self._inputs, outputs=self._outputs, dt=0.0)
        return self._joint_efforts
