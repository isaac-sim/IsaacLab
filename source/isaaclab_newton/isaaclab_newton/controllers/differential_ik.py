# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.controllers import ControllerDifferentialIKModelFree, DifferentialIKMethod

from .utils import bind_inputs, spatial_vector

if TYPE_CHECKING:
    from .differential_ik_cfg import NewtonDifferentialIKControllerCfg


class NewtonDifferentialIKController:
    """Differential inverse kinematics through Newton's model-free solver.

    Wraps :class:`newton.controllers.ControllerDifferentialIKModelFree` for batched torch inputs. The caller
    provides the end-effector pose and Jacobian, so the controller works with any physics backend. Poses,
    targets, and the Jacobian must share one frame, for example the robot root frame.

    Inputs are bound to Newton without copies when they are contiguous float32 tensors. The returned joint
    targets are a view of Newton's output buffer and are overwritten by the next :meth:`compute`.
    """

    def __init__(self, cfg: NewtonDifferentialIKControllerCfg, num_envs: int, num_joints: int, device: str):
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
        self._device = device

        # wide placeholder limits keep avoidance inactive until set_joint_pos_limits() provides real ones
        num_dofs = num_envs * num_joints
        use_limits = cfg.use_joint_limit_avoidance
        self._controller = ControllerDifferentialIKModelFree(
            controlled_dofs_per_robot=wp.full(num_envs, num_joints, dtype=wp.int32, device=device),
            axis_weight=spatial_vector(cfg.axis_weight),
            bandwidth=cfg.bandwidth,
            damping=cfg.damping,
            ik_method=DifferentialIKMethod(cfg.ik_method),
            adaptive_damping_min=cfg.adaptive_damping_min,
            adaptive_damping_max=cfg.adaptive_damping_max,
            adaptive_damping_threshold=cfg.adaptive_damping_threshold,
            truncated_svd_threshold=cfg.truncated_svd_threshold,
            use_joint_limit_avoidance=use_limits,
            joint_limit_avoidance_gain=cfg.joint_limit_avoidance_gain,
            joint_limit_avoidance_margin=cfg.joint_limit_avoidance_margin,
            joint_pos_lower=wp.full(num_dofs, -1.0e9, dtype=wp.float32, device=device) if use_limits else None,
            joint_pos_upper=wp.full(num_dofs, 1.0e9, dtype=wp.float32, device=device) if use_limits else None,
            use_null_space_posture_control=cfg.use_null_space_posture_control,
            null_space_stiffness=cfg.null_space_stiffness,
            null_space_damping=cfg.null_space_damping,
            null_space_axes=spatial_vector(cfg.null_space_axes),
            device=device,
        )
        self._inputs = self._controller.input()
        self._outputs = self._controller.output()
        self._joint_pos_des = wp.to_torch(self._outputs.joint_q_target).view(num_envs, num_joints)

    @property
    def newton_controller(self) -> ControllerDifferentialIKModelFree:
        """The wrapped Newton controller."""
        return self._controller

    def set_joint_pos_limits(self, lower: torch.Tensor, upper: torch.Tensor) -> None:
        """Set the joint position limits used by joint-limit avoidance.

        Args:
            lower: Lower joint-position limits [m or rad, depending on joint type], shape (num_joints,) or
                (num_envs, num_joints).
            upper: Upper joint-position limits [m or rad, depending on joint type], same shape as ``lower``.
        """
        shape = (self.num_envs, self.num_joints)
        lower = torch.broadcast_to(lower.to(self._device, torch.float32), shape).reshape(-1).contiguous()
        upper = torch.broadcast_to(upper.to(self._device, torch.float32), shape).reshape(-1).contiguous()
        self._controller.set_joint_limits(joint_pos_lower=wp.from_torch(lower), joint_pos_upper=wp.from_torch(upper))

    def compute(
        self,
        ee_pose: torch.Tensor,
        ee_pose_des: torch.Tensor,
        jacobian: torch.Tensor,
        joint_pos: torch.Tensor,
        dt: float,
        *,
        bandwidth: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
        null_space_joint_pos_target: torch.Tensor | None = None,
        null_space_stiffness: torch.Tensor | None = None,
        null_space_damping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the one-step-ahead joint position targets.

        Keyword inputs are only accepted when the configuration leaves the matching gain live or enables the
        matching feature. ``None`` keeps the previously bound tensor, or zeros if none was bound.

        Args:
            ee_pose: Current end-effector pose ``(x, y, z, qx, qy, qz, qw)`` [m, unitless], shape (num_envs, 7).
            ee_pose_des: Desired end-effector pose [m, unitless], shape (num_envs, 7).
            jacobian: End-effector Jacobian, shape (num_envs, 6, num_joints).
            joint_pos: Current joint positions [m or rad, depending on joint type], shape (num_envs, num_joints).
            dt: Step duration [s].
            bandwidth: Output velocity gain [1/s], shape (num_envs, num_joints).
            damping: Damped-least-squares regularization, shape (num_envs,).
            null_space_joint_pos_target: Posture target [m or rad, depending on joint type],
                shape (num_envs, num_joints).
            null_space_stiffness: Posture-control gain, shape (num_envs, num_joints).
            null_space_damping: Null-space projector regularization, shape (num_envs,).

        Returns:
            The joint position targets [m or rad, depending on joint type], shape (num_envs, num_joints).
        """
        bind_inputs(
            self._inputs,
            tool_pose_world=ee_pose,
            desired_tool_pose_world=ee_pose_des,
            jacobian_tool_world=jacobian,
            joint_q=joint_pos,
            bandwidth=bandwidth,
            damping=damping,
            q_des_null=null_space_joint_pos_target,
            null_space_stiffness=null_space_stiffness,
            null_space_damping=null_space_damping,
        )
        self._controller.step(inputs=self._inputs, outputs=self._outputs, dt=dt)
        return self._joint_pos_des
