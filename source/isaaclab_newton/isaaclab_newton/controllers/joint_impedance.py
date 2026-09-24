# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp
from newton.controllers import ControllerJointImpedanceModelFree

from .utils import bind_inputs, per_joint

if TYPE_CHECKING:
    from .joint_impedance_cfg import NewtonJointImpedanceControllerCfg


class NewtonJointImpedanceController:
    """Joint impedance control through Newton's model-free controller.

    Wraps :class:`newton.controllers.ControllerJointImpedanceModelFree` for batched torch inputs. The caller
    provides the joint state and dynamics terms, so the controller works with any physics backend.

    Inputs are bound to Newton without copies when they are contiguous float32 tensors. The returned efforts
    are a view of Newton's output buffer and are overwritten by the next :meth:`compute`.
    """

    def __init__(self, cfg: NewtonJointImpedanceControllerCfg, num_envs: int, num_joints: int, device: str):
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

        self._controller = ControllerJointImpedanceModelFree(
            controlled_dofs_per_robot=wp.full(num_envs, num_joints, dtype=wp.int32, device=device),
            stiffness=per_joint(cfg.stiffness, num_envs, device),
            damping=per_joint(cfg.damping, num_envs, device),
            use_gravity_compensation=cfg.use_gravity_compensation,
            use_coriolis_compensation=cfg.use_coriolis_compensation,
            use_inertia_decoupling=cfg.use_inertia_decoupling,
            has_qdd_feedforward=cfg.use_qdd_feedforward,
            device=device,
        )
        self._inputs = self._controller.input()
        self._outputs = self._controller.output()
        self._joint_efforts = wp.to_torch(self._outputs.joint_f).view(num_envs, num_joints)

    @property
    def newton_controller(self) -> ControllerJointImpedanceModelFree:
        """The wrapped Newton controller."""
        return self._controller

    def compute(
        self,
        joint_pos_des: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        *,
        joint_vel_des: torch.Tensor | None = None,
        joint_acc_des: torch.Tensor | None = None,
        mass_matrix: torch.Tensor | None = None,
        gravity: torch.Tensor | None = None,
        coriolis: torch.Tensor | None = None,
        stiffness: torch.Tensor | None = None,
        damping: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the joint efforts.

        Keyword inputs are only accepted when the configuration enables the matching feature or leaves the
        matching gain live. ``None`` keeps the previously bound tensor, or zeros if none was bound.

        Args:
            joint_pos_des: Desired joint positions [m or rad, depending on joint type], shape (num_envs, num_joints).
            joint_pos: Current joint positions [m or rad, depending on joint type], shape (num_envs, num_joints).
            joint_vel: Current joint velocities [m/s or rad/s, depending on joint type],
                shape (num_envs, num_joints).
            joint_vel_des: Desired joint velocities [m/s or rad/s, depending on joint type],
                shape (num_envs, num_joints).
            joint_acc_des: Desired joint accelerations [m/s² or rad/s², depending on joint type],
                shape (num_envs, num_joints).
            mass_matrix: Joint-space mass matrix, shape (num_envs, num_joints, num_joints).
            gravity: Gravity generalized forces [N or N·m, depending on joint type], shape (num_envs, num_joints).
            coriolis: Coriolis generalized forces [N or N·m, depending on joint type], shape (num_envs, num_joints).
            stiffness: Position-error gain, shape (num_envs, num_joints).
            damping: Velocity-error gain, shape (num_envs, num_joints).

        Returns:
            The joint efforts [N or N·m, depending on joint type], shape (num_envs, num_joints).
        """
        bind_inputs(
            self._inputs,
            joint_q_des=joint_pos_des,
            joint_q=joint_pos,
            joint_qd=joint_vel,
            joint_qd_des=joint_vel_des,
            joint_qdd=joint_acc_des,
            mass_matrix=mass_matrix,
            gravity_force=gravity,
            coriolis_force=coriolis,
            stiffness=stiffness,
            damping=damping,
        )
        # the impedance law does not integrate, so dt is unused
        self._controller.step(inputs=self._inputs, outputs=self._outputs, dt=0.0)
        return self._joint_efforts
