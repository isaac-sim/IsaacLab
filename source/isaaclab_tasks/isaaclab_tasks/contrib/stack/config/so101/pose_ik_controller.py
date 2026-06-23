# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrist-only-orientation differential IK for the SO-101.

The SO-101 is a 5-DOF arm, so a full 6-DOF pose target is over-determined by one DOF. This is
handled with features that now live in the core :class:`~isaaclab.controllers.DifferentialIKController`:

* ``ik_method="adaptive_dls"`` -- manipulability-aware damped least squares.
* ``orientation_weight`` -- per-axis soft weighting of the orientation task rows.
* ``joint_limit_avoidance_gain`` / ``joint_limit_avoidance_margin`` -- null-space joint-limit avoidance.

The only SO-101-specific behavior left here is the **orientation joint mask**: restricting which
joints serve the orientation rows so the base (``shoulder_pan``) stays position-only and the wrist
serves orientation. See :attr:`SO101PoseIKControllerCfg.orientation_joint_names`.

.. note::
    The quaternion convention is **xyzw** (scalar-last), matching the IsaacLab asset convention and
    :func:`isaaclab.utils.math.compute_pose_error`. The core controller renormalizes the commanded
    quaternion in :meth:`~isaaclab.controllers.DifferentialIKController.set_command`.
"""

from __future__ import annotations

import torch

from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.utils.configclass import configclass


@configclass
class SO101PoseIKControllerCfg(DifferentialIKControllerCfg):
    """Core differential-IK config plus the SO-101 wrist-only orientation joint mask.

    Adaptive-DLS damping (``ik_method="adaptive_dls"`` with ``ik_params`` ``lambda_min`` /
    ``lambda_max`` / ``sigma_thresh``), per-axis :attr:`~DifferentialIKControllerCfg.orientation_weight`,
    and null-space joint-limit avoidance (:attr:`~DifferentialIKControllerCfg.joint_limit_avoidance_gain` /
    :attr:`~DifferentialIKControllerCfg.joint_limit_avoidance_margin`) are inherited from the core config.
    """

    orientation_joint_names: tuple[str, ...] | None = None
    """Names of the joints permitted to serve the orientation task rows. When set, every other
    joint's orientation-Jacobian columns are zeroed, so those joints serve **position only** while
    orientation is solved purely by the listed joints (position still uses all joints). ``None``
    (default) lets all joints serve orientation.

    For the SO-101's down-pointing gripper this is set to the wrist joints
    ``("wrist_flex", "wrist_roll")``: ``wrist_roll`` takes the gripper spin about the (vertical)
    approach axis -- the DOF otherwise redundant with ``shoulder_pan`` -- and ``wrist_flex`` takes
    the tilt, leaving ``shoulder_pan`` free to serve position (heading) so the base never swings to
    satisfy a commanded orientation. The action term resolves these names to Jacobian columns
    (asset-ordered, so the mask is order-proof) and pushes the column mask via
    :meth:`SO101PoseIKController.set_orientation_joint_mask`."""


class SO101PoseIKController(DifferentialIKController):
    """Core differential-IK controller plus an optional wrist-only orientation joint mask.

    All of the heavy lifting -- adaptive-DLS damping, per-axis orientation weighting, and
    null-space joint-limit avoidance -- is inherited from
    :class:`~isaaclab.controllers.DifferentialIKController`. This subclass only adds the orientation
    joint mask: zeroing the orientation-Jacobian columns of the masked-out joints so they serve
    position only (see :attr:`SO101PoseIKControllerCfg.orientation_joint_names`).
    """

    cfg: SO101PoseIKControllerCfg

    def __init__(self, cfg: SO101PoseIKControllerCfg, num_envs: int, device: str):
        super().__init__(cfg, num_envs, device)
        # Column mask (1 = joint may serve the orientation rows) over the IK joints, pushed by the
        # action term once it has resolved ``orientation_joint_names`` to Jacobian columns. ``None``
        # leaves all joints free to serve orientation (the default).
        self._ori_joint_mask: torch.Tensor | None = None

    def set_orientation_joint_mask(self, mask: torch.Tensor) -> None:
        """Restrict which joints serve the orientation task rows.

        Args:
            mask: Per-IK-joint multiplier [dimensionless], shape (num_joints,), in the controller's
                joint (Jacobian-column) order: 1 for joints allowed to serve orientation, 0 for
                joints that should serve position only. See
                :attr:`SO101PoseIKControllerCfg.orientation_joint_names`.
        """
        self._ori_joint_mask = mask.to(self._device)

    def _compute_pose_task(
        self, ee_pos: torch.Tensor, ee_quat: torch.Tensor, jacobian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the core (orientation-weighted) pose task, then mask the orientation columns.

        Zeroing the orientation rows of the masked-out joints means only the allowed joints can
        reduce the orientation error, so the base (``shoulder_pan``) serves position only and the
        redundant spin-about-vertical DOF is routed to ``wrist_roll``. Position rows are untouched.
        """
        task_jacobian, task_error = super()._compute_pose_task(ee_pos, ee_quat, jacobian)
        if self._ori_joint_mask is not None:
            task_jacobian = task_jacobian.clone()
            task_jacobian[:, 3:6, :] = task_jacobian[:, 3:6, :] * self._ori_joint_mask.view(1, 1, -1)
        return task_jacobian, task_error
