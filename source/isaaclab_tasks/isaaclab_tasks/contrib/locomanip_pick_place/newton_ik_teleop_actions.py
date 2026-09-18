# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapter letting the teleoperation retargeter drive a Newton IK action term.

Isaac Lab Teleop already rebases its output poses into the robot's base frame when
:attr:`~isaaclab_teleop.IsaacTeleopCfg.target_frame_prim_path` names the base link, which is what
the Newton IK action term expects. What remains is quaternion order: the retargeter emits Isaac
Lab's ``wxyz`` while Warp's quaternion constructor reads ``xyzw``, so an identity rotation would
otherwise be read as a 180 degree one and the arms thrash.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab_newton.envs.mdp.actions.newton_ik_actions import NewtonInverseKinematicsAction
from isaaclab_newton.envs.mdp.actions.newton_ik_actions_cfg import NewtonInverseKinematicsActionCfg

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    pass

_POSE_DIM = 7
"""Action dimensions per pose objective: three for position, four for the quaternion."""


class TeleopNewtonInverseKinematicsAction(NewtonInverseKinematicsAction):
    """Newton IK action term fed by base-frame teleoperation targets in ``wxyz`` order."""

    def process_actions(self, actions: torch.Tensor) -> None:
        converted = actions.clone()
        # Every pose objective owns a leading 7-wide slice, in configuration order.
        num_poses = min(len(self.cfg.objectives), converted.shape[1] // _POSE_DIM)
        for i in range(num_poses):
            base = i * _POSE_DIM
            converted[:, base + 3 : base + 7] = actions[:, [base + 4, base + 5, base + 6, base + 3]]
        super().process_actions(converted)


@configclass
class TeleopNewtonInverseKinematicsActionCfg(NewtonInverseKinematicsActionCfg):
    """Configuration for :class:`TeleopNewtonInverseKinematicsAction`."""

    class_type: type[NewtonInverseKinematicsAction] | str = (
        "isaaclab_tasks.contrib.locomanip_pick_place.newton_ik_teleop_actions:"
        "TeleopNewtonInverseKinematicsAction"
    )
