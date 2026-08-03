# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_newton.ik.newton_ik_objectives_cfg import NewtonIKObjectiveCfg
from isaaclab_newton.ik.newton_ik_solver_cfg import NewtonIKSolverCfg

if TYPE_CHECKING:
    from .newton_ik_actions import NewtonInverseKinematicsAction


@configclass
class NewtonInverseKinematicsActionCfg(ActionTermCfg):
    """Configuration for a Newton inverse-kinematics action term.

    The action solves IK as a single list of objectives. Pose objectives
    (:class:`~isaaclab_newton.ik.NewtonIKPoseObjectiveCfg`) are command-driven
    and contribute action dimensions -- one drives a single-body solve, several
    drive a multi-body solve. Constraint objectives such as
    :class:`~isaaclab_newton.ik.NewtonIKJointLimitObjectiveCfg` add residuals
    but no action dimensions. The action vector is the concatenation of every
    pose objective's slice, in list order.

    The action currently supports fixed-base articulations only. Each pose
    objective's body and the configured joints must resolve both in Isaac Lab
    and in the registered Newton prototype model for the controlled asset.
    """

    class_type: type[NewtonInverseKinematicsAction] | str = (
        "isaaclab_newton.envs.mdp.actions.newton_ik_actions:NewtonInverseKinematicsAction"
    )

    joint_names: list[str] = MISSING
    """Joints actuated by the action.

    The Newton solve resolves the whole prototype joint configuration jointly
    against all objectives, so this is the single set of joints written back to
    the articulation -- not a per-objective property.
    """

    objectives: list[NewtonIKObjectiveCfg] = MISSING
    """Ordered IK objectives. Must contain at least one pose objective."""

    controller: NewtonIKSolverCfg = field(default_factory=NewtonIKSolverCfg)
    """Configuration for the Newton IK solver (solver hyperparameters only)."""
