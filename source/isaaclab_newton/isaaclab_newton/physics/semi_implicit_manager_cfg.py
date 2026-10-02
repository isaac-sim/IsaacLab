# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Newton's semi-implicit solver."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .newton_manager_cfg import NewtonSolverCfg

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonManager


@configclass
class SemiImplicitSolverCfg(NewtonSolverCfg):
    """A maximal-coordinate semi-implicit integrator using symplectic Euler.

    The solver supports prismatic, revolute, ball, fixed, free, distance, and
    D6 joints. It does not support rod joints, equality or mimic constraints,
    joint armature, joint friction, effort or velocity limits, or target mode.
    Ball-joint limits and targets are not enforced. Stiff systems can require
    a smaller time step because semi-implicit integration is not
    unconditionally stable.
    """

    class_type: type[NewtonManager] | str = "{DIR}.semi_implicit_manager:NewtonSemiImplicitManager"
    """Manager class for the semi-implicit solver."""

    solver_type: str = "semi_implicit"
    """Solver type. Can be ``"semi_implicit"``."""

    angular_damping: float = 0.05
    """Angular damping coefficient [1/s] for rigid-body integration."""

    friction_smoothing: float = 1.0
    """Huber-norm delta [m/s] used to normalize friction velocity."""

    joint_attach_ke: float = 1.0e4
    """Joint attachment spring stiffness [N/m for translation, N m/rad for rotation]."""

    joint_attach_kd: float = 1.0e2
    """Joint attachment damping [N s/m for translation, N m s/rad for rotation]."""

    enable_tri_contact: bool = True
    """Whether to enable triangle-triangle contact forces."""
