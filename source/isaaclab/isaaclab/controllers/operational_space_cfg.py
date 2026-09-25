# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from ..utils import configclass

if TYPE_CHECKING:
    from .operational_space import OperationalSpaceController


@configclass
class OperationalSpaceControllerCfg:
    """Configuration for operational-space controller."""

    class_type: type[OperationalSpaceController] | str = "{DIR}.operational_space:OperationalSpaceController"
    """The associated controller class."""

    target_types: Sequence[str] = MISSING
    """Type of task-space targets.

    It has two sub-strings joined by underscore:
        - type of task-space target: ``"pose"``, ``"wrench"``
        - reference for the task-space targets: ``"abs"`` (absolute), ``"rel"`` (relative, only for pose)
    """

    motion_control_axes_task: Sequence[int] = (1, 1, 1, 1, 1, 1)
    """Motion direction to control in task reference frame. Mark as ``0/1`` for each axis."""

    contact_wrench_control_axes_task: Sequence[int] = (0, 0, 0, 0, 0, 0)
    """Contact wrench direction to control in task reference frame. Mark as 0/1 for each axis."""

    inertial_dynamics_decoupling: bool = False
    """Whether to perform inertial dynamics decoupling for motion control (inverse dynamics)."""

    partial_inertial_dynamics_decoupling: bool = False
    """Whether to ignore the inertial coupling between the translational & rotational motions."""

    inertia_conditioning_thresholds: tuple[float, float] = (1.0e-5, 1.0e-4)
    """Lower and upper relative eigenvalue thresholds for inertial decoupling near singularities.

    Each eigenvalue of :math:`J M^{-1} J^T` is divided by its largest eigenvalue. Directions at or below
    the lower threshold receive damping equal to the lower threshold times the largest eigenvalue,
    added before inversion. A cubic smoothstep reduces this damping to zero at the upper threshold,
    where the usual inverse is recovered. This bounds amplification without discarding weak directions.
    Inertia calculations use double precision to limit cancellation near singularities.
    Thresholds must satisfy ``0 < lower < upper <= 1``. With partial decoupling, each block is filtered
    separately. The ratios depend on the task's translational and rotational scaling.

    Full inertial decoupling uses the same damping for posture control: undamped directions remain
    dynamically decoupled, while weak directions gradually become available to posture control.
    Decoupling is approximate in damped directions. For singularity handling in operational space, see
    `Chang and Khatib (1995) <https://khatib.stanford.edu/publications/pdfs/Chang_1995.pdf>`_.
    Actuator effort limits must still be enforced separately.
    """

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""

    impedance_mode: str = "fixed"
    """Type of gains for motion control: ``"fixed"``, ``"variable"``, ``"variable_kp"``."""

    motion_stiffness_task: float | Sequence[float] = (100.0, 100.0, 100.0, 100.0, 100.0, 100.0)
    """The positional gain for determining operational space command forces based on task-space pose error."""

    motion_damping_ratio_task: float | Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    """The damping ratio is used in-conjunction with positional gain to compute operational space command forces
    based on task-space velocity error.

    The following math operation is performed for computing velocity gains:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`.
    """

    motion_stiffness_limits_task: tuple[float, float] = (0, 1000)
    """Minimum and maximum values for positional gains.

    Note: Used only when :obj:`impedance_mode` is ``"variable"`` or ``"variable_kp"``.
    """

    motion_damping_ratio_limits_task: tuple[float, float] = (0, 100)
    """Minimum and maximum values for damping ratios used to compute velocity gains.

    Note: Used only when :obj:`impedance_mode` is ``"variable"``.
    """

    contact_wrench_stiffness_task: float | Sequence[float] | None = None
    """The proportional gain for determining operational space command forces for closed-loop contact force control.

    If ``None``, then open-loop control of desired contact wrench is performed.

    Note: since only the linear forces could be measured at the moment,
    only the first three elements are used for the feedback loop.
    """

    nullspace_control: str = "none"
    """The null space control method for redundant manipulators: ``"none"``, ``"position"``.

    Note: ``"position"`` is used to drive the redundant manipulator to zero configuration by default. If
    ``target_joint_pos`` is provided in the ``compute()`` method, it will be driven to this configuration.
    """

    nullspace_stiffness: float = 10.0
    """The stiffness for null space control."""

    nullspace_damping_ratio: float = 1.0
    """The damping ratio for null space control."""
