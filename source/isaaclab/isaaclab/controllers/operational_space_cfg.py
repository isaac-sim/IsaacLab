# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass

from .operational_space import OperationalSpaceController


@configclass
class OperationalSpaceControllerCfg:
    """Configuration for operational-space controller."""

    class_type: type = OperationalSpaceController
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
