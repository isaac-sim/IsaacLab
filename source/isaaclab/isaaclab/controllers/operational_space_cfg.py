# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils.configclass import configclass

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

    inertial_decoupling_method: str = "inv"
    """Method used to invert the task-space inertia :math:`J M^{-1} J^T`: ``"inv"``, ``"cond_clamp"``.

    The task-space inertia loses rank at kinematic singularities, where a plain inverse produces
    unbounded command forces. Non-redundant (6-DoF) arms are most exposed, since they cannot
    reconfigure through a singularity in the null space, but redundant arms reach such
    configurations too.

    - Plain inverse (``"inv"``): no regularization. Matches the behavior of releases before this
      option existed and remains the default.
    - Condition-number clamp (``"cond_clamp"``): damps :math:`J M^{-1} J^T` by an amount derived
      from its own magnitude, bounding its condition number and so capping the resulting command
      forces.
        - ``"max_condition_number"``: approximate upper bound on the ratio between the largest and
          smallest eigenvalue (default: 1e6). The bound is approached within a factor of the task
          dimension, since the damping is keyed off the largest diagonal entry rather than the
          largest eigenvalue.

    Because the damping is set by a ratio, it is independent of the robot's mass and link scale.
    Away from singularities it perturbs the inverse by roughly ``1/max_condition_number`` in
    relative terms, which is small but not zero: expect some change in tracking even on
    well-conditioned setups. For reference, the task-space inertia of a UR10 or a Franka sits around
    1e4 to 3e5 during healthy tracking and climbs past 1e7 as the arm diverges. Lower the bound to
    intervene earlier, at the cost of tracking accuracy near singularities.
    """

    inertial_decoupling_params: dict[str, float] | None = None
    """Parameters for the given :attr:`inertial_decoupling_method`.

    Unspecified entries fall back to the defaults documented on :attr:`inertial_decoupling_method`.
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

    def __post_init__(self):
        # check valid input
        if self.inertial_decoupling_method not in ["inv", "cond_clamp"]:
            raise ValueError(f"Unsupported inertial decoupling method: {self.inertial_decoupling_method}.")
        # default parameters for each inversion method
        default_params = {
            "inv": {},
            "cond_clamp": {"max_condition_number": 1e6},
        }
        # update parameters for the chosen method if not provided
        params = default_params[self.inertial_decoupling_method].copy()
        if self.inertial_decoupling_params is not None:
            params.update(self.inertial_decoupling_params)
        self.inertial_decoupling_params = params
        # validate the clamp bound
        if (
            self.inertial_decoupling_method == "cond_clamp"
            and self.inertial_decoupling_params["max_condition_number"] <= 1.0
        ):
            raise ValueError(
                "cond_clamp max_condition_number must be > 1, got"
                f" {self.inertial_decoupling_params['max_condition_number']}."
            )
