# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .operational_space import NewtonOperationalSpaceController


@configclass
class NewtonOperationalSpaceControllerCfg:
    """Configuration for :class:`NewtonOperationalSpaceController`.

    The fields map one-to-one onto :class:`newton.controllers.ControllerOperationalSpaceModelFree`, whose
    defaults they mirror, and Newton validates the combination at construction. Per-axis values are a scalar or
    a ``(x, y, z, roll, pitch, yaw)`` tuple expressed in the operational frame. A gain or frame left at ``None``
    is read live from :meth:`~NewtonOperationalSpaceController.compute`.

    Motion and null-space gains are in [1/s²] and [1/s] with :attr:`use_inertia_decoupling`, otherwise in
    force per unit error.
    """

    class_type: type[NewtonOperationalSpaceController] | str = (
        "{DIR}.operational_space:NewtonOperationalSpaceController"
    )
    """The associated controller class."""

    motion_stiffness: float | tuple[float, float, float, float, float, float] | None = MISSING
    """Task-space pose-error gain Kp."""

    motion_damping: float | tuple[float, float, float, float, float, float] | None = MISSING
    """Task-space velocity-error gain Kd."""

    operational_frame_pose: tuple[float, ...] | None = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    """Pose ``(x, y, z, qx, qy, qz, qw)`` [m, unitless] of the frame that gains and targets are expressed in.

    Defaults to the frame of the other inputs.
    """

    use_inertia_decoupling: bool = True
    """Whether to decouple the task dynamics with the task-space inertia. Requires at least six joints."""

    use_partial_inertia_decoupling: bool = False
    """Whether to ignore coupling between translational and rotational inertia when decoupling."""

    use_gravity_compensation: bool = True
    """Whether to add the gravity generalized forces to the output."""

    use_wrench_feedforward: bool = False
    """Whether to command the desired wrench directly on the wrench-selected axes."""

    use_wrench_feedback: bool = False
    """Whether to correct the wrench command with the measured-wrench error."""

    motion_selection_axes: tuple[float, float, float, float, float, float] | None = None
    """Motion-controlled axes. Only used with wrench control; defaults to all axes."""

    wrench_selection_axes: tuple[float, float, float, float, float, float] | None = None
    """Wrench-controlled axes. Required with wrench control."""

    wrench_stiffness: float | tuple[float, float, float, float, float, float] | None = None
    """Dimensionless wrench-error gain. Only used with :attr:`use_wrench_feedback`."""

    linear_selection_frame: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 1.0)
    """Orientation ``(qx, qy, qz, qw)`` of the frame for the linear selection axes, in the operational frame."""

    angular_selection_frame: tuple[float, float, float, float] | None = (0.0, 0.0, 0.0, 1.0)
    """Orientation ``(qx, qy, qz, qw)`` of the frame for the angular selection axes, in the operational frame."""

    use_null_space_control: bool = False
    """Whether to track a joint posture target in the task null space. Requires more than six joints."""

    null_space_stiffness: float | Sequence[float] | None = None
    """Posture position-error gain, scalar or per joint. Only used with :attr:`use_null_space_control`."""

    null_space_damping: float | Sequence[float] | None = None
    """Posture velocity-error gain, scalar or per joint. Only used with :attr:`use_null_space_control`."""
