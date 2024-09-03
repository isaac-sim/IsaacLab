# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from collections.abc import Sequence

from omni.isaac.lab.utils import configclass

from .operational_space import OperationSpaceController


@configclass
class OperationSpaceControllerCfg:
    """Configuration for operation-space controller."""

    class_type: type = OperationSpaceController
    """The associated controller class."""

    command_type: Sequence[str] = MISSING
    """Type of command.

    It has two sub-strings joined by underscore:
        - type of command mode: "position", "pose", "force"
        - type of command resolving: "abs" (absolute), "rel" (relative)
    """

    impedance_mode: str = MISSING
    """Type of gains for motion control: "fixed", "variable", "variable_kp"."""

    uncouple_motion_wrench: bool = False
    """Whether to decouple the wrench computation from task-space pose (motion) error."""

    motion_control_axes: Sequence[int] = (1, 1, 1, 1, 1, 1)
    """Motion direction to control. Mark as 0/1 for each axis."""
    force_control_axes: Sequence[int] = (0, 0, 0, 0, 0, 0)
    """Force direction to control. Mark as 0/1 for each axis."""

    inertial_compensation: bool = False
    """Whether to perform inertial compensation for motion control (inverse dynamics)."""

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""

    stiffness: float | Sequence[float] = MISSING
    """The positional gain for determining wrenches based on task-space pose error."""

    damping_ratio: float | Sequence[float] | None = None
    """The damping ratio is used in-conjunction with positional gain to compute wrenches
    based on task-space velocity error.

    The following math operation is performed for computing velocity gains:
        :math:`d_gains = 2 * sqrt(p_gains) * damping_ratio`.
    """

    stiffness_limits: tuple[float, float] = (0, 300)
    """Minimum and maximum values for positional gains.

    Note: Used only when :obj:`impedance_mode` is "variable" or "variable_kp".
    """

    damping_ratio_limits: tuple[float, float] = (0, 100)
    """Minimum and maximum values for damping ratios used to compute velocity gains.

    Note: Used only when :obj:`impedance_mode` is "variable".
    """

    force_stiffness: float | Sequence[float] = None
    """The positional gain for determining wrenches for closed-loop force control.

    If obj:`None`, then open-loop control of desired forces is performed.
    """

    position_command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the position command received. Used only in relative mode."""
    rotation_command_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Scaling of the rotation command received. Used only in relative mode."""
