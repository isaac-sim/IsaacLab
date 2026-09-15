# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class JointImpedanceControllerCfg:
    """Configuration for joint impedance regulation controller."""

    class_type: type | str = "isaaclab.controllers.joint_impedance:JointImpedanceController"
    """The associated controller class."""

    command_type: str = "p_abs"
    """Type of command: p_abs (absolute) or p_rel (relative)."""

    dof_pos_offset: Sequence[float] | None = None
    """Offset to DOF position command given to controller. (default: None).

    If None then position offsets are set to zero.
    """

    impedance_mode: str = MISSING
    """Type of gains: "fixed", "variable", "variable_kp"."""

    inertial_compensation: bool = False
    """Whether to perform inertial compensation (inverse dynamics)."""

    gravity_compensation: bool = False
    """Whether to perform gravity compensation."""

    stiffness: float | Sequence[float] = MISSING
    """The positional gain for determining desired torques based on joint position error."""

    damping_ratio: float | Sequence[float] | None = None
    """The damping ratio is used in-conjunction with positional gain to compute desired torques
    based on joint velocity error.

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
