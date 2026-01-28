# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class LeeControllerBaseCfg:
    """Base configuration for Lee-style geometric quadrotor controllers.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    The controller gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.

    Note:
        To disable randomization, set the min and max values to be identical.
        For example: K_rot_range = ((1.85, 1.85, 0.4), (1.85, 1.85, 0.4))
    """

    K_rot_range: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    """Orientation (rotation) error proportional gain range about body axes [unitless].

    This is a tuple of two tuples containing the minimum and maximum gains for roll, pitch, and yaw.
    Format: ((min_roll, min_pitch, min_yaw), (max_roll, max_pitch, max_yaw))

    To disable randomization, set both tuples to the same values.

    Example (with randomization):
        ((1.6, 1.6, 0.25), (1.85, 1.85, 0.4)) for ARL Robot 1

    Example (without randomization):
        ((1.85, 1.85, 0.4), (1.85, 1.85, 0.4)) for fixed gains
    """

    K_angvel_range: tuple[tuple[float, float, float], tuple[float, float, float]] = MISSING
    """Body angular-velocity error proportional gain range [unitless].

    This is a tuple of two tuples containing the minimum and maximum gains for roll, pitch, and yaw rates.
    Format: ((min_roll_rate, min_pitch_rate, min_yaw_rate), (max_roll_rate, max_pitch_rate, max_yaw_rate))

    To disable randomization, set both tuples to the same values.

    Example (with randomization):
        ((0.4, 0.4, 0.075), (0.5, 0.5, 0.09)) for ARL Robot 1

    Example (without randomization):
        ((0.5, 0.5, 0.09), (0.5, 0.5, 0.09)) for fixed gains
    """

    max_inclination_angle_rad: float = MISSING
    """Maximum allowed roll/pitch magnitude (inclination) in radians.

    This limits the maximum tilt angle of the quadrotor during control.
    Typical range: 0.5 to 1.57 radians (30° to 90°)

    Example:
        1.0471975511965976 (60° in radians) for ARL Robot 1
    """

    max_yaw_rate: float = MISSING
    """Maximum allowed yaw rate command [rad/s].

    This limits the maximum rotational velocity about the z-axis.
    Typical range: 0.5 to 2.0 rad/s

    Example:
        1.0471975511965976 (60°/s in radians) for ARL Robot 1
    """
