# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .lee_acc_controller import LeeController


@configclass
class LeeAccControllerCfg:
    """Configuration for a Lee-style geometric quadrotor controller.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    When :attr:`randomize_params` is True, gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.
    """

    class_type: type[LeeController] = LeeController
    """Concrete controller class to instantiate."""

    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    """World gravity vector used by the controller [m/s^2]."""

    K_rot_range: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.8, 0.8, 0.4), (1.2, 1.2, 0.6))
    """Orientation (rotation) error proportional gain range about body axes [unitless]."""

    K_angvel_range: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.1, 0.1, 0.1), (0.2, 0.2, 0.2))
    """Body angular-velocity error proportional gain range (roll, pitch, yaw) [unitless]."""

    max_inclination_angle_rad: float = 1.0471975511965976
    """Maximum allowed roll/pitch magnitude (inclination) in radians."""

    max_yaw_rate: float = 1.0471975511965976
    """Maximum allowed yaw rate command [rad/s]."""

    randomize_params: bool = False
    """If True, sample controller gains uniformly between the provided min/max bounds at resets."""
