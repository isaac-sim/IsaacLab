# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2023, Autonomous Robots Lab, Norwegian University of Science and Technology.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .lee_controller import BaseLeeController


@configclass
class LeeControllerCfg:
    """Configuration for a Lee-style geometric quadrotor controller.

    Unless otherwise noted, vectors are ordered as (x, y, z) in the simulation world/body frames.
    When :attr:`randomize_params` is True, gains are sampled uniformly per environment between
    their corresponding ``*_min`` and ``*_max`` bounds at reset.
    """

    class_type: type[BaseLeeController] = BaseLeeController
    """Concrete controller class to instantiate."""

    gravity: list[float] = [0.0, 0.0, -9.81]
    """World gravity vector used by the controller [m/s^2]."""

    K_angvel_max: list[float] = [0.2, 0.2, 0.2]
    """Maximum proportional gains for body angular-velocity error (roll, pitch, yaw) [unitless]."""

    K_angvel_min: list[float] = [0.1, 0.1, 0.1]
    """Minimum proportional gains for body angular-velocity error (roll, pitch, yaw) [unitless]."""

    K_pos_max: list[float] = [3.0, 3.0, 2.0]
    """Maximum proportional gains for position error in world frame [1/s^2]."""

    K_pos_min: list[float] = [2.0, 2.0, 1.0]
    """Minimum proportional gains for position error in world frame [1/s^2]."""

    K_rot_max: list[float] = [1.2, 1.2, 0.6]
    """Maximum proportional gains for orientation (rotation) error about body axes [unitless]."""

    K_rot_min: list[float] = [0.8, 0.8, 0.4]
    """Minimum proportional gains for orientation (rotation) error about body axes [unitless]."""

    K_vel_max: list[float] = [3.0, 3.0, 3.0]
    """Maximum proportional gains for linear-velocity error in world frame [1/s]."""

    K_vel_min: list[float] = [2.0, 2.0, 2.0]
    """Minimum proportional gains for linear-velocity error in world frame [1/s]."""

    max_inclination_angle_rad: float = 1.0471975511965976
    """Maximum allowed roll/pitch magnitude (inclination) in radians."""

    max_yaw_rate: float = 1.0471975511965976
    """Maximum allowed yaw rate command [rad/s]."""

    num_actions: int = 4
    """Length of the action vector expected by the controller.

    Convention: ``[thrust_scale, roll_cmd, pitch_cmd, yaw_rate_cmd]`` where
    ``thrust_scale`` is mapped to collective thrust, commands are in radians/rad/s.
    """

    randomize_params: bool = False
    """If True, sample controller gains uniformly between the provided min/max bounds at resets."""
