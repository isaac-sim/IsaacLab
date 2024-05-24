# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Callable

from omni.isaac.orbit.sim.spawners.spawner_cfg import RigidObjectSpawnerCfg
from omni.isaac.orbit.utils import configclass

from . import race_quadcopter


@configclass
class RaceQuadcopterCfg(RigidObjectSpawnerCfg):
    """Configuration for the quadcopter in FLU body frame convention.

    The center of body frame is the crossing point of the arms,
    on the upper surface of the arm rectangles.

    Collision shape is the minimum bounding box of the quadcopter,
    and is automatically computed.

    Additional mass properties are defined here instead of in `MassPropertiesCfg`
    to avoid breaking existing code and tests.
    """

    func: Callable = race_quadcopter.spawn_race_quadcopter

    # visual

    arm_length_rear: float = 0.14
    """Length of the two rear arms [m]."""

    arm_length_front: float = 0.14
    """Length of the two front arms [m]."""

    arm_thickness: float = 0.01
    """Thickness of the arm plate [m]."""

    arm_front_angle: float = 100.0 * math.pi / 180
    """Separation angle between two front arms [rad]."""

    motor_diameter: float = 0.023
    """Diameter of the motor cylinder [m]."""

    motor_height: float = 0.006
    """Height of the motor cylinder [m]."""

    central_body_length_x: float = 0.15
    """X-dimension of the cnetral body cuboid [m]."""

    central_body_length_y: float = 0.05
    """Y-dimension of the cnetral body cuboid [m]."""

    central_body_length_z: float = 0.05
    """Z-dimension of the cnetral body cuboid [m]."""

    central_body_center_x: float = 0.0
    """X-position of the cnetral body cuboid [m]."""

    central_body_center_y: float = 0.0
    """Y-position of the cnetral body cuboid [m]."""

    central_body_center_z: float = 0.015
    """Y-position of the cnetral body cuboid [m]."""

    propeller_diameter: float = 6 * 2.54 * 0.01
    """Diameter of the propeller cylinder [m]."""

    propeller_height: float = 0.01
    """Height of the propeller cylinder [m]."""

    # additional mass properties

    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Center of mass position in body frame [m]."""

    diagonal_inertia: tuple[float, float, float] = (0.0025, 0.0025, 0.0045)
    """Diagonal inertia [kg m^2]."""

    principal_axes_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    """Quaternion representing the same rotation as the principal axes matrix."""
