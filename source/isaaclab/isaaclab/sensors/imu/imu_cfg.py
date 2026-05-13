# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from ..sensor_base_cfg import SensorBaseCfg

if TYPE_CHECKING:
    from .imu import Imu


@configclass
class ImuCfg(SensorBaseCfg):
    """Configuration for an Inertial Measurement Unit (IMU) sensor.

    This configures a sensor that provides the two physical quantities measured by a
    real IMU: angular velocity (gyroscope) and linear acceleration (accelerometer).
    For a richer sensor that also provides pose, velocity, and angular acceleration,
    see :class:`~isaaclab.sensors.PvaCfg`.
    """

    class_type: type[Imu] | str = "{DIR}.imu:Imu"

    @configclass
    class OffsetCfg:
        """The offset pose of the sensor's frame from the sensor's parent frame."""

        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
        """Translation w.r.t. the parent frame [m]. Defaults to (0.0, 0.0, 0.0)."""

        rot: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
        """Quaternion rotation (x, y, z, w) w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0, 1.0)."""

    offset: OffsetCfg = OffsetCfg()
    """The offset pose of the sensor's frame from the sensor's parent frame. Defaults to identity."""
