# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module for PhysX IMU sensor."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .imu import Imu
    from .imu_data import ImuData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("imu", "Imu"),
    ("imu_data", "ImuData"),
)
