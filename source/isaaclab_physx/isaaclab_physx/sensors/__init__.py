# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing PhysX-specific sensor implementations."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .contact_sensor import ContactSensor, ContactSensorData
    from .frame_transformer import FrameTransformer, FrameTransformerData
    from .imu import Imu, ImuData

from isaaclab.utils.module import lazy_export

lazy_export(
    ("contact_sensor", ["ContactSensor", "ContactSensorData"]),
    ("frame_transformer", ["FrameTransformer", "FrameTransformerData"]),
    ("imu", ["Imu", "ImuData"]),
)
