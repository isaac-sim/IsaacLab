# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock sensor interfaces for testing without Isaac Sim."""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from .mock_contact_sensor import MockContactSensor, MockContactSensorData
    from .mock_frame_transformer import MockFrameTransformer, MockFrameTransformerData
    from .mock_imu import MockImu, MockImuData
    from .factories import create_mock_contact_sensor, create_mock_foot_contact_sensor, create_mock_frame_transformer, create_mock_imu

from isaaclab.utils.module import lazy_export

lazy_export(
    ("mock_contact_sensor", ["MockContactSensor", "MockContactSensorData"]),
    ("mock_frame_transformer", ["MockFrameTransformer", "MockFrameTransformerData"]),
    ("mock_imu", ["MockImu", "MockImuData"]),
    ("factories", [
        "create_mock_contact_sensor",
        "create_mock_foot_contact_sensor",
        "create_mock_frame_transformer",
        "create_mock_imu",
    ]),
)
