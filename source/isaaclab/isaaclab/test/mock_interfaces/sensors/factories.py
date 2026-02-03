# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Factory functions for creating pre-configured mock sensors."""

from __future__ import annotations

import torch

from .mock_contact_sensor import MockContactSensor
from .mock_frame_transformer import MockFrameTransformer
from .mock_imu import MockImu


def create_mock_imu(
    num_instances: int = 1,
    device: str = "cpu",
    gravity: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> MockImu:
    """Create a mock IMU sensor with default configuration.

    Args:
        num_instances: Number of sensor instances.
        device: Device for tensor allocation.
        gravity: Default gravity direction in body frame.

    Returns:
        Configured MockImu instance.
    """
    imu = MockImu(num_instances=num_instances, device=device)
    imu.data.set_projected_gravity_b(torch.tensor([gravity], device=device).expand(num_instances, -1).clone())
    return imu


def create_mock_contact_sensor(
    num_instances: int = 1,
    num_bodies: int = 1,
    body_names: list[str] | None = None,
    device: str = "cpu",
    history_length: int = 0,
    num_filter_bodies: int = 0,
) -> MockContactSensor:
    """Create a mock contact sensor with default configuration.

    Args:
        num_instances: Number of environment instances.
        num_bodies: Number of bodies with contact sensors.
        body_names: Names of bodies with contact sensors.
        device: Device for tensor allocation.
        history_length: Length of history buffer for forces.
        num_filter_bodies: Number of filter bodies for force matrix.

    Returns:
        Configured MockContactSensor instance.
    """
    return MockContactSensor(
        num_instances=num_instances,
        num_bodies=num_bodies,
        body_names=body_names,
        device=device,
        history_length=history_length,
        num_filter_bodies=num_filter_bodies,
    )


def create_mock_foot_contact_sensor(
    num_instances: int = 1,
    num_feet: int = 4,
    foot_names: list[str] | None = None,
    device: str = "cpu",
    history_length: int = 0,
) -> MockContactSensor:
    """Create a mock foot contact sensor for quadruped robots.

    Args:
        num_instances: Number of environment instances.
        num_feet: Number of feet (default 4 for quadruped).
        foot_names: Names of feet. Defaults to FL, FR, RL, RR.
        device: Device for tensor allocation.
        history_length: Length of history buffer for forces.

    Returns:
        Configured MockContactSensor instance for foot contacts.
    """
    if foot_names is None:
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"][:num_feet]

    return MockContactSensor(
        num_instances=num_instances,
        num_bodies=num_feet,
        body_names=foot_names,
        device=device,
        history_length=history_length,
    )


def create_mock_frame_transformer(
    num_instances: int = 1,
    num_target_frames: int = 1,
    target_frame_names: list[str] | None = None,
    device: str = "cpu",
) -> MockFrameTransformer:
    """Create a mock frame transformer sensor with default configuration.

    Args:
        num_instances: Number of environment instances.
        num_target_frames: Number of target frames to track.
        target_frame_names: Names of target frames.
        device: Device for tensor allocation.

    Returns:
        Configured MockFrameTransformer instance.
    """
    return MockFrameTransformer(
        num_instances=num_instances,
        num_target_frames=num_target_frames,
        target_frame_names=target_frame_names,
        device=device,
    )
