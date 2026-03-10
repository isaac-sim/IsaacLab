# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact sensor parity tests for the ovphysx backend.

Mirrors the structure of isaaclab_physx/test/sensors/check_contact_sensor.py.
Contact sensors are not yet supported by the ovphysx backend, so all tests
are skipped with an explanatory message.
"""

import pytest


def test_contact_sensor_creation():
    """Verify contact sensor can be created on the ovphysx backend."""
    pytest.skip("Contact sensor not yet supported by ovphysx backend.")


def test_contact_sensor_data_reading():
    """Verify contact sensor data can be read after a simulation step."""
    pytest.skip("Contact sensor not yet supported by ovphysx backend.")


def test_contact_sensor_reset():
    """Verify contact sensor state resets correctly."""
    pytest.skip("Contact sensor not yet supported by ovphysx backend.")


def test_contact_sensor_air_time_tracking():
    """Verify contact sensor air time tracking."""
    pytest.skip("Contact sensor not yet supported by ovphysx backend.")


def test_contact_sensor_friction_forces():
    """Verify contact sensor friction force reporting."""
    pytest.skip("Contact sensor not yet supported by ovphysx backend.")
