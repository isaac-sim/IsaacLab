# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Mock interfaces for IsaacLab sensors and assets.

This module provides mock implementations of sensor and asset classes for unit testing
without requiring the full Isaac Sim simulation environment.

Example usage:

    .. code-block:: python

        from isaaclab.test.mock_interfaces.sensors import MockContactSensor
        from isaaclab.test.mock_interfaces.assets import MockArticulation

        # Create mock sensor
        sensor = MockContactSensor(num_instances=4, num_bodies=4, device="cpu")
        sensor.data.set_mock_data(net_forces_w=torch.randn(4, 4, 3))

        # Create mock articulation
        robot = MockArticulation(
            num_instances=4,
            num_joints=12,
            num_bodies=13,
            joint_names=["joint_" + str(i) for i in range(12)],
            device="cpu"
        )

"""

from .assets import *
from .sensors import *
from .utils import *
