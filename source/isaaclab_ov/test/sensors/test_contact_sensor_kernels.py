# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysx contact-sensor Warp kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.sensors.contact_sensor.kernels import unpack_contact_buffer_data

from isaaclab.test.utils import DeviceScope, test_devices


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.parametrize("use_mask, capacity", [(False, None), (True, 5)])
def test_unpack_contact_buffer_data_pattern_major(device: str, use_mask: bool, capacity: int | None):
    """Preserve body/environment order, masked values and partial or absent contact positions."""
    num_envs, num_sensors, num_filters = 2, 2, 2
    positions = np.array(
        [[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9], [9, 10, 11], [11, 12, 13], [13, 14, 15], [15, 16, 17]],
        dtype=np.float32,
    )[:capacity]
    counts = wp.array([[2, 0], [1, 1], [0, 2], [1, 1]], dtype=wp.uint32, device=device)
    starts = wp.array([[0, 2], [2, 3], [4, 4], [6, 7]], dtype=wp.uint32, device=device)
    absent = [np.nan, np.nan, np.nan]
    expected = np.array(
        [
            [[[2, 3, 4], absent], [absent, [10, 11, 12]]],
            [[[5, 6, 7], [7, 8, 9]], [[13, 14, 15], [15, 16, 17]]],
        ],
        dtype=np.float32,
    )
    if capacity is not None:
        expected[0, 1, 1] = [9, 10, 11]
        expected[1, 1] = np.nan
    if use_mask:
        expected[1] = -1.0
    mask = wp.array([True, False], dtype=wp.bool, device=device) if use_mask else None
    output = wp.full((num_envs, num_sensors, num_filters), value=wp.vec3f(-1.0), dtype=wp.vec3f, device=device)
    wp.launch(
        unpack_contact_buffer_data,
        dim=(num_envs, num_sensors, num_filters),
        inputs=[wp.array(positions, dtype=wp.vec3f, device=device), counts, starts, mask, num_envs],
        outputs=[output],
        device=device,
    )
    np.testing.assert_allclose(output.numpy(), expected, equal_nan=True)
