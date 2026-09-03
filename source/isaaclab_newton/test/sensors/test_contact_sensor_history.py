# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton contact sensor history buffers."""

# pyright: reportPrivateUsage=none

import torch
import warp as wp
from isaaclab_newton.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab_newton.sensors.contact_sensor.contact_sensor_kernels import update_contact_sensor_kernel


def test_force_matrix_history_rolls_newest_first_and_honors_mask():
    """Test newest-first ordering without advancing masked environments."""
    data = ContactSensorData()
    data.create_buffers(2, 1, 1, 3, True, False, False, "cpu")
    timestamp = wp.ones((2,), dtype=wp.float32, device="cpu")
    timestamp_last_update = wp.zeros((2,), dtype=wp.float32, device="cpu")

    for value, mask_values in ((1.0, [True, True]), (2.0, [True, True]), (3.0, [True, False])):
        wp.to_torch(data._force_matrix_w).fill_(value)
        wp.launch(
            update_contact_sensor_kernel,
            dim=(2, 1),
            inputs=[
                3,
                1,
                0.0,
                wp.array(mask_values, dtype=wp.bool, device="cpu"),
                data._net_forces_w,
                data._force_matrix_w,
                timestamp,
                timestamp_last_update,
                data._net_forces_w_history,
                data._force_matrix_w_history,
                None,
                None,
                None,
                None,
            ],
            device="cpu",
        )

    history = data.force_matrix_w_history.torch
    for env, values in enumerate(((3.0, 2.0, 1.0), (2.0, 1.0, 0.0))):
        for history_index, value in enumerate(values):
            torch.testing.assert_close(history[env, history_index], torch.full_like(history[env, history_index], value))
