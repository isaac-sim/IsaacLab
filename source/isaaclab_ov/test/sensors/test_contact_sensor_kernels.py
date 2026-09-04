# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysx contact-sensor Warp kernels."""

import numpy as np
import warp as wp
from isaaclab_ov.sensors.contact_sensor.kernels import reset_contact_sensor_kernel


def test_reset_contact_sensor_kernel_clears_selected_force_matrix_history():
    """Reset clears filtered-force history only for selected environments."""
    num_envs = 2
    num_sensors = 1
    history_length = 2
    num_filter_shapes = 1
    device = "cpu"
    env_mask = wp.array([True, False], dtype=wp.bool, device=device)

    net_normal_forces_w = wp.zeros((num_envs, num_sensors), dtype=wp.vec3f, device=device)
    net_normal_forces_w_history = wp.zeros((num_envs, history_length, num_sensors), dtype=wp.vec3f, device=device)
    normal_force_matrix_w = wp.zeros((num_envs, num_sensors, num_filter_shapes), dtype=wp.vec3f, device=device)
    normal_force_matrix_w_history = wp.array(
        np.ones((num_envs, history_length, num_sensors, num_filter_shapes, 3), dtype=np.float32),
        dtype=wp.vec3f,
        device=device,
    )
    current_air_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
    last_air_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
    current_contact_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)
    last_contact_time = wp.zeros((num_envs, num_sensors), dtype=wp.float32, device=device)

    wp.launch(
        reset_contact_sensor_kernel,
        dim=(num_envs, num_sensors),
        inputs=[
            history_length,
            num_filter_shapes,
            env_mask,
            net_normal_forces_w,
            net_normal_forces_w_history,
            normal_force_matrix_w,
            normal_force_matrix_w_history,
        ],
        outputs=[
            current_air_time,
            last_air_time,
            current_contact_time,
            last_contact_time,
            None,
            None,
            None,
        ],
        device=device,
    )

    np.testing.assert_array_equal(normal_force_matrix_w_history.numpy()[0], 0.0)
    np.testing.assert_array_equal(normal_force_matrix_w_history.numpy()[1], 1.0)
