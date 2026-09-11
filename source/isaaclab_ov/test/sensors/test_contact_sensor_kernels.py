# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OvPhysx contact-sensor Warp kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_ov.sensors.contact_sensor.kernels import reset_contact_sensor_kernel, unpack_contact_buffer_data


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
    friction_force_matrix_w = wp.ones_like(normal_force_matrix_w)
    friction_force_matrix_w_history = wp.ones_like(normal_force_matrix_w_history)
    contact_pos_w = wp.ones_like(normal_force_matrix_w)

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
            friction_force_matrix_w,
            friction_force_matrix_w_history,
            contact_pos_w,
        ],
        device=device,
    )

    np.testing.assert_array_equal(normal_force_matrix_w_history.numpy()[0], 0.0)
    np.testing.assert_array_equal(normal_force_matrix_w_history.numpy()[1], 1.0)
    for forces in (friction_force_matrix_w, friction_force_matrix_w_history):
        np.testing.assert_array_equal(forces.numpy()[0], 0.0)
        np.testing.assert_array_equal(forces.numpy()[1], 1.0)
    assert np.isnan(contact_pos_w.numpy()[0]).all()
    np.testing.assert_array_equal(contact_pos_w.numpy()[1], 1.0)


@pytest.mark.parametrize("avg", [False, True])
def test_unpack_contact_buffer_data_pattern_major(avg: bool):
    """Aggregate variable-length pairs in body-major order, preserving unselected environments."""
    num_envs, num_sensors, num_filters = 3, 4, 2
    counts = np.arange(num_envs * num_sensors * num_filters, dtype=np.uint32) % 3
    starts = np.cumsum(counts, dtype=np.uint32) - counts
    flat = np.arange(int(counts.sum()) * 3, dtype=np.float32).reshape(-1, 3)
    pair_shape = (num_envs * num_sensors, num_filters)
    counts, starts = counts.reshape(pair_shape), starts.reshape(pair_shape)
    default = float("nan") if avg else 0.0
    expected = np.full((num_envs, num_sensors, num_filters, 3), -1.0, dtype=np.float32)
    for env in (0, 2):
        for sensor in range(num_sensors):
            for partner in range(num_filters):
                row = sensor * num_envs + env
                start, count = int(starts[row, partner]), int(counts[row, partner])
                values = flat[start : start + count]
                expected[env, sensor, partner] = (values.mean(0) if avg else values.sum(0)) if count else default
    output = wp.full(
        (num_envs, num_sensors, num_filters), value=wp.vec3f(wp.float32(-1.0)), dtype=wp.vec3f, device="cpu"
    )
    wp.launch(
        unpack_contact_buffer_data,
        dim=(num_envs, num_sensors, num_filters),
        inputs=[
            wp.array(flat, dtype=wp.vec3f, device="cpu"),
            wp.array(counts, dtype=wp.uint32, device="cpu"),
            wp.array(starts, dtype=wp.uint32, device="cpu"),
            wp.array([True, False, True], dtype=wp.bool, device="cpu"),
            num_envs,
            avg,
            default,
        ],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_allclose(output.numpy(), expected, equal_nan=True)
