# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton contact sensor history buffers."""

# pyright: reportPrivateUsage=none

import warnings

import torch
import warp as wp
from isaaclab_newton.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab_newton.sensors.contact_sensor.contact_sensor_kernels import (
    copy_from_newton_kernel,
    update_contact_sensor_kernel,
)


def test_force_matrix_history_rolls_newest_first_and_honors_mask():
    """Test newest-first ordering without advancing masked environments."""
    data = ContactSensorData()
    data.create_buffers(2, 1, 1, 3, True, False, False, "cpu")
    timestamp = wp.ones((2,), dtype=wp.float32, device="cpu")
    timestamp_last_update = wp.zeros((2,), dtype=wp.float32, device="cpu")

    for value, mask_values in ((1.0, [True, True]), (2.0, [True, True]), (3.0, [True, False])):
        wp.to_torch(data._normal_force_matrix_w).fill_(value)
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
                data._net_normal_forces_w,
                data._normal_force_matrix_w,
                data._net_friction_forces_w,
                data._friction_force_matrix_w,
                timestamp,
                timestamp_last_update,
                data._net_forces_w_history,
                data._force_matrix_w_history,
                data._net_normal_forces_w_history,
                data._normal_force_matrix_w_history,
                data._net_friction_forces_w_history,
                data._friction_force_matrix_w_history,
                None,
                None,
                None,
                None,
            ],
            device="cpu",
        )

    history = data.normal_force_matrix_w_history.torch
    for env, values in enumerate(((3.0, 2.0, 1.0), (2.0, 1.0, 0.0))):
        for history_index, value in enumerate(values):
            torch.testing.assert_close(history[env, history_index], torch.full_like(history[env, history_index], value))


def test_copy_from_newton_decomposes_normal_and_friction_forces():
    """Test aggregate and filtered total-force decomposition."""
    data = ContactSensorData()
    data.create_buffers(1, 1, 2, 1, True, False, False, "cpu", track_friction_forces=True)
    total_force = wp.array([(3.0, 4.0, 0.0)], dtype=wp.vec3f, device="cpu")
    total_friction = wp.array([(0.0, 4.0, 0.0)], dtype=wp.vec3f, device="cpu")
    force_matrix = wp.array(
        [[(1.0, 2.0, 0.0), (0.0, 0.0, 3.0)]],
        dtype=wp.vec3f,
        ndim=2,
        device="cpu",
    )
    friction_matrix = wp.array(
        [[(0.0, 2.0, 0.0), (0.0, 0.0, 1.0)]],
        dtype=wp.vec3f,
        ndim=2,
        device="cpu",
    )
    positions = wp.zeros((1, 2), dtype=wp.vec3f, device="cpu")

    wp.launch(
        copy_from_newton_kernel,
        dim=(1, 1, 2),
        inputs=[
            wp.array([True], dtype=wp.bool, device="cpu"),
            1,
            total_force,
            total_friction,
            force_matrix,
            friction_matrix,
            positions,
            wp.ones((1,), dtype=wp.float32, device="cpu"),
        ],
        outputs=[
            data._net_forces_w,
            data._net_normal_forces_w,
            data._force_matrix_w,
            data._normal_force_matrix_w,
            data._net_friction_forces_w,
            data._friction_force_matrix_w,
            data._contact_pos_w,
        ],
        device="cpu",
    )

    torch.testing.assert_close(wp.to_torch(data._net_forces_w), torch.tensor([[[3.0, 4.0, 0.0]]]))
    torch.testing.assert_close(data.net_normal_forces_w.torch, torch.tensor([[[3.0, 0.0, 0.0]]]))
    torch.testing.assert_close(data.net_friction_forces_w.torch, torch.tensor([[[0.0, 4.0, 0.0]]]))
    torch.testing.assert_close(
        wp.to_torch(data._force_matrix_w),
        torch.tensor([[[[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]]),
    )
    torch.testing.assert_close(
        data.normal_force_matrix_w.torch,
        torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]]]]),
    )
    torch.testing.assert_close(
        data.friction_force_matrix_w.torch,
        torch.tensor([[[[0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]]]),
    )


def test_net_forces_w_is_newton_total_without_warning():
    """Test Newton total-force properties return totals without a warning."""
    data = ContactSensorData()
    data.create_buffers(1, 1, 1, 1, True, False, False, "cpu", track_friction_forces=True)
    wp.to_torch(data._net_forces_w).fill_(7.0)
    wp.to_torch(data._force_matrix_w).fill_(8.0)
    wp.to_torch(data._friction_force_matrix_w).fill_(9.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        total = data.net_forces_w
        matrix = data.force_matrix_w
        history = data.net_forces_w_history
        matrix_history = data.force_matrix_w_history
        friction = data.friction_forces_w
    assert total is not None
    assert matrix is not None
    assert history is not None
    assert matrix_history is not None
    assert friction is data.net_friction_forces_w
    assert not [item for item in caught if issubclass(item.category, UserWarning)]


def test_friction_force_history_rolls_newest_first():
    """Test friction history buffers roll newest-first."""
    data = ContactSensorData()
    data.create_buffers(1, 1, 1, 3, True, False, False, "cpu", track_friction_forces=True)
    timestamp = wp.ones((1,), dtype=wp.float32, device="cpu")
    timestamp_last_update = wp.zeros((1,), dtype=wp.float32, device="cpu")

    for value in (1.0, 2.0, 3.0):
        wp.to_torch(data._net_friction_forces_w).fill_(value)
        wp.to_torch(data._friction_force_matrix_w).fill_(value)
        wp.launch(
            update_contact_sensor_kernel,
            dim=(1, 1),
            inputs=[
                3,
                1,
                0.0,
                wp.array([True], dtype=wp.bool, device="cpu"),
                data._net_forces_w,
                data._force_matrix_w,
                data._net_normal_forces_w,
                data._normal_force_matrix_w,
                data._net_friction_forces_w,
                data._friction_force_matrix_w,
                timestamp,
                timestamp_last_update,
                data._net_forces_w_history,
                data._force_matrix_w_history,
                data._net_normal_forces_w_history,
                data._normal_force_matrix_w_history,
                data._net_friction_forces_w_history,
                data._friction_force_matrix_w_history,
                None,
                None,
                None,
                None,
            ],
            device="cpu",
        )

    torch.testing.assert_close(
        data.net_friction_forces_w_history.torch,
        torch.tensor([[[[3.0, 3.0, 3.0]], [[2.0, 2.0, 2.0]], [[1.0, 1.0, 1.0]]]]),
    )
    torch.testing.assert_close(
        data.friction_force_matrix_w_history.torch,
        torch.tensor([[[[[3.0, 3.0, 3.0]]], [[[2.0, 2.0, 2.0]]], [[[1.0, 1.0, 1.0]]]]]),
    )
