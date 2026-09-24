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
    """Test newest-first ordering of every history buffer without advancing masked environments.

    Newton's total-force properties return the totals without the base-class PhysX-limitation warning.
    """
    data = ContactSensorData()
    data.create_buffers(2, 1, 1, 3, True, False, False, "cpu", track_friction_forces=True)
    timestamp = wp.ones((2,), dtype=wp.float32, device="cpu")
    timestamp_last_update = wp.zeros((2,), dtype=wp.float32, device="cpu")

    # Each buffer gets a distinct scale so a mis-wired history shows up as a wrong value.
    buffers = {
        "net_forces_w": 1.0,
        "force_matrix_w": 10.0,
        "normal_force_matrix_w": 100.0,
        "net_friction_forces_w": 1000.0,
        "friction_force_matrix_w": 10000.0,
    }
    for value, mask_values in ((1.0, [True, True]), (2.0, [True, True]), (3.0, [True, False])):
        for name, scale in buffers.items():
            wp.to_torch(getattr(data, f"_{name}")).fill_(scale * value)
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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        histories = {name: getattr(data, f"{name}_history").torch for name in buffers}
        torch.testing.assert_close(data.net_forces_w.torch, torch.full((2, 1, 3), 3.0))
        torch.testing.assert_close(data.force_matrix_w.torch, torch.full((2, 1, 1, 3), 30.0))
        assert data.friction_forces_w is data.net_friction_forces_w
    assert not [item for item in caught if issubclass(item.category, UserWarning)]

    for name, scale in buffers.items():
        history = histories[name]
        for env, values in enumerate(((3.0, 2.0, 1.0), (2.0, 1.0, 0.0))):
            for history_index, value in enumerate(values):
                expected = torch.full_like(history[env, history_index], scale * value)
                torch.testing.assert_close(history[env, history_index], expected, msg=name)


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

    torch.testing.assert_close(data.net_forces_w.torch, torch.tensor([[[3.0, 4.0, 0.0]]]))
    torch.testing.assert_close(data.net_normal_forces_w.torch, torch.tensor([[[3.0, 0.0, 0.0]]]))
    torch.testing.assert_close(data.net_friction_forces_w.torch, torch.tensor([[[0.0, 4.0, 0.0]]]))
    torch.testing.assert_close(
        data.force_matrix_w.torch,
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
