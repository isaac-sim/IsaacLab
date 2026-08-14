# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton contact sensor history buffers."""

# pyright: reportPrivateUsage=none

import torch
import warp as wp
from isaaclab_newton.sensors.contact_sensor.contact_sensor_data import ContactSensorData
from isaaclab_newton.sensors.contact_sensor.contact_sensor_kernels import (
    reset_contact_sensor_kernel,
    update_contact_sensor_kernel,
)


def _create_data(history_length: int, num_filter_objects: int = 2) -> ContactSensorData:
    """Create CPU contact sensor data for deterministic kernel tests."""
    data = ContactSensorData()
    data.create_buffers(
        num_envs=2,
        num_sensors=1,
        num_filter_objects=num_filter_objects,
        history_length=history_length,
        generate_force_matrix=num_filter_objects > 0,
        track_air_time=False,
        track_pose=False,
        device="cpu",
    )
    return data


def _update_history(data: ContactSensorData, sample: torch.Tensor, env_mask: list[bool]) -> None:
    """Store a force-matrix sample and advance selected environments."""
    wp.to_torch(data._force_matrix_w).copy_(sample)
    mask = wp.array(env_mask, dtype=wp.bool, device="cpu")
    timestamp = wp.ones((2,), dtype=wp.float32, device="cpu")
    timestamp_last_update = wp.zeros((2,), dtype=wp.float32, device="cpu")
    wp.launch(
        update_contact_sensor_kernel,
        dim=(2, 1),
        inputs=[
            3,
            2,
            0.0,
            mask,
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


def test_create_buffers_matches_physx_history_shapes():
    """Test that zero configured history retains one sample like PhysX."""
    data = _create_data(history_length=0)

    assert data.net_forces_w_history is not None
    assert data.force_matrix_w_history is not None
    assert data.net_forces_w_history.torch.shape == (2, 1, 1, 3)
    assert data.force_matrix_w_history.torch.shape == (2, 1, 1, 2, 3)

    data_without_filters = _create_data(history_length=0, num_filter_objects=0)
    assert data_without_filters.force_matrix_w is None
    assert data_without_filters.force_matrix_w_history is None


def test_force_matrix_history_rolls_newest_first_and_honors_mask():
    """Test newest-first ordering without advancing masked environments."""
    data = _create_data(history_length=3)
    first = torch.tensor([[[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], [[[11.0, 0.0, 0.0], [12.0, 0.0, 0.0]]]])
    second = first + 20.0
    third = first + 40.0

    _update_history(data, first, [True, True])
    _update_history(data, second, [True, True])
    _update_history(data, third, [True, False])

    history = data.force_matrix_w_history.torch
    torch.testing.assert_close(history[0], torch.stack((third[0], second[0], first[0])))
    torch.testing.assert_close(history[1, 0], second[1])
    torch.testing.assert_close(history[1, 1], first[1])
    torch.testing.assert_close(history[1, 2], torch.zeros_like(first[1]))


def test_reset_clears_only_selected_force_history():
    """Test that reset clears current and historical forces for selected environments."""
    data = _create_data(history_length=3)
    wp.to_torch(data._net_forces_w).fill_(1.0)
    wp.to_torch(data._net_forces_w_history).fill_(1.0)
    wp.to_torch(data._force_matrix_w).fill_(1.0)
    wp.to_torch(data._force_matrix_w_history).fill_(1.0)
    mask = wp.array([True, False], dtype=wp.bool, device="cpu")

    wp.launch(
        reset_contact_sensor_kernel,
        dim=(2, 1),
        inputs=[
            3,
            2,
            mask,
            data._net_forces_w,
            data._net_forces_w_history,
            data._force_matrix_w,
            data._force_matrix_w_history,
            None,
        ],
        outputs=[None, None, None, None],
        device="cpu",
    )

    for buffer in (
        data._net_forces_w,
        data._net_forces_w_history,
        data._force_matrix_w,
        data._force_matrix_w_history,
    ):
        tensor = wp.to_torch(buffer)
        torch.testing.assert_close(tensor[0], torch.zeros_like(tensor[0]))
        torch.testing.assert_close(tensor[1], torch.ones_like(tensor[1]))
