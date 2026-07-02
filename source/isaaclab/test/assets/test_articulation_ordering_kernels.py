# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for shared articulation ordering Warp kernels."""

import numpy as np
import pytest
import warp as wp

from isaaclab.assets.articulation.ordering_kernels import (
    reorder_2d_backend_to_user,
    reorder_2d_user_to_backend,
    reorder_3d_backend_to_user,
    write_float_user_to_backend_with_indices,
    write_float_user_to_backend_with_mask,
    write_joint_state_user_to_backend_with_indices,
    write_joint_state_user_to_backend_with_mask,
    write_joint_vel_user_to_backend_with_indices,
    write_joint_vel_user_to_backend_with_mask,
)


def test_reorder_2d_backend_to_user_gathers_user_axis() -> None:
    """Reorder a 2-D backend buffer into public user order."""
    backend_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    user_to_backend = wp.array(np.asarray([2, 0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(backend_data)

    wp.launch(
        reorder_2d_backend_to_user,
        dim=user_data.shape,
        inputs=[backend_data, user_to_backend],
        outputs=[user_data],
        device="cpu",
    )

    expected = np.asarray([[12.0, 10.0, 11.0], [22.0, 20.0, 21.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected)


def test_reorder_2d_user_to_backend_scatters_backend_axis() -> None:
    """Reorder a 2-D public user buffer into backend order."""
    user_data = wp.array(
        np.asarray(
            [
                [12.0, 10.0, 11.0],
                [22.0, 20.0, 21.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    backend_to_user = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    backend_data = wp.zeros_like(user_data)

    wp.launch(
        reorder_2d_user_to_backend,
        dim=backend_data.shape,
        inputs=[user_data, backend_to_user],
        outputs=[backend_data],
        device="cpu",
    )

    expected = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)
    np.testing.assert_allclose(backend_data.numpy(), expected)


def test_reorder_3d_backend_to_user_gathers_user_axis() -> None:
    """Reorder a 3-D backend buffer whose second axis is the public user axis."""
    backend_data_np = np.asarray(
        [
            [[100.0, 101.0], [110.0, 111.0], [120.0, 121.0]],
            [[200.0, 201.0], [210.0, 211.0], [220.0, 221.0]],
        ],
        dtype=np.float32,
    )
    backend_data = wp.array(backend_data_np, dtype=wp.float32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(backend_data)

    wp.launch(
        reorder_3d_backend_to_user,
        dim=user_data.shape,
        inputs=[backend_data, user_to_backend],
        outputs=[user_data],
        device="cpu",
    )

    np.testing.assert_allclose(user_data.numpy(), backend_data_np[:, [1, 2, 0], :])


@pytest.mark.parametrize("selection", ["indices", "mask"])
@pytest.mark.parametrize("writer", ["scalar", "float", "velocity", "state"])
def test_fused_identity_writer_supports_aliased_outputs(selection: str, writer: str) -> None:
    """Preserve identity-order writes when public and backend outputs alias."""
    user_to_backend = wp.array(np.asarray([0, 1, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    env_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([1], dtype=np.int32), dtype=wp.int32, device="cpu")
    env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, False]), dtype=wp.bool, device="cpu")
    initial = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    input_data = wp.array(
        np.asarray([[101.0, 102.0, 103.0], [104.0, 105.0, 106.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )

    def expected_with_selected_value(source: np.ndarray, value: float) -> np.ndarray:
        expected = source.copy()
        expected[0, 1] = value
        return expected

    if writer == "scalar":
        data = wp.array(initial, dtype=wp.float32, device="cpu")
        if selection == "indices":
            wp.launch(
                write_float_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[42.0, env_ids, user_ids, user_to_backend, False, False],
                outputs=[data, data],
                device="cpu",
            )
        else:
            wp.launch(
                write_float_user_to_backend_with_mask,
                dim=data.shape,
                inputs=[42.0, env_mask, user_mask, user_to_backend, False],
                outputs=[data, data],
                device="cpu",
            )
        np.testing.assert_allclose(data.numpy(), expected_with_selected_value(initial, 42.0))
        return

    if writer == "float":
        data = wp.array(initial, dtype=wp.float32, device="cpu")
        if selection == "indices":
            wp.launch(
                write_float_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[input_data, env_ids, user_ids, user_to_backend, False, True],
                outputs=[data, data],
                device="cpu",
            )
        else:
            wp.launch(
                write_float_user_to_backend_with_mask,
                dim=data.shape,
                inputs=[input_data, env_mask, user_mask, user_to_backend, False],
                outputs=[data, data],
                device="cpu",
            )
        np.testing.assert_allclose(data.numpy(), expected_with_selected_value(initial, 102.0))
        return

    initial_velocity = initial + 10.0
    initial_previous_velocity = initial + 20.0
    initial_acceleration = initial + 30.0
    velocity = wp.array(initial_velocity, dtype=wp.float32, device="cpu")
    previous_velocity = wp.array(initial_previous_velocity, dtype=wp.float32, device="cpu")
    acceleration = wp.array(initial_acceleration, dtype=wp.float32, device="cpu")

    if writer == "velocity":
        if selection == "indices":
            wp.launch(
                write_joint_vel_user_to_backend_with_indices,
                dim=(env_ids.shape[0], user_ids.shape[0]),
                inputs=[input_data, env_ids, user_ids, user_to_backend, False, True],
                outputs=[velocity, previous_velocity, acceleration, velocity],
                device="cpu",
            )
        else:
            wp.launch(
                write_joint_vel_user_to_backend_with_mask,
                dim=velocity.shape,
                inputs=[input_data, env_mask, user_mask, user_to_backend, False],
                outputs=[velocity, previous_velocity, acceleration, velocity],
                device="cpu",
            )
        np.testing.assert_allclose(velocity.numpy(), expected_with_selected_value(initial_velocity, 102.0))
        np.testing.assert_allclose(
            previous_velocity.numpy(), expected_with_selected_value(initial_previous_velocity, 102.0)
        )
        np.testing.assert_allclose(acceleration.numpy(), expected_with_selected_value(initial_acceleration, 0.0))
        return

    position_data = wp.array(
        np.asarray([[201.0, 202.0, 203.0], [204.0, 205.0, 206.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    velocity_data = wp.array(
        np.asarray([[301.0, 302.0, 303.0], [304.0, 305.0, 306.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    initial_position = initial + 40.0
    position = wp.array(initial_position, dtype=wp.float32, device="cpu")
    if selection == "indices":
        wp.launch(
            write_joint_state_user_to_backend_with_indices,
            dim=(env_ids.shape[0], user_ids.shape[0]),
            inputs=[position_data, velocity_data, env_ids, user_ids, user_to_backend, False, True],
            outputs=[position, velocity, previous_velocity, acceleration, position, velocity],
            device="cpu",
        )
    else:
        wp.launch(
            write_joint_state_user_to_backend_with_mask,
            dim=position.shape,
            inputs=[position_data, velocity_data, env_mask, user_mask, user_to_backend, False],
            outputs=[position, velocity, previous_velocity, acceleration, position, velocity],
            device="cpu",
        )
    np.testing.assert_allclose(position.numpy(), expected_with_selected_value(initial_position, 202.0))
    np.testing.assert_allclose(velocity.numpy(), expected_with_selected_value(initial_velocity, 302.0))
    np.testing.assert_allclose(
        previous_velocity.numpy(), expected_with_selected_value(initial_previous_velocity, 302.0)
    )
    np.testing.assert_allclose(acceleration.numpy(), expected_with_selected_value(initial_acceleration, 0.0))


def test_write_float_scalar_user_to_backend_with_indices_updates_user_and_backend_buffers() -> None:
    """Fuse indexed scalar writes into user and backend-order buffers."""
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_float_user_to_backend_with_indices,
        dim=(env_ids.shape[0], user_ids.shape[0]),
        inputs=[4.5, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[4.5, 0.0, 4.5], [0.0, 0.0, 0.0], [4.5, 0.0, 4.5]], dtype=np.float32)
    expected_backend = np.asarray([[4.5, 4.5, 0.0], [0.0, 0.0, 0.0], [4.5, 4.5, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_float_scalar_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked scalar writes into user and backend-order buffers."""
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((2, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_float_user_to_backend_with_mask,
        dim=user_data.shape,
        inputs=[7.25, env_mask, user_mask, user_to_backend, True],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 7.25, 7.25], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[7.25, 0.0, 7.25], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_float_array_user_to_backend_with_indices_updates_user_and_backend_buffers() -> None:
    """Fuse partial user-order writes into user and backend-order buffers."""
    input_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0],
                [20.0, 21.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_data = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_float_user_to_backend_with_indices,
        dim=input_data.shape,
        inputs=[input_data, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[11.0, 0.0, 10.0], [0.0, 0.0, 0.0], [21.0, 0.0, 20.0]], dtype=np.float32)
    expected_backend = np.asarray([[10.0, 11.0, 0.0], [0.0, 0.0, 0.0], [20.0, 21.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_float_array_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked user-order writes into user and backend-order buffers."""
    input_data = wp.array(
        np.asarray(
            [
                [10.0, 11.0, 12.0],
                [20.0, 21.0, 22.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.float32,
        device="cpu",
    )
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(input_data)
    backend_data = wp.zeros_like(input_data)

    wp.launch(
        write_float_user_to_backend_with_mask,
        dim=input_data.shape,
        inputs=[input_data, env_mask, user_mask, user_to_backend, True],
        outputs=[user_data, backend_data],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[12.0, 0.0, 11.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_write_joint_state_user_to_backend_with_indices_updates_history_and_backend_buffers() -> None:
    """Fuse indexed joint state writes into user cache, history, and backend buffers."""
    position = wp.array(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    velocity = wp.array(np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    env_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_pos = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_prev_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    user_acc = wp.ones((3, 3), dtype=wp.float32, device="cpu")
    backend_pos = wp.zeros((3, 3), dtype=wp.float32, device="cpu")
    backend_vel = wp.zeros((3, 3), dtype=wp.float32, device="cpu")

    wp.launch(
        write_joint_state_user_to_backend_with_indices,
        dim=position.shape,
        inputs=[position, velocity, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_pos, user_vel, user_prev_vel, user_acc, backend_pos, backend_vel],
        device="cpu",
    )

    expected_user_pos = np.asarray([[2.0, 0.0, 1.0], [0.0, 0.0, 0.0], [4.0, 0.0, 3.0]], dtype=np.float32)
    expected_user_vel = np.asarray([[6.0, 0.0, 5.0], [0.0, 0.0, 0.0], [8.0, 0.0, 7.0]], dtype=np.float32)
    expected_backend_pos = np.asarray([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]], dtype=np.float32)
    expected_backend_vel = np.asarray([[5.0, 6.0, 0.0], [0.0, 0.0, 0.0], [7.0, 8.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_pos.numpy(), expected_user_pos)
    np.testing.assert_allclose(user_vel.numpy(), expected_user_vel)
    np.testing.assert_allclose(user_prev_vel.numpy(), expected_user_vel)
    np.testing.assert_allclose(backend_pos.numpy(), expected_backend_pos)
    np.testing.assert_allclose(backend_vel.numpy(), expected_backend_vel)
    np.testing.assert_allclose(user_acc.numpy()[[0, 2]][:, [0, 2]], np.zeros((2, 2), dtype=np.float32))


def test_write_joint_vel_user_to_backend_with_mask_updates_selected_entries() -> None:
    """Fuse masked joint velocity writes into user cache, history, and backend buffers."""
    velocity = wp.array(
        np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32),
        dtype=wp.float32,
        device="cpu",
    )
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_vel = wp.zeros_like(velocity)
    user_prev_vel = wp.zeros_like(velocity)
    user_acc = wp.ones_like(velocity)
    backend_vel = wp.zeros_like(velocity)

    wp.launch(
        write_joint_vel_user_to_backend_with_mask,
        dim=velocity.shape,
        inputs=[velocity, env_mask, user_mask, user_to_backend, True],
        outputs=[user_vel, user_prev_vel, user_acc, backend_vel],
        device="cpu",
    )

    expected_user = np.asarray([[0.0, 11.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    expected_backend = np.asarray([[12.0, 0.0, 11.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(user_vel.numpy(), expected_user)
    np.testing.assert_allclose(user_prev_vel.numpy(), expected_user)
    np.testing.assert_allclose(backend_vel.numpy(), expected_backend)
    np.testing.assert_allclose(user_acc.numpy()[0, 1:], np.zeros((2,), dtype=np.float32))
