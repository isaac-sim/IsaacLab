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
    reorder_body_state_backend_to_user,
    reorder_joint_state_backend_to_user,
    reorder_joint_targets_user_to_backend,
    write_2d_user_to_backend_with_indices,
    write_2d_user_to_backend_with_mask,
    write_3d_user_to_backend_with_indices,
    write_3d_user_to_backend_with_mask,
    write_float_user_to_backend_with_indices,
    write_float_user_to_backend_with_mask,
    write_joint_state_user_to_backend_with_indices,
    write_joint_state_user_to_backend_with_indices_kernel,
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


def test_reorder_joint_state_backend_to_user_gathers_position_and_velocity() -> None:
    """Fuse the two joint-state publish gathers under a single shared joint map."""
    backend_pos_np = np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32)
    backend_vel_np = backend_pos_np + 100.0
    # Non-identity permutation: public joint i draws from backend joint user_to_backend[i].
    user_to_backend_np = np.asarray([2, 0, 1], dtype=np.int32)

    backend_pos = wp.array(backend_pos_np, dtype=wp.float32, device="cpu")
    backend_vel = wp.array(backend_vel_np, dtype=wp.float32, device="cpu")
    user_to_backend = wp.array(user_to_backend_np, dtype=wp.int32, device="cpu")
    user_pos = wp.zeros_like(backend_pos)
    user_vel = wp.zeros_like(backend_vel)

    wp.launch(
        reorder_joint_state_backend_to_user,
        dim=(backend_pos_np.shape[0], backend_pos_np.shape[1]),
        inputs=[backend_pos, backend_vel, user_to_backend],
        outputs=[user_pos, user_vel],
        device="cpu",
    )

    np.testing.assert_array_equal(user_pos.numpy(), backend_pos_np[:, user_to_backend_np])
    np.testing.assert_array_equal(user_vel.numpy(), backend_vel_np[:, user_to_backend_np])


def test_reorder_body_state_backend_to_user_gathers_pose_and_velocity() -> None:
    """Fuse the transform and spatial-vector body-state publish gathers under one body map."""
    num_envs, num_bodies = 2, 3
    backend_pose_np = np.arange(num_envs * num_bodies * 7, dtype=np.float32).reshape(num_envs, num_bodies, 7)
    backend_vel_np = np.arange(num_envs * num_bodies * 6, dtype=np.float32).reshape(num_envs, num_bodies, 6) + 500.0
    # Non-identity permutation: public body i draws from backend body user_to_backend[i].
    user_to_backend_np = np.asarray([2, 0, 1], dtype=np.int32)

    backend_pose = wp.array(backend_pose_np, dtype=wp.transformf, device="cpu")
    backend_vel = wp.array(backend_vel_np, dtype=wp.spatial_vectorf, device="cpu")
    user_to_backend = wp.array(user_to_backend_np, dtype=wp.int32, device="cpu")
    user_pose = wp.zeros_like(backend_pose)
    user_vel = wp.zeros_like(backend_vel)

    wp.launch(
        reorder_body_state_backend_to_user,
        dim=(num_envs, num_bodies),
        inputs=[backend_pose, backend_vel, user_to_backend],
        outputs=[user_pose, user_vel],
        device="cpu",
    )

    np.testing.assert_array_equal(user_pose.numpy(), backend_pose_np[:, user_to_backend_np, :])
    np.testing.assert_array_equal(user_vel.numpy(), backend_vel_np[:, user_to_backend_np, :])


@pytest.mark.parametrize(
    "combo, write_effort, write_pos, write_vel, write_act",
    [
        ("effort_only", True, False, False, False),
        ("all_targets", True, True, True, False),
        ("newton_four", True, True, True, True),
    ],
)
def test_reorder_joint_targets_user_to_backend_writes_only_flagged_outputs(
    combo: str,
    write_effort: bool,
    write_pos: bool,
    write_vel: bool,
    write_act: bool,
) -> None:
    """Fuse the write-path target reorders, honoring per-output flags over a non-trivial permutation."""
    user_effort_np = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    user_pos_np = user_effort_np + 10.0
    user_vel_np = user_effort_np + 20.0
    # Non-identity permutation: backend joint i draws from public joint backend_to_user[i].
    backend_to_user_np = np.asarray([2, 0, 1], dtype=np.int32)

    user_effort = wp.array(user_effort_np, dtype=wp.float32, device="cpu")
    user_pos = wp.array(user_pos_np, dtype=wp.float32, device="cpu")
    user_vel = wp.array(user_vel_np, dtype=wp.float32, device="cpu")
    backend_to_user = wp.array(backend_to_user_np, dtype=wp.int32, device="cpu")

    # Distinct sentinels so gated-off outputs are provably untouched.
    effort_sentinel = np.full_like(user_effort_np, -1.0)
    pos_sentinel = np.full_like(user_effort_np, -2.0)
    vel_sentinel = np.full_like(user_effort_np, -3.0)
    act_sentinel = np.full_like(user_effort_np, -4.0)
    backend_effort = wp.array(effort_sentinel, dtype=wp.float32, device="cpu")
    backend_pos = wp.array(pos_sentinel, dtype=wp.float32, device="cpu")
    backend_vel = wp.array(vel_sentinel, dtype=wp.float32, device="cpu")
    backend_act = wp.array(act_sentinel, dtype=wp.float32, device="cpu")

    wp.launch(
        reorder_joint_targets_user_to_backend,
        dim=(user_effort_np.shape[0], user_effort_np.shape[1]),
        inputs=[user_effort, user_pos, user_vel, backend_to_user, write_effort, write_pos, write_vel, write_act],
        outputs=[backend_effort, backend_pos, backend_vel, backend_act],
        device="cpu",
    )

    expected_effort = user_effort_np[:, backend_to_user_np] if write_effort else effort_sentinel
    expected_pos = user_pos_np[:, backend_to_user_np] if write_pos else pos_sentinel
    expected_vel = user_vel_np[:, backend_to_user_np] if write_vel else vel_sentinel
    expected_act = user_effort_np[:, backend_to_user_np] if write_act else act_sentinel
    np.testing.assert_array_equal(backend_effort.numpy(), expected_effort)
    np.testing.assert_array_equal(backend_pos.numpy(), expected_pos)
    np.testing.assert_array_equal(backend_vel.numpy(), expected_vel)
    np.testing.assert_array_equal(backend_act.numpy(), expected_act)


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
            write_float_user_to_backend_with_indices(
                42.0, env_ids, user_ids, user_to_backend, False, False, data, data, device="cpu"
            )
        else:
            write_float_user_to_backend_with_mask(
                42.0, env_mask, user_mask, user_to_backend, False, data, data, device="cpu"
            )
        np.testing.assert_allclose(data.numpy(), expected_with_selected_value(initial, 42.0))
        return

    if writer == "float":
        data = wp.array(initial, dtype=wp.float32, device="cpu")
        if selection == "indices":
            write_float_user_to_backend_with_indices(
                input_data, env_ids, user_ids, user_to_backend, False, True, data, data, device="cpu"
            )
        else:
            write_float_user_to_backend_with_mask(
                input_data, env_mask, user_mask, user_to_backend, False, data, data, device="cpu"
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

    write_float_user_to_backend_with_indices(
        4.5, env_ids, user_ids, user_to_backend, True, False, user_data, backend_data, device="cpu"
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

    write_float_user_to_backend_with_mask(
        7.25, env_mask, user_mask, user_to_backend, True, user_data, backend_data, device="cpu"
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

    write_float_user_to_backend_with_indices(
        input_data, env_ids, user_ids, user_to_backend, True, False, user_data, backend_data, device="cpu"
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

    write_float_user_to_backend_with_mask(
        input_data, env_mask, user_mask, user_to_backend, True, user_data, backend_data, device="cpu"
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


@pytest.mark.parametrize("selection", ["indices", "mask"])
def test_write_2d_user_to_backend_updates_structured_buffers(selection: str) -> None:
    """Fuse selected structured writes into public and backend-order buffers."""
    input_data_np = np.asarray(
        [
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]],
            [[40.0, 41.0, 42.0], [50.0, 51.0, 52.0], [60.0, 61.0, 62.0]],
        ],
        dtype=np.float32,
    )
    input_data = wp.array(input_data_np, dtype=wp.vec3f, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(input_data)
    backend_data = wp.zeros_like(input_data)

    if selection == "indices":
        env_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
        user_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
        write_2d_user_to_backend_with_indices(
            input_data,
            env_ids,
            user_ids,
            user_to_backend,
            True,
            True,
            user_data,
            backend_data,
            dtype=wp.vec3f,
            device="cpu",
        )
    else:
        env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
        user_mask = wp.array(np.asarray([True, False, True]), dtype=wp.bool, device="cpu")
        write_2d_user_to_backend_with_mask(
            input_data,
            env_mask,
            user_mask,
            user_to_backend,
            True,
            user_data,
            backend_data,
            dtype=wp.vec3f,
            device="cpu",
        )

    expected_user = np.zeros_like(input_data_np)
    expected_user[0, [0, 2]] = input_data_np[0, [0, 2]]
    expected_backend = np.zeros_like(input_data_np)
    expected_backend[0, [1, 0]] = input_data_np[0, [0, 2]]
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


@pytest.mark.parametrize("selection", ["indices", "mask"])
def test_write_3d_user_to_backend_updates_component_buffers(selection: str) -> None:
    """Fuse selected component-array writes into public and backend-order buffers."""
    input_data_np = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4) + 1.0
    input_data = wp.array(input_data_np, dtype=wp.float32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_data = wp.zeros_like(input_data)
    backend_data = wp.zeros_like(input_data)

    if selection == "indices":
        env_ids = wp.array(np.asarray([0], dtype=np.int32), dtype=wp.int32, device="cpu")
        user_ids = wp.array(np.asarray([0, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
        write_3d_user_to_backend_with_indices(
            input_data,
            env_ids,
            user_ids,
            user_to_backend,
            True,
            True,
            user_data,
            backend_data,
            dtype=wp.float32,
            device="cpu",
        )
    else:
        env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
        user_mask = wp.array(np.asarray([True, False, True]), dtype=wp.bool, device="cpu")
        write_3d_user_to_backend_with_mask(
            input_data,
            env_mask,
            user_mask,
            user_to_backend,
            True,
            user_data,
            backend_data,
            dtype=wp.float32,
            device="cpu",
        )

    expected_user = np.zeros_like(input_data_np)
    expected_user[0, [0, 2]] = input_data_np[0, [0, 2]]
    expected_backend = np.zeros_like(input_data_np)
    expected_backend[0, [1, 0]] = input_data_np[0, [0, 2]]
    np.testing.assert_allclose(user_data.numpy(), expected_user)
    np.testing.assert_allclose(backend_data.numpy(), expected_backend)


def test_directional_reorder_names_alias_the_gather_kernels() -> None:
    """The four direction names are aliases: one kernel body per rank."""
    from isaaclab.assets.articulation import ordering_kernels as k

    assert k.reorder_2d_backend_to_user is k.gather_2d
    assert k.reorder_2d_user_to_backend is k.gather_2d
    assert k.reorder_3d_backend_to_user is k.gather_3d
    assert k.reorder_3d_user_to_backend is k.gather_3d


def test_write_2d_wrapper_adapts_torch_vec3_inputs() -> None:
    """The launch wrapper accepts torch tensors and folds trailing vec3 dims."""
    import torch

    from isaaclab.assets.articulation import ordering_kernels as k

    num_envs, num_items = 2, 3
    input_data = torch.arange(num_envs * num_items * 3, dtype=torch.float32, device="cpu").reshape(
        num_envs, num_items, 3
    )
    user_data = torch.zeros_like(input_data)
    backend_data = torch.zeros_like(input_data)
    env_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    user_ids = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    user_to_backend = wp.array([2, 0, 1], dtype=wp.int32, device="cpu")

    k.write_2d_user_to_backend_with_indices(
        input_data,
        env_ids,
        user_ids,
        user_to_backend,
        True,
        True,
        user_data,
        backend_data,
        dtype=wp.vec3f,
        device="cpu",
    )
    assert torch.equal(user_data, input_data)
    assert torch.equal(backend_data[:, [2, 0, 1], :], input_data)


def test_write_float_wrapper_accepts_scalar_input() -> None:
    """The float wrapper passes Python scalars straight to the scalar overload."""
    import torch

    from isaaclab.assets.articulation import ordering_kernels as k

    user_data = torch.zeros((2, 3), dtype=torch.float32, device="cpu")
    backend_data = torch.zeros((2, 3), dtype=torch.float32, device="cpu")
    env_ids = wp.array([0, 1], dtype=wp.int32, device="cpu")
    user_ids = wp.array([1], dtype=wp.int32, device="cpu")
    user_to_backend = wp.array([2, 0, 1], dtype=wp.int32, device="cpu")

    k.write_float_user_to_backend_with_indices(
        7.5, env_ids, user_ids, user_to_backend, True, False, user_data, backend_data, device="cpu"
    )
    assert user_data[0, 1].item() == 7.5 and user_data[1, 1].item() == 7.5
    assert backend_data[0, 0].item() == 7.5  # user index 1 -> backend index 0


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_joint_state_writer_accepts_selector_widths(env_dtype: type, joint_dtype: type) -> None:
    env_ids = wp.array([1, 0], dtype=env_dtype, device="cpu")
    user_ids = wp.array([2, 0], dtype=joint_dtype, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    position = wp.array([[110.0, 111.0], [120.0, 121.0]], dtype=wp.float32, device="cpu")
    velocity = wp.array([[10.0, 11.0], [20.0, 21.0]], dtype=wp.float32, device="cpu")
    user_position = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_velocity = wp.zeros_like(user_position)
    previous_velocity = wp.zeros_like(user_velocity)
    acceleration = wp.ones_like(user_velocity)
    backend_position = wp.zeros_like(user_position)
    backend_velocity = wp.zeros_like(user_velocity)
    kernel = write_joint_state_user_to_backend_with_indices
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = write_joint_state_user_to_backend_with_indices_kernel(env_ids, user_ids)
    wp.launch(
        kernel,
        dim=(env_ids.shape[0], user_ids.shape[0]),
        inputs=[position, velocity, env_ids, user_ids, user_to_backend, True, False],
        outputs=[user_position, user_velocity, previous_velocity, acceleration, backend_position, backend_velocity],
        device="cpu",
    )

    np.testing.assert_array_equal(user_position.numpy(), [[121.0, 0.0, 120.0], [111.0, 0.0, 110.0]])
    np.testing.assert_array_equal(backend_position.numpy(), [[120.0, 121.0, 0.0], [110.0, 111.0, 0.0]])
