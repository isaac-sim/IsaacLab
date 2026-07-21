# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for PhysX articulation Warp kernels."""

import numpy as np
import pytest
import warp as wp
from isaaclab_physx.assets.articulation.kernels import write_joint_state_data, write_joint_vel_data


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_write_joint_vel_data_scatters_nonidentity_selectors(env_dtype: type, joint_dtype: type) -> None:
    """Scatter compact joint velocities for every supported selector-width combination."""
    in_data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_acc = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")

    wp.launch(
        write_joint_vel_data,
        dim=(2, 2),
        inputs=[in_data, env_ids, joint_ids, False],
        outputs=[joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_velocity = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
    expected_acceleration = np.asarray([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(prev_joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(joint_acc.numpy(), expected_acceleration)


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_write_joint_state_data_scatters_nonidentity_selectors(env_dtype: type, joint_dtype: type) -> None:
    """Scatter compact joint state for every supported selector-width combination."""
    pos_data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
    vel_data = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_pos = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
    joint_acc = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")

    wp.launch(
        write_joint_state_data,
        dim=(2, 2),
        inputs=[pos_data, vel_data, env_ids, joint_ids, False],
        outputs=[joint_pos, joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_position = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
    expected_velocity = np.asarray([[122.0, -1.0, 121.0], [112.0, -1.0, 111.0]], dtype=np.float32)
    expected_acceleration = np.asarray([[0.0, -1.0, 0.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(joint_pos.numpy(), expected_position)
    np.testing.assert_array_equal(joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(prev_joint_vel.numpy(), expected_velocity)
    np.testing.assert_array_equal(joint_acc.numpy(), expected_acceleration)
