# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import warp as wp
from isaaclab_newton.assets.articulation import kernels as articulation_kernels


def test_write_joint_limit_data_to_user_and_backend_index_reorders_backend_buffers() -> None:
    """Write partial user-order joint limits into user and backend-order buffers."""
    limits_np = np.asarray([[[1.0, 3.0], [2.0, 5.0]], [[-1.0, 1.0], [4.0, 8.0]]], dtype=np.float32)
    limits = wp.array(limits_np, dtype=wp.vec2f, device="cpu")
    env_ids = wp.array(np.asarray([0, 1], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_ids = wp.array(np.asarray([2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    backend_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    default_pos = wp.array(
        np.asarray([[0.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32), dtype=wp.float32, device="cpu"
    )
    clamped_defaults = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_limit_data_to_user_and_backend_index,
        dim=limits.shape,
        inputs=[limits, 1.0, env_ids, user_ids, user_to_backend, True],
        outputs=[
            user_lower,
            user_upper,
            user_limits,
            backend_lower,
            backend_upper,
            soft_limits,
            default_pos,
            clamped_defaults,
        ],
        device="cpu",
    )

    np.testing.assert_allclose(user_lower.numpy(), np.asarray([[2.0, 0.0, 1.0], [4.0, 0.0, -1.0]], dtype=np.float32))
    np.testing.assert_allclose(user_upper.numpy(), np.asarray([[5.0, 0.0, 3.0], [8.0, 0.0, 1.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_lower.numpy(), np.asarray([[1.0, 2.0, 0.0], [-1.0, 4.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_upper.numpy(), np.asarray([[3.0, 5.0, 0.0], [1.0, 8.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        user_limits.numpy(),
        np.asarray([[[2.0, 5.0], [0.0, 0.0], [1.0, 3.0]], [[4.0, 8.0], [0.0, 0.0], [-1.0, 1.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(default_pos.numpy(), np.asarray([[2.0, 0.0, 3.0], [4.0, 0.0, 0.0]], dtype=np.float32))
    assert clamped_defaults.numpy()[0] == 3


def test_write_joint_limit_data_to_user_and_backend_mask_reorders_backend_buffers() -> None:
    """Write masked user-order joint limits into user and backend-order buffers."""
    limits_np = np.asarray(
        [[[1.0, 2.0], [3.0, 6.0], [4.0, 9.0]], [[-2.0, 2.0], [5.0, 7.0], [8.0, 10.0]]], dtype=np.float32
    )
    limits = wp.array(limits_np, dtype=wp.vec2f, device="cpu")
    env_mask = wp.array(np.asarray([True, False], dtype=bool), dtype=wp.bool, device="cpu")
    user_mask = wp.array(np.asarray([False, True, True], dtype=bool), dtype=wp.bool, device="cpu")
    user_to_backend = wp.array(np.asarray([1, 2, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    user_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    user_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    backend_lower = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    backend_upper = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
    default_pos = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    clamped_defaults = wp.zeros(1, dtype=wp.int32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_limit_data_to_user_and_backend_mask,
        dim=limits.shape,
        inputs=[limits, 1.0, env_mask, user_mask, user_to_backend, True],
        outputs=[
            user_lower,
            user_upper,
            user_limits,
            backend_lower,
            backend_upper,
            soft_limits,
            default_pos,
            clamped_defaults,
        ],
        device="cpu",
    )

    np.testing.assert_allclose(user_lower.numpy(), np.asarray([[0.0, 3.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(user_upper.numpy(), np.asarray([[0.0, 6.0, 9.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_lower.numpy(), np.asarray([[4.0, 0.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(backend_upper.numpy(), np.asarray([[9.0, 0.0, 6.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    np.testing.assert_allclose(
        user_limits.numpy(),
        np.asarray([[[0.0, 0.0], [3.0, 6.0], [4.0, 9.0]], [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]], dtype=np.float32),
    )
    np.testing.assert_allclose(default_pos.numpy(), np.asarray([[0.0, 3.0, 4.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    assert clamped_defaults.numpy()[0] == 2
