# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for Newton external-wrench packing kernels."""

import numpy as np
import warp as wp
from isaaclab_newton.assets import kernels as shared_kernels
from isaaclab_newton.assets.articulation import kernels as articulation_kernels

_IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)
_QUARTER_TURN_Z_QUAT = (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5))


def _body_link_poses_w(
    poses: list[tuple[tuple[float, float, float], tuple[float, float, float, float]]],
) -> wp.array:
    """Create one-environment body-link poses from translations and quaternions."""
    poses = np.asarray([[[*translation, *quaternion] for translation, quaternion in poses]], dtype=np.float32)
    return wp.array(poses, dtype=wp.transformf, device="cpu")


def test_update_wrench_array_rotates_body_wrenches_to_world_frame() -> None:
    """Rotate body-frame force and torque into the world-frame wrench."""
    forces = wp.array(np.asarray([[[2.0, 0.0, 0.0]]], dtype=np.float32), dtype=wp.vec3f, device="cpu")
    torques = wp.array(np.asarray([[[3.0, 0.0, 0.0]]], dtype=np.float32), dtype=wp.vec3f, device="cpu")
    # A COM wrench rotation must not add a link-origin moment from this translation.
    body_link_pose_w = _body_link_poses_w([((7.0, -5.0, 11.0), _QUARTER_TURN_Z_QUAT)])
    wrench = wp.zeros((1, 1), dtype=wp.spatial_vectorf, device="cpu")
    env_mask = wp.array(np.asarray([True]), dtype=wp.bool, device="cpu")
    body_mask = wp.array(np.asarray([True]), dtype=wp.bool, device="cpu")

    wp.launch(
        shared_kernels.update_wrench_array_with_force_and_torque,
        dim=(1, 1),
        inputs=[forces, torques, body_link_pose_w, wrench, env_mask, body_mask],
        device="cpu",
    )

    np.testing.assert_allclose(
        wrench.numpy(), np.asarray([[[0.0, 2.0, 0.0, 0.0, 3.0, 0.0]]], dtype=np.float32), atol=1e-6
    )


def test_update_wrench_array_ordered_rotates_and_scatter_wrenches_to_backend_order() -> None:
    """Rotate public-order body wrenches and scatter them into backend order."""
    forces = wp.array(np.asarray([[[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], dtype=np.float32), dtype=wp.vec3f, device="cpu")
    torques = wp.array(np.asarray([[[2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]], dtype=np.float32), dtype=wp.vec3f, device="cpu")
    # Link poses are backend ordered: public body 1 is backend body 0. The
    # rotated body's nonzero translation must not add a p x f moment.
    body_link_pose_w = _body_link_poses_w(
        [((2.0, -3.0, 4.0), _QUARTER_TURN_Z_QUAT), ((-1.0, 6.0, 8.0), _IDENTITY_QUAT)]
    )
    user_to_backend = wp.array(np.asarray([1, 0], dtype=np.int32), dtype=wp.int32, device="cpu")
    wrench = wp.zeros((1, 2), dtype=wp.spatial_vectorf, device="cpu")
    env_mask = wp.array(np.asarray([True]), dtype=wp.bool, device="cpu")
    body_mask = wp.array(np.asarray([True, True]), dtype=wp.bool, device="cpu")

    wp.launch(
        articulation_kernels.update_wrench_array_with_force_and_torque_ordered,
        dim=(1, 2),
        inputs=[forces, torques, body_link_pose_w, user_to_backend, wrench, env_mask, body_mask],
        device="cpu",
    )

    np.testing.assert_allclose(
        wrench.numpy(),
        np.asarray([[[0.0, 3.0, 0.0, 0.0, 4.0, 0.0], [1.0, 0.0, 0.0, 2.0, 0.0, 0.0]]], dtype=np.float32),
        atol=1e-6,
    )
