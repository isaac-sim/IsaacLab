# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation import kernels as articulation_kernels


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_write_joint_limit_data_to_user_and_backend_index_accepts_index_dtypes(
    env_dtype: type, joint_dtype: type
) -> None:
    """Write partial user-order joint limits into user and backend-order buffers."""
    limits_np = np.asarray([[[1.0, 3.0], [2.0, 5.0]], [[-1.0, 1.0], [4.0, 8.0]]], dtype=np.float32)
    limits = wp.array(limits_np, dtype=wp.vec2f, device="cpu")
    env_ids = _selector([0, 1], env_dtype)
    user_ids = _selector([2, 0], joint_dtype)
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


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_deprecated_write_joint_vel_data_index_accepts_index_dtypes(env_dtype: type, joint_dtype: type) -> None:
    """Deprecated kernel still scatters selected joint velocities and resets acceleration."""
    in_data = wp.array(np.asarray([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_acc = wp.array(np.ones((2, 3), dtype=np.float32), dtype=wp.float32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_vel_data_index,
        dim=in_data.shape,
        inputs=[in_data, env_ids, joint_ids],
        outputs=[joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_vel = np.asarray([[21.0, 0.0, 20.0], [11.0, 0.0, 10.0]], dtype=np.float32)
    np.testing.assert_allclose(joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(prev_joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(joint_acc.numpy(), np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))


def test_deprecated_write_joint_vel_data_mask_writes_only_masked_cells() -> None:
    """Deprecated masked kernel only touches cells selected by both masks."""
    in_data = wp.array(
        np.asarray([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]], dtype=np.float32), dtype=wp.float32, device="cpu"
    )
    env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
    joint_mask = wp.array(np.asarray([True, False, True]), dtype=wp.bool, device="cpu")
    joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_acc = wp.array(np.ones((2, 3), dtype=np.float32), dtype=wp.float32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_vel_data_mask,
        dim=in_data.shape,
        inputs=[in_data, env_mask, joint_mask],
        outputs=[joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    expected_vel = np.asarray([[10.0, 0.0, 12.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(prev_joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(joint_acc.numpy(), np.asarray([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("joint_dtype", [wp.int32, wp.int64])
def test_deprecated_write_joint_state_data_index_accepts_index_dtypes(env_dtype: type, joint_dtype: type) -> None:
    """Deprecated kernel scatters selected joint positions/velocities and resets acceleration."""
    pos_data = wp.array(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    vel_data = wp.array(np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
    env_ids = _selector([1, 0], env_dtype)
    joint_ids = _selector([2, 0], joint_dtype)
    joint_pos = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_acc = wp.array(np.ones((2, 3), dtype=np.float32), dtype=wp.float32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_state_data_index,
        dim=pos_data.shape,
        inputs=[pos_data, vel_data, env_ids, joint_ids],
        outputs=[joint_pos, joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    np.testing.assert_allclose(joint_pos.numpy(), np.asarray([[4.0, 0.0, 3.0], [2.0, 0.0, 1.0]], dtype=np.float32))
    expected_vel = np.asarray([[8.0, 0.0, 7.0], [6.0, 0.0, 5.0]], dtype=np.float32)
    np.testing.assert_allclose(joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(prev_joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(joint_acc.numpy(), np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32))


def test_deprecated_write_joint_state_data_mask_writes_only_masked_cells() -> None:
    """Deprecated masked kernel only writes joint state where both masks select the cell."""
    pos_data = wp.array(
        np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32), dtype=wp.float32, device="cpu"
    )
    vel_data = wp.array(
        np.asarray([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=np.float32), dtype=wp.float32, device="cpu"
    )
    env_mask = wp.array(np.asarray([True, False]), dtype=wp.bool, device="cpu")
    joint_mask = wp.array(np.asarray([False, True, True]), dtype=wp.bool, device="cpu")
    joint_pos = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    prev_joint_vel = wp.zeros((2, 3), dtype=wp.float32, device="cpu")
    joint_acc = wp.array(np.ones((2, 3), dtype=np.float32), dtype=wp.float32, device="cpu")

    wp.launch(
        articulation_kernels.write_joint_state_data_mask,
        dim=pos_data.shape,
        inputs=[pos_data, vel_data, env_mask, joint_mask],
        outputs=[joint_pos, joint_vel, prev_joint_vel, joint_acc],
        device="cpu",
    )

    np.testing.assert_allclose(joint_pos.numpy(), np.asarray([[0.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32))
    expected_vel = np.asarray([[0.0, 8.0, 9.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(prev_joint_vel.numpy(), expected_vel)
    np.testing.assert_allclose(joint_acc.numpy(), np.asarray([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32))


@pytest.mark.parametrize("index_dtype", [wp.int32, wp.int64])
def test_scatter_reset_masks_from_ids_accepts_index_dtype(index_dtype: type) -> None:
    """Set exact world and articulation reset masks from nonidentity environment IDs."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "^'RigidBodyMaterialCfg' is deprecated and will be removed in 5\\.0\\. Use "
                "'isaaclab_physx\\.sim\\.spawners\\.materials\\.PhysxRigidBodyMaterialCfg' for PhysX properties, "
                "or 'isaaclab\\.sim\\.spawners\\.materials\\.RigidBodyMaterialBaseCfg' for solver-common properties "
                "only\\.$"
            ),
            category=DeprecationWarning,
        )
        from isaaclab_newton.physics import newton_manager

    env_ids = _selector([2, 0], index_dtype)
    articulation_ids = wp.array(np.asarray([[0, 1], [2, 3], [4, 5]], dtype=np.int32), dtype=int, device="cpu")
    world_mask = wp.zeros(3, dtype=wp.bool, device="cpu")
    fk_mask = wp.zeros(6, dtype=wp.bool, device="cpu")

    wp.launch(
        newton_manager._scatter_reset_masks_from_ids,
        dim=(env_ids.shape[0], articulation_ids.shape[1]),
        inputs=[env_ids, articulation_ids],
        outputs=[world_mask, fk_mask],
        device="cpu",
    )

    np.testing.assert_array_equal(world_mask.numpy(), np.asarray([True, False, True]))
    np.testing.assert_array_equal(fk_mask.numpy(), np.asarray([True, True, False, False, True, True]))
