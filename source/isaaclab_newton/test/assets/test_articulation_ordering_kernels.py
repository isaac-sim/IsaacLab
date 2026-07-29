# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import warnings

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.assets.articulation import kernels as articulation_kernels

from isaaclab.utils.warp import ProxyArray


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


def _articulation_class():
    """Import the Newton articulation class without unrelated config deprecations."""
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
        from isaaclab_newton.assets.articulation.articulation import Articulation

    return Articulation


@pytest.mark.parametrize(
    "resolver_name",
    [
        "_resolve_env_ids",
        "_resolve_joint_ids",
        "_resolve_body_ids",
        "_resolve_fixed_tendon_ids",
        "_resolve_spatial_tendon_ids",
    ],
)
def test_selector_resolvers_unwrap_proxy_array_without_copy(resolver_name: str) -> None:
    """Return the exact Warp allocation cached by a proxy selector."""
    Articulation = _articulation_class()

    selector_array = _selector([1, 0], wp.int32)
    selector = ProxyArray(selector_array)
    articulation = type(
        "ResolverStub",
        (),
        {
            "_ALL_INDICES": _selector([0, 1], wp.int32),
            "_ALL_JOINT_INDICES": _selector([0, 1], wp.int32),
            "_ALL_BODY_INDICES": _selector([0, 1], wp.int32),
            "_ALL_FIXED_TENDON_INDICES": _selector([0, 1], wp.int32),
            "_ALL_SPATIAL_TENDON_INDICES": _selector([0, 1], wp.int32),
            "device": "cpu",
        },
    )()

    resolved = getattr(Articulation, resolver_name)(articulation, selector)

    assert resolved is selector_array
    assert selector._torch_cache is None  # noqa: SLF001


def test_public_joint_velocity_kernel_rejects_int16_direct_launch() -> None:
    """The deprecated public raw symbol remains a concrete int32 compatibility kernel."""
    data = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    env_ids = _selector([0], wp.int16)
    joint_ids = _selector([0], wp.int32)
    outputs = [wp.zeros((1, 1), dtype=wp.float32, device="cpu") for _ in range(3)]
    with pytest.raises(RuntimeError):
        wp.launch(
            articulation_kernels.write_joint_vel_data_index,
            dim=(1, 1),
            inputs=[data, env_ids, joint_ids],
            outputs=outputs,
            device="cpu",
        )


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
    kernel = articulation_kernels.write_joint_limit_data_to_user_and_backend_index
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = articulation_kernels.write_joint_limit_data_to_user_and_backend_index_kernel(env_ids, user_ids)

    wp.launch(
        kernel,
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
    kernel = articulation_kernels.write_joint_vel_data_index
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = articulation_kernels.write_joint_vel_data_index_kernel(env_ids, joint_ids)

    wp.launch(
        kernel,
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

    kernel = articulation_kernels.write_joint_state_data_index
    if env_dtype != wp.int32 or joint_dtype != wp.int32:
        kernel = articulation_kernels.write_joint_state_data_index_kernel(env_ids, joint_ids)
    wp.launch(
        kernel,
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


def test_public_joint_velocity_kernel_factory_launches_int64_specialization() -> None:
    """Launch the deprecated Newton worker's validated int64 specialization."""
    data = wp.array([[3.0]], dtype=wp.float32, device="cpu")
    env_ids = _selector([0], wp.int64)
    joint_ids = _selector([0], wp.int64)
    outputs = [wp.zeros((1, 1), dtype=wp.float32, device="cpu") for _ in range(3)]
    wp.launch(
        articulation_kernels.write_joint_vel_data_index_kernel(env_ids, joint_ids),
        dim=(1, 1),
        inputs=[data, env_ids, joint_ids],
        outputs=outputs,
        device="cpu",
    )
    assert outputs[0].numpy()[0, 0] == 3.0


def test_public_joint_velocity_kernel_factory_launches_proxy_specialization() -> None:
    """Launch the deprecated Newton worker from exact proxy Warp allocations."""
    env_array = _selector([0], wp.int32)
    joint_array = _selector([0], wp.int32)
    env_ids = ProxyArray(env_array)
    joint_ids = ProxyArray(joint_array)
    factory = articulation_kernels.write_joint_vel_data_index_kernel
    assert "ProxyArray" in factory.__annotations__["env_ids"]
    data = wp.array([[3.0]], dtype=wp.float32, device="cpu")
    outputs = [wp.zeros((1, 1), dtype=wp.float32, device="cpu") for _ in range(3)]
    wp.launch(
        factory(env_ids, joint_ids),
        dim=(1, 1),
        inputs=[data, env_ids.warp, joint_ids.warp],
        outputs=outputs,
        device="cpu",
    )
    assert env_ids.warp is env_array
    assert joint_ids.warp is joint_array
    assert outputs[0].numpy()[0, 0] == 3.0


def test_public_joint_velocity_kernel_factory_launches_empty_proxy_selectors() -> None:
    """Launch the Newton factory with empty proxy selectors without touching outputs."""
    env_ids = ProxyArray(_selector([], wp.int32))
    joint_ids = ProxyArray(_selector([], wp.int32))
    data = wp.zeros((0, 0), dtype=wp.float32, device="cpu")
    outputs = [wp.full((1, 1), value=-1.0, dtype=wp.float32, device="cpu") for _ in range(3)]
    wp.launch(
        articulation_kernels.write_joint_vel_data_index_kernel(env_ids, joint_ids),
        dim=(0, 0),
        inputs=[data, env_ids.warp, joint_ids.warp],
        outputs=outputs,
        device="cpu",
    )
    for output in outputs:
        np.testing.assert_array_equal(output.numpy(), np.asarray([[-1.0]], dtype=np.float32))


def test_public_joint_velocity_kernel_factory_scatters_noncontiguous_proxy_selectors() -> None:
    """Scatter Newton values selected by noncontiguous logical proxy indices."""
    env_ids = ProxyArray(_selector([2, 0], wp.int32))
    joint_ids = ProxyArray(_selector([2, 0], wp.int32))
    data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
    outputs = [wp.full((3, 3), value=-1.0, dtype=wp.float32, device="cpu") for _ in range(3)]
    wp.launch(
        articulation_kernels.write_joint_vel_data_index_kernel(env_ids, joint_ids),
        dim=(2, 2),
        inputs=[data, env_ids.warp, joint_ids.warp],
        outputs=outputs,
        device="cpu",
    )
    expected_velocity = np.asarray([[22.0, -1.0, 21.0], [-1.0, -1.0, -1.0], [12.0, -1.0, 11.0]], dtype=np.float32)
    expected_acceleration = np.asarray([[0.0, -1.0, 0.0], [-1.0, -1.0, -1.0], [0.0, -1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(outputs[0].numpy(), expected_velocity)
    np.testing.assert_array_equal(outputs[1].numpy(), expected_velocity)
    np.testing.assert_array_equal(outputs[2].numpy(), expected_acceleration)


def test_public_joint_velocity_kernel_factory_rejects_unsupported_proxy_dtype_before_launch() -> None:
    """Reject unsupported Newton proxy selector dtypes while selecting the worker."""
    env_ids = ProxyArray(_selector([0], wp.int16))
    joint_ids = ProxyArray(_selector([0], wp.int32))
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit"):
        articulation_kernels.write_joint_vel_data_index_kernel(env_ids, joint_ids)
