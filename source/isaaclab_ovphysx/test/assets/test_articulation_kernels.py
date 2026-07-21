# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for OVPhysX articulation Warp kernels."""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_ovphysx.assets import kernels as shared_kernels
from isaaclab_ovphysx.assets.articulation import kernels as articulation_kernels


def _selector(values: list[int], dtype: type) -> wp.array:
    """Create a CPU Warp selector with the requested integer width."""
    return wp.array(values, dtype=dtype, device="cpu")


def test_public_root_pose_kernel_rejects_int16_direct_launch() -> None:
    """The public OVPhysX raw symbol remains a concrete int32 compatibility kernel."""
    data = wp.zeros(1, dtype=wp.transformf, device="cpu")
    env_ids = _selector([0], wp.int16)
    output = wp.zeros(1, dtype=wp.transformf, device="cpu")
    with pytest.raises(RuntimeError):
        wp.launch(
            shared_kernels.set_root_link_pose_to_sim_index,
            dim=1,
            inputs=[data, env_ids],
            outputs=[output],
            device="cpu",
        )


def _articulation_class():
    """Import the OVPhysX articulation class without unrelated config deprecations."""
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
        from isaaclab_ovphysx.assets.articulation.articulation import Articulation

    return Articulation


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize(
    "kernel_name",
    [
        "set_root_link_pose_to_sim_index",
        "set_root_com_pose_to_sim_index",
        "set_root_com_velocity_to_sim_index",
        "set_root_link_velocity_to_sim_index",
    ],
)
def test_root_index_workers_accept_selector_widths(kernel_name: str, env_dtype: type) -> None:
    """Launch every root-index worker with either supported environment selector width."""
    env_ids = _selector([1, 0], env_dtype)
    kernel = getattr(shared_kernels, kernel_name)
    if env_dtype != wp.int32:
        kernel = getattr(shared_kernels, f"{kernel_name}_kernel")(env_ids)
    if "pose" in kernel_name:
        data = wp.array(
            np.asarray([[11.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
            dtype=wp.transformf,
            device="cpu",
        )
        root_link_pose = wp.zeros(2, dtype=wp.transformf, device="cpu")
        if kernel_name == "set_root_link_pose_to_sim_index":
            wp.launch(
                kernel,
                dim=2,
                inputs=[data, env_ids],
                outputs=[root_link_pose],
                device="cpu",
            )
            output = root_link_pose
        else:
            body_com_pose = wp.array(
                np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_com_pose = wp.zeros(2, dtype=wp.transformf, device="cpu")
            wp.launch(
                kernel,
                dim=2,
                inputs=[data, body_com_pose, env_ids],
                outputs=[root_com_pose, root_link_pose],
                device="cpu",
            )
            output = root_com_pose
    else:
        data = wp.array(
            np.asarray([[11.0, 12.0, 13.0, 14.0, 15.0, 16.0], [21.0, 22.0, 23.0, 24.0, 25.0, 26.0]]),
            dtype=wp.spatial_vectorf,
            device="cpu",
        )
        root_com_velocity = wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu")
        body_acceleration = wp.zeros((2, 1), dtype=wp.spatial_vectorf, device="cpu")
        if kernel_name == "set_root_com_velocity_to_sim_index":
            wp.launch(
                kernel,
                dim=2,
                inputs=[data, env_ids, 1],
                outputs=[root_com_velocity, body_acceleration],
                device="cpu",
            )
            output = root_com_velocity
        else:
            body_com_pose = wp.array(
                np.asarray([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_link_pose = wp.array(
                np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]] * 2),
                dtype=wp.transformf,
                device="cpu",
            )
            root_link_velocity = wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu")
            wp.launch(
                kernel,
                dim=2,
                inputs=[data, body_com_pose, root_link_pose, env_ids, 1],
                outputs=[root_link_velocity, root_com_velocity, body_acceleration],
                device="cpu",
            )
            output = root_link_velocity

    np.testing.assert_array_equal(output.numpy(), data.numpy()[[1, 0]])


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("item_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize(
    "kernel_name",
    [
        "write_2d_data_to_buffer_with_indices",
        "write_joint_state_to_buffer_with_indices",
        "write_joint_position_limit_to_buffer_index",
        "clamp_default_joint_pos_and_update_soft_limits_index",
        "write_joint_friction_data_to_buffer_index",
    ],
)
def test_item_index_workers_accept_selector_widths(kernel_name: str, env_dtype: type, item_dtype: type) -> None:
    """Launch every item-index worker with all supported selector-width combinations."""
    env_ids = _selector([1, 0], env_dtype)
    item_ids = _selector([2, 0], item_dtype)
    if kernel_name in {
        "clamp_default_joint_pos_and_update_soft_limits_index",
        "write_joint_friction_data_to_buffer_index",
    }:
        kernel_module = articulation_kernels
    else:
        kernel_module = shared_kernels
    kernel = getattr(kernel_module, kernel_name)
    if env_dtype != wp.int32 or item_dtype != wp.int32:
        kernel = getattr(kernel_module, f"{kernel_name}_kernel")(env_ids, item_ids)
    if kernel_name == "write_2d_data_to_buffer_with_indices":
        data = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        output = wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu")
        wp.launch(
            kernel,
            dim=(2, 2),
            inputs=[data, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        expected = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
        np.testing.assert_array_equal(output.numpy(), expected)
    elif kernel_name == "write_joint_state_to_buffer_with_indices":
        position = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        velocity = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
        outputs = [wp.full((2, 3), value=-1.0, dtype=wp.float32, device="cpu") for _ in range(4)]
        wp.launch(
            kernel,
            dim=(2, 2),
            inputs=[position, velocity, env_ids, item_ids],
            outputs=outputs,
            device="cpu",
        )
        expected_position = np.asarray([[22.0, -1.0, 21.0], [12.0, -1.0, 11.0]], dtype=np.float32)
        expected_velocity = np.asarray([[122.0, -1.0, 121.0], [112.0, -1.0, 111.0]], dtype=np.float32)
        np.testing.assert_array_equal(outputs[0].numpy(), expected_position)
        np.testing.assert_array_equal(outputs[1].numpy(), expected_velocity)
        np.testing.assert_array_equal(outputs[2].numpy(), expected_velocity)
        np.testing.assert_array_equal(outputs[3].numpy(), (expected_position != -1.0).astype(np.float32) - 1.0)
    elif kernel_name == "write_joint_position_limit_to_buffer_index":
        data = wp.array(
            np.asarray([[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]]], dtype=np.float32),
            device="cpu",
        )
        output = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
        wp.launch(
            kernel,
            dim=(2, 2),
            inputs=[data, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        assert output.numpy()[0, 0].tolist() == [23.0, 24.0]
        assert output.numpy()[1, 2].tolist() == [11.0, 12.0]
    elif kernel_name == "clamp_default_joint_pos_and_update_soft_limits_index":
        limits = wp.array(
            np.asarray([[[-1.0, 1.0]] * 3, [[-2.0, 2.0]] * 3], dtype=np.float32),
            dtype=wp.vec2f,
            device="cpu",
        )
        defaults = wp.array(np.asarray([[3.0, 0.0, -3.0], [4.0, 0.0, -4.0]], dtype=np.float32), device="cpu")
        soft_limits = wp.zeros((2, 3), dtype=wp.vec2f, device="cpu")
        clamped_count = wp.zeros(1, dtype=wp.int32, device="cpu")
        wp.launch(
            kernel,
            dim=(2, 2),
            inputs=[limits, env_ids, item_ids, 0.5],
            outputs=[defaults, soft_limits, clamped_count],
            device="cpu",
        )
        assert clamped_count.numpy()[0] == 4
        assert defaults.numpy()[0, 0] == pytest.approx(1.0)
        assert defaults.numpy()[1, 2] == pytest.approx(-2.0)
    else:
        static = wp.array(np.asarray([[11.0, 12.0], [21.0, 22.0]], dtype=np.float32), device="cpu")
        dynamic = wp.array(np.asarray([[111.0, 112.0], [121.0, 122.0]], dtype=np.float32), device="cpu")
        viscous = wp.array(np.asarray([[211.0, 212.0], [221.0, 222.0]], dtype=np.float32), device="cpu")
        output = wp.full((2, 3, 3), value=-1.0, dtype=wp.float32, device="cpu")
        wp.launch(
            kernel,
            dim=(2, 2),
            inputs=[static, dynamic, viscous, env_ids, item_ids],
            outputs=[output],
            device="cpu",
        )
        assert output.numpy()[0, 0].tolist() == [22.0, 122.0, 222.0]
        assert output.numpy()[1, 2].tolist() == [11.0, 111.0, 211.0]


@pytest.mark.parametrize(("torch_dtype", "warp_dtype"), [(torch.int32, wp.int32), (torch.int64, wp.int64)])
def test_selector_resolvers_preserve_item_width_and_alias_source(torch_dtype: torch.dtype, warp_dtype: type) -> None:
    """Alias int32/int64 item selectors while keeping environment IDs int32 for OVPhysX bindings."""
    Articulation = _articulation_class()
    articulation = type("ResolverStub", (), {"_device": "cpu"})()
    selector = torch.tensor([1, 0], dtype=torch_dtype)

    assert Articulation._resolve_env_ids(articulation, selector).dtype == wp.int32
    for resolver_name in (
        "_resolve_joint_ids",
        "_resolve_body_ids",
        "_resolve_fixed_tendon_ids",
        "_resolve_spatial_tendon_ids",
    ):
        resolved = getattr(Articulation, resolver_name)(articulation, selector)
        assert resolved.dtype == warp_dtype
        assert resolved.ptr == selector.data_ptr()


@pytest.mark.parametrize("env_dtype", [wp.int32, wp.int64])
@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
def test_external_env_resolvers_return_int32_with_preserved_values(resolver_name: str, env_dtype: type) -> None:
    """Normalize every supported Warp environment selector before an OVPhysX boundary."""
    Articulation = _articulation_class()
    all_indices = _selector([0, 1], wp.int32)
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=wp.clone(all_indices, device="cpu"),
        _device="cpu",
    )
    selector = _selector([1, 0], env_dtype)

    resolved = getattr(Articulation, resolver_name)(articulation, selector)

    assert resolved.dtype == wp.int32
    np.testing.assert_array_equal(resolved.numpy(), [1, 0])
    if resolver_name == "_resolve_env_ids" and env_dtype == wp.int32:
        assert resolved.ptr == selector.ptr


@pytest.mark.parametrize("torch_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
def test_external_env_resolvers_normalize_torch_widths(resolver_name: str, torch_dtype: torch.dtype) -> None:
    """Normalize every supported Torch environment selector before an OVPhysX boundary."""
    Articulation = _articulation_class()
    all_indices = _selector([0, 1], wp.int32)
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=wp.clone(all_indices, device="cpu"),
        _device="cpu",
    )
    selector = torch.tensor([1, 0], dtype=torch_dtype)

    resolved = getattr(Articulation, resolver_name)(articulation, selector)

    assert resolved.dtype == wp.int32
    np.testing.assert_array_equal(resolved.numpy(), [1, 0])


@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
@pytest.mark.parametrize(
    "selector",
    [
        pytest.param(lambda: _selector([0], wp.int16), id="warp_int16"),
        pytest.param(lambda: torch.tensor([0], dtype=torch.int16), id="torch_int16"),
    ],
)
def test_external_env_resolvers_reject_unsupported_dtypes_before_boundary(resolver_name: str, selector) -> None:
    """Reject unsupported environment selector widths before calling an OVPhysX binding."""
    Articulation = _articulation_class()
    all_indices = _selector([0], wp.int32)
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=wp.clone(all_indices, device="cpu"),
        _device="cpu",
    )

    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit integers"):
        getattr(Articulation, resolver_name)(articulation, selector())


@pytest.mark.parametrize("resolver_name", ["_resolve_env_ids", "_get_cpu_env_ids"])
def test_external_env_resolvers_reject_numpy_inputs_before_boundary(resolver_name: str) -> None:
    """Reject unsupported NumPy environment selectors before calling an OVPhysX binding."""
    Articulation = _articulation_class()
    all_indices = _selector([0, 1], wp.int32)
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=wp.clone(all_indices, device="cpu"),
        _device="cpu",
    )
    selector = np.asarray([0, 1], dtype=np.int64)

    with pytest.raises(TypeError, match="Torch tensor or Warp array"):
        getattr(Articulation, resolver_name)(articulation, selector)


def test_cpu_env_ids_all_returns_pinned_fast_path() -> None:
    """Return the pre-allocated pinned CPU selector for OVPhysX all-environment writes."""
    Articulation = _articulation_class()
    all_indices = _selector([0, 1], wp.int32)
    cpu_env_ids_all = wp.clone(all_indices, device="cpu")
    articulation = SimpleNamespace(
        _ALL_INDICES=all_indices,
        _cpu_env_ids_all=cpu_env_ids_all,
        _device="cpu",
    )

    resolved = Articulation._get_cpu_env_ids(articulation, all_indices)

    assert resolved is cpu_env_ids_all
    assert resolved.ptr == cpu_env_ids_all.ptr
    np.testing.assert_array_equal(resolved.numpy(), [0, 1])


@pytest.mark.parametrize("state_write", [False, True], ids=["velocity", "state"])
def test_ordered_joint_writes_reject_unsupported_selector_before_launch(monkeypatch, state_write: bool) -> None:
    """Reject an unsupported item selector through the shared factory before calling Warp launch."""
    Articulation = _articulation_class()
    articulation = Articulation.__new__(Articulation)
    articulation._initialize_handle = None
    articulation._invalidate_initialize_handle = None
    articulation._prim_deletion_handle = None
    articulation._device = "cpu"
    articulation.assert_shape_and_dtype = lambda *args, **kwargs: None
    articulation._joint_user_to_backend_map = lambda: wp.zeros(1, dtype=wp.int32, device="cpu")

    def _buffer() -> SimpleNamespace:
        return SimpleNamespace(data=wp.zeros((1, 1), dtype=wp.float32, device="cpu"))

    joint_pos_backend = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    joint_vel_backend = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    articulation._data = SimpleNamespace(
        joint_ordering=None,
        _joint_pos_buf=_buffer(),
        _joint_vel_buf=_buffer(),
        _previous_joint_vel=wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        _joint_acc=_buffer(),
        _get_joint_pos_write_buffer=lambda partial: joint_pos_backend,
        _get_joint_vel_write_buffer=lambda partial: joint_vel_backend,
    )
    monkeypatch.setattr(wp, "launch", lambda *args, **kwargs: pytest.fail("wp.launch must not be called"))

    env_ids = wp.array([0], dtype=wp.int32, device="cpu")
    joint_ids = wp.array([0], dtype=wp.float32, device="cpu")
    position = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    velocity = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    with pytest.raises(TypeError, match="signed 32-bit or signed 64-bit integers"):
        if state_write:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                articulation.write_joint_state_to_sim(position, velocity, joint_ids=joint_ids, env_ids=env_ids)
        else:
            articulation.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids, env_ids=env_ids)


def test_public_root_pose_kernel_factory_launches_int64_specialization() -> None:
    """Launch the OVPhysX root-pose worker's validated int64 specialization."""
    data = wp.array([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transformf, device="cpu")
    env_ids = _selector([0], wp.int64)
    output = wp.zeros(1, dtype=wp.transformf, device="cpu")
    wp.launch(
        shared_kernels.set_root_link_pose_to_sim_index_kernel(env_ids),
        dim=1,
        inputs=[data, env_ids],
        outputs=[output],
        device="cpu",
    )
    np.testing.assert_array_equal(output.numpy(), data.numpy())
