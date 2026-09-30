# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""
Checks that the rigid object collection interfaces are consistent across backends, and are providing
the exact same data as what the base rigid object collection class advertises. All rigid object
collection interfaces need to comply with the same interface contract.

The setup is a bit convoluted so that we can run these tests without requiring Isaac Sim or GPU simulation.
"""

import math
from unittest.mock import patch

import numpy as np
import pytest
import torch
import warp as wp
from _rigid_object_collection_iface_test_utils import BACKENDS, get_rigid_object_collection

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.integration

# Distinct instance and body counts make swapped axes visible.
_NUM_INSTANCES, _NUM_BODIES = 2, 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_proxy_array(arr, *, expected_shape: tuple, expected_dtype: type, name: str):
    """Assert that `arr` is a ProxyArray with the expected shape and dtype."""
    from isaaclab.utils.warp import ProxyArray

    assert isinstance(arr, ProxyArray), f"{name}: expected ProxyArray, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


# Common parametrize decorators. Pure bookkeeping (counts, names, finders, aliases) runs on CPU only;
# getters and writers keep every test device because PhysX stages through CPU-pinned buffers on CUDA.
_backends = pytest.mark.parametrize("backend", BACKENDS, indirect=False)
_devices = pytest.mark.parametrize("device", test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
_index_resolution_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)
_reshape_3d_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton", "ovphysx") if backend in BACKENDS], indirect=False
)
_production_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton", "ovphysx") if backend in BACKENDS], indirect=False
)


# ---------------------------------------------------------------------------
# Writer/setter test helpers
# ---------------------------------------------------------------------------

_WP_DTYPE_TO_TRAILING = {
    wp.transformf: 7,
    wp.spatial_vectorf: 6,
    wp.vec2f: 2,
    wp.float32: 0,
}


def _make_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create valid torch test data for a given warp dtype."""
    trailing = _WP_DTYPE_TO_TRAILING[wp_dtype]
    if trailing:
        full_shape = (*shape, trailing)
    else:
        full_shape = shape
    data = torch.zeros(full_shape, device=device, dtype=torch.float32)
    if wp_dtype == wp.transformf:
        data[..., 6] = 1.0
    elif wp_dtype == wp.float32:
        data.fill_(1.0)
    return data


def _make_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create valid warp test data for a given warp dtype."""
    t = _make_data_torch(shape, device, wp_dtype)
    if wp_dtype == wp.float32:
        return wp.from_torch(t, dtype=wp.float32)
    return wp.from_torch(t.contiguous(), dtype=wp_dtype)


def _make_payload_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create valid torch data whose entries differ per element, for writer read-back checks.

    Transforms get distinct positions and a fixed 90-degree rotation about Z.
    """
    values = torch.arange(1, math.prod(shape) + 1, dtype=torch.float32, device=device).reshape(shape) + 0.25
    if wp_dtype == wp.spatial_vectorf:
        return values.unsqueeze(-1) + torch.arange(6, dtype=torch.float32, device=device) / 10.0
    if wp_dtype == wp.transformf:
        data = torch.zeros((*shape, 7), dtype=torch.float32, device=device)
        data[..., :3] = values.unsqueeze(-1) + torch.arange(3, dtype=torch.float32, device=device) / 10.0
        data[..., 5] = data[..., 6] = 2.0**-0.5
        return data
    raise ValueError(f"Unsupported payload dtype: {wp_dtype}")


def _make_payload_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create the per-element distinct payload of :func:`_make_payload_torch` as a warp array."""
    return wp.from_torch(_make_payload_torch(shape, device, wp_dtype).contiguous(), dtype=wp_dtype)


def _assert_reads_back(value, expected: torch.Tensor, name: str) -> None:
    """Assert a data getter (ProxyArray) returns the values a writer just wrote."""
    torch.testing.assert_close(value.torch, expected, atol=1e-5, rtol=1e-5, msg=lambda msg: f"{name}: {msg}")


def _make_com_data(backend: str, shape: tuple[int, ...], device: str) -> wp.array:
    """Create backend-compatible center-of-mass test data."""
    if backend == "newton":
        return wp.zeros(shape, dtype=wp.vec3f, device=device)
    return _make_data_warp(shape, device, wp.transformf)


def _prime_timestamped_properties(data, property_buffer_pairs: list[tuple[str, str]]):
    """Prime public lazy properties and return their concrete timestamped buffers."""
    buffers = []
    for property_name, buffer_name in property_buffer_pairs:
        getattr(data, property_name)
        buffer = getattr(data, buffer_name)
        assert buffer is not None, buffer_name
        buffer.timestamp = data._sim_timestamp
        buffers.append((buffer_name, buffer))
    return buffers


def _assert_buffers_stale(data, buffers) -> None:
    for name, buffer in buffers:
        assert buffer.timestamp < data._sim_timestamp, name


def _make_bad_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create torch data with wrong leading shape for negative testing."""
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_torch(bad_shape, device, wp_dtype)


def _make_bad_data_warp(shape: tuple, device: str, wp_dtype=wp.float32) -> wp.array:
    """Create warp data with wrong leading shape for negative testing."""
    bad_shape = (shape[0] + 1,) + shape[1:]
    return _make_data_warp(bad_shape, device, wp_dtype)


def _make_env_mask(num_instances: int, device: str, partial: bool) -> wp.array | None:
    """Create an env_mask: None for all envs, or a partial bool mask."""
    if not partial:
        return None
    mask_np = np.zeros(num_instances, dtype=bool)
    mask_np[0] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


def _make_env_ids(device: str, subset: bool) -> torch.Tensor | None:
    """Create env_ids: None for all envs, or [0] for a subset."""
    if not subset:
        return None
    return torch.tensor([0], dtype=torch.int32, device=device)


def _make_body_ids(device: str, subset_ids: list[int] | None) -> torch.Tensor | None:
    """Create body_ids: None for all bodies, or a list for a subset."""
    if subset_ids is None:
        return None
    return torch.tensor(subset_ids, dtype=torch.int32, device=device)


def _make_item_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Create a bool warp mask with True at `selected` indices, False elsewhere."""
    mask_np = np.zeros(total, dtype=bool)
    for i in selected:
        mask_np[i] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


class TestCollectionIndexResolution:
    """Test backend-specific index resolution helpers."""

    @_production_backends
    @_devices
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend, device):
        obj, _ = get_rigid_object_collection(backend, num_instances=4, device=device)

        env_ids = torch.arange(4, dtype=torch.int32, device=device)
        resolved_full = obj._resolve_env_ids(env_ids)
        resolved_view = obj._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2
        cached = wp.to_torch(obj._ALL_ENV_INDICES)
        for selection in (slice(None), slice(1, None, 2), slice(0, 0)):
            resolved = wp.to_torch(obj._resolve_env_ids(selection))
            torch.testing.assert_close(resolved, cached[selection])
            assert resolved.data_ptr() == cached[selection].data_ptr()
            assert resolved.stride() == cached[selection].stride()

    @_index_resolution_backends
    def test_resolve_body_ids_handles_tensor_view_shape(self, backend):
        obj, _ = get_rigid_object_collection(backend, num_bodies=4, device="cpu")

        body_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = obj._resolve_body_ids(body_ids)
        resolved_view = obj._resolve_body_ids(body_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: View reshape helpers
# ---------------------------------------------------------------------------


class TestCollectionViewReshape:
    """Test backend-specific view reshape helpers."""

    @_reshape_3d_backends
    @_devices
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_reshape_data_to_view_3d_accepts_torch_tensor(self, backend, device, dtype):
        num_instances = 2
        num_bodies = 3
        data_dim = 4
        obj, _ = get_rigid_object_collection(backend, num_instances=num_instances, num_bodies=num_bodies, device=device)
        data = torch.arange(num_instances * num_bodies * data_dim, dtype=dtype, device=device).reshape(
            num_instances, num_bodies, data_dim
        )

        view = obj.reshape_data_to_view_3d(data, data_dim, device=device)

        assert isinstance(view, torch.Tensor)
        assert view.dtype == dtype
        assert view.device == data.device
        assert view.is_contiguous()
        torch.testing.assert_close(view, data.permute(1, 0, 2).reshape(num_bodies * num_instances, data_dim))

        if dtype == torch.float32:
            # Warp inputs keep returning a warp array with the same body-major layout.
            warp_view = obj.reshape_data_to_view_3d(wp.from_torch(data, dtype=wp.float32), data_dim, device=device)
            assert isinstance(warp_view, wp.array)
            assert warp_view.shape == (num_bodies * num_instances, data_dim)
            assert warp_view.dtype == wp.float32
            assert str(warp_view.device) == device
            torch.testing.assert_close(wp.to_torch(warp_view), view)

    @_reshape_3d_backends
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_reshape_data_to_view_3d_moves_torch_tensor_to_requested_device(self, backend):
        num_instances = 2
        num_bodies = 3
        data_dim = 4
        obj, _ = get_rigid_object_collection(
            backend, num_instances=num_instances, num_bodies=num_bodies, device="cuda:0"
        )
        data = torch.arange(num_instances * num_bodies * data_dim, dtype=torch.float32, device="cuda:0").reshape(
            num_instances, num_bodies, data_dim
        )

        view = obj.reshape_data_to_view_3d(data, data_dim, device="cpu")

        assert view.device.type == "cpu"
        torch.testing.assert_close(view, data.permute(1, 0, 2).reshape(num_bodies * num_instances, data_dim).cpu())


# ---------------------------------------------------------------------------
# Tests: Collection properties and finders
# ---------------------------------------------------------------------------


class TestCollectionProperties:
    """Test that collection properties return the correct types/values."""

    @_backends
    def test_collection_counts_and_names(self, backend):
        from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import (
            BaseRigidObjectCollectionData,
        )

        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, "cpu")

        assert obj.num_instances == _NUM_INSTANCES
        assert obj.num_bodies == _NUM_BODIES
        names = obj.body_names
        assert isinstance(names, list)
        assert len(names) == _NUM_BODIES
        assert all(isinstance(n, str) for n in names)
        assert isinstance(obj.data, BaseRigidObjectCollectionData)


class TestCollectionFinderReturnModes:
    """Test finder return modes on production collection backends."""

    @_production_backends
    def test_find_bodies_returns_legacy_tensor_or_cached_proxy(self, backend):
        collection, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")

        indices, names = collection.find_bodies(".*")
        proxy, proxy_names = collection.find_bodies(".*", as_proxy=True)

        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.int32
        assert isinstance(names, list)
        assert len(names) == 3
        assert indices.tolist() == proxy.torch.tolist()
        assert names == proxy_names
        assert proxy is collection.find_bodies(".*", as_proxy=True)[0]
        assert proxy.dtype == wp.int32
        assert str(proxy.device) == collection.device

        first_body = collection.body_names[0]
        _, first_names = collection.find_bodies(first_body)
        assert first_names == [first_body]

    @_production_backends
    def test_find_objects_forwards_return_mode_with_alias_warning(self, backend):
        collection, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")

        with pytest.warns(DeprecationWarning):
            proxy, _ = collection.find_objects(".*", as_proxy=True)

        assert proxy is collection.find_bodies(".*", as_proxy=True)[0]


# ---------------------------------------------------------------------------
# Tests: Collection data property contract
# ---------------------------------------------------------------------------

# (property, shape kind, dtype). Shape kinds: "NB" = (num_instances, num_bodies),
# "NB9" = (num_instances, num_bodies, 9).
_COLLECTION_DATA_PROPERTIES = [
    # body state
    ("body_link_pose_w", "NB", wp.transformf),
    ("body_link_vel_w", "NB", wp.spatial_vectorf),
    ("body_com_pose_w", "NB", wp.transformf),
    ("body_com_vel_w", "NB", wp.spatial_vectorf),
    ("body_com_acc_w", "NB", wp.spatial_vectorf),
    ("body_com_pose_b", "NB", wp.transformf),
    # sliced
    ("body_link_pos_w", "NB", wp.vec3f),
    ("body_link_quat_w", "NB", wp.quatf),
    ("body_link_lin_vel_w", "NB", wp.vec3f),
    ("body_link_ang_vel_w", "NB", wp.vec3f),
    ("body_com_pos_w", "NB", wp.vec3f),
    ("body_com_quat_w", "NB", wp.quatf),
    ("body_com_lin_vel_w", "NB", wp.vec3f),
    ("body_com_ang_vel_w", "NB", wp.vec3f),
    ("body_com_lin_acc_w", "NB", wp.vec3f),
    ("body_com_ang_acc_w", "NB", wp.vec3f),
    ("body_com_pos_b", "NB", wp.vec3f),
    ("body_com_quat_b", "NB", wp.quatf),
    # derived
    ("projected_gravity_b", "NB", wp.vec3f),
    ("heading_w", "NB", wp.float32),
    ("body_link_lin_vel_b", "NB", wp.vec3f),
    ("body_link_ang_vel_b", "NB", wp.vec3f),
    ("body_com_lin_vel_b", "NB", wp.vec3f),
    ("body_com_ang_vel_b", "NB", wp.vec3f),
    # mass properties
    ("body_mass", "NB", wp.float32),
    ("body_inertia", "NB9", wp.float32),
    # defaults
    ("default_body_pose", "NB", wp.transformf),
    ("default_body_vel", "NB", wp.spatial_vectorf),
]


class TestCollectionDataProperties:
    """Test that every data property is a ProxyArray with the advertised shape and dtype."""

    @_backends
    @_devices
    def test_collection_data_property_contract(self, backend, device):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        obj.data.update(dt=0.01)
        shapes = {"NB": (_NUM_INSTANCES, _NUM_BODIES), "NB9": (_NUM_INSTANCES, _NUM_BODIES, 9)}
        for name, shape_kind, dtype in _COLLECTION_DATA_PROPERTIES:
            _check_proxy_array(
                getattr(obj.data, name), expected_shape=shapes[shape_kind], expected_dtype=dtype, name=name
            )


# ---------------------------------------------------------------------------
# Tests: Body pose/velocity writers
# ---------------------------------------------------------------------------

# writer suffix -> data getter that reads the written quantity back
_BODY_POSE_METHODS = {
    "body_pose": "body_link_pose_w",
    "body_link_pose": "body_link_pose_w",
    "body_com_pose": "body_com_pose_w",
}
_BODY_VEL_METHODS = {
    "body_velocity": "body_com_vel_w",
    "body_com_velocity": "body_com_vel_w",
    "body_link_velocity": "body_link_vel_w",
}


class TestCollectionCacheInvalidation:
    @_production_backends
    def test_pose_write_invalidates_pose_dependent_caches(self, backend):
        obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
        obj.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            obj.data,
            [
                ("body_link_vel_w", "_body_link_vel_w"),
                ("projected_gravity_b", "_projected_gravity_b"),
                ("heading_w", "_heading_w"),
                ("body_link_lin_vel_b", "_body_link_lin_vel_b"),
                ("body_link_ang_vel_b", "_body_link_ang_vel_b"),
                ("body_com_lin_vel_b", "_body_com_lin_vel_b"),
                ("body_com_ang_vel_b", "_body_com_ang_vel_b"),
            ],
        )
        body_pose = _make_data_warp((obj.num_instances, obj.num_bodies), "cpu", wp.transformf)
        obj.write_body_link_pose_to_sim_index(body_poses=body_pose)
        _assert_buffers_stale(obj.data, buffers)

    @_production_backends
    def test_velocity_write_invalidates_body_frame_caches(self, backend):
        obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
        obj.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            obj.data,
            [
                ("body_link_lin_vel_b", "_body_link_lin_vel_b"),
                ("body_link_ang_vel_b", "_body_link_ang_vel_b"),
                ("body_com_lin_vel_b", "_body_com_lin_vel_b"),
                ("body_com_ang_vel_b", "_body_com_ang_vel_b"),
            ],
        )
        body_velocity = _make_data_warp((obj.num_instances, obj.num_bodies), "cpu", wp.spatial_vectorf)
        obj.write_body_com_velocity_to_sim_index(body_velocities=body_velocity)
        _assert_buffers_stale(obj.data, buffers)

    @_production_backends
    @pytest.mark.parametrize("setter_kind", ["index", "mask"])
    def test_set_coms_invalidates_same_timestamp_dependents(self, backend, setter_kind):
        obj, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")
        obj.data.update(dt=0.01)
        common_pairs = [
            ("body_com_pose_w", "_body_com_pose_w"),
            ("body_link_vel_w", "_body_link_vel_w"),
            ("body_link_lin_vel_b", "_body_link_lin_vel_b"),
            ("body_link_ang_vel_b", "_body_link_ang_vel_b"),
            ("body_com_lin_vel_b", "_body_com_lin_vel_b"),
            ("body_com_ang_vel_b", "_body_com_ang_vel_b"),
            ("body_state_w", "_body_state_w"),
            ("body_link_state_w", "_body_link_state_w"),
            ("body_com_state_w", "_body_com_state_w"),
        ]
        if backend != "newton":
            common_pairs.append(("body_com_vel_w", "_body_com_vel_w"))
        # Prime public properties before resolving private buffers so Newton allocates lazy caches.
        buffers = _prime_timestamped_properties(obj.data, common_pairs)
        if backend == "newton":
            buffers += _prime_timestamped_properties(obj.data, [("body_com_pose_b", "_body_com_pose_b")])
        coms = _make_com_data(backend, (obj.num_instances, obj.num_bodies), "cpu")

        def set_coms() -> None:
            if setter_kind == "index":
                obj.set_coms_index(coms=coms)
            else:
                obj.set_coms_mask(coms=coms)

        if backend == "newton":
            from isaaclab_newton.physics import NewtonManager

            with patch.object(NewtonManager, "add_model_change"):
                set_coms()
        else:
            set_coms()
        _assert_buffers_stale(obj.data, buffers)


class TestCollectionWritersPose:
    """Test body pose/velocity writers with all input combinations."""

    # -- index variants for pose --

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _BODY_POSE_METHODS)
    def test_write_body_pose_to_sim_index(self, backend, device, method_suffix):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs + all bodies
        method(body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf))
        # torch, subset envs
        method(body_poses=_make_data_torch((1, num_bodies), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # torch, subset bodies
        method(
            body_poses=_make_data_torch((num_instances, 1), device, wp.transformf), body_ids=_make_body_ids(device, [0])
        )
        # torch, subset both
        method(
            body_poses=_make_data_torch((1, 1), device, wp.transformf),
            env_ids=_make_env_ids(device, True),
            body_ids=_make_body_ids(device, [0]),
        )
        # warp, all envs + all bodies: the matching getter reads the written poses back
        method(body_poses=_make_payload_warp((num_instances, num_bodies), device, wp.transformf))
        getter = _BODY_POSE_METHODS[method_suffix]
        expected = _make_payload_torch((num_instances, num_bodies), device, wp.transformf)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, subset
        method(
            body_poses=_make_data_warp((1, 1), device, wp.transformf),
            env_ids=_make_env_ids(device, True),
            body_ids=_make_body_ids(device, [0]),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_poses=_make_bad_data_torch((num_instances, num_bodies), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_poses=_make_bad_data_warp((num_instances, num_bodies), device, wp.transformf))

    # -- index variants for velocity --

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _BODY_VEL_METHODS)
    def test_write_body_velocity_to_sim_index(self, backend, device, method_suffix):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs + all bodies
        method(body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf))
        # torch, subset envs
        method(
            body_velocities=_make_data_torch((1, num_bodies), device, wp.spatial_vectorf),
            env_ids=_make_env_ids(device, True),
        )
        # torch, subset bodies
        method(
            body_velocities=_make_data_torch((num_instances, 1), device, wp.spatial_vectorf),
            body_ids=_make_body_ids(device, [0]),
        )
        # torch, subset both
        method(
            body_velocities=_make_data_torch((1, 1), device, wp.spatial_vectorf),
            env_ids=_make_env_ids(device, True),
            body_ids=_make_body_ids(device, [0]),
        )
        # warp, all envs + all bodies: the matching getter reads the written velocities back
        method(body_velocities=_make_payload_warp((num_instances, num_bodies), device, wp.spatial_vectorf))
        getter = _BODY_VEL_METHODS[method_suffix]
        expected = _make_payload_torch((num_instances, num_bodies), device, wp.spatial_vectorf)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, subset
        method(
            body_velocities=_make_data_warp((1, 1), device, wp.spatial_vectorf),
            env_ids=_make_env_ids(device, True),
            body_ids=_make_body_ids(device, [0]),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_velocities=_make_bad_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_velocities=_make_bad_data_warp((num_instances, num_bodies), device, wp.spatial_vectorf))

    # -- mask variants for pose --

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _BODY_POSE_METHODS)
    def test_write_body_pose_to_sim_mask(self, backend, device, method_suffix):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask (all)
        method(body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf))
        # torch, partial env_mask
        method(
            body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # torch, partial body_mask
        method(
            body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf),
            body_mask=_make_item_mask(num_bodies, [0], device),
        )
        # torch, both masks
        method(
            body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
            body_mask=_make_item_mask(num_bodies, [0], device),
        )
        # warp, no mask: the matching getter reads the written poses back
        method(body_poses=_make_payload_warp((num_instances, num_bodies), device, wp.transformf))
        getter = _BODY_POSE_METHODS[method_suffix]
        expected = _make_payload_torch((num_instances, num_bodies), device, wp.transformf)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, partial env_mask
        method(
            body_poses=_make_data_warp((num_instances, num_bodies), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_poses=_make_bad_data_torch((num_instances, num_bodies), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_poses=_make_bad_data_warp((num_instances, num_bodies), device, wp.transformf))

    # -- mask variants for velocity --

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _BODY_VEL_METHODS)
    def test_write_body_velocity_to_sim_mask(self, backend, device, method_suffix):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask
        method(body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf))
        # torch, partial env_mask
        method(
            body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # torch, partial body_mask
        method(
            body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf),
            body_mask=_make_item_mask(num_bodies, [0], device),
        )
        # warp, no mask: the matching getter reads the written velocities back
        method(body_velocities=_make_payload_warp((num_instances, num_bodies), device, wp.spatial_vectorf))
        getter = _BODY_VEL_METHODS[method_suffix]
        expected = _make_payload_torch((num_instances, num_bodies), device, wp.spatial_vectorf)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, partial env_mask
        method(
            body_velocities=_make_data_warp((num_instances, num_bodies), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_velocities=_make_bad_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(body_velocities=_make_bad_data_warp((num_instances, num_bodies), device, wp.spatial_vectorf))


# ---------------------------------------------------------------------------
# Tests: Body property setters
# ---------------------------------------------------------------------------

_BODY_METHODS = [("set_masses", "masses"), ("set_coms", "coms"), ("set_inertias", "inertias")]


def _body_writer_layout(backend: str, method_base: str) -> tuple[type, int, str]:
    """Return the warp dtype, torch trailing size, and read-back getter of a body property writer."""
    if method_base == "set_masses":
        return wp.float32, 0, "body_mass"
    if method_base == "set_inertias":
        return wp.float32, 9, "body_inertia"
    if backend == "newton":
        # Newton stores the COM as a position only.
        return wp.vec3f, 3, "body_com_pos_b"
    return wp.transformf, 7, "body_com_pose_b"


def _make_body_torch(shape: tuple[int, int], device: str, wp_dtype: type, trailing: int, payload: bool = False):
    """Create body-property torch data: valid ones by default, or per-element distinct values."""
    full_shape = (*shape, trailing) if trailing else shape
    if payload:
        data = torch.arange(1, math.prod(full_shape) + 1, dtype=torch.float32, device=device).reshape(full_shape)
    else:
        data = torch.ones(full_shape, device=device, dtype=torch.float32)
    if wp_dtype == wp.transformf:
        data[..., 3:6] = 0.0
        data[..., 6] = 1.0
        if not payload:
            data[..., :3] = 0.0
    return data


def _make_body_warp(shape: tuple[int, int], device: str, wp_dtype: type, trailing: int, payload: bool = False):
    """Create body-property data as a warp array (structured dtypes collapse the trailing dim)."""
    t = _make_body_torch(shape, device, wp_dtype, trailing, payload).contiguous()
    return wp.from_torch(t, dtype=wp.float32 if wp_dtype == wp.float32 else wp_dtype)


class TestCollectionWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_devices
    @pytest.mark.parametrize("method_base, kwarg", _BODY_METHODS, ids=[m[0] for m in _BODY_METHODS])
    def test_body_writer_index(self, backend, device, method_base, kwarg):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        wp_dtype, trailing, getter = _body_writer_layout(backend, method_base)
        obj.data.update(dt=0.01)
        method = getattr(obj, f"{method_base}_index")
        sub_body_ids = [0]

        # torch, all envs + all bodies
        method(**{kwarg: _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing)})
        # torch, subset
        method(
            **{
                kwarg: _make_body_torch((1, 1), device, wp_dtype, trailing),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies: the matching getter reads the written values back
        method(**{kwarg: _make_body_warp((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)})
        expected = _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, subset
        method(
            **{
                kwarg: _make_body_warp((1, 1), device, wp_dtype, trailing),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # negative: bad torch shape (extra env)
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_body_torch((num_instances + 1, num_bodies), device, wp_dtype, trailing)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_body_warp((num_instances + 1, num_bodies), device, wp_dtype, trailing)})

    @_backends
    @_devices
    @pytest.mark.parametrize("method_base, kwarg", _BODY_METHODS, ids=[m[0] for m in _BODY_METHODS])
    def test_body_writer_mask(self, backend, device, method_base, kwarg):
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        wp_dtype, trailing, getter = _body_writer_layout(backend, method_base)
        obj.data.update(dt=0.01)
        method = getattr(obj, f"{method_base}_mask")
        sub_body_sel = [0]

        # torch, no mask
        method(**{kwarg: _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing)})
        # torch, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask: the matching getter reads the written values back
        method(**{kwarg: _make_body_warp((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)})
        expected = _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)
        _assert_reads_back(getattr(obj.data, getter), expected, getter)
        # warp, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_body_warp((num_instances, num_bodies), device, wp_dtype, trailing),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_body_torch((num_instances + 1, num_bodies), device, wp_dtype, trailing)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_body_warp((num_instances + 1, num_bodies), device, wp_dtype, trailing)})


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------

_COLLECTION_ALIASES = [
    ("body_pose_w", "body_link_pose_w"),
    ("body_pos_w", "body_link_pos_w"),
    ("body_quat_w", "body_link_quat_w"),
    ("body_vel_w", "body_com_vel_w"),
    ("body_lin_vel_w", "body_com_lin_vel_w"),
    ("body_ang_vel_w", "body_com_ang_vel_w"),
    ("body_acc_w", "body_com_acc_w"),
    ("body_lin_acc_w", "body_com_lin_acc_w"),
    ("body_ang_acc_w", "body_com_ang_acc_w"),
    ("com_pos_b", "body_com_pos_b"),
    ("com_quat_b", "body_com_quat_b"),
]


class TestCollectionDataAliases:
    """Test that alias properties return the values of their canonical counterparts."""

    @_backends
    def test_aliases_match_canonical_values(self, backend):
        # Random mock state makes link and COM quantities differ, so a retargeted alias fails.
        obj, _ = get_rigid_object_collection(backend, _NUM_INSTANCES, _NUM_BODIES, "cpu")
        obj.data.update(dt=0.01)
        d = obj.data

        for alias, canonical in _COLLECTION_ALIASES:
            alias_value, canonical_value = getattr(d, alias), getattr(d, canonical)
            assert alias_value.shape == canonical_value.shape, alias
            assert alias_value.dtype == canonical_value.dtype, alias
            assert torch.equal(alias_value.torch, canonical_value.torch), alias
