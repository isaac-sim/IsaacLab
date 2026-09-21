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

import numpy as np
import pytest
import torch
import warp as wp
from _rigid_object_collection_iface_test_utils import BACKENDS, get_rigid_object_collection

pytestmark = pytest.mark.integration


@pytest.fixture
def collection_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    num_bodies = request.getfixturevalue("num_bodies")
    device = request.getfixturevalue("device")
    result = get_rigid_object_collection(backend, num_instances, num_bodies, device)
    if backend != "newton":
        yield result
        return
    # The Newton collection's ``body_link_pose_w`` triggers ``_ensure_fk_fresh()`` ->
    # ``NewtonManager.forward()``, which runs ``eval_fk`` against a live simulation state. The mocked
    # interface has no such state (``_state_0`` is ``None``), so stub ``forward()`` to a no-op for the
    # test body; the mock view supplies the cached pose data directly.
    from unittest.mock import patch

    from isaaclab_newton.physics import NewtonManager

    with patch.object(NewtonManager, "forward"):
        yield result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_proxy_array(arr, *, expected_shape: tuple, expected_dtype: type, name: str):
    """Assert that `arr` is a ProxyArray with the expected shape and dtype."""
    from isaaclab.utils.warp import ProxyArray

    assert isinstance(arr, ProxyArray), f"{name}: expected ProxyArray, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


# Common parametrize decorators
_backends = pytest.mark.parametrize("backend", BACKENDS, indirect=False)
_default_dims = pytest.mark.parametrize("num_instances", [1, 2, 100])

_default_bodies = pytest.mark.parametrize("num_bodies", [1, 3])

_default_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])
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

    @_index_resolution_backends
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend):
        obj, _ = get_rigid_object_collection(backend, num_instances=4, device="cpu")

        env_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = obj._resolve_env_ids(env_ids)
        resolved_view = obj._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2

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
    @_default_devices
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_reshape_data_to_view_3d_accepts_torch_tensor(self, backend, device, dtype):
        if device.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

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

    @_reshape_3d_backends
    @_default_devices
    def test_reshape_data_to_view_3d_keeps_warp_array_behavior(self, backend, device):
        if device.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA is not available")

        num_instances = 2
        num_bodies = 3
        data_dim = 4
        obj, _ = get_rigid_object_collection(backend, num_instances=num_instances, num_bodies=num_bodies, device=device)
        data = torch.arange(num_instances * num_bodies * data_dim, dtype=torch.float32, device=device).reshape(
            num_instances, num_bodies, data_dim
        )

        torch_view = obj.reshape_data_to_view_3d(data, data_dim, device=device)
        warp_view = obj.reshape_data_to_view_3d(wp.from_torch(data, dtype=wp.float32), data_dim, device=device)

        assert isinstance(warp_view, wp.array)
        assert warp_view.shape == (num_bodies * num_instances, data_dim)
        assert warp_view.dtype == wp.float32
        assert str(warp_view.device) == device
        torch.testing.assert_close(wp.to_torch(warp_view), torch_view)


# ---------------------------------------------------------------------------
# Tests: Collection properties
# ---------------------------------------------------------------------------


class TestCollectionProperties:
    """Test that collection properties return the correct types/values."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_num_instances(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        assert obj.num_instances == num_instances

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_num_bodies(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        assert obj.num_bodies == num_bodies

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_names(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        names = obj.body_names
        assert isinstance(names, list)
        assert len(names) == num_bodies
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_data_returns_collection_data(self, backend, num_instances, num_bodies, device, collection_iface):
        from isaaclab.assets.rigid_object_collection.base_rigid_object_collection_data import (
            BaseRigidObjectCollectionData,
        )

        obj, _ = collection_iface
        assert isinstance(obj.data, BaseRigidObjectCollectionData)


# ---------------------------------------------------------------------------
# Tests: Collection finder methods
# ---------------------------------------------------------------------------


class TestCollectionFinders:
    """Test that finder methods return correct results."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_find_bodies_all(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        mask, names = obj.find_bodies(".*")
        assert isinstance(mask, torch.Tensor)
        assert isinstance(names, list)
        assert len(names) == num_bodies

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_find_bodies_single(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        first_body = obj.body_names[0]
        mask, names = obj.find_bodies(first_body)
        assert len(names) == 1
        assert names == [first_body]


class TestCollectionFinderReturnModes:
    """Test finder return modes on production collection backends."""

    @_production_backends
    def test_find_bodies_returns_legacy_tensor_or_cached_proxy(self, backend):
        collection, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")

        indices, names = collection.find_bodies(".*")
        proxy, proxy_names = collection.find_bodies(".*", as_proxy=True)

        assert isinstance(indices, torch.Tensor)
        assert indices.dtype == torch.int32
        assert indices.tolist() == proxy.torch.tolist()
        assert names == proxy_names
        assert proxy is collection.find_bodies(".*", as_proxy=True)[0]
        assert proxy.dtype == wp.int32
        assert str(proxy.device) == collection.device

    @_production_backends
    def test_find_objects_forwards_return_mode_with_alias_warning(self, backend):
        collection, _ = get_rigid_object_collection(backend, num_instances=2, num_bodies=3, device="cpu")

        with pytest.warns(DeprecationWarning):
            proxy, _ = collection.find_objects(".*", as_proxy=True)

        assert proxy is collection.find_bodies(".*", as_proxy=True)[0]


# ---------------------------------------------------------------------------
# Tests: Body state properties
# ---------------------------------------------------------------------------


class TestCollectionDataBodyState:
    """Test data properties for body state."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_pose_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_pose_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_acc_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_acc_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_acc_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_pose_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pose_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_b",
        )


# ---------------------------------------------------------------------------
# Tests: Sliced properties
# ---------------------------------------------------------------------------


class TestCollectionDataSliced:
    """Test sliced data properties."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_pos_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_pos_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_quat_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_link_quat_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_lin_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_lin_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_ang_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_ang_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_pos_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_quat_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_lin_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_lin_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_ang_vel_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_ang_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_lin_acc_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_lin_acc_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_lin_acc_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_ang_acc_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_ang_acc_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_ang_acc_w",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_pos_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pos_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_b",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_quat_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_quat_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_b",
        )


# ---------------------------------------------------------------------------
# Tests: Derived properties
# ---------------------------------------------------------------------------


class TestCollectionDataDerived:
    """Test derived/computed data properties."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_projected_gravity_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.projected_gravity_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="projected_gravity_b",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_heading_w(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.heading_w, expected_shape=(num_instances, num_bodies), expected_dtype=wp.float32, name="heading_w"
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_lin_vel_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_lin_vel_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_link_ang_vel_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_ang_vel_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_b",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_lin_vel_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_lin_vel_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_com_ang_vel_b(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_ang_vel_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_ang_vel_b",
        )


# ---------------------------------------------------------------------------
# Tests: Mass properties
# ---------------------------------------------------------------------------


class TestCollectionDataMass:
    """Test body mass/inertia properties."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_mass(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_mass, expected_shape=(num_instances, num_bodies), expected_dtype=wp.float32, name="body_mass"
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_inertia(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_inertia,
            expected_shape=(num_instances, num_bodies, 9),
            expected_dtype=wp.float32,
            name="body_inertia",
        )


# ---------------------------------------------------------------------------
# Tests: Default state properties
# ---------------------------------------------------------------------------


class TestCollectionDataDefaults:
    """Test default state properties."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_default_body_pose(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.default_body_pose,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="default_body_pose",
        )

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_default_body_vel(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.default_body_vel,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="default_body_vel",
        )


# ---------------------------------------------------------------------------
# Tests: Body pose/velocity writers
# ---------------------------------------------------------------------------

_BODY_POSE_METHODS = ["body_pose", "body_link_pose", "body_com_pose"]
_BODY_VEL_METHODS = ["body_velocity", "body_com_velocity", "body_link_velocity"]


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
            from unittest.mock import patch

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
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _BODY_POSE_METHODS)
    def test_write_body_pose_to_sim_index(
        self, backend, num_instances, num_bodies, device, collection_iface, method_suffix
    ):
        obj, _ = collection_iface
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
        # warp, all envs + all bodies
        method(body_poses=_make_data_warp((num_instances, num_bodies), device, wp.transformf))
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
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _BODY_VEL_METHODS)
    def test_write_body_velocity_to_sim_index(
        self, backend, num_instances, num_bodies, device, collection_iface, method_suffix
    ):
        obj, _ = collection_iface
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
        # warp, all envs + all bodies
        method(body_velocities=_make_data_warp((num_instances, num_bodies), device, wp.spatial_vectorf))
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
    # Note: write_body_pose_to_sim_mask accepts body_mask, but write_body_link_pose_to_sim_mask
    # and write_body_com_pose_to_sim_mask use body_ids instead. We only test body_mask on body_pose.

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _BODY_POSE_METHODS)
    def test_write_body_pose_to_sim_mask(
        self, backend, num_instances, num_bodies, device, collection_iface, method_suffix
    ):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        has_body_mask = method_suffix == "body_pose"

        # torch, no mask (all)
        method(body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf))
        # torch, partial env_mask
        method(
            body_poses=_make_data_torch((num_instances, num_bodies), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        if has_body_mask:
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
        # warp, no mask
        method(body_poses=_make_data_warp((num_instances, num_bodies), device, wp.transformf))
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
    # Note: write_body_velocity_to_sim_mask accepts body_mask, but the _link_/_com_ variants use body_ids.

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _BODY_VEL_METHODS)
    def test_write_body_velocity_to_sim_mask(
        self, backend, num_instances, num_bodies, device, collection_iface, method_suffix
    ):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        has_body_mask = method_suffix == "body_velocity"

        # torch, no mask
        method(body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf))
        # torch, partial env_mask
        method(
            body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        if has_body_mask:
            # torch, partial body_mask
            method(
                body_velocities=_make_data_torch((num_instances, num_bodies), device, wp.spatial_vectorf),
                body_mask=_make_item_mask(num_bodies, [0], device),
            )
        # warp, no mask
        method(body_velocities=_make_data_warp((num_instances, num_bodies), device, wp.spatial_vectorf))
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

_BODY_METHODS = [
    ("set_masses", "masses", wp.float32, 0),
    ("set_coms", "coms", wp.transformf, 7),
    ("set_inertias", "inertias", wp.float32, 9),
]


class TestCollectionWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_index(
        self, backend, num_instances, num_bodies, device, collection_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"{method_base}_index")

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_body_ids = [0]

        # torch, all envs + all bodies
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, subset
        method(
            **{
                kwarg: _make_torch(1, 1),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, subset
        method(
            **{
                kwarg: _make_warp(1, 1),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # negative: bad torch shape (extra env)
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_mask(
        self, backend, num_instances, num_bodies, device, collection_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"{method_base}_mask")

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _make_torch(n_envs, n_bods):
            shape = _torch_shape(n_envs, n_bods)
            data = torch.ones(shape, device=device, dtype=torch.float32)
            if wp_dtype == wp.transformf:
                data[..., :3] = 0.0
                data[..., 3:6] = 0.0
                data[..., 6] = 1.0
            return data

        def _make_warp(n_envs, n_bods):
            t = _make_torch(n_envs, n_bods)
            if wp_dtype == wp.transformf:
                return wp.from_torch(t.contiguous(), dtype=wp.transformf)
            return wp.from_torch(t.contiguous(), dtype=wp.float32)

        sub_body_sel = [0]

        # torch, no mask
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_torch(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, partial env_mask + body_mask
        method(
            **{
                kwarg: _make_warp(num_instances, num_bodies),
                "body_mask": _make_item_mask(num_bodies, sub_body_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_torch(num_instances + 1, num_bodies)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_warp(num_instances + 1, num_bodies)})


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------


class TestCollectionDataAliases:
    """Test that alias properties return the same shape/dtype as their canonical counterparts."""

    @_backends
    @_default_dims
    @_default_bodies
    @_default_devices
    def test_body_aliases(self, backend, num_instances, num_bodies, device, collection_iface):
        obj, _ = collection_iface
        obj.data.update(dt=0.01)
        d = obj.data

        # Pose aliases
        assert d.body_pose_w.shape == d.body_link_pose_w.shape
        assert d.body_pose_w.dtype == d.body_link_pose_w.dtype
        assert d.body_pos_w.shape == d.body_link_pos_w.shape
        assert d.body_pos_w.dtype == d.body_link_pos_w.dtype
        assert d.body_quat_w.shape == d.body_link_quat_w.shape
        assert d.body_quat_w.dtype == d.body_link_quat_w.dtype

        # Velocity aliases
        assert d.body_vel_w.shape == d.body_com_vel_w.shape
        assert d.body_vel_w.dtype == d.body_com_vel_w.dtype
        assert d.body_lin_vel_w.shape == d.body_com_lin_vel_w.shape
        assert d.body_lin_vel_w.dtype == d.body_com_lin_vel_w.dtype
        assert d.body_ang_vel_w.shape == d.body_com_ang_vel_w.shape
        assert d.body_ang_vel_w.dtype == d.body_com_ang_vel_w.dtype

        # Acceleration aliases
        assert d.body_acc_w.shape == d.body_com_acc_w.shape
        assert d.body_acc_w.dtype == d.body_com_acc_w.dtype
        assert d.body_lin_acc_w.shape == d.body_com_lin_acc_w.shape
        assert d.body_lin_acc_w.dtype == d.body_com_lin_acc_w.dtype
        assert d.body_ang_acc_w.shape == d.body_com_ang_acc_w.shape
        assert d.body_ang_acc_w.dtype == d.body_com_ang_acc_w.dtype

        # CoM body frame aliases
        assert d.com_pos_b.shape == d.body_com_pos_b.shape
        assert d.com_pos_b.dtype == d.body_com_pos_b.dtype
        assert d.com_quat_b.shape == d.body_com_quat_b.shape
        assert d.com_quat_b.dtype == d.body_com_quat_b.dtype
