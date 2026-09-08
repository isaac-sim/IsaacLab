# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""
Checks that the rigid object interfaces are consistent across backends, and are providing the exact same data as what
the base rigid object class advertises. All rigid object interfaces need to comply with the same interface contract.

The setup is a bit convoluted so that we can run these tests without requiring Isaac Sim or GPU simulation.
"""

import numpy as np
import pytest
import torch
import warp as wp
from _rigid_object_iface_test_utils import BACKENDS, get_rigid_object

pytestmark = pytest.mark.integration


@pytest.fixture
def rigid_object_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    device = request.getfixturevalue("device")
    return get_rigid_object(backend, num_instances, device)


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

_default_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])
_index_resolution_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)
_production_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton", "ovphysx") if backend in BACKENDS], indirect=False
)


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


class TestRigidObjectIndexResolution:
    """Test backend-specific index resolution helpers."""

    @_index_resolution_backends
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend):
        obj, _ = get_rigid_object(backend, num_instances=4, device="cpu")

        env_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = obj._resolve_env_ids(env_ids)
        resolved_view = obj._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: RigidObject properties
# ---------------------------------------------------------------------------


class TestRigidObjectProperties:
    """Test that rigid object properties return the correct types/values."""

    @_backends
    @_default_dims
    @_default_devices
    def test_num_instances(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        assert obj.num_instances == num_instances

    @_backends
    @_default_dims
    @_default_devices
    def test_num_bodies(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        assert obj.num_bodies == 1

    @_backends
    @_default_dims
    @_default_devices
    def test_body_names(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        names = obj.body_names
        assert isinstance(names, list)
        assert len(names) == 1
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_data_returns_rigid_object_data(self, backend, num_instances, device, rigid_object_iface):
        from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData

        obj, _ = rigid_object_iface
        assert isinstance(obj.data, BaseRigidObjectData)


# ---------------------------------------------------------------------------
# Tests: RigidObject finder methods
# ---------------------------------------------------------------------------


class TestRigidObjectFinders:
    """Test that finder methods return (list[int], list[str]) tuples."""

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_all(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        indices, names = obj.find_bodies(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == 1
        assert len(names) == 1
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_single(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        first_body = obj.body_names[0]
        indices, names = obj.find_bodies(first_body)
        assert indices == [0]
        assert names == [first_body]


class TestRigidObjectFinderReturnModes:
    """Test finder return modes on production rigid-object backends."""

    @_production_backends
    def test_find_bodies_returns_legacy_list_or_cached_proxy(self, backend):
        obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")

        indices, names = obj.find_bodies(".*")
        proxy, proxy_names = obj.find_bodies(".*", as_proxy=True)

        assert isinstance(indices, list)
        assert indices == proxy.torch.tolist()
        assert names == proxy_names
        assert proxy is obj.find_bodies(".*", as_proxy=True)[0]
        assert proxy.dtype == wp.int32
        assert str(proxy.device) == obj.device


# ---------------------------------------------------------------------------
# Tests: RigidObjectData root state properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataRootState:
    """Test data properties for root rigid body state."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_link_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_w",
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData derived properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataDerivedProperties:
    """Test derived/computed data properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_projected_gravity_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.projected_gravity_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="projected_gravity_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_heading_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.heading_w, expected_shape=(num_instances,), expected_dtype=wp.float32, name="heading_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_link_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.root_com_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_b",
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData body state properties
# ---------------------------------------------------------------------------


class TestRigidObjectDataBodyState:
    """Test data properties for all body states."""

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_pose_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pose_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_acc_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_acc_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_acc_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pose_b,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.transformf,
            name="body_com_pose_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_mass(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_mass, expected_shape=(num_instances, 1), expected_dtype=wp.float32, name="body_mass"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_inertia(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_inertia, expected_shape=(num_instances, 1, 9), expected_dtype=wp.float32, name="body_inertia"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_pos_w, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_quat_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.quatf,
            name="body_link_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_lin_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_lin_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_ang_vel_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_link_ang_vel_w,
            expected_shape=(num_instances, 1),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pos_w, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_w(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_quat_w, expected_shape=(num_instances, 1), expected_dtype=wp.quatf, name="body_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_pos_b, expected_shape=(num_instances, 1), expected_dtype=wp.vec3f, name="body_com_pos_b"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_b(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.body_com_quat_b, expected_shape=(num_instances, 1), expected_dtype=wp.quatf, name="body_com_quat_b"
        )


# ---------------------------------------------------------------------------
# Tests: RigidObjectData defaults
# ---------------------------------------------------------------------------


class TestRigidObjectDataDefaults:
    """Test default state properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_pose(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.default_root_pose,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="default_root_pose",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_vel(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        _check_proxy_array(
            obj.data.default_root_vel,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="default_root_vel",
        )


# ---------------------------------------------------------------------------
# Writer/setter test helpers
# ---------------------------------------------------------------------------

# Map warp structured dtypes to their torch trailing dimension size.
_WP_DTYPE_TO_TRAILING = {
    wp.transformf: 7,
    wp.spatial_vectorf: 6,
    wp.vec2f: 2,
    wp.float32: 0,  # no trailing dimension
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
        data[..., 6] = 1.0  # identity quat w
    elif wp_dtype == wp.vec2f:
        data[..., 0] = -1.0
        data[..., 1] = 1.0
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


def _make_item_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Create a bool warp mask with True at `selected` indices, False elsewhere."""
    mask_np = np.zeros(total, dtype=bool)
    for i in selected:
        mask_np[i] = True
    return wp.array(mask_np, dtype=wp.bool, device=device)


# ---------------------------------------------------------------------------
# Tests: Root writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_ROOT_POSE_METHODS = ["root_pose", "root_link_pose", "root_com_pose"]
_ROOT_VEL_METHODS = ["root_velocity", "root_link_velocity", "root_com_velocity"]


class TestRigidObjectCacheInvalidation:
    @_production_backends
    def test_pose_write_invalidates_pose_dependent_caches(self, backend):
        obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
        obj.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            obj.data,
            [
                ("root_link_vel_w", "_root_link_vel_w"),
                ("projected_gravity_b", "_projected_gravity_b"),
                ("heading_w", "_heading_w"),
                ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
                ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
                ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
                ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
            ],
        )
        root_pose = _make_data_warp((obj.num_instances,), "cpu", wp.transformf)
        obj.write_root_link_pose_to_sim_index(root_pose=root_pose)
        _assert_buffers_stale(obj.data, buffers)

    @_production_backends
    def test_velocity_write_invalidates_body_frame_caches(self, backend):
        obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
        obj.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            obj.data,
            [
                ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
                ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
                ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
                ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
            ],
        )
        root_velocity = _make_data_warp((obj.num_instances,), "cpu", wp.spatial_vectorf)
        obj.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)
        _assert_buffers_stale(obj.data, buffers)

    @_production_backends
    @pytest.mark.parametrize("setter_kind", ["index", "mask"])
    def test_set_coms_invalidates_same_timestamp_dependents(self, backend, setter_kind):
        obj, _ = get_rigid_object(backend, num_instances=2, device="cpu")
        obj.data.update(dt=0.01)
        common_pairs = [
            ("root_com_pose_w", "_root_com_pose_w"),
            ("root_link_vel_w", "_root_link_vel_w"),
            ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
            ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
            ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
            ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
            ("root_state_w", "_root_state_w"),
            ("root_link_state_w", "_root_link_state_w"),
            ("root_com_state_w", "_root_com_state_w"),
        ]
        if backend != "newton":
            common_pairs.append(("root_com_vel_w", "_root_com_vel_w"))
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


class TestRigidObjectWritersRoot:
    """Test root pose/velocity writers with all input combinations."""

    # -- index variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_index(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, subset
        method(root_pose=_make_data_torch((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, subset
        method(root_pose=_make_data_warp((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_index(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, subset
        method(root_velocity=_make_data_torch((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # warp, all envs
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, subset
        method(root_velocity=_make_data_warp((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))

    # -- mask variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_mask(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask (all)
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, partial mask
        method(
            root_pose=_make_data_torch((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_pose=_make_data_warp((num_instances,), device, wp.transformf))
        # warp, partial mask
        method(
            root_pose=_make_data_warp((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_mask(self, backend, num_instances, device, rigid_object_iface, method_suffix):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, partial mask
        method(
            root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask
        method(root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf))
        # warp, partial mask
        method(
            root_velocity=_make_data_warp((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_torch((num_instances,), device, wp.spatial_vectorf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_velocity=_make_bad_data_warp((num_instances,), device, wp.spatial_vectorf))


# ---------------------------------------------------------------------------
# Tests: Body writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, trailing_dim)
_BODY_METHODS = [
    ("set_masses", "masses", wp.float32, 0),
    ("set_coms", "coms", wp.transformf, 7),
    ("set_inertias", "inertias", wp.float32, 9),
]


class TestRigidObjectWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_index(
        self, backend, num_instances, device, rigid_object_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        num_bodies = 1
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

        sub_b = 1  # rigid object always has 1 body
        sub_body_ids = [0]

        # torch, all envs + all bodies
        method(**{kwarg: _make_torch(num_instances, num_bodies)})
        # torch, subset
        method(
            **{
                kwarg: _make_torch(1, sub_b),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies
        method(**{kwarg: _make_warp(num_instances, num_bodies)})
        # warp, subset
        method(
            **{
                kwarg: _make_warp(1, sub_b),
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
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, trailing",
        _BODY_METHODS,
        ids=[m[0] for m in _BODY_METHODS],
    )
    def test_body_writer_mask(
        self, backend, num_instances, device, rigid_object_iface, method_base, kwarg, wp_dtype, trailing
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton set_coms expects vec3f (position only), not transformf (pose)")
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        num_bodies = 1
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


class TestRigidObjectDataAliases:
    """Test that alias properties return the same shape/dtype as their canonical counterparts."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_aliases(self, backend, num_instances, device, rigid_object_iface):
        """root_pose_w == root_link_pose_w, root_vel_w == root_com_vel_w, etc."""
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        d = obj.data

        assert d.root_pose_w.shape == d.root_link_pose_w.shape
        assert d.root_pose_w.dtype == d.root_link_pose_w.dtype
        assert d.root_pos_w.shape == d.root_link_pos_w.shape
        assert d.root_quat_w.shape == d.root_link_quat_w.shape

        assert d.root_vel_w.shape == d.root_com_vel_w.shape
        assert d.root_vel_w.dtype == d.root_com_vel_w.dtype
        assert d.root_lin_vel_w.shape == d.root_com_lin_vel_w.shape
        assert d.root_ang_vel_w.shape == d.root_com_ang_vel_w.shape

    @_backends
    @_default_dims
    @_default_devices
    def test_body_aliases(self, backend, num_instances, device, rigid_object_iface):
        obj, _ = rigid_object_iface
        obj.data.update(dt=0.01)
        d = obj.data

        assert d.body_pose_w.shape == d.body_link_pose_w.shape
        assert d.body_pos_w.shape == d.body_link_pos_w.shape
        assert d.body_quat_w.shape == d.body_link_quat_w.shape
        assert d.body_vel_w.shape == d.body_com_vel_w.shape
        assert d.body_lin_vel_w.shape == d.body_com_lin_vel_w.shape
        assert d.body_ang_vel_w.shape == d.body_com_ang_vel_w.shape
