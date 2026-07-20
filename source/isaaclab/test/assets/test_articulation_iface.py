# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""
Checks that the articulation interfaces are consistent across backends, and are providing the exact same data as what
the base articulation class advertises. All articulation interfaces need to comply with the same interface contract.

The setup is a bit convoluted so that we can run these tests without requiring Isaac Sim or GPU simulation.
"""

import numpy as np
import pytest
import torch
import warp as wp
from _articulation_iface_test_utils import BACKENDS, get_articulation

pytestmark = pytest.mark.integration


@pytest.fixture
def articulation_iface(request):
    backend = request.getfixturevalue("backend")
    num_instances = request.getfixturevalue("num_instances")
    num_joints = request.getfixturevalue("num_joints")
    num_bodies = request.getfixturevalue("num_bodies")
    device = request.getfixturevalue("device")
    try:
        num_fixed_tendons = request.getfixturevalue("num_fixed_tendons")
    except pytest.FixtureLookupError:
        num_fixed_tendons = 0
    try:
        num_spatial_tendons = request.getfixturevalue("num_spatial_tendons")
    except pytest.FixtureLookupError:
        num_spatial_tendons = 0
    return get_articulation(
        backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_proxy_array(arr, *, expected_shape: tuple, expected_dtype: type, name: str):
    """Assert that `arr` is a ProxyArray with the expected shape and dtype."""
    from isaaclab.utils.warp import ProxyArray

    assert isinstance(arr, ProxyArray), f"{name}: expected ProxyArray, got {type(arr)}"
    assert arr.shape == expected_shape, f"{name}: expected shape {expected_shape}, got {arr.shape}"
    assert arr.dtype == expected_dtype, f"{name}: expected dtype {expected_dtype}, got {arr.dtype}"


# Common parametrize decorator for all interface tests
_backends = pytest.mark.parametrize("backend", BACKENDS, indirect=False)

# We also need to provide the fixture params that articulation_iface reads:
_default_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies",
    [(1, 1, 1), (1, 2, 2), (2, 6, 7), (100, 8, 13)],
)

_default_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])
_index_resolution_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


class TestArticulationIndexResolution:
    """Test backend-specific index resolution helpers."""

    @_index_resolution_backends
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_instances=4, device="cpu")

        env_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_env_ids(env_ids)
        resolved_view = art._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2

    @_index_resolution_backends
    def test_resolve_joint_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_joints=4, device="cpu")

        joint_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_joint_ids(joint_ids)
        resolved_view = art._resolve_joint_ids(joint_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2

    @_index_resolution_backends
    def test_resolve_body_ids_handles_tensor_view_shape(self, backend):
        art, _ = get_articulation(backend, num_bodies=4, device="cpu")

        body_ids = torch.arange(4, dtype=torch.int32, device="cpu")
        resolved_full = art._resolve_body_ids(body_ids)
        resolved_view = art._resolve_body_ids(body_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: Articulation properties
# ---------------------------------------------------------------------------


class TestArticulationProperties:
    """Test that articulation properties return the correct types/values."""

    @_backends
    @_default_dims
    @_default_devices
    def test_num_instances(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_instances == num_instances

    @_backends
    @_default_dims
    @_default_devices
    def test_num_joints(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_joints == num_joints

    @_backends
    @_default_dims
    @_default_devices
    def test_num_bodies(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert art.num_bodies == num_bodies

    @_backends
    @_default_dims
    @_default_devices
    def test_is_fixed_base(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        assert isinstance(art.is_fixed_base, bool)

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_names(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        names = art.joint_names
        assert isinstance(names, list)
        assert len(names) == num_joints
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_body_names(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        names = art.body_names
        assert isinstance(names, list)
        assert len(names) == num_bodies
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_data_returns_articulation_data(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface
    ):
        from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

        art, _ = articulation_iface
        assert isinstance(art.data, BaseArticulationData)


# ---------------------------------------------------------------------------
# Tests: Articulation finder methods
# ---------------------------------------------------------------------------


class TestArticulationFinders:
    """Test that finder methods return (list[int], list[str]) tuples."""

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_all(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        indices, names = art.find_bodies(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_bodies
        assert len(names) == num_bodies
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_backends
    @_default_dims
    @_default_devices
    def test_find_joints_all(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        indices, names = art.find_joints(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_joints
        assert len(names) == num_joints

    @_backends
    @_default_dims
    @_default_devices
    def test_find_bodies_single(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        first_body = art.body_names[0]
        indices, names = art.find_bodies(first_body)
        assert indices == [0]
        assert names == [first_body]

    @_backends
    @_default_dims
    @_default_devices
    def test_find_joints_single(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        first_joint = art.joint_names[0]
        indices, names = art.find_joints(first_joint)
        assert indices == [0]
        assert names == [first_joint]


# ---------------------------------------------------------------------------
# Tests: resolve_matching_names caching behavior
# ---------------------------------------------------------------------------


_non_mock_backends = pytest.mark.parametrize(
    "backend", [backend for backend in BACKENDS if backend.lower() != "mock"], indirect=False
)


class TestResolveMatchingNamesCache:
    """Test that resolve_matching_names caching returns correct, isolated results."""

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_unmatched_regex_raises(self, backend, num_instances, num_joints, num_bodies, device):
        """ValueError from resolve_matching_names propagates correctly."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        with pytest.raises(ValueError):
            art.find_bodies("nonexistent_body_xyz")
        with pytest.raises(ValueError):
            art.find_joints("nonexistent_joint_xyz")

    @_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_mutating_result_does_not_corrupt_cache(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface
    ):
        """Mutating returned lists must not affect future cached results."""
        art, _ = articulation_iface

        for finder, expected_len in [("find_bodies", num_bodies), ("find_joints", num_joints)]:
            idx1, names1 = getattr(art, finder)(".*")
            assert len(idx1) == expected_len

            idx1.clear()
            names1.append("corrupted")

            idx2, names2 = getattr(art, finder)(".*")
            assert len(idx2) == expected_len
            assert "corrupted" not in names2

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_find_with_multiple_patterns(self, backend, num_instances, num_joints, num_bodies, device):
        """Passing a list of regex patterns works correctly."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        idx, names = art.find_joints(["joint_0", "joint_1"])
        assert "joint_0" in names
        assert "joint_1" in names
        assert len(names) == 2

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 6, 7)])
    @_default_devices
    def test_find_with_preserve_order(self, backend, num_instances, num_joints, num_bodies, device):
        """preserve_order=True returns names in the order of the input patterns."""
        art, _ = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        idx_fwd, names_fwd = art.find_joints(["joint_1", "joint_0"], preserve_order=True)
        assert names_fwd == ["joint_1", "joint_0"]

        idx_rev, names_rev = art.find_joints(["joint_0", "joint_1"], preserve_order=True)
        assert names_rev == ["joint_0", "joint_1"]


# ---------------------------------------------------------------------------
# Tests: ArticulationData root state properties
# ---------------------------------------------------------------------------


class TestArticulationDataRootState:
    """Test data properties for root rigid body state."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_pose_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="root_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="root_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_link_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_link_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_pos_w, expected_shape=(num_instances,), expected_dtype=wp.vec3f, name="root_com_pos_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_quat_w, expected_shape=(num_instances,), expected_dtype=wp.quatf, name="root_com_quat_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_lin_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_ang_vel_w,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_w",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData derived properties
# ---------------------------------------------------------------------------


class TestArticulationDataDerivedProperties:
    """Test derived/computed data properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_projected_gravity_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.projected_gravity_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="projected_gravity_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_heading_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.heading_w, expected_shape=(num_instances,), expected_dtype=wp.float32, name="heading_w"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_lin_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_link_ang_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_link_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_link_ang_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_lin_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_lin_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_lin_vel_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_root_com_ang_vel_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.root_com_ang_vel_b,
            expected_shape=(num_instances,),
            expected_dtype=wp.vec3f,
            name="root_com_ang_vel_b",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData body state properties
# ---------------------------------------------------------------------------


class TestArticulationDataBodyState:
    """Test data properties for all body states."""

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_link_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_link_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pose_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_acc_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_acc_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.spatial_vectorf,
            name="body_com_acc_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pose_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        if backend == "newton":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pose_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.transformf,
            name="body_com_pose_b",
        )

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_body_com_pose_b_is_cached_across_sim_timestamps(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_set_coms_index_updates_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        coms = wp.zeros((art.num_instances, art.num_bodies), dtype=wp.transformf, device="cpu")
        art.set_coms_index(coms=coms, full_data=True)
        art.data.body_com_pose_b

        assert num_get_coms_calls == 0

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_joint_position_write_preserves_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        art.data.update(dt=0.01)
        art.data.body_com_pose_b
        assert num_get_coms_calls == 1

        joint_pos = torch.zeros((art.num_instances, art.num_joints), device="cpu")
        art.write_joint_position_to_sim_index(position=joint_pos, full_data=True)
        art.data.body_com_pose_b

        assert num_get_coms_calls == 1

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_partial_set_coms_index_initializes_cold_body_com_pose_b_cache(self):
        art, view = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")
        initial_coms = view.get_coms().numpy().copy()

        num_get_coms_calls = 0
        get_coms = view.get_coms

        def counted_get_coms():
            nonlocal num_get_coms_calls
            num_get_coms_calls += 1
            return get_coms()

        view.get_coms = counted_get_coms

        coms = wp.zeros((1, 1), dtype=wp.transformf, device="cpu")
        art.set_coms_index(
            coms=coms,
            env_ids=wp.array([0], dtype=wp.int32, device="cpu"),
            body_ids=wp.array([0], dtype=wp.int32, device="cpu"),
        )
        body_com_pose_b = art.data.body_com_pose_b.torch

        assert num_get_coms_calls == 1
        torch.testing.assert_close(body_com_pose_b[1, 1], torch.from_numpy(initial_coms[1, 1]))

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_set_coms_index_invalidates_body_com_pose_b_dependents(self):
        art, _ = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        art.data.update(dt=0.01)
        dependent_buffers = [
            ("root_com_pose_w", art.data._root_com_pose_w),
            ("root_com_vel_w", art.data._root_com_vel_w),
            ("root_link_vel_w", art.data._root_link_vel_w),
            ("body_com_pose_w", art.data._body_com_pose_w),
            ("body_com_vel_w", art.data._body_com_vel_w),
            ("body_link_vel_w", art.data._body_link_vel_w),
            ("root_link_lin_vel_b", art.data._root_link_lin_vel_b),
            ("root_link_ang_vel_b", art.data._root_link_ang_vel_b),
            ("root_com_lin_vel_b", art.data._root_com_lin_vel_b),
            ("root_com_ang_vel_b", art.data._root_com_ang_vel_b),
            ("root_state_w", art.data._root_state_w),
            ("root_link_state_w", art.data._root_link_state_w),
            ("root_com_state_w", art.data._root_com_state_w),
            ("body_state_w", art.data._body_state_w),
            ("body_link_state_w", art.data._body_link_state_w),
            ("body_com_state_w", art.data._body_com_state_w),
            ("body_com_jacobian_w", art.data._body_com_jacobian_w),
            ("mass_matrix", art.data._mass_matrix),
            ("gravity_compensation_forces", art.data._gravity_compensation_forces),
        ]
        for _, buffer in dependent_buffers:
            buffer.timestamp = art.data._sim_timestamp

        coms = wp.zeros((art.num_instances, art.num_bodies), dtype=wp.transformf, device="cpu")
        art.set_coms_index(coms=coms, full_data=True)

        for name, buffer in dependent_buffers:
            assert buffer.timestamp < art.data._sim_timestamp, name

    @pytest.mark.skipif("physx" not in BACKENDS, reason="PhysX backend unavailable")
    def test_physx_identity_ordering_reuses_dynamics_proxy_array_across_refreshes(self):
        """With identity ordering (the default), ``body_com_jacobian_w``, ``mass_matrix``, and
        ``gravity_compensation_forces`` alias stable, pre-allocated PhysX buffers, so their
        ``ProxyArray`` wrapper must be created once and reused on every subsequent refresh —
        matching the pattern already used by, e.g., ``root_link_pose_w``. Rebuilding the wrapper
        on every step forces a redundant ``.torch`` view rebuild even though the underlying
        device pointer never changes.
        """
        art, _ = get_articulation("physx", num_instances=2, num_joints=3, num_bodies=4, device="cpu")

        art.data.update(dt=0.01)
        jacobian_wrapper_a = art.data.body_com_jacobian_w
        mass_matrix_wrapper_a = art.data.mass_matrix
        gravity_wrapper_a = art.data.gravity_compensation_forces

        art.data.update(dt=0.01)
        jacobian_wrapper_b = art.data.body_com_jacobian_w
        mass_matrix_wrapper_b = art.data.mass_matrix
        gravity_wrapper_b = art.data.gravity_compensation_forces

        assert jacobian_wrapper_a is jacobian_wrapper_b
        assert mass_matrix_wrapper_a is mass_matrix_wrapper_b
        assert gravity_wrapper_a is gravity_wrapper_b

    @_backends
    @_default_dims
    @_default_devices
    def test_body_mass(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_mass, expected_shape=(num_instances, num_bodies), expected_dtype=wp.float32, name="body_mass"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_inertia(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_inertia,
            expected_shape=(num_instances, num_bodies, 9),
            expected_dtype=wp.float32,
            name="body_inertia",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_pos_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_link_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_lin_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_lin_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_lin_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_link_ang_vel_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_link_ang_vel_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_link_ang_vel_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pos_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_w(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_quat_w,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_w",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_pos_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_pos_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.vec3f,
            name="body_com_pos_b",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_body_com_quat_b(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        if backend == "newton":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.body_com_quat_b,
            expected_shape=(num_instances, num_bodies),
            expected_dtype=wp.quatf,
            name="body_com_quat_b",
        )


class TestArticulationDataJointState:
    """Test data properties for joint state and joint properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_pos"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_vel"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_acc(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_acc, expected_shape=(num_instances, num_joints), expected_dtype=wp.float32, name="joint_acc"
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_stiffness(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_stiffness,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_stiffness",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_damping(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_damping,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_damping",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_armature(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_armature,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_armature",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_friction_coeff(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_friction_coeff,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_friction_coeff",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.vec2f,
            name="joint_pos_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_vel_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_effort_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_effort_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_effort_limits",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_soft_joint_pos_limits(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.soft_joint_pos_limits,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.vec2f,
            name="soft_joint_pos_limits",
        )


# ---------------------------------------------------------------------------
# Tests: ArticulationData defaults and command targets
# ---------------------------------------------------------------------------


class TestArticulationDataDefaults:
    """Test default state and command target properties."""

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_pose(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_root_pose,
            expected_shape=(num_instances,),
            expected_dtype=wp.transformf,
            name="default_root_pose",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_root_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_root_vel,
            expected_shape=(num_instances,),
            expected_dtype=wp.spatial_vectorf,
            name="default_root_vel",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_joint_pos(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_joint_pos,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="default_joint_pos",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_default_joint_vel(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.default_joint_vel,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="default_joint_vel",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_pos_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_pos_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_pos_target",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_vel_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_vel_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_vel_target",
        )

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_effort_target(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.joint_effort_target,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="joint_effort_target",
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
    """Create valid torch test data for a given warp dtype.

    For transformf shapes, appends a trailing dim of 7 and sets quat w=1.
    For spatial_vectorf, appends trailing 6.
    For vec2f, appends trailing 2 with [-1, 1].
    For float32, no trailing dim.
    """
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
    """Create valid warp test data for a given warp dtype.

    Warp structured types collapse the trailing dim into the dtype,
    so a (N,) transformf array is equivalent to (N, 7) float32 in torch.
    """
    t = _make_data_torch(shape, device, wp_dtype)
    if wp_dtype == wp.float32:
        return wp.from_torch(t, dtype=wp.float32)
    # For structured types, the torch tensor has the trailing dim; convert to warp
    return wp.from_torch(t.contiguous(), dtype=wp_dtype)


def _make_bad_data_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create torch data with wrong leading shape for negative testing.

    Adds +1 to the first dimension so the shape doesn't match.
    """
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
# Tests: Articulation operations
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: Root writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_ROOT_POSE_METHODS = ["root_pose", "root_link_pose", "root_com_pose"]
_ROOT_VEL_METHODS = ["root_velocity", "root_link_velocity", "root_com_velocity"]


class TestArticulationWritersRoot:
    """Test root pose/velocity writers with all input combinations."""

    # -- index variants --

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_index(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

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
    def test_write_root_velocity_to_sim_index(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

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
    def test_write_root_pose_to_sim_mask(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

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
    def test_write_root_velocity_to_sim_mask(
        self, backend, num_instances, num_joints, num_bodies, device, articulation_iface, method_suffix
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

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
# Tests: Joint writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, accepts_float)
_JOINT_METHODS = [
    ("write_joint_position_to_sim", "position", wp.float32, False),
    ("write_joint_velocity_to_sim", "velocity", wp.float32, False),
    ("write_joint_stiffness_to_sim", "stiffness", wp.float32, True),
    ("write_joint_damping_to_sim", "damping", wp.float32, True),
    ("write_joint_position_limit_to_sim", "limits", wp.vec2f, True),
    ("write_joint_velocity_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_effort_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_armature_to_sim", "armature", wp.float32, True),
    ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", wp.float32, False),
    ("set_joint_position_target", "target", wp.float32, False),
    ("set_joint_velocity_target", "target", wp.float32, False),
    ("set_joint_effort_target", "target", wp.float32, False),
]


class TestArticulationWritersJoint:
    """Test joint writers/setters with all input combinations."""

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_j = min(2, num_joints)
        sub_joint_ids = list(range(sub_j))

        # torch, all envs + all joints
        method(**{kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype)})
        # torch, subset envs + subset joints
        method(
            **{
                kwarg: _make_data_torch((1, sub_j), device, wp_dtype),
                "joint_ids": sub_joint_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all joints
        method(**{kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_j), device, wp_dtype),
                "joint_ids": sub_joint_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f position_limit)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # float scalar for vec2f position_limit should raise ValueError
        if accepts_float and wp_dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_joints), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_joints), device, wp_dtype)})

    @_backends
    @_default_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_joint_sel = list(range(min(2, num_joints)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype)})
        # torch, partial env_mask + joint_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_joints), device, wp_dtype),
                "joint_mask": _make_item_mask(num_joints, sub_joint_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype)})
        # warp, partial env_mask + joint_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_joints), device, wp_dtype),
                "joint_mask": _make_item_mask(num_joints, sub_joint_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_joints), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_joints), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Body writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, trailing_dim)
_BODY_METHODS = [
    ("set_masses", "masses", wp.float32, 0),
    ("set_coms", "coms", wp.transformf, 7),
    ("set_inertias", "inertias", wp.float32, 9),
]


class TestArticulationWritersBody:
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
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        trailing,
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")

        # For inertias, the shape is (N, B, 9) always (no structured warp type)
        # For coms, torch shape is (N, B, 7), warp shape is (N, B) transformf
        # For masses, shape is (N, B)

        def _torch_shape(n_envs, n_bods):
            if trailing:
                return (n_envs, n_bods, trailing)
            return (n_envs, n_bods)

        def _warp_shape(n_envs, n_bods):
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

        sub_b = min(2, num_bodies)
        sub_body_ids = list(range(sub_b))

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
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        trailing,
    ):
        if backend == "newton" and method_base == "set_coms":
            pytest.xfail("Newton only stores CoM position, not orientation")
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")

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

        sub_body_sel = list(range(min(2, num_bodies)))

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


class TestArticulationDataAliases:
    """Test that alias properties return the same shape/dtype as their canonical counterparts."""

    @_backends
    @_default_dims
    @_default_devices
    def test_root_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        """root_pose_w == root_link_pose_w, root_vel_w == root_com_vel_w, etc."""
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

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
    def test_body_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

        assert d.body_pose_w.shape == d.body_link_pose_w.shape
        assert d.body_pos_w.shape == d.body_link_pos_w.shape
        assert d.body_quat_w.shape == d.body_link_quat_w.shape
        assert d.body_vel_w.shape == d.body_com_vel_w.shape
        assert d.body_lin_vel_w.shape == d.body_com_lin_vel_w.shape
        assert d.body_ang_vel_w.shape == d.body_com_ang_vel_w.shape

    @_backends
    @_default_dims
    @_default_devices
    def test_joint_aliases(self, backend, num_instances, num_joints, num_bodies, device, articulation_iface):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        d = art.data

        assert d.joint_limits.shape == d.joint_pos_limits.shape
        assert d.joint_friction.shape == d.joint_friction_coeff.shape


# ---------------------------------------------------------------------------
# Tendon tests — parametrize, properties, finders, data, writers
# ---------------------------------------------------------------------------

# Newton does not support tendons (always 0), so exclude it from tendon tests.
_tendon_backends = pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "newton"], indirect=False)

_tendon_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons",
    [
        (1, 2, 2, 1, 0),  # fixed only
        (2, 6, 7, 3, 2),  # both types
        (100, 8, 13, 4, 3),  # large, both types
    ],
)


class TestArticulationTendonProperties:
    """Test that tendon-related articulation properties return the correct types/values."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_num_fixed_tendons(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        assert art.num_fixed_tendons == num_fixed_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_num_spatial_tendons(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        assert art.num_spatial_tendons == num_spatial_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_names(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        names = art.fixed_tendon_names
        assert isinstance(names, list)
        assert len(names) == num_fixed_tendons
        assert all(isinstance(n, str) for n in names)

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_names(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        names = art.spatial_tendon_names
        assert isinstance(names, list)
        assert len(names) == num_spatial_tendons
        assert all(isinstance(n, str) for n in names)


class TestArticulationTendonFinders:
    """Test that tendon finder methods return (list[int], list[str]) tuples."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_fixed_tendons_all(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        indices, names = art.find_fixed_tendons(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_fixed_tendons
        assert len(names) == num_fixed_tendons
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_fixed_tendons_single(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        first = art.fixed_tendon_names[0]
        indices, names = art.find_fixed_tendons(first)
        assert indices == [0]
        assert names == [first]

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_spatial_tendons_all(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        indices, names = art.find_spatial_tendons(".*")
        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == num_spatial_tendons
        assert len(names) == num_spatial_tendons

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_find_spatial_tendons_single(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        first = art.spatial_tendon_names[0]
        indices, names = art.find_spatial_tendons(first)
        assert indices == [0]
        assert names == [first]


class TestArticulationDataTendonState:
    """Test data properties for tendon state (fixed and spatial)."""

    # -- Fixed tendon data properties --

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_stiffness,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_damping(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_damping,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_damping",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_limit_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_limit_stiffness,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_limit_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_rest_length(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_rest_length,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_rest_length",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_offset(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.fixed_tendon_offset,
            expected_shape=(num_instances, num_fixed_tendons),
            expected_dtype=wp.float32,
            name="fixed_tendon_offset",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_fixed_tendon_pos_limits(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        art.data.update(dt=0.01)
        from isaaclab.utils.warp import ProxyArray

        arr = art.data.fixed_tendon_pos_limits
        assert isinstance(arr, ProxyArray), f"fixed_tendon_pos_limits: expected ProxyArray, got {type(arr)}"
        if num_fixed_tendons == 0:
            # When no tendons, shape is (N, 0, 2) float32
            assert arr.shape == (num_instances, 0, 2)
            assert arr.dtype == wp.float32
        else:
            # PhysX returns (N, T, 2) float32; Mock returns (N, T) vec2f
            assert arr.shape in ((num_instances, num_fixed_tendons), (num_instances, num_fixed_tendons, 2))
            assert arr.dtype in (wp.vec2f, wp.float32)

    # -- Spatial tendon data properties --

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_stiffness,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_damping(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_damping,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_damping",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_limit_stiffness(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_limit_stiffness,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_limit_stiffness",
        )

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_spatial_tendon_offset(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        _check_proxy_array(
            art.data.spatial_tendon_offset,
            expected_shape=(num_instances, num_spatial_tendons),
            expected_dtype=wp.float32,
            name="spatial_tendon_offset",
        )


# ---------------------------------------------------------------------------
# Tests: Fixed tendon writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, accepts_float)
_FIXED_TENDON_METHODS = [
    ("set_fixed_tendon_stiffness", "stiffness", wp.float32, True),
    ("set_fixed_tendon_damping", "damping", wp.float32, True),
    ("set_fixed_tendon_limit_stiffness", "limit_stiffness", wp.float32, True),
    ("set_fixed_tendon_rest_length", "rest_length", wp.float32, True),
    ("set_fixed_tendon_offset", "offset", wp.float32, True),
]
# Note: set_fixed_tendon_position_limit is excluded because the PhysX backend stores
# pos_limits as (N, T, 2) float32 while the setter validates (N, T) float32. This data
# layout mismatch prevents consistent testing across mock and PhysX backends.


class TestArticulationWritersFixedTendon:
    """Test fixed tendon writers/setters with all input combinations."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _FIXED_TENDON_METHODS,
        ids=[m[0] for m in _FIXED_TENDON_METHODS],
    )
    def test_fixed_tendon_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_t = min(2, num_fixed_tendons)
        sub_tendon_ids = list(range(sub_t))

        # torch, all envs + all tendons
        method(**{kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # torch, subset envs + subset tendons
        method(
            **{
                kwarg: _make_data_torch((1, sub_t), device, wp_dtype),
                "fixed_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all tendons
        method(**{kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_t), device, wp_dtype),
                "fixed_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # float scalar for vec2f should raise ValueError
        if accepts_float and wp_dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _FIXED_TENDON_METHODS,
        ids=[m[0] for m in _FIXED_TENDON_METHODS],
    )
    def test_fixed_tendon_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_tendon_sel = list(range(min(2, num_fixed_tendons)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # torch, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_fixed_tendons), device, wp_dtype),
                "fixed_tendon_mask": _make_item_mask(num_fixed_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})
        # warp, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_fixed_tendons), device, wp_dtype),
                "fixed_tendon_mask": _make_item_mask(num_fixed_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar (only for accepts_float methods, and NOT for vec2f)
        if accepts_float and wp_dtype != wp.vec2f:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_fixed_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_fixed_tendons), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Spatial tendon writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

_SPATIAL_TENDON_METHODS = [
    ("set_spatial_tendon_stiffness", "stiffness", wp.float32, True),
    ("set_spatial_tendon_damping", "damping", wp.float32, True),
    ("set_spatial_tendon_limit_stiffness", "limit_stiffness", wp.float32, True),
    ("set_spatial_tendon_offset", "offset", wp.float32, True),
]


class TestArticulationWritersSpatialTendon:
    """Test spatial tendon writers/setters with all input combinations."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _SPATIAL_TENDON_METHODS,
        ids=[m[0] for m in _SPATIAL_TENDON_METHODS],
    )
    def test_spatial_tendon_writer_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_t = min(2, num_spatial_tendons)
        sub_tendon_ids = list(range(sub_t))

        # torch, all envs + all tendons
        method(**{kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # torch, subset envs + subset tendons
        method(
            **{
                kwarg: _make_data_torch((1, sub_t), device, wp_dtype),
                "spatial_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all tendons
        method(**{kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})
        # warp, subset
        method(
            **{
                kwarg: _make_data_warp((1, sub_t), device, wp_dtype),
                "spatial_tendon_ids": sub_tendon_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # float scalar
        if accepts_float:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float",
        _SPATIAL_TENDON_METHODS,
        ids=[m[0] for m in _SPATIAL_TENDON_METHODS],
    )
    def test_spatial_tendon_writer_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
        method_base,
        kwarg,
        wp_dtype,
        accepts_float,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_tendon_sel = list(range(min(2, num_spatial_tendons)))

        # torch, no mask
        method(**{kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # torch, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_torch((num_instances, num_spatial_tendons), device, wp_dtype),
                "spatial_tendon_mask": _make_item_mask(num_spatial_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # warp, no mask
        method(**{kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})
        # warp, partial env_mask + tendon_mask
        method(
            **{
                kwarg: _make_data_warp((num_instances, num_spatial_tendons), device, wp_dtype),
                "spatial_tendon_mask": _make_item_mask(num_spatial_tendons, sub_tendon_sel, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
        )
        # float scalar
        if accepts_float:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch((num_instances, num_spatial_tendons), device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp((num_instances, num_spatial_tendons), device, wp_dtype)})


# ---------------------------------------------------------------------------
# Tests: Tendon write-to-sim smoke tests
# ---------------------------------------------------------------------------


class TestArticulationWritersTendonToSim:
    """Smoke test write_*_tendon_properties_to_sim_index/mask methods."""

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_fixed_tendon_properties_to_sim_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        # all envs
        art.write_fixed_tendon_properties_to_sim_index()
        # subset envs
        art.write_fixed_tendon_properties_to_sim_index(env_ids=_make_env_ids(device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_fixed_tendon_properties_to_sim_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_fixed_tendons == 0:
            pytest.skip("No fixed tendons configured")
        art.data.update(dt=0.01)
        # no mask
        art.write_fixed_tendon_properties_to_sim_mask()
        # partial env mask
        art.write_fixed_tendon_properties_to_sim_mask(env_mask=_make_env_mask(num_instances, device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_spatial_tendon_properties_to_sim_index(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        # all envs
        art.write_spatial_tendon_properties_to_sim_index()
        # subset envs
        art.write_spatial_tendon_properties_to_sim_index(env_ids=_make_env_ids(device, True))

    @_tendon_backends
    @_tendon_dims
    @_default_devices
    def test_write_spatial_tendon_properties_to_sim_mask(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        articulation_iface,
    ):
        art, _ = articulation_iface
        if num_spatial_tendons == 0:
            pytest.skip("No spatial tendons configured")
        art.data.update(dt=0.01)
        # no mask
        art.write_spatial_tendon_properties_to_sim_mask()
        # partial env mask
        art.write_spatial_tendon_properties_to_sim_mask(env_mask=_make_env_mask(num_instances, device, True))
