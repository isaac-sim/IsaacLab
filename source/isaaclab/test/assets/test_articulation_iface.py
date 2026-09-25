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

import math
import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp
from _articulation_iface_test_utils import BACKEND_UNAVAILABLE_REASONS, BACKENDS, get_articulation

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.integration


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
# One fixture with distinct instance, joint, and body counts so any swapped axis shows up.
_NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES = 2, 6, 7
_index_resolution_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton") if backend in BACKENDS], indirect=False
)
_production_backends = pytest.mark.parametrize(
    "backend", [backend for backend in ("physx", "newton", "ovphysx") if backend in BACKENDS], indirect=False
)


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


class TestArticulationIndexResolution:
    """Test backend-specific index resolution helpers."""

    @_index_resolution_backends
    @_devices
    def test_resolve_env_ids_handles_tensor_view_shape(self, backend, device):
        art, _ = get_articulation(backend, num_instances=4, device=device)

        env_ids = torch.arange(4, dtype=torch.int32, device=device)
        resolved_full = art._resolve_env_ids(env_ids)
        resolved_view = art._resolve_env_ids(env_ids[:2])

        assert resolved_full.shape[0] == 4
        assert resolved_view.shape[0] == 2
        # Native indexed writers can use a strided view of cached IDs without copying or uploading them.
        cached = wp.to_torch(art._ALL_INDICES)
        for selection in (slice(None), slice(1, None, 2), slice(0, 0)):
            resolved = wp.to_torch(art._resolve_env_ids(selection))
            torch.testing.assert_close(resolved, cached[selection])
            assert resolved.data_ptr() == cached[selection].data_ptr()
            assert resolved.stride() == cached[selection].stride()

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
# Tests: Articulation properties and finders
# ---------------------------------------------------------------------------


class TestArticulationProperties:
    """Test that articulation properties return the correct types/values."""

    @_backends
    @pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
    def test_articulation_counts_names_and_finders(self, backend, is_fixed_base):
        from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

        art, _ = get_articulation(
            backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device="cpu", is_fixed_base=is_fixed_base
        )

        assert art.num_instances == _NUM_INSTANCES
        assert art.num_joints == _NUM_JOINTS
        assert art.num_bodies == _NUM_BODIES
        assert art.is_fixed_base is is_fixed_base
        for names, count in ((art.joint_names, _NUM_JOINTS), (art.body_names, _NUM_BODIES)):
            assert isinstance(names, list)
            assert len(names) == count
            assert all(isinstance(n, str) for n in names)
        assert isinstance(art.data, BaseArticulationData)


class TestArticulationFinderReturnModes:
    """Test finder return modes on production articulation backends."""

    @_production_backends
    @pytest.mark.parametrize(
        "finder_name, names_attr, count",
        [
            ("find_bodies", "body_names", 4),
            ("find_joints", "joint_names", 3),
            ("find_fixed_tendons", "fixed_tendon_names", 2),
            ("find_spatial_tendons", "spatial_tendon_names", 5),
        ],
    )
    def test_finder_returns_legacy_list_or_cached_proxy(self, backend, finder_name, names_attr, count):
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=3,
            num_bodies=4,
            num_fixed_tendons=2,
            num_spatial_tendons=5,
            device="cpu",
        )
        if backend == "newton" and finder_name == "find_spatial_tendons":
            pytest.skip("Newton does not support spatial tendons.")
        finder = getattr(art, finder_name)

        indices, names = finder(".*")
        proxy, proxy_names = finder(".*", as_proxy=True)

        assert isinstance(indices, list) and isinstance(names, list)
        assert len(indices) == count
        assert len(names) == count
        assert all(isinstance(i, int) for i in indices)
        assert all(isinstance(n, str) for n in names)
        assert indices == proxy.torch.tolist()
        assert names == proxy_names
        assert proxy is finder(".*", as_proxy=True)[0]
        assert proxy.dtype == wp.int32
        assert str(proxy.device) == art.device

        first_name = getattr(art, names_attr)[0]
        assert finder(first_name) == ([0], [first_name])


class TestFixedTendonTargetScheduling:
    """Commanding a tendon target is what schedules the write, not merely having tendons."""

    @staticmethod
    def _articulation(backend):
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=2,
            num_bodies=2,
            num_fixed_tendons=2,
            num_spatial_tendons=0,
            device="cpu",
        )
        return art

    @_production_backends
    def test_commanding_a_target_schedules_the_write(self, backend):
        """The target setter marks the write pending; a static offset write keeps its own contract."""
        if backend == "newton":
            pytest.skip("Newton needs a MuJoCo tendon actuator, covered by the delegation tests below.")
        art = self._articulation(backend)
        art.set_fixed_tendon_position_target_index(target=torch.zeros((2, 2), dtype=torch.float32))
        assert art._fixed_tendon_target_dirty is True

    def test_newton_reports_a_missing_tendon_actuator(self):
        """Without a MuJoCo tendon actuator the solver has no adapter, so commanding must say so."""
        if "newton" not in BACKENDS:
            pytest.skip(BACKEND_UNAVAILABLE_REASONS.get("newton", "newton backend unavailable"))
        art = self._articulation("newton")
        with pytest.raises(RuntimeError, match="no MuJoCo tendon actuator"):
            art.set_fixed_tendon_position_target_index(target=torch.zeros((2, 2), dtype=torch.float32))

    def test_a_solver_without_tendon_transmission_refuses_to_build_an_adapter(self):
        """A solver with no tendon transmission says so, rather than returning nothing."""
        if "newton" not in BACKENDS:
            pytest.skip(BACKEND_UNAVAILABLE_REASONS.get("newton", "newton backend unavailable"))
        from isaaclab_newton.physics import NewtonManager

        art = self._articulation("newton")
        with pytest.raises(NotImplementedError, match="does not drive fixed tendons"):
            NewtonManager.create_fixed_tendon_control(art)


# ---------------------------------------------------------------------------
# Tests: resolve_matching_names caching behavior
# ---------------------------------------------------------------------------


class TestResolveMatchingNamesCache:
    """Test that resolve_matching_names caching returns correct, isolated results."""

    @_backends
    def test_unmatched_regex_raises(self, backend):
        """ValueError from resolve_matching_names propagates correctly."""
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device="cpu")
        with pytest.raises(ValueError):
            art.find_bodies("nonexistent_body_xyz")
        with pytest.raises(ValueError):
            art.find_joints("nonexistent_joint_xyz")

    @_backends
    def test_mutating_result_does_not_corrupt_cache(self, backend):
        """Mutating returned lists must not affect future cached results."""
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device="cpu")

        for finder, expected_len in [("find_bodies", _NUM_BODIES), ("find_joints", _NUM_JOINTS)]:
            idx1, names1 = getattr(art, finder)(".*")
            assert len(idx1) == expected_len

            idx1.clear()
            names1.append("corrupted")

            idx2, names2 = getattr(art, finder)(".*")
            assert len(idx2) == expected_len
            assert "corrupted" not in names2

    @_backends
    def test_find_with_preserve_order(self, backend):
        """A list of patterns resolves each name; preserve_order=True keeps the pattern order."""
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device="cpu")
        idx_fwd, names_fwd = art.find_joints(["joint_1", "joint_0"], preserve_order=True)
        assert names_fwd == ["joint_1", "joint_0"]

        idx_rev, names_rev = art.find_joints(["joint_0", "joint_1"], preserve_order=True)
        assert names_rev == ["joint_0", "joint_1"]


# ---------------------------------------------------------------------------
# Tests: ArticulationData property contract
# ---------------------------------------------------------------------------

# (property, shape kind, dtype). Shape kinds: "N" = (num_instances,), "NB" = (num_instances, num_bodies),
# "NB9" = (num_instances, num_bodies, 9), "NJ" = (num_instances, num_joints).
_ARTICULATION_DATA_PROPERTIES = [
    # root state
    ("root_link_pose_w", "N", wp.transformf),
    ("root_link_vel_w", "N", wp.spatial_vectorf),
    ("root_com_pose_w", "N", wp.transformf),
    ("root_com_vel_w", "N", wp.spatial_vectorf),
    ("root_link_pos_w", "N", wp.vec3f),
    ("root_link_quat_w", "N", wp.quatf),
    ("root_link_lin_vel_w", "N", wp.vec3f),
    ("root_link_ang_vel_w", "N", wp.vec3f),
    ("root_com_pos_w", "N", wp.vec3f),
    ("root_com_quat_w", "N", wp.quatf),
    ("root_com_lin_vel_w", "N", wp.vec3f),
    ("root_com_ang_vel_w", "N", wp.vec3f),
    # derived
    ("projected_gravity_b", "N", wp.vec3f),
    ("heading_w", "N", wp.float32),
    ("root_link_lin_vel_b", "N", wp.vec3f),
    ("root_link_ang_vel_b", "N", wp.vec3f),
    ("root_com_lin_vel_b", "N", wp.vec3f),
    ("root_com_ang_vel_b", "N", wp.vec3f),
    # body state
    ("body_link_pose_w", "NB", wp.transformf),
    ("body_link_vel_w", "NB", wp.spatial_vectorf),
    ("body_com_pose_w", "NB", wp.transformf),
    ("body_com_vel_w", "NB", wp.spatial_vectorf),
    ("body_com_acc_w", "NB", wp.spatial_vectorf),
    ("body_com_pose_b", "NB", wp.transformf),
    ("body_mass", "NB", wp.float32),
    ("body_inertia", "NB9", wp.float32),
    ("body_link_pos_w", "NB", wp.vec3f),
    ("body_link_quat_w", "NB", wp.quatf),
    ("body_link_lin_vel_w", "NB", wp.vec3f),
    ("body_link_ang_vel_w", "NB", wp.vec3f),
    ("body_com_pos_w", "NB", wp.vec3f),
    ("body_com_quat_w", "NB", wp.quatf),
    ("body_com_pos_b", "NB", wp.vec3f),
    ("body_com_quat_b", "NB", wp.quatf),
    # joint state and properties
    ("joint_pos", "NJ", wp.float32),
    ("joint_vel", "NJ", wp.float32),
    ("joint_acc", "NJ", wp.float32),
    ("joint_stiffness", "NJ", wp.float32),
    ("joint_damping", "NJ", wp.float32),
    ("joint_armature", "NJ", wp.float32),
    ("joint_friction_coeff", "NJ", wp.float32),
    ("joint_pos_limits", "NJ", wp.vec2f),
    ("joint_vel_limits", "NJ", wp.float32),
    ("joint_effort_limits", "NJ", wp.float32),
    ("soft_joint_pos_limits", "NJ", wp.vec2f),
    # defaults and command targets
    ("default_root_pose", "N", wp.transformf),
    ("default_root_vel", "N", wp.spatial_vectorf),
    ("default_joint_pos", "NJ", wp.float32),
    ("default_joint_vel", "NJ", wp.float32),
    ("joint_pos_target", "NJ", wp.float32),
    ("joint_vel_target", "NJ", wp.float32),
    ("joint_effort_target", "NJ", wp.float32),
]


class TestArticulationDataProperties:
    """Test that every data property is a ProxyArray with the advertised shape and dtype."""

    @_backends
    @_devices
    def test_articulation_data_property_contract(self, backend, device):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        art.data.update(dt=0.01)
        shapes = {
            "N": (_NUM_INSTANCES,),
            "NB": (_NUM_INSTANCES, _NUM_BODIES),
            "NB9": (_NUM_INSTANCES, _NUM_BODIES, 9),
            "NJ": (_NUM_INSTANCES, _NUM_JOINTS),
        }
        for name, shape_kind, dtype in _ARTICULATION_DATA_PROPERTIES:
            if backend == "newton" and name == "body_com_pose_b":
                # Newton stores a position-only COM and warns that the pose appends a unit quaternion.
                with pytest.warns(UserWarning, match="unit quaternion"):
                    value = getattr(art.data, name)
            else:
                value = getattr(art.data, name)
            _check_proxy_array(value, expected_shape=shapes[shape_kind], expected_dtype=dtype, name=name)

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

    @_production_backends
    def test_actuator_compatibility_projections_are_stable(self, backend):
        num_instances, num_joints = 2, 4
        art, _ = get_articulation(backend, num_instances, num_joints, 5, device="cpu")
        soft_joint_vel_limits = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], dtype=torch.float32)
        wp.copy(art.actuators._soft_joint_vel_limits, wp.from_torch(soft_joint_vel_limits))

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            soft_joint_vel_limits_data = art.data.soft_joint_vel_limits
            soft_joint_vel_limits_repeat = art.data.soft_joint_vel_limits

        _check_proxy_array(
            soft_joint_vel_limits_data,
            expected_shape=(num_instances, num_joints),
            expected_dtype=wp.float32,
            name="soft_joint_vel_limits",
        )
        torch.testing.assert_close(soft_joint_vel_limits_data.torch, soft_joint_vel_limits, rtol=0.0, atol=0.0)
        assert soft_joint_vel_limits_data.warp.ptr == soft_joint_vel_limits_repeat.warp.ptr
        assert soft_joint_vel_limits_data.warp.ptr == art.actuators._soft_joint_vel_limits.ptr
        assert not [warning for warning in caught_warnings if warning.category is DeprecationWarning]


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------

_ARTICULATION_ALIASES = [
    ("root_pose_w", "root_link_pose_w"),
    ("root_pos_w", "root_link_pos_w"),
    ("root_quat_w", "root_link_quat_w"),
    ("root_vel_w", "root_com_vel_w"),
    ("root_lin_vel_w", "root_com_lin_vel_w"),
    ("root_ang_vel_w", "root_com_ang_vel_w"),
    ("body_pose_w", "body_link_pose_w"),
    ("body_pos_w", "body_link_pos_w"),
    ("body_quat_w", "body_link_quat_w"),
    ("body_vel_w", "body_com_vel_w"),
    ("body_lin_vel_w", "body_com_lin_vel_w"),
    ("body_ang_vel_w", "body_com_ang_vel_w"),
    ("joint_limits", "joint_pos_limits"),
    ("joint_friction", "joint_friction_coeff"),
]


class TestArticulationDataAliases:
    """Test that alias properties return the values of their canonical counterparts."""

    @_backends
    def test_aliases_match_canonical_values(self, backend):
        # Random mock state makes link and COM quantities differ, so a retargeted alias fails.
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device="cpu")
        art.data.update(dt=0.01)
        d = art.data

        for alias, canonical in _ARTICULATION_ALIASES:
            alias_value, canonical_value = getattr(d, alias), getattr(d, canonical)
            assert alias_value.shape == canonical_value.shape, alias
            assert alias_value.dtype == canonical_value.dtype, alias
            assert torch.equal(alias_value.torch, canonical_value.torch), alias


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


def _make_payload_torch(shape: tuple, device: str, wp_dtype=wp.float32) -> torch.Tensor:
    """Create valid torch data whose entries differ per element, for writer read-back checks.

    Transforms get distinct positions and a fixed 90-degree rotation about Z; ``vec2f`` limits get
    distinct ``[-v, v]`` pairs.
    """
    count = math.prod(shape)
    values = torch.arange(1, count + 1, dtype=torch.float32, device=device).reshape(shape) + 0.25
    if wp_dtype == wp.float32:
        return values
    if wp_dtype == wp.vec2f:
        return torch.stack((-values, values), dim=-1)
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
# Tests: Root writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# writer suffix -> data getter that reads the written quantity back
_ROOT_POSE_METHODS = {
    "root_pose": "root_link_pose_w",
    "root_link_pose": "root_link_pose_w",
    "root_com_pose": "root_com_pose_w",
}
_ROOT_VEL_METHODS = {
    "root_velocity": "root_com_vel_w",
    "root_link_velocity": "root_link_vel_w",
    "root_com_velocity": "root_com_vel_w",
}


class TestArticulationWritersRoot:
    """Test root pose/velocity writers with all input combinations."""

    @_production_backends
    @pytest.mark.parametrize(
        "body_ordering",
        [None, ("body_0", "body_3", "body_2", "body_1")],
    )
    def test_pose_write_invalidates_pose_dependent_caches(self, backend, body_ordering):
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=3,
            num_bodies=4,
            device="cpu",
            body_ordering=body_ordering,
        )
        art.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            art.data,
            [
                ("root_link_vel_w", "_root_link_vel_w"),
                ("body_link_vel_w", "_body_link_vel_w"),
                *([("body_com_vel_w", "_body_com_vel_w")] if backend != "newton" else []),
                ("projected_gravity_b", "_projected_gravity_b"),
                ("heading_w", "_heading_w"),
                ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
                ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
                ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
                ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
            ],
        )
        if backend == "ovphysx" and art.data._body_com_vel_w_backend is not None:
            buffers.append(("_body_com_vel_w_backend", art.data._body_com_vel_w_backend))
        root_pose = _make_data_warp((art.num_instances,), "cpu", wp.transformf)
        art.write_root_link_pose_to_sim_index(root_pose=root_pose)
        _assert_buffers_stale(art.data, buffers)

    @_production_backends
    @pytest.mark.parametrize(
        "body_ordering",
        [None, ("body_0", "body_3", "body_2", "body_1")],
    )
    def test_velocity_write_invalidates_body_frame_caches(self, backend, body_ordering):
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=3,
            num_bodies=4,
            device="cpu",
            body_ordering=body_ordering,
        )
        art.data.update(dt=0.01)
        buffers = _prime_timestamped_properties(
            art.data,
            [
                ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
                ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
                ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
                ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
            ],
        )
        root_velocity = _make_data_warp((art.num_instances,), "cpu", wp.spatial_vectorf)
        art.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)
        _assert_buffers_stale(art.data, buffers)

    @_production_backends
    @pytest.mark.parametrize("setter_kind", ["index", "mask"])
    @pytest.mark.parametrize(
        "body_ordering",
        [None, ("body_0", "body_3", "body_2", "body_1")],
    )
    def test_set_coms_invalidates_same_timestamp_dependents(self, backend, setter_kind, body_ordering):
        art, _ = get_articulation(
            backend,
            num_instances=2,
            num_joints=3,
            num_bodies=4,
            device="cpu",
            body_ordering=body_ordering,
        )
        art.data.update(dt=0.01)
        common_pairs = [
            ("root_com_pose_w", "_root_com_pose_w"),
            ("root_link_vel_w", "_root_link_vel_w"),
            ("body_com_pose_w", "_body_com_pose_w"),
            ("body_link_vel_w", "_body_link_vel_w"),
            ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
            ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
            ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
            ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
        ]
        if backend != "newton":
            common_pairs += [
                ("root_com_vel_w", "_root_com_vel_w"),
                ("body_com_vel_w", "_body_com_vel_w"),
            ]
        if backend == "physx":
            # COM changes also move the COM Jacobian and the generalized dynamics.
            common_pairs += [
                ("body_com_jacobian_w", "_body_com_jacobian_w"),
                ("mass_matrix", "_mass_matrix"),
                ("gravity_compensation_forces", "_gravity_compensation_forces"),
            ]
        state_buffer_suffix = "_buf" if backend == "ovphysx" else ""
        common_pairs += [
            ("root_state_w", f"_root_state_w{state_buffer_suffix}"),
            ("root_link_state_w", f"_root_link_state_w{state_buffer_suffix}"),
            ("root_com_state_w", f"_root_com_state_w{state_buffer_suffix}"),
            ("body_state_w", f"_body_state_w{state_buffer_suffix}"),
            ("body_link_state_w", f"_body_link_state_w{state_buffer_suffix}"),
            ("body_com_state_w", f"_body_com_state_w{state_buffer_suffix}"),
        ]
        # Prime public properties before resolving private buffers so Newton allocates lazy caches.
        buffers = _prime_timestamped_properties(art.data, common_pairs)
        if backend == "newton":
            buffers += _prime_timestamped_properties(art.data, [("body_com_pose_b", "_body_com_pose_b")])
        if backend == "ovphysx" and art.data._body_com_vel_w_backend is not None:
            art.data._body_com_vel_w_backend.timestamp = art.data._sim_timestamp
            buffers.append(("_body_com_vel_w_backend", art.data._body_com_vel_w_backend))
        coms = _make_com_data(backend, (art.num_instances, art.num_bodies), "cpu")

        def set_coms() -> None:
            if setter_kind == "index":
                kwargs = {} if backend != "physx" else {"full_data": True}
                art.set_coms_index(coms=coms, **kwargs)
            else:
                art.set_coms_mask(coms=coms)

        if backend == "newton":
            from unittest.mock import patch

            from isaaclab_newton.physics import NewtonManager

            with patch.object(NewtonManager, "add_model_change"):
                set_coms()
        else:
            set_coms()
        _assert_buffers_stale(art.data, buffers)

    # -- index variants --

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_index(self, backend, device, method_suffix):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances = _NUM_INSTANCES
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, subset
        method(root_pose=_make_data_torch((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # warp, all envs: the matching getter reads the written poses back
        method(root_pose=_make_payload_warp((num_instances,), device, wp.transformf))
        getter = _ROOT_POSE_METHODS[method_suffix]
        _assert_reads_back(
            getattr(art.data, getter), _make_payload_torch((num_instances,), device, wp.transformf), getter
        )
        # warp, subset
        method(root_pose=_make_data_warp((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))

    @_backends
    @_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_index(self, backend, device, method_suffix):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances = _NUM_INSTANCES
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_index")

        # torch, all envs
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, subset
        method(root_velocity=_make_data_torch((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
        # warp, all envs: the matching getter reads the written velocities back
        method(root_velocity=_make_payload_warp((num_instances,), device, wp.spatial_vectorf))
        getter = _ROOT_VEL_METHODS[method_suffix]
        _assert_reads_back(
            getattr(art.data, getter), _make_payload_torch((num_instances,), device, wp.spatial_vectorf), getter
        )
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
    @_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
    def test_write_root_pose_to_sim_mask(self, backend, device, method_suffix):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances = _NUM_INSTANCES
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask (all)
        method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
        # torch, partial mask
        method(
            root_pose=_make_data_torch((num_instances,), device, wp.transformf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask: the matching getter reads the written poses back
        method(root_pose=_make_payload_warp((num_instances,), device, wp.transformf))
        getter = _ROOT_POSE_METHODS[method_suffix]
        _assert_reads_back(
            getattr(art.data, getter), _make_payload_torch((num_instances,), device, wp.transformf), getter
        )
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
    @_devices
    @pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
    def test_write_root_velocity_to_sim_mask(self, backend, device, method_suffix):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances = _NUM_INSTANCES
        art.data.update(dt=0.01)
        method = getattr(art, f"write_{method_suffix}_to_sim_mask")

        # torch, no mask
        method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
        # torch, partial mask
        method(
            root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf),
            env_mask=_make_env_mask(num_instances, device, True),
        )
        # warp, no mask: the matching getter reads the written velocities back
        method(root_velocity=_make_payload_warp((num_instances,), device, wp.spatial_vectorf))
        getter = _ROOT_VEL_METHODS[method_suffix]
        _assert_reads_back(
            getattr(art.data, getter), _make_payload_torch((num_instances,), device, wp.spatial_vectorf), getter
        )
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


@pytest.mark.parametrize("backend", BACKENDS)
def test_deprecated_joint_friction_writers(backend):
    """The deprecated joint friction writers warn and forward to the index writer on every backend."""
    num_instances, num_joints = 2, 4
    art, _ = get_articulation(backend, num_instances, num_joints, 5, device="cpu")
    for writer_name, value in (("write_joint_friction_coefficient_to_sim", 0.5), ("write_joint_friction_to_sim", 0.25)):
        friction = torch.full((num_instances, num_joints), value)
        with pytest.warns(DeprecationWarning):
            getattr(art, writer_name)(friction)
        _assert_reads_back(art.data.joint_friction_coeff, friction, writer_name)


# ---------------------------------------------------------------------------
# Tests: Joint writers — torch/warp × index/mask × all/subset × negative
# ---------------------------------------------------------------------------

# (method_name, kwarg_name, wp_dtype, accepts_float, read-back getter)
_JOINT_METHODS = [
    ("write_joint_position_to_sim", "position", wp.float32, False, "joint_pos"),
    ("write_joint_velocity_to_sim", "velocity", wp.float32, False, "joint_vel"),
    ("write_joint_stiffness_to_sim", "stiffness", wp.float32, True, "joint_stiffness"),
    ("write_joint_damping_to_sim", "damping", wp.float32, True, "joint_damping"),
    ("write_joint_position_limit_to_sim", "limits", wp.vec2f, True, "joint_pos_limits"),
    ("write_joint_velocity_limit_to_sim", "limits", wp.float32, True, "joint_vel_limits"),
    ("write_joint_effort_limit_to_sim", "limits", wp.float32, True, "joint_effort_limits"),
    ("write_joint_armature_to_sim", "armature", wp.float32, True, "joint_armature"),
    ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", wp.float32, False, "joint_friction_coeff"),
    ("set_joint_position_target", "target", wp.float32, False, "actuators.target_command.position"),
    ("set_joint_velocity_target", "target", wp.float32, False, "actuators.target_command.velocity"),
    ("set_joint_effort_target", "target", wp.float32, False, "actuators.target_command.effort"),
]


def _read_joint_writer_target(art, getter: str):
    """Resolve a read-back getter: a data property, or a dotted path on the articulation."""
    if "." not in getter:
        return getattr(art.data, getter)
    value = art
    for attr in getter.split("."):
        value = getattr(value, attr)
    return value


class TestArticulationWritersJoint:
    """Test joint writers/setters with all input combinations."""

    @_backends
    @_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float, getter",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_index(self, backend, device, method_base, kwarg, wp_dtype, accepts_float, getter):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances, num_joints = _NUM_INSTANCES, _NUM_JOINTS
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
        # warp, all envs + all joints: the matching getter reads the written values back
        method(**{kwarg: _make_payload_warp((num_instances, num_joints), device, wp_dtype)})
        _assert_reads_back(
            _read_joint_writer_target(art, getter),
            _make_payload_torch((num_instances, num_joints), device, wp_dtype),
            getter,
        )
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
    @_devices
    @pytest.mark.parametrize(
        "method_base, kwarg, wp_dtype, accepts_float, getter",
        _JOINT_METHODS,
        ids=[m[0] for m in _JOINT_METHODS],
    )
    def test_joint_writer_mask(self, backend, device, method_base, kwarg, wp_dtype, accepts_float, getter):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances, num_joints = _NUM_INSTANCES, _NUM_JOINTS
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
        # warp, no mask: the matching getter reads the written values back
        method(**{kwarg: _make_payload_warp((num_instances, num_joints), device, wp_dtype)})
        _assert_reads_back(
            _read_joint_writer_target(art, getter),
            _make_payload_torch((num_instances, num_joints), device, wp_dtype),
            getter,
        )
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

# (method_name, kwarg_name, read-back getter)
_BODY_METHODS = [
    ("set_masses", "masses", "body_mass"),
    ("set_coms", "coms", "body_com_pose_b"),
    ("set_inertias", "inertias", "body_inertia"),
]


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


class TestArticulationWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_backends
    @_devices
    @pytest.mark.parametrize("method_base, kwarg", [m[:2] for m in _BODY_METHODS], ids=[m[0] for m in _BODY_METHODS])
    def test_body_writer_index(self, backend, device, method_base, kwarg):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        wp_dtype, trailing, getter = _body_writer_layout(backend, method_base)
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_index")
        sub_b = min(2, num_bodies)
        sub_body_ids = list(range(sub_b))

        # torch, all envs + all bodies
        method(**{kwarg: _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing)})
        # torch, subset
        method(
            **{
                kwarg: _make_body_torch((1, sub_b), device, wp_dtype, trailing),
                "body_ids": sub_body_ids,
                "env_ids": _make_env_ids(device, True),
            }
        )
        # warp, all envs + all bodies: the matching getter reads the written values back
        method(**{kwarg: _make_body_warp((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)})
        expected = _make_body_torch((num_instances, num_bodies), device, wp_dtype, trailing, payload=True)
        _assert_reads_back(getattr(art.data, getter), expected, getter)
        # warp, subset
        method(
            **{
                kwarg: _make_body_warp((1, sub_b), device, wp_dtype, trailing),
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
    @pytest.mark.parametrize("method_base, kwarg", [m[:2] for m in _BODY_METHODS], ids=[m[0] for m in _BODY_METHODS])
    def test_body_writer_mask(self, backend, device, method_base, kwarg):
        art, _ = get_articulation(backend, _NUM_INSTANCES, _NUM_JOINTS, _NUM_BODIES, device=device)
        num_instances, num_bodies = _NUM_INSTANCES, _NUM_BODIES
        wp_dtype, trailing, getter = _body_writer_layout(backend, method_base)
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_mask")
        sub_body_sel = list(range(min(2, num_bodies)))

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
        _assert_reads_back(getattr(art.data, getter), expected, getter)
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
# Tendon tests — counts, names, data, writers
# ---------------------------------------------------------------------------

# Newton's fixed-tendon solver integration is covered in isaaclab_newton/test/assets/test_articulation.py.
# This fixture matrix also requires spatial tendons and PhysX-specific properties.
_tendon_backends = pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "newton"], indirect=False)

# The first row covers the zero-spatial-tendon edge; the second has distinct counts on every axis.
_tendon_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons",
    [
        (1, 2, 2, 1, 0),  # fixed only
        (2, 6, 7, 3, 2),  # both types
    ],
)

# (property, tendon kind, dtype)
_TENDON_DATA_PROPERTIES = [
    ("fixed_tendon_stiffness", "fixed", wp.float32),
    ("fixed_tendon_damping", "fixed", wp.float32),
    ("fixed_tendon_limit_stiffness", "fixed", wp.float32),
    ("fixed_tendon_rest_length", "fixed", wp.float32),
    ("fixed_tendon_offset", "fixed", wp.float32),
    ("fixed_tendon_pos_limits", "fixed", wp.vec2f),
    ("spatial_tendon_stiffness", "spatial", wp.float32),
    ("spatial_tendon_damping", "spatial", wp.float32),
    ("spatial_tendon_limit_stiffness", "spatial", wp.float32),
    ("spatial_tendon_offset", "spatial", wp.float32),
]


class TestArticulationTendons:
    """Test tendon counts, names, and data properties."""

    @_tendon_backends
    @_tendon_dims
    @_devices
    def test_articulation_tendon_contract(
        self, backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
    ):
        art, _ = get_articulation(
            backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
        )
        counts = {"fixed": num_fixed_tendons, "spatial": num_spatial_tendons}
        assert art.num_fixed_tendons == num_fixed_tendons
        assert art.num_spatial_tendons == num_spatial_tendons
        for names, count in (
            (art.fixed_tendon_names, num_fixed_tendons),
            (art.spatial_tendon_names, num_spatial_tendons),
        ):
            assert isinstance(names, list)
            assert len(names) == count
            assert all(isinstance(n, str) for n in names)

        art.data.update(dt=0.01)
        for name, kind, dtype in _TENDON_DATA_PROPERTIES:
            if counts[kind] == 0:
                continue
            expected_shape = (num_instances, counts[kind])
            if backend == "physx" and name == "fixed_tendon_pos_limits":
                # Known inconsistency: PhysX exposes (N, T, 2) float32 although its docstring and OVPhysX
                # advertise (N, T) vec2f. Pin the current layout so a change on either side is noticed.
                expected_shape, dtype = (*expected_shape, 2), wp.float32
            _check_proxy_array(getattr(art.data, name), expected_shape=expected_shape, expected_dtype=dtype, name=name)


# (method_name, kwarg_name, read-back data property)
_FIXED_TENDON_METHODS = [
    ("set_fixed_tendon_stiffness", "stiffness", "fixed_tendon_stiffness"),
    ("set_fixed_tendon_damping", "damping", "fixed_tendon_damping"),
    ("set_fixed_tendon_limit_stiffness", "limit_stiffness", "fixed_tendon_limit_stiffness"),
    ("set_fixed_tendon_rest_length", "rest_length", "fixed_tendon_rest_length"),
    ("set_fixed_tendon_offset", "offset", "fixed_tendon_offset"),
    ("set_fixed_tendon_position_limit", "limit", "fixed_tendon_pos_limits"),
]
_SPATIAL_TENDON_METHODS = [
    ("set_spatial_tendon_stiffness", "stiffness", "spatial_tendon_stiffness"),
    ("set_spatial_tendon_damping", "damping", "spatial_tendon_damping"),
    ("set_spatial_tendon_limit_stiffness", "limit_stiffness", "spatial_tendon_limit_stiffness"),
    ("set_spatial_tendon_offset", "offset", "spatial_tendon_offset"),
]
_TENDON_METHODS = [("fixed", *m) for m in _FIXED_TENDON_METHODS] + [("spatial", *m) for m in _SPATIAL_TENDON_METHODS]


class TestArticulationWritersTendon:
    """Test tendon writers/setters with all input combinations."""

    @_tendon_backends
    @_tendon_dims
    @_devices
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("kind, method_base, kwarg, getter", _TENDON_METHODS, ids=[m[1] for m in _TENDON_METHODS])
    def test_tendon_writer(
        self,
        backend,
        num_instances,
        num_joints,
        num_bodies,
        num_fixed_tendons,
        num_spatial_tendons,
        device,
        selection,
        kind,
        method_base,
        kwarg,
        getter,
    ):
        num_tendons = num_fixed_tendons if kind == "fixed" else num_spatial_tendons
        if num_tendons == 0:
            pytest.skip(f"No {kind} tendons configured")
        art, _ = get_articulation(
            backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device
        )
        art.data.update(dt=0.01)
        method = getattr(art, f"{method_base}_{selection}")
        wp_dtype = wp.vec2f if getter == "fixed_tendon_pos_limits" else wp.float32
        full_shape = (num_instances, num_tendons)
        sub_t = min(2, num_tendons)
        sub_tendons = list(range(sub_t))
        if selection == "index":
            subset = {f"{kind}_tendon_ids": sub_tendons, "env_ids": _make_env_ids(device, True)}
            subset_shape = (1, sub_t)
        else:
            subset = {
                f"{kind}_tendon_mask": _make_item_mask(num_tendons, sub_tendons, device),
                "env_mask": _make_env_mask(num_instances, device, True),
            }
            subset_shape = full_shape

        # torch, all envs + all tendons
        method(**{kwarg: _make_data_torch(full_shape, device, wp_dtype)})
        # torch, subset
        method(**{kwarg: _make_data_torch(subset_shape, device, wp_dtype)}, **subset)
        # warp, all envs + all tendons: the matching data property reads the written values back
        method(**{kwarg: _make_payload_warp(full_shape, device, wp_dtype)})
        _assert_reads_back(getattr(art.data, getter), _make_payload_torch(full_shape, device, wp_dtype), getter)
        # warp, subset
        method(**{kwarg: _make_data_warp(subset_shape, device, wp_dtype)}, **subset)
        # Position limits require a lower/upper pair; other properties accept scalar values.
        if wp_dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        else:
            method(**{kwarg: 1.0})
        # negative: bad torch shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_torch(full_shape, device, wp_dtype)})
        # negative: bad warp shape
        with pytest.raises((AssertionError, RuntimeError)):
            method(**{kwarg: _make_bad_data_warp(full_shape, device, wp_dtype)})

    @_tendon_backends
    @_devices
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("kind", ["fixed", "spatial"])
    def test_write_tendon_properties_to_sim_selects_envs(self, backend, device, selection, kind):
        """Pushing tendon properties writes the selected environments to the backend."""
        num_instances = 3
        art, raw_backend = get_articulation(backend, num_instances, 2, 2, 2, 2, device)
        art.data.update(dt=0.01)
        env_writes = []
        if backend == "physx":

            def capture(*args, indices=None, **kwargs):
                env_writes.append(indices.numpy().tolist())

            setattr(raw_backend, f"set_{kind}_tendon_properties", MagicMock(side_effect=capture))
        else:
            from isaaclab_ov import tensor_types as TT

            prefix = "FIXED_TENDON" if kind == "fixed" else "SPATIAL_TENDON"
            tendon_types = {getattr(TT, name) for name in dir(TT) if name.startswith(prefix)}
            set_attribute = art._root_view.set_attribute

            def capture(name, values, *, indices=None, mask=None):
                if name in tendon_types:
                    if mask is not None:
                        env_writes.append(np.flatnonzero(mask.numpy()).tolist())
                    else:
                        env_writes.append(list(range(num_instances)) if indices is None else indices.numpy().tolist())
                set_attribute(name, values, indices=indices, mask=mask)

            object.__setattr__(art._root_view, "set_attribute", capture)
        write = getattr(art, f"write_{kind}_tendon_properties_to_sim_{selection}")

        for selected_envs in (list(range(num_instances)), [0, 2]):
            env_writes.clear()
            if selection == "index":
                write(
                    env_ids=None if len(selected_envs) == num_instances else torch.tensor(selected_envs, device=device)
                )
            else:
                mask = None
                if len(selected_envs) < num_instances:
                    mask = wp.array([i in selected_envs for i in range(num_instances)], dtype=wp.bool, device=device)
                write(env_mask=mask)
            assert env_writes, "no tendon properties were written"
            assert all(envs == selected_envs for envs in env_writes), env_writes
