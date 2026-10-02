# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared rigid-object contracts across the production backends.

Checks that every rigid-object backend provides the data and writer behavior the base rigid-object class advertises.
The backends run on mocked views, so these cases need neither Isaac Sim nor a GPU simulation.
"""

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import warp as wp

from .rigid_object_factory import get_rigid_object

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
_NUM_INSTANCES = 2


# ---------------------------------------------------------------------------
# Tests: Index resolution helpers
# ---------------------------------------------------------------------------


def test_resolve_env_ids_handles_tensor_view_shape(monkeypatch, backend, device):
    obj, _ = get_rigid_object(backend, num_instances=4, device=device, monkeypatch=monkeypatch)

    env_ids = torch.arange(4, dtype=torch.int32, device=device)
    resolved_full = obj._resolve_env_ids(env_ids)
    resolved_view = obj._resolve_env_ids(env_ids[:2])

    assert resolved_full.shape[0] == 4
    assert resolved_view.shape[0] == 2
    cached = wp.to_torch(obj._ALL_INDICES)
    for selection in (slice(None), slice(1, None, 2), slice(0, 0)):
        resolved = wp.to_torch(obj._resolve_env_ids(selection))
        torch.testing.assert_close(resolved, cached[selection])
        assert resolved.data_ptr() == cached[selection].data_ptr()
        assert resolved.stride() == cached[selection].stride()


# ---------------------------------------------------------------------------
# Tests: RigidObject properties and finders
# ---------------------------------------------------------------------------


def test_rigid_object_counts_and_names(monkeypatch, backend):
    from isaaclab.assets.rigid_object.base_rigid_object_data import BaseRigidObjectData

    obj, _ = get_rigid_object(backend, _NUM_INSTANCES, device="cpu", monkeypatch=monkeypatch)

    assert obj.num_instances == _NUM_INSTANCES
    assert obj.num_bodies == 1
    names = obj.body_names
    assert isinstance(names, list)
    assert len(names) == 1
    assert all(isinstance(n, str) for n in names)
    assert isinstance(obj.data, BaseRigidObjectData)


def test_find_bodies_returns_legacy_list_or_cached_proxy(monkeypatch, backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu", monkeypatch=monkeypatch)

    indices, names = obj.find_bodies(".*")
    proxy, proxy_names = obj.find_bodies(".*", as_proxy=True)

    assert isinstance(indices, list) and isinstance(names, list)
    assert len(indices) == 1
    assert len(names) == 1
    assert all(isinstance(i, int) for i in indices)
    assert all(isinstance(n, str) for n in names)
    assert indices == proxy.torch.tolist()
    assert names == proxy_names
    assert proxy is obj.find_bodies(".*", as_proxy=True)[0]
    assert proxy.dtype == wp.int32
    assert str(proxy.device) == obj.device

    first_body = obj.body_names[0]
    assert obj.find_bodies(first_body) == ([0], [first_body])


# ---------------------------------------------------------------------------
# Tests: RigidObjectData property contract
# ---------------------------------------------------------------------------

# (property, shape kind, dtype). Shape kinds: "N" = (num_instances,), "N1" = (num_instances, 1),
# "N19" = (num_instances, 1, 9).
_RIGID_OBJECT_DATA_PROPERTIES = [
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
    ("body_link_pose_w", "N1", wp.transformf),
    ("body_link_vel_w", "N1", wp.spatial_vectorf),
    ("body_com_pose_w", "N1", wp.transformf),
    ("body_com_vel_w", "N1", wp.spatial_vectorf),
    ("body_com_acc_w", "N1", wp.spatial_vectorf),
    ("body_com_pose_b", "N1", wp.transformf),
    ("body_mass", "N1", wp.float32),
    ("body_inertia", "N19", wp.float32),
    ("body_link_pos_w", "N1", wp.vec3f),
    ("body_link_quat_w", "N1", wp.quatf),
    ("body_link_lin_vel_w", "N1", wp.vec3f),
    ("body_link_ang_vel_w", "N1", wp.vec3f),
    ("body_com_pos_w", "N1", wp.vec3f),
    ("body_com_quat_w", "N1", wp.quatf),
    ("body_com_pos_b", "N1", wp.vec3f),
    ("body_com_quat_b", "N1", wp.quatf),
    # defaults
    ("default_root_pose", "N", wp.transformf),
    ("default_root_vel", "N", wp.spatial_vectorf),
]


def test_rigid_object_data_property_contract(monkeypatch, backend, device):
    obj, _ = get_rigid_object(backend, _NUM_INSTANCES, device, monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    if backend == "newton":
        for name in ("root_link_vel_w", "root_com_pose_w", "body_com_pose_b", "projected_gravity_b", "heading_w"):
            assert getattr(obj.data, "_" + name).data is None
    shapes = {"N": (_NUM_INSTANCES,), "N1": (_NUM_INSTANCES, 1), "N19": (_NUM_INSTANCES, 1, 9)}
    for name, shape_kind, dtype in _RIGID_OBJECT_DATA_PROPERTIES:
        _check_proxy_array(getattr(obj.data, name), expected_shape=shapes[shape_kind], expected_dtype=dtype, name=name)

    for frame in ("root_link", "root_com", "body_link", "body_com"):
        for quantity, components in (("pose", ("pos", "quat")), ("vel", ("lin_vel", "ang_vel"))):
            packed = getattr(obj.data, f"{frame}_{quantity}_w").torch
            for component, expected in zip(components, (packed[..., :3], packed[..., 3:]), strict=True):
                view = getattr(obj.data, f"{frame}_{component}_w").torch
                torch.testing.assert_close(view, expected)
                assert view.data_ptr() == expected.data_ptr()
                assert view.stride() == expected.stride()


# ---------------------------------------------------------------------------
# Tests: Alias/shorthand properties
# ---------------------------------------------------------------------------

_RIGID_OBJECT_ALIASES = [
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
]


def test_aliases_match_canonical_values(monkeypatch, backend):
    # Random mock state makes link and COM quantities differ, so a retargeted alias fails.
    obj, _ = get_rigid_object(backend, _NUM_INSTANCES, device="cpu", monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    d = obj.data

    for alias, canonical in _RIGID_OBJECT_ALIASES:
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


def test_pose_write_invalidates_pose_dependent_caches(monkeypatch, backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu", monkeypatch=monkeypatch)
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


def test_velocity_write_invalidates_body_frame_caches(monkeypatch, backend):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu", monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    body_velocity = obj.data.body_link_vel_w
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
    # Read the body alias first; zero angular velocity makes link and COM velocities equal.
    _assert_reads_back(obj.data.body_link_vel_w, wp.to_torch(root_velocity).unsqueeze(1), "body_link_vel_w")
    assert obj.data.body_link_vel_w is body_velocity


@pytest.mark.parametrize("setter_kind", ["index", "mask"])
def test_set_coms_invalidates_same_timestamp_dependents(monkeypatch, backend, setter_kind):
    obj, _ = get_rigid_object(backend, num_instances=2, device="cpu", monkeypatch=monkeypatch)
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
        from isaaclab_newton.physics import NewtonManager

        with patch.object(NewtonManager, "add_model_change"):
            set_coms()
    else:
        set_coms()
    _assert_buffers_stale(obj.data, buffers)


# -- index variants --


@pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
def test_write_root_pose_to_sim_index(monkeypatch, backend, device, method_suffix):
    num_instances = _NUM_INSTANCES
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    method = getattr(obj, f"write_{method_suffix}_to_sim_index")

    # torch, all envs
    method(root_pose=_make_data_torch((num_instances,), device, wp.transformf))
    # torch, subset
    method(root_pose=_make_data_torch((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
    # warp, all envs: the matching getter reads the written poses back
    method(root_pose=_make_payload_warp((num_instances,), device, wp.transformf))
    getter = _ROOT_POSE_METHODS[method_suffix]
    _assert_reads_back(getattr(obj.data, getter), _make_payload_torch((num_instances,), device, wp.transformf), getter)
    # warp, subset
    method(root_pose=_make_data_warp((1,), device, wp.transformf), env_ids=_make_env_ids(device, True))
    # negative: bad torch shape
    with pytest.raises((AssertionError, RuntimeError)):
        method(root_pose=_make_bad_data_torch((num_instances,), device, wp.transformf))
    # negative: bad warp shape
    with pytest.raises((AssertionError, RuntimeError)):
        method(root_pose=_make_bad_data_warp((num_instances,), device, wp.transformf))


@pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
def test_write_root_velocity_to_sim_index(monkeypatch, backend, device, method_suffix):
    num_instances = _NUM_INSTANCES
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    method = getattr(obj, f"write_{method_suffix}_to_sim_index")

    # torch, all envs
    method(root_velocity=_make_data_torch((num_instances,), device, wp.spatial_vectorf))
    # torch, subset
    method(root_velocity=_make_data_torch((1,), device, wp.spatial_vectorf), env_ids=_make_env_ids(device, True))
    # warp, all envs: the matching getter reads the written velocities back
    method(root_velocity=_make_payload_warp((num_instances,), device, wp.spatial_vectorf))
    getter = _ROOT_VEL_METHODS[method_suffix]
    _assert_reads_back(
        getattr(obj.data, getter), _make_payload_torch((num_instances,), device, wp.spatial_vectorf), getter
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


@pytest.mark.parametrize("method_suffix", _ROOT_POSE_METHODS)
def test_write_root_pose_to_sim_mask(monkeypatch, backend, device, method_suffix):
    num_instances = _NUM_INSTANCES
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

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
    _assert_reads_back(getattr(obj.data, getter), _make_payload_torch((num_instances,), device, wp.transformf), getter)
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


@pytest.mark.parametrize("method_suffix", _ROOT_VEL_METHODS)
def test_write_root_velocity_to_sim_mask(monkeypatch, backend, device, method_suffix):
    num_instances = _NUM_INSTANCES
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
    obj.data.update(dt=0.01)
    method = getattr(obj, f"write_{method_suffix}_to_sim_mask")

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
        getattr(obj.data, getter), _make_payload_torch((num_instances,), device, wp.spatial_vectorf), getter
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


# ---------------------------------------------------------------------------
# Tests: Body writers — torch/warp × index/mask × all/subset × negative
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


def test_external_wrench_frames(monkeypatch, backend):
    """Forward local and world wrenches through the real writer in each backend's frame."""
    device = "cpu"  # the composer and wrench-packing kernels are device-independent
    obj, raw_backend = get_rigid_object(backend, num_instances=2, device=device, monkeypatch=monkeypatch)
    # Seed a known 90-degree rotation about Z so the local-frame expectation is independent of production.
    root_pose = torch.tensor(
        [[1.0, 2.0, 3.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5], [4.0, 5.0, 6.0, 0.0, 0.0, 2.0**-0.5, 2.0**-0.5]],
        device=device,
    )
    obj.write_root_link_pose_to_sim_index(root_pose=root_pose)
    if backend == "newton":
        # Seed the body pose the native FK would publish.
        obj.data._sim_bind_body_link_pose_w.assign(wp.from_torch(root_pose, dtype=wp.transformf))
    composer = obj.permanent_wrench_composer
    forces = torch.arange(1.0, 7.0, device=device).reshape(2, 1, 3)
    torques = forces + 10.0
    if backend == "physx":
        raw_backend.apply_forces_and_torques_at_position = MagicMock()

    for is_global in (False, True):
        composer.reset()
        composer.set_forces_and_torques_index(forces=forces, torques=torques, is_global=is_global)
        with patch.object(composer, "compose_to_body_frame", wraps=composer.compose_to_body_frame) as compose:
            obj.write_data_to_sim()
        assert compose.call_count == int(is_global and backend == "newton")

        expected_force, expected_torque = forces, torques
        if backend == "physx":
            call = raw_backend.apply_forces_and_torques_at_position.call_args.kwargs
            assert call["is_global"] is is_global
            assert call["position_data"] is None
            actual_force = call["force_data"].numpy().reshape(2, 1, 3)
            actual_torque = call["torque_data"].numpy().reshape(2, 1, 3)
        else:
            if not is_global:
                # A 90-degree Z rotation maps body (x, y, z) to world (-y, x, z).
                expected_force = torch.stack((-forces[..., 1], forces[..., 0], forces[..., 2]), dim=-1)
                expected_torque = torch.stack((-torques[..., 1], torques[..., 0], torques[..., 2]), dim=-1)
            if backend == "newton":
                packed = obj.data._sim_bind_body_external_wrench.numpy()
            else:
                from isaaclab_ov import tensor_types as TT

                packed = raw_backend.bindings[TT.RIGID_BODY_WRENCH]._data.reshape(2, 1, 9)
                np.testing.assert_allclose(packed[..., 6:9], root_pose[:, None, :3].numpy())
            actual_force, actual_torque = packed[..., :3], packed[..., 3:6]
        np.testing.assert_allclose(actual_force, expected_force.cpu().numpy(), atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(actual_torque, expected_torque.cpu().numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("method_base, kwarg", _BODY_METHODS, ids=[m[0] for m in _BODY_METHODS])
def test_body_writer_index(monkeypatch, backend, device, method_base, kwarg):
    num_instances, num_bodies = _NUM_INSTANCES, 1
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
    wp_dtype, trailing, getter = _body_writer_layout(backend, method_base)
    obj.data.update(dt=0.01)
    method = getattr(obj, f"{method_base}_index")
    sub_b = 1  # rigid object always has 1 body
    sub_body_ids = [0]

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
    _assert_reads_back(getattr(obj.data, getter), expected, getter)
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


@pytest.mark.parametrize("method_base, kwarg", _BODY_METHODS, ids=[m[0] for m in _BODY_METHODS])
def test_body_writer_mask(monkeypatch, backend, device, method_base, kwarg):
    num_instances, num_bodies = _NUM_INSTANCES, 1
    obj, _ = get_rigid_object(backend, num_instances, device, monkeypatch=monkeypatch)
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
# Tests: partial writes reach only the selected backend rows
# ---------------------------------------------------------------------------

# quantity -> (writer base name, writer keyword, literal per-environment payload)
_PARTIAL_WRITES = {
    "root_pose": (
        "write_root_link_pose_to_sim",
        "root_pose",
        [[31.0, 32.0, 33.0, 0.0, 0.0, 0.0, 1.0], [41.0, 42.0, 43.0, 0.0, 0.0, 0.0, 1.0]],
    ),
    "root_velocity": (
        "write_root_com_velocity_to_sim",
        "root_velocity",
        [[21.0, 22.0, 23.0, 24.0, 25.0, 26.0], [71.0, 72.0, 73.0, 74.0, 75.0, 76.0]],
    ),
    "mass": ("set_masses", "masses", [[51.0], [61.0]]),
}


def _read_backend_rows(backend: str, raw_backend, quantity: str) -> torch.Tensor:
    """Read one rigid-object quantity from backend storage as one row per environment."""
    if backend == "physx":
        getter = {
            "root_pose": raw_backend.get_transforms,
            "root_velocity": raw_backend.get_velocities,
            "mass": raw_backend.get_masses,
        }
        values = getter[quantity]()
    elif backend == "newton":
        if quantity == "mass":
            values = raw_backend.get_attribute("body_mass", None)
        else:
            getter = {"root_pose": raw_backend.get_root_transforms, "root_velocity": raw_backend.get_root_velocities}
            values = getter[quantity](None)
    else:
        from isaaclab_ov import tensor_types as TT

        binding = {"root_pose": TT.RIGID_BODY_POSE, "root_velocity": TT.RIGID_BODY_VELOCITY, "mass": TT.RIGID_BODY_MASS}
        values = raw_backend.bindings[binding[quantity]]._data
    values = wp.to_torch(values) if isinstance(values, wp.array) else torch.as_tensor(values)
    return values.reshape(2, -1).cpu().clone()


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("quantity", _PARTIAL_WRITES)
def test_partial_write_preserves_unselected_backend_rows(monkeypatch, backend, selection, quantity):
    obj, raw_backend = get_rigid_object(backend, num_instances=2, device="cpu", monkeypatch=monkeypatch)
    writer, kwarg, payload = _PARTIAL_WRITES[quantity]
    values = torch.tensor(payload, dtype=torch.float32)
    before = _read_backend_rows(backend, raw_backend, quantity)
    # Select the second environment so a writer that ignores the selection cannot pass by writing row 0.
    selected, unselected = 1, 0
    if selection == "index":
        kwargs = {kwarg: values[selected : selected + 1], "env_ids": torch.tensor([selected], dtype=torch.int32)}
        if quantity == "mass":
            kwargs["body_ids"] = [0]
    else:
        kwargs = {kwarg: values, "env_mask": _make_item_mask(2, [selected], "cpu")}
        if quantity == "mass":
            kwargs["body_mask"] = _make_item_mask(1, [0], "cpu")

    getattr(obj, f"{writer}_{selection}")(**kwargs)

    after = _read_backend_rows(backend, raw_backend, quantity)
    torch.testing.assert_close(after[selected], values[selected], rtol=0.0, atol=0.0)
    torch.testing.assert_close(after[unselected], before[unselected], rtol=0.0, atol=0.0)
