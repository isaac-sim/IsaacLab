# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Cross-backend articulation interface contract checks on mocked views (no Isaac Sim or physics scene)."""

import warnings
from unittest.mock import patch

import pytest
import torch
import warp as wp
from _iface_test_utils import (
    assert_buffers_stale,
    backends_parametrize,
    check_aliases,
    check_data_properties,
    exercise_writer,
    get_articulation,
    make_com_data,
    make_data_warp,
    make_env_ids,
    make_mask,
    prime_timestamped_properties,
    requires_backend,
)

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData

pytestmark = pytest.mark.integration

_backends = backends_parametrize()
_production_backends = backends_parametrize("physx", "newton", "ovphysx")
_index_resolution_backends = backends_parametrize("physx", "newton")
# Newton does not support tendons.
_tendon_backends = backends_parametrize("physx", "ovphysx")
_dims = pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(1, 1, 1), (2, 6, 7)])
_tendon_dims = pytest.mark.parametrize(
    "num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons", [(1, 2, 2, 1, 0), (2, 6, 7, 3, 2)]
)
_devices = pytest.mark.parametrize("device", ["cuda:0", "cpu"])
_body_orderings = pytest.mark.parametrize("body_ordering", [None, ("body_0", "body_3", "body_2", "body_1")])


@pytest.fixture
def articulation(request):
    kwargs = {
        name: request.getfixturevalue(name)
        for name in ("num_instances", "num_joints", "num_bodies", "device", "num_fixed_tendons", "num_spatial_tendons")
        if name in request.fixturenames
    }
    art, _ = get_articulation(request.getfixturevalue("backend"), **kwargs)
    art.data.update(dt=0.01)
    return art


def _small_articulation(backend: str, **kwargs):
    art, view = get_articulation(backend, num_instances=2, num_joints=3, num_bodies=4, device="cpu", **kwargs)
    art.data.update(dt=0.01)
    return art, view


def _count_calls(obj, method_name: str) -> list[int]:
    """Wrap ``obj.method_name`` so its call count is tracked in the returned single-element list."""
    calls = [0]
    original = getattr(obj, method_name)

    def counted(*args, **kwargs):
        calls[0] += 1
        return original(*args, **kwargs)

    setattr(obj, method_name, counted)
    return calls


"""
Data property tables: property name -> (leading shape kind, dtype).
"""

_ROOT_PROPERTIES = {
    "root_link_pose_w": wp.transformf,
    "root_link_vel_w": wp.spatial_vectorf,
    "root_com_pose_w": wp.transformf,
    "root_com_vel_w": wp.spatial_vectorf,
    "root_link_pos_w": wp.vec3f,
    "root_link_quat_w": wp.quatf,
    "root_link_lin_vel_w": wp.vec3f,
    "root_link_ang_vel_w": wp.vec3f,
    "root_com_pos_w": wp.vec3f,
    "root_com_quat_w": wp.quatf,
    "root_com_lin_vel_w": wp.vec3f,
    "root_com_ang_vel_w": wp.vec3f,
    "projected_gravity_b": wp.vec3f,
    "heading_w": wp.float32,
    "root_link_lin_vel_b": wp.vec3f,
    "root_link_ang_vel_b": wp.vec3f,
    "root_com_lin_vel_b": wp.vec3f,
    "root_com_ang_vel_b": wp.vec3f,
    "default_root_pose": wp.transformf,
    "default_root_vel": wp.spatial_vectorf,
}
_BODY_PROPERTIES = {
    "body_link_pose_w": wp.transformf,
    "body_link_vel_w": wp.spatial_vectorf,
    "body_com_pose_w": wp.transformf,
    "body_com_vel_w": wp.spatial_vectorf,
    "body_com_acc_w": wp.spatial_vectorf,
    "body_com_pose_b": wp.transformf,
    "body_mass": wp.float32,
    "body_link_pos_w": wp.vec3f,
    "body_link_quat_w": wp.quatf,
    "body_link_lin_vel_w": wp.vec3f,
    "body_link_ang_vel_w": wp.vec3f,
    "body_com_pos_w": wp.vec3f,
    "body_com_quat_w": wp.quatf,
    "body_com_pos_b": wp.vec3f,
    "body_com_quat_b": wp.quatf,
}
# Newton only stores the CoM position, not its orientation.
_NEWTON_UNSUPPORTED_BODY_PROPERTIES = {"body_com_pose_b", "body_com_quat_b"}
_JOINT_PROPERTIES = {
    "joint_pos": wp.float32,
    "joint_vel": wp.float32,
    "joint_acc": wp.float32,
    "joint_stiffness": wp.float32,
    "joint_damping": wp.float32,
    "joint_armature": wp.float32,
    "joint_friction_coeff": wp.float32,
    "joint_pos_limits": wp.vec2f,
    "joint_vel_limits": wp.float32,
    "joint_effort_limits": wp.float32,
    "soft_joint_pos_limits": wp.vec2f,
    "default_joint_pos": wp.float32,
    "default_joint_vel": wp.float32,
    "joint_pos_target": wp.float32,
    "joint_vel_target": wp.float32,
    "joint_effort_target": wp.float32,
}
_ALIASES = {
    "root_pose_w": "root_link_pose_w",
    "root_pos_w": "root_link_pos_w",
    "root_quat_w": "root_link_quat_w",
    "root_vel_w": "root_com_vel_w",
    "root_lin_vel_w": "root_com_lin_vel_w",
    "root_ang_vel_w": "root_com_ang_vel_w",
    "body_pose_w": "body_link_pose_w",
    "body_pos_w": "body_link_pos_w",
    "body_quat_w": "body_link_quat_w",
    "body_vel_w": "body_com_vel_w",
    "body_lin_vel_w": "body_com_lin_vel_w",
    "body_ang_vel_w": "body_com_ang_vel_w",
    "joint_limits": "joint_pos_limits",
    "joint_friction": "joint_friction_coeff",
}
_FIXED_TENDON_PROPERTIES = (
    "fixed_tendon_stiffness",
    "fixed_tendon_damping",
    "fixed_tendon_limit_stiffness",
    "fixed_tendon_rest_length",
    "fixed_tendon_offset",
)
_SPATIAL_TENDON_PROPERTIES = (
    "spatial_tendon_stiffness",
    "spatial_tendon_damping",
    "spatial_tendon_limit_stiffness",
    "spatial_tendon_offset",
)

"""
Writer tables: (writer name, data keyword, dtype, scalar handling).
"""

_ROOT_WRITERS = [
    ("write_root_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_link_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_com_pose_to_sim", "root_pose", wp.transformf),
    ("write_root_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
    ("write_root_link_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
    ("write_root_com_velocity_to_sim", "root_velocity", wp.spatial_vectorf),
]
_JOINT_WRITERS = [
    ("write_joint_position_to_sim", "position", wp.float32, None),
    ("write_joint_velocity_to_sim", "velocity", wp.float32, None),
    ("write_joint_stiffness_to_sim", "stiffness", wp.float32, True),
    ("write_joint_damping_to_sim", "damping", wp.float32, True),
    ("write_joint_position_limit_to_sim", "limits", wp.vec2f, (ValueError, TypeError)),
    ("write_joint_velocity_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_effort_limit_to_sim", "limits", wp.float32, True),
    ("write_joint_armature_to_sim", "armature", wp.float32, True),
    ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", wp.float32, None),
    ("set_joint_position_target", "target", wp.float32, None),
    ("set_joint_velocity_target", "target", wp.float32, None),
    ("set_joint_effort_target", "target", wp.float32, None),
]
_BODY_WRITERS = [
    ("set_masses", "masses", wp.float32, None),
    ("set_coms", "coms", wp.transformf, None),
    ("set_inertias", "inertias", wp.float32, 9),
]
# ``set_fixed_tendon_position_limit`` is excluded: PhysX stores the limits as (N, T, 2) float32 while
# the setter validates (N, T), so the two backends cannot share one input matrix.
_FIXED_TENDON_WRITERS = [
    ("set_fixed_tendon_stiffness", "stiffness"),
    ("set_fixed_tendon_damping", "damping"),
    ("set_fixed_tendon_limit_stiffness", "limit_stiffness"),
    ("set_fixed_tendon_rest_length", "rest_length"),
    ("set_fixed_tendon_offset", "offset"),
]
_TENDON_WRITERS = _FIXED_TENDON_WRITERS + [
    ("set_spatial_tendon_stiffness", "stiffness"),
    ("set_spatial_tendon_damping", "damping"),
    ("set_spatial_tendon_limit_stiffness", "limit_stiffness"),
    ("set_spatial_tendon_offset", "offset"),
]


"""
Properties, finders, and data layout.
"""


@_index_resolution_backends
def test_resolve_ids_handle_tensor_views(backend):
    art, _ = get_articulation(backend, num_instances=4, num_joints=4, num_bodies=4, device="cpu")
    ids = torch.arange(4, dtype=torch.int32)
    for resolve in (art._resolve_env_ids, art._resolve_joint_ids, art._resolve_body_ids):
        assert resolve(ids).shape[0] == 4
        assert resolve(ids[:2]).shape[0] == 2


@_backends
@_dims
@_devices
def test_properties_and_finders(backend, num_instances, num_joints, num_bodies, device, articulation):
    art = articulation
    assert isinstance(art.data, BaseArticulationData)
    assert (art.num_instances, art.num_joints, art.num_bodies) == (num_instances, num_joints, num_bodies)
    assert isinstance(art.is_fixed_base, bool)
    for names, count, finder in (
        (art.joint_names, num_joints, art.find_joints),
        (art.body_names, num_bodies, art.find_bodies),
    ):
        assert isinstance(names, list) and len(names) == count and all(isinstance(n, str) for n in names)
        indices, found = finder(".*")
        assert indices == list(range(count)) and found == names
        assert finder(names[0]) == ([0], [names[0]])
        with pytest.raises(ValueError):
            finder("nonexistent_xyz")
        # Mutating the returned lists must not corrupt the finder cache.
        indices.clear()
        found.append("corrupted")
        assert finder(".*") == (list(range(count)), names)
    if num_joints > 1:
        assert art.find_joints(["joint_0", "joint_1"])[1] == ["joint_0", "joint_1"]
        assert art.find_joints(["joint_1", "joint_0"], preserve_order=True)[1] == ["joint_1", "joint_0"]


@_production_backends
@pytest.mark.parametrize("finder_name", ["find_bodies", "find_joints", "find_fixed_tendons", "find_spatial_tendons"])
def test_finder_returns_legacy_list_or_cached_proxy(backend, finder_name):
    if backend == "newton" and finder_name == "find_spatial_tendons":
        pytest.skip("Newton does not support spatial tendons.")
    art, _ = get_articulation(
        backend, num_instances=2, num_joints=2, num_bodies=2, num_fixed_tendons=2, num_spatial_tendons=2, device="cpu"
    )
    finder = getattr(art, finder_name)
    indices, names = finder(".*")
    proxy, proxy_names = finder(".*", as_proxy=True)
    assert isinstance(indices, list)
    assert indices == proxy.torch.tolist()
    assert names == proxy_names
    assert proxy is finder(".*", as_proxy=True)[0]
    assert proxy.dtype == wp.int32
    assert str(proxy.device) == art.device


@_backends
@_dims
@_devices
def test_data_property_layout(backend, num_instances, num_joints, num_bodies, device, articulation):
    body_properties = dict(_BODY_PROPERTIES)
    if backend == "newton":
        for name in _NEWTON_UNSUPPORTED_BODY_PROPERTIES:
            body_properties.pop(name)
    expected = {name: ((num_instances,), dtype) for name, dtype in _ROOT_PROPERTIES.items()}
    expected |= {name: ((num_instances, num_bodies), dtype) for name, dtype in body_properties.items()}
    expected |= {name: ((num_instances, num_joints), dtype) for name, dtype in _JOINT_PROPERTIES.items()}
    expected["body_inertia"] = ((num_instances, num_bodies, 9), wp.float32)
    check_data_properties(articulation.data, expected)
    check_aliases(articulation.data, _ALIASES)


@_production_backends
def test_soft_joint_vel_limits_alias_actuator_buffer_without_deprecation_warning(backend):
    art, _ = get_articulation(backend, num_instances=2, num_joints=4, num_bodies=5, device="cpu")
    soft_joint_vel_limits = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.5, 3.0, 3.5, 4.0]], dtype=torch.float32)
    wp.copy(art.actuators._soft_joint_vel_limits, wp.from_torch(soft_joint_vel_limits))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = art.data.soft_joint_vel_limits
        second = art.data.soft_joint_vel_limits

    check_data_properties(art.data, {"soft_joint_vel_limits": ((2, 4), wp.float32)})
    torch.testing.assert_close(first.torch, soft_joint_vel_limits, rtol=0.0, atol=0.0)
    assert first.warp.ptr == second.warp.ptr == art.actuators._soft_joint_vel_limits.ptr
    assert not [w for w in caught if w.category is DeprecationWarning]


"""
Fixed tendon target scheduling: commanding a target schedules the write, not merely having tendons.
"""


def _tendon_articulation(backend: str):
    art, _ = get_articulation(backend, num_instances=2, num_joints=2, num_bodies=2, num_fixed_tendons=2, device="cpu")
    return art


@_production_backends
def test_tendon_target_scheduling(backend):
    art = _tendon_articulation(backend)
    assert art.num_fixed_tendons == 2
    assert art._fixed_tendon_target_dirty is False
    target = torch.zeros((2, 2), dtype=torch.float32)
    if backend == "newton":
        # Without a MuJoCo tendon actuator the solver has no adapter, so commanding must say so.
        with pytest.raises(RuntimeError, match="no MuJoCo tendon actuator"):
            art.set_fixed_tendon_position_target_index(target=target)
    else:
        art.set_fixed_tendon_position_target_index(target=target)
        assert art._fixed_tendon_target_dirty is True


@requires_backend("newton")
def test_newton_solver_without_tendon_transmission_refuses_to_build_adapter():
    from isaaclab_newton.physics import NewtonManager

    with pytest.raises(NotImplementedError, match="does not drive fixed tendons"):
        NewtonManager.create_fixed_tendon_control(_tendon_articulation("newton"))


"""
PhysX center-of-mass cache.
"""


@requires_backend("physx")
def test_physx_body_com_pose_b_cache_survives_sim_steps_and_joint_writes():
    art, view = _small_articulation("physx")
    get_coms_calls = _count_calls(view, "get_coms")

    art.data.body_com_pose_b
    art.data.update(dt=0.01)
    art.data.body_com_pose_b
    assert get_coms_calls[0] == 1

    joint_pos = torch.zeros((art.num_instances, art.num_joints))
    art.write_joint_position_to_sim_index(position=joint_pos, full_data=True)
    art.data.body_com_pose_b
    assert get_coms_calls[0] == 1


@requires_backend("physx")
def test_physx_set_coms_populates_body_com_pose_b_cache():
    art, view = _small_articulation("physx")
    get_coms_calls = _count_calls(view, "get_coms")

    coms = wp.zeros((art.num_instances, art.num_bodies), dtype=wp.transformf, device="cpu")
    art.set_coms_index(coms=coms, full_data=True)
    art.data.body_com_pose_b
    assert get_coms_calls[0] == 0


@requires_backend("physx")
def test_physx_partial_set_coms_initializes_cold_cache_from_backend():
    art, view = _small_articulation("physx")
    initial_coms = view.get_coms().numpy().copy()
    get_coms_calls = _count_calls(view, "get_coms")

    selector = wp.array([0], dtype=wp.int32, device="cpu")
    art.set_coms_index(coms=wp.zeros((1, 1), dtype=wp.transformf, device="cpu"), env_ids=selector, body_ids=selector)
    body_com_pose_b = art.data.body_com_pose_b.torch

    assert get_coms_calls[0] == 1
    torch.testing.assert_close(body_com_pose_b[1, 1], torch.from_numpy(initial_coms[1, 1]))


@requires_backend("physx")
def test_physx_identity_ordering_reuses_dynamics_proxy_arrays():
    """Identity-ordered dynamics quantities alias stable PhysX buffers, so their wrappers are reused."""
    art, _ = _small_articulation("physx")
    first = (art.data.body_com_jacobian_w, art.data.mass_matrix, art.data.gravity_compensation_forces)
    art.data.update(dt=0.01)
    second = (art.data.body_com_jacobian_w, art.data.mass_matrix, art.data.gravity_compensation_forces)
    assert all(a is b for a, b in zip(first, second))


"""
Cache invalidation on writes.
"""

_BODY_FRAME_VELOCITY_CACHES = [
    ("root_link_lin_vel_b", "_root_link_lin_vel_b"),
    ("root_link_ang_vel_b", "_root_link_ang_vel_b"),
    ("root_com_lin_vel_b", "_root_com_lin_vel_b"),
    ("root_com_ang_vel_b", "_root_com_ang_vel_b"),
]


@_production_backends
@_body_orderings
def test_pose_write_invalidates_pose_dependent_caches(backend, body_ordering):
    art, _ = _small_articulation(backend, body_ordering=body_ordering)
    pairs = [
        ("root_link_vel_w", "_root_link_vel_w"),
        ("body_link_vel_w", "_body_link_vel_w"),
        ("projected_gravity_b", "_projected_gravity_b"),
        ("heading_w", "_heading_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
    ]
    if backend != "newton":
        pairs.append(("body_com_vel_w", "_body_com_vel_w"))
    buffers = prime_timestamped_properties(art.data, pairs)
    if backend == "ovphysx" and art.data._body_com_vel_w_backend is not None:
        buffers.append(("_body_com_vel_w_backend", art.data._body_com_vel_w_backend))
    art.write_root_link_pose_to_sim_index(root_pose=make_data_warp((art.num_instances,), "cpu", wp.transformf))
    assert_buffers_stale(art.data, buffers)


@_production_backends
@_body_orderings
def test_velocity_write_invalidates_body_frame_caches(backend, body_ordering):
    art, _ = _small_articulation(backend, body_ordering=body_ordering)
    buffers = prime_timestamped_properties(art.data, _BODY_FRAME_VELOCITY_CACHES)
    art.write_root_com_velocity_to_sim_index(
        root_velocity=make_data_warp((art.num_instances,), "cpu", wp.spatial_vectorf)
    )
    assert_buffers_stale(art.data, buffers)


@_production_backends
@pytest.mark.parametrize("setter_kind", ["index", "mask"])
@_body_orderings
def test_set_coms_invalidates_same_timestamp_dependents(backend, setter_kind, body_ordering):
    art, _ = _small_articulation(backend, body_ordering=body_ordering)
    pairs = [
        ("root_com_pose_w", "_root_com_pose_w"),
        ("root_link_vel_w", "_root_link_vel_w"),
        ("body_com_pose_w", "_body_com_pose_w"),
        ("body_link_vel_w", "_body_link_vel_w"),
        *_BODY_FRAME_VELOCITY_CACHES,
    ]
    if backend != "newton":
        pairs += [("root_com_vel_w", "_root_com_vel_w"), ("body_com_vel_w", "_body_com_vel_w")]
    suffix = "_buf" if backend == "ovphysx" else ""
    pairs += [
        (name, f"_{name}{suffix}")
        for name in (
            "root_state_w",
            "root_link_state_w",
            "root_com_state_w",
            "body_state_w",
            "body_link_state_w",
            "body_com_state_w",
        )
    ]
    if backend == "physx":
        pairs += [
            ("body_com_jacobian_w", "_body_com_jacobian_w"),
            ("mass_matrix", "_mass_matrix"),
            ("gravity_compensation_forces", "_gravity_compensation_forces"),
        ]
    # Prime public properties before resolving private buffers so Newton allocates its lazy caches.
    buffers = prime_timestamped_properties(art.data, pairs)
    if backend == "newton":
        buffers += prime_timestamped_properties(art.data, [("body_com_pose_b", "_body_com_pose_b")])
    coms = make_com_data(backend, (art.num_instances, art.num_bodies), "cpu")

    def set_coms() -> None:
        if setter_kind == "index":
            art.set_coms_index(coms=coms, **({"full_data": True} if backend == "physx" else {}))
        else:
            art.set_coms_mask(coms=coms)

    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        with patch.object(NewtonManager, "add_model_change"):
            set_coms()
    else:
        set_coms()
    assert_buffers_stale(art.data, buffers)


"""
Writers: torch/warp x index/mask x all/subset x malformed shape.
"""


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype", _ROOT_WRITERS, ids=[w[0] for w in _ROOT_WRITERS])
def test_root_writers(backend, num_instances, num_joints, num_bodies, device, articulation, name, kwarg, wp_dtype):
    exercise_writer(articulation, name, kwarg, wp_dtype, shape=(num_instances,), device=device)


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype, scalar", _JOINT_WRITERS, ids=[w[0] for w in _JOINT_WRITERS])
def test_joint_writers(
    backend, num_instances, num_joints, num_bodies, device, articulation, name, kwarg, wp_dtype, scalar
):
    exercise_writer(
        articulation,
        name,
        kwarg,
        wp_dtype,
        shape=(num_instances, num_joints),
        device=device,
        item_axis=("joint_ids", "joint_mask"),
        scalar=scalar,
    )


@_backends
@_dims
@_devices
@pytest.mark.parametrize("name, kwarg, wp_dtype, trailing", _BODY_WRITERS, ids=[w[0] for w in _BODY_WRITERS])
def test_body_writers(
    backend, num_instances, num_joints, num_bodies, device, articulation, name, kwarg, wp_dtype, trailing
):
    if backend == "newton" and name == "set_coms":
        pytest.xfail("Newton only stores CoM position, not orientation")
    exercise_writer(
        articulation,
        name,
        kwarg,
        wp_dtype,
        shape=(num_instances, num_bodies),
        device=device,
        item_axis=("body_ids", "body_mask"),
        trailing=trailing,
    )


"""
Tendons.
"""


@_tendon_backends
@_tendon_dims
@_devices
def test_tendon_properties_and_data_layout(
    backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device, articulation
):
    art = articulation
    assert (art.num_fixed_tendons, art.num_spatial_tendons) == (num_fixed_tendons, num_spatial_tendons)
    expected = {name: ((num_instances, num_fixed_tendons), wp.float32) for name in _FIXED_TENDON_PROPERTIES}
    for names, count, finder in (
        (art.fixed_tendon_names, num_fixed_tendons, art.find_fixed_tendons),
        (art.spatial_tendon_names, num_spatial_tendons, art.find_spatial_tendons),
    ):
        assert isinstance(names, list) and len(names) == count and all(isinstance(n, str) for n in names)
        if count:
            assert finder(".*") == (list(range(count)), names)
            assert finder(names[0]) == ([0], [names[0]])
    if num_spatial_tendons:
        expected |= {name: ((num_instances, num_spatial_tendons), wp.float32) for name in _SPATIAL_TENDON_PROPERTIES}
    check_data_properties(art.data, expected)
    # PhysX exposes (N, T, 2) float32 limits while the mock view exposes (N, T) vec2f.
    limits = art.data.fixed_tendon_pos_limits
    assert limits.shape in ((num_instances, num_fixed_tendons), (num_instances, num_fixed_tendons, 2))
    assert limits.dtype in (wp.vec2f, wp.float32)


@_tendon_backends
@_tendon_dims
@_devices
@pytest.mark.parametrize("name, kwarg", _TENDON_WRITERS, ids=[w[0] for w in _TENDON_WRITERS])
def test_tendon_writers(
    backend,
    num_instances,
    num_joints,
    num_bodies,
    num_fixed_tendons,
    num_spatial_tendons,
    device,
    articulation,
    name,
    kwarg,
):
    kind = "fixed" if "fixed" in name else "spatial"
    count = num_fixed_tendons if kind == "fixed" else num_spatial_tendons
    if count == 0:
        pytest.skip(f"No {kind} tendons configured")
    exercise_writer(
        articulation,
        name,
        kwarg,
        wp.float32,
        shape=(num_instances, count),
        device=device,
        item_axis=(f"{kind}_tendon_ids", f"{kind}_tendon_mask"),
        scalar=True,
    )


@_tendon_backends
@_tendon_dims
@_devices
@pytest.mark.parametrize("kind", ["fixed", "spatial"])
def test_write_tendon_properties_to_sim(
    backend, num_instances, num_joints, num_bodies, num_fixed_tendons, num_spatial_tendons, device, articulation, kind
):
    count = num_fixed_tendons if kind == "fixed" else num_spatial_tendons
    if count == 0:
        pytest.skip(f"No {kind} tendons configured")
    index_writer = getattr(articulation, f"write_{kind}_tendon_properties_to_sim_index")
    mask_writer = getattr(articulation, f"write_{kind}_tendon_properties_to_sim_mask")
    index_writer()
    index_writer(env_ids=make_env_ids(device))
    index_writer(**{f"{kind}_tendon_ids": wp.array([0], dtype=wp.int64, device=device)})
    mask_writer()
    mask_writer(env_mask=make_mask(num_instances, [0], device))
    mask_writer(**{f"{kind}_tendon_mask": make_mask(count, [0], device)})
