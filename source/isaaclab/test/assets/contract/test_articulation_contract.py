# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared articulation contracts across the production backends.

Checks that every articulation backend provides the data, writer, and joint/body ordering behavior the base
articulation class advertises. The backends run on mocked views, so these cases need neither Isaac Sim nor a GPU
simulation. Pure bookkeeping and ordering run on CPU only; getters and writers also run on CUDA for the backends that
stage through CPU-pinned buffers there.
"""

import math
import warnings
from operator import attrgetter
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
from isaaclab.utils import replace
from isaaclab.utils.warp import ProxyArray

from .articulation_factory import get_articulation
from .backends import backends, read_backend_joint_state, requires

pytestmark = pytest.mark.integration

# Distinct instance, joint, body, and tendon counts so any swapped axis shows up.
N, J, B, FT, ST = 2, 6, 7, 3, 2
_AXIS_COUNTS = {"joint": J, "body": B, "fixed_tendon": FT, "spatial_tendon": ST}


def _articulation(backend: str, device: str, monkeypatch: pytest.MonkeyPatch, **kwargs):
    """Create the shared mocked articulation; keyword arguments override the default counts."""
    counts = dict(num_instances=N, num_joints=J, num_bodies=B, num_fixed_tendons=FT, num_spatial_tendons=ST)
    return get_articulation(backend, device=device, monkeypatch=monkeypatch, **(counts | kwargs))


def _mask(total: int, selected, device: str) -> wp.array:
    """Return a boolean Warp mask with the selected indices set."""
    mask = np.zeros(total, dtype=bool)
    mask[list(selected)] = True
    return wp.array(mask, dtype=wp.bool, device=device)


def _payload(shape: tuple[int, ...], trailing: int, dtype, device: str, offset: float = 0.0) -> torch.Tensor:
    """Return valid writer data whose entries differ per element.

    Transforms carry a fixed 90-degree rotation about Z, and ``vec2f`` limits a ``[-v, v]`` pair.
    """
    values = torch.arange(1, math.prod(shape) + 1, dtype=torch.float32, device=device).reshape(shape) + offset + 0.25
    if dtype == wp.vec2f:
        return torch.stack((-values, values), dim=-1)
    if trailing:
        values = values.unsqueeze(-1) + torch.arange(trailing, dtype=torch.float32, device=device) / 10.0
    if dtype == wp.transformf:
        values[..., 3:] = torch.tensor([0.0, 0.0, 0.5**0.5, 0.5**0.5], device=device)
    return values


def _assert_reads_back(value: ProxyArray, expected: torch.Tensor, name: str) -> None:
    """Assert a data getter returns the values a writer just wrote."""
    torch.testing.assert_close(value.torch, expected, atol=1e-5, rtol=1e-5, msg=lambda msg: f"{name}: {msg}")


def _prime_timestamped_properties(data, buffer_names: list[str]) -> list:
    """Read lazy public properties and stamp their backing buffers as current."""
    buffers = []
    for name in buffer_names:
        getattr(data, name.removeprefix("_").removesuffix("_buf"))
        buffer = getattr(data, name)
        assert buffer is not None, name
        buffer.timestamp = data._sim_timestamp
        buffers.append((name, buffer))
    return buffers


def _assert_buffers_stale(data, buffers) -> None:
    for name, buffer in buffers:
        assert buffer.timestamp < data._sim_timestamp, name


def _ignore_newton_model_changes(backend: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """The mocked Newton asset has no solver to receive model-change notifications."""
    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        monkeypatch.setattr(NewtonManager, "add_model_change", MagicMock())


# ---------------------------------------------------------------------------
# Counts, names, finders, and selectors
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
def test_counts_and_names(monkeypatch, backend, is_fixed_base):
    art, _ = _articulation(backend, "cpu", monkeypatch, is_fixed_base=is_fixed_base)
    # Newton has no spatial tendons.
    num_spatial = 0 if backend == "newton" else ST

    assert isinstance(art.data, BaseArticulationData)
    assert art.num_instances == N
    assert art.is_fixed_base is is_fixed_base
    for names, count, expected in (
        (art.joint_names, art.num_joints, J),
        (art.body_names, art.num_bodies, B),
        (art.fixed_tendon_names, art.num_fixed_tendons, FT),
        (art.spatial_tendon_names, art.num_spatial_tendons, num_spatial),
    ):
        assert count == expected
        assert isinstance(names, list) and len(names) == expected
        assert all(isinstance(name, str) for name in names)


def test_finders(monkeypatch, backend):
    """Finders return fresh lists or a cached index proxy, honor pattern order, and reject unmatched patterns."""
    art, _ = _articulation(backend, "cpu", monkeypatch)
    finders = {"find_bodies": B, "find_joints": J, "find_fixed_tendons": FT}
    if backend != "newton":
        finders["find_spatial_tendons"] = ST

    for finder_name, count in finders.items():
        finder = getattr(art, finder_name)
        indices, names = finder(".*")
        proxy, proxy_names = finder(".*", as_proxy=True)
        assert indices == list(range(count)) and len(names) == count
        assert all(isinstance(name, str) for name in names)
        assert indices == proxy.torch.tolist() and names == proxy_names
        assert proxy is finder(".*", as_proxy=True)[0]
        assert proxy.dtype == wp.int32 and str(proxy.device) == art.device
        assert finder(names[0]) == ([0], [names[0]])
        # Mutating a returned list must not corrupt the cache.
        indices.clear()
        names.append("corrupted")
        assert finder(".*") == (list(range(count)), proxy_names)
        assert finder([proxy_names[1], proxy_names[0]], preserve_order=True) == ([1, 0], proxy_names[1::-1])
        with pytest.raises(ValueError):
            finder("nonexistent_xyz")


def test_resolve_ids_accept_tensor_views(monkeypatch, backend):
    """Selector resolution honors tensor views, and environment slices alias the cached indices without copies."""
    art, _ = _articulation(backend, "cpu", monkeypatch, num_instances=4, num_joints=4, num_bodies=4)
    ids = torch.arange(4, dtype=torch.int32)
    for resolve in (art._resolve_env_ids, art._resolve_joint_ids, art._resolve_body_ids):
        assert resolve(ids).shape[0] == 4
        assert resolve(ids[:2]).shape[0] == 2

    cached = wp.to_torch(art._ALL_INDICES)
    for selection in (slice(None), slice(1, None, 2), slice(0, 0)):
        resolved = wp.to_torch(art._resolve_env_ids(selection))
        torch.testing.assert_close(resolved, cached[selection])
        assert resolved.data_ptr() == cached[selection].data_ptr()
        assert resolved.stride() == cached[selection].stride()


# ---------------------------------------------------------------------------
# Fixed-tendon targets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", backends("physx", "ovphysx"))
def test_commanding_a_tendon_target_schedules_the_write(monkeypatch, backend):
    art, _ = _articulation(backend, "cpu", monkeypatch, num_fixed_tendons=2)
    art.set_fixed_tendon_position_target_index(target=torch.zeros((N, 2)))
    assert art._fixed_tendon_target_dirty is True


@requires("newton")
def test_newton_tendon_target_requires_a_tendon_actuator(monkeypatch):
    """Without a MuJoCo tendon actuator Newton cannot command targets, and an untransmitting solver says so."""
    from isaaclab_newton.physics import NewtonManager

    art, _ = _articulation("newton", "cpu", monkeypatch, num_fixed_tendons=2)
    with pytest.raises(RuntimeError, match="no MuJoCo tendon actuator"):
        art.set_fixed_tendon_position_target_index(target=torch.zeros((N, 2)))
    with pytest.raises(NotImplementedError, match="does not drive fixed tendons"):
        NewtonManager.create_fixed_tendon_control(art)


# ---------------------------------------------------------------------------
# Data properties
# ---------------------------------------------------------------------------


def _properties(suffix: tuple[int, ...], dtype, names: str) -> list[tuple[str, tuple[int, ...], type]]:
    return [(name, suffix, dtype) for name in names.split()]


# (property, shape after the instance axis, dtype)
_DATA_PROPERTIES = [
    *_properties((), wp.transformf, "root_link_pose_w root_com_pose_w default_root_pose"),
    *_properties((), wp.spatial_vectorf, "root_link_vel_w root_com_vel_w default_root_vel"),
    *_properties((), wp.quatf, "root_link_quat_w root_com_quat_w"),
    *_properties(
        (),
        wp.vec3f,
        """root_link_pos_w root_link_lin_vel_w root_link_ang_vel_w root_com_pos_w root_com_lin_vel_w
        root_com_ang_vel_w projected_gravity_b root_link_lin_vel_b root_link_ang_vel_b root_com_lin_vel_b
        root_com_ang_vel_b""",
    ),
    *_properties((), wp.float32, "heading_w"),
    *_properties((B,), wp.transformf, "body_link_pose_w body_com_pose_w body_com_pose_b"),
    *_properties((B,), wp.spatial_vectorf, "body_link_vel_w body_com_vel_w body_com_acc_w"),
    *_properties((B,), wp.quatf, "body_link_quat_w body_com_quat_w body_com_quat_b"),
    *_properties(
        (B,), wp.vec3f, "body_link_pos_w body_link_lin_vel_w body_link_ang_vel_w body_com_pos_w body_com_pos_b"
    ),
    *_properties((B,), wp.float32, "body_mass"),
    *_properties((B, 9), wp.float32, "body_inertia"),
    *_properties(
        (J,),
        wp.float32,
        """joint_pos joint_vel joint_acc joint_stiffness joint_damping joint_armature joint_friction_coeff
        joint_vel_limits joint_effort_limits default_joint_pos default_joint_vel""",
    ),
    *_properties((J,), wp.vec2f, "joint_pos_limits soft_joint_pos_limits"),
    *_properties((FT,), wp.float32, "fixed_tendon_stiffness fixed_tendon_damping"),
    *_properties((FT,), wp.vec2f, "fixed_tendon_pos_limits"),
]
# Tendon properties that Newton does not implement.
_PHYSX_FAMILY_TENDON_PROPERTIES = [
    *_properties((FT,), wp.float32, "fixed_tendon_limit_stiffness fixed_tendon_rest_length fixed_tendon_offset"),
    *_properties(
        (ST,),
        wp.float32,
        "spatial_tendon_stiffness spatial_tendon_damping spatial_tendon_limit_stiffness spatial_tendon_offset",
    ),
]
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


def test_data_properties(monkeypatch, backend, device):
    """Every data property is a ProxyArray of the advertised shape and dtype; components and aliases share data."""
    art, _ = _articulation(backend, device, monkeypatch)
    if backend == "newton":
        # Native selections may stride over other articulations in each world.
        for name in ("root_link_pose_w", "root_com_vel_w", "body_link_pose_w", "body_com_vel_w"):
            binding = getattr(art.data, "_sim_bind_" + name)
            view = wp.empty((2 * binding.shape[0], *binding.shape[1:]), dtype=binding.dtype, device=device)[::2]
            view.assign(binding)
            setattr(art.data, "_sim_bind_" + name, view)
        art.data._pin_proxy_arrays()
    art.data.update(dt=0.01)
    if backend == "newton":
        # Derived Newton quantities stay unallocated until read.
        for name in ("root_link_vel_w", "body_link_vel_w", "body_com_pose_b", "root_com_pose_w", "body_com_pose_w"):
            assert getattr(art.data, "_" + name).data is None, name
        for name in ("projected_gravity_b", "heading_w"):
            assert getattr(art.data, "_" + name).data is None, name

    properties = _DATA_PROPERTIES + (_PHYSX_FAMILY_TENDON_PROPERTIES if backend != "newton" else [])
    for name, suffix, dtype in properties:
        if backend == "newton" and name == "body_com_pose_b":
            # Newton stores a position-only COM and warns that the pose appends a unit quaternion.
            with pytest.warns(UserWarning, match="unit quaternion"):
                value = getattr(art.data, name)
        else:
            value = getattr(art.data, name)
        shape = (N, *suffix)
        if backend == "physx" and name == "fixed_tendon_pos_limits":
            # Known inconsistency: PhysX exposes (N, T, 2) float32 although its docstring and OVPhysX
            # advertise (N, T) vec2f. Pin the current layout so a change on either side is noticed.
            shape, dtype = (*shape, 2), wp.float32
        assert isinstance(value, ProxyArray), name
        assert (value.shape, value.dtype) == (shape, dtype), name

    for frame in ("root_link", "root_com", "body_link", "body_com"):
        for quantity, components in (("pose", ("pos", "quat")), ("vel", ("lin_vel", "ang_vel"))):
            packed = getattr(art.data, f"{frame}_{quantity}_w").torch
            for component, expected in zip(components, (packed[..., :3], packed[..., 3:]), strict=True):
                view = getattr(art.data, f"{frame}_{component}_w").torch
                torch.testing.assert_close(view, expected)
                assert view.data_ptr() == expected.data_ptr()
                assert view.stride() == expected.stride()

    # Random mock state makes link and COM quantities differ, so a retargeted alias fails.
    for alias, canonical in _ALIASES.items():
        alias_value, canonical_value = getattr(art.data, alias), getattr(art.data, canonical)
        assert (alias_value.shape, alias_value.dtype) == (canonical_value.shape, canonical_value.dtype), alias
        assert torch.equal(alias_value.torch, canonical_value.torch), alias


def test_actuator_backed_data_properties(monkeypatch, backend):
    """Soft velocity limits alias the actuator buffer without warning; deprecated command properties warn and alias."""
    art, _ = _articulation(backend, "cpu", monkeypatch, num_joints=4)
    actuators = art.actuators
    soft_joint_vel_limits = torch.arange(1, N * 4 + 1, dtype=torch.float32).reshape(N, 4) / 2.0
    wp.copy(actuators._soft_joint_vel_limits, wp.from_torch(soft_joint_vel_limits))

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        value = art.data.soft_joint_vel_limits
        assert value.warp.ptr == art.data.soft_joint_vel_limits.warp.ptr == actuators._soft_joint_vel_limits.ptr
    assert (value.shape, value.dtype) == ((N, 4), wp.float32)
    torch.testing.assert_close(value.torch, soft_joint_vel_limits, rtol=0.0, atol=0.0)

    aliases = {
        "joint_pos_target": actuators.target_command.position,
        "joint_vel_target": actuators.target_command.velocity,
        "joint_effort_target": actuators.target_command.effort,
        "computed_torque": actuators.computed_effort,
        "applied_torque": actuators.applied_effort,
    }
    for name, collection_buffer in aliases.items():
        with pytest.warns(DeprecationWarning, match=name):
            assert getattr(art.data, name) is collection_buffer, name


@pytest.mark.parametrize("backend", backends("physx", "ovphysx"))
def test_dynamics_buffers_allocate_on_read_and_reuse_across_refreshes(monkeypatch, backend, device):
    """Optional dynamics allocate independently on demand and retain stable public views."""
    art, _ = _articulation(backend, device, monkeypatch, num_joints=3, num_bodies=4)
    names = ("body_com_jacobian_w", "mass_matrix", "gravity_compensation_forces")
    art.data.update(dt=0.01)
    for index, name in enumerate(names):
        for unread in names[index:]:
            assert getattr(art.data, "_" + unread).data is None
        wrapper = getattr(art.data, name)
        assert wrapper is getattr(art.data, "_" + name).data
        art.data.update(dt=0.01)
        assert getattr(art.data, name) is wrapper
        assert art.data._body_link_jacobian_w is None
    wrapper = art.data.body_link_jacobian_w
    art.data.update(dt=0.01)
    assert art.data.body_link_jacobian_w is wrapper


# ---------------------------------------------------------------------------
# Cache invalidation
# ---------------------------------------------------------------------------

_BODY_ORDERINGS = pytest.mark.parametrize(
    "body_ordering", [None, ("body_0", "body_3", "body_2", "body_1")], ids=["backend_order", "reordered"]
)
_ROOT_BODY_FRAME_VELOCITIES = ["_root_link_lin_vel_b", "_root_link_ang_vel_b", "_root_com_lin_vel_b"]
_ROOT_BODY_FRAME_VELOCITIES += ["_root_com_ang_vel_b"]


@_BODY_ORDERINGS
def test_root_writes_invalidate_dependent_caches(monkeypatch, backend, body_ordering):
    art, _ = _articulation(backend, "cpu", monkeypatch, num_joints=3, num_bodies=4, body_ordering=body_ordering)
    art.data.update(dt=0.01)
    pose_dependents = ["_root_link_vel_w", "_body_link_vel_w", "_projected_gravity_b", "_heading_w"]
    if backend != "newton":
        pose_dependents.append("_body_com_vel_w")

    buffers = _prime_timestamped_properties(art.data, pose_dependents + _ROOT_BODY_FRAME_VELOCITIES)
    if backend == "ovphysx" and art.data._body_com_vel_w_backend is not None:
        buffers.append(("_body_com_vel_w_backend", art.data._body_com_vel_w_backend))
    art.write_root_link_pose_to_sim_index(root_pose=_payload((N,), 7, wp.transformf, "cpu"))
    _assert_buffers_stale(art.data, buffers)

    buffers = _prime_timestamped_properties(art.data, _ROOT_BODY_FRAME_VELOCITIES)
    art.write_root_com_velocity_to_sim_index(root_velocity=_payload((N,), 6, wp.spatial_vectorf, "cpu"))
    _assert_buffers_stale(art.data, buffers)


@_BODY_ORDERINGS
def test_set_coms_invalidates_same_timestamp_dependents(monkeypatch, backend, body_ordering):
    _ignore_newton_model_changes(backend, monkeypatch)
    art, _ = _articulation(backend, "cpu", monkeypatch, num_joints=3, num_bodies=4, body_ordering=body_ordering)
    art.data.update(dt=0.01)
    names = ["_root_com_pose_w", "_root_link_vel_w", "_body_com_pose_w", "_body_link_vel_w"]
    names += _ROOT_BODY_FRAME_VELOCITIES
    if backend == "newton":
        names.append("_body_com_pose_b")
    else:
        names += ["_root_com_vel_w", "_body_com_vel_w"]
    if backend == "physx":
        # COM changes also move the COM Jacobian and the generalized dynamics.
        names += ["_body_com_jacobian_w", "_mass_matrix", "_gravity_compensation_forces"]
    for frame in ("", "_link", "_com"):
        for owner in ("root", "body"):
            names.append(f"_{owner}{frame}_state_w{'_buf' if backend == 'ovphysx' else ''}")
    if backend == "newton":
        coms = wp.zeros((N, 4), dtype=wp.vec3f, device="cpu")
    else:
        coms = wp.from_torch(_payload((N, 4), 7, wp.transformf, "cpu"), dtype=wp.transformf)

    for set_coms in (art.set_coms_index, art.set_coms_mask):
        buffers = _prime_timestamped_properties(art.data, names)
        if backend == "ovphysx" and art.data._body_com_vel_w_backend is not None:
            art.data._body_com_vel_w_backend.timestamp = art.data._sim_timestamp
            buffers.append(("_body_com_vel_w_backend", art.data._body_com_vel_w_backend))
        set_coms(coms=coms)
        _assert_buffers_stale(art.data, buffers)


@requires("physx")
def test_physx_com_cache_survives_com_and_joint_writes(monkeypatch):
    """A COM write refreshes the PhysX COM cache directly, and a joint write leaves it valid."""
    art, view = _articulation("physx", "cpu", monkeypatch, num_joints=3, num_bodies=4)
    view.get_coms = MagicMock(wraps=view.get_coms)

    art.set_coms_index(coms=wp.zeros((N, 4), dtype=wp.transformf, device="cpu"), full_data=True)
    art.data.body_com_pose_b
    art.write_joint_position_to_sim_index(position=torch.zeros((N, 3)), full_data=True)
    art.data.body_com_pose_b

    assert view.get_coms.call_count == 0


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

# (writer, keyword, selected item axis, trailing size, dtype, read-back getter)
_WRITERS = [
    ("write_root_pose_to_sim", "root_pose", None, 7, wp.transformf, "data.root_link_pose_w"),
    ("write_root_link_pose_to_sim", "root_pose", None, 7, wp.transformf, "data.root_link_pose_w"),
    ("write_root_com_pose_to_sim", "root_pose", None, 7, wp.transformf, "data.root_com_pose_w"),
    ("write_root_velocity_to_sim", "root_velocity", None, 6, wp.spatial_vectorf, "data.root_com_vel_w"),
    ("write_root_link_velocity_to_sim", "root_velocity", None, 6, wp.spatial_vectorf, "data.root_link_vel_w"),
    ("write_root_com_velocity_to_sim", "root_velocity", None, 6, wp.spatial_vectorf, "data.root_com_vel_w"),
    ("write_joint_position_to_sim", "position", "joint", 0, wp.float32, "data.joint_pos"),
    ("write_joint_velocity_to_sim", "velocity", "joint", 0, wp.float32, "data.joint_vel"),
    ("write_joint_stiffness_to_sim", "stiffness", "joint", 0, wp.float32, "data.joint_stiffness"),
    ("write_joint_damping_to_sim", "damping", "joint", 0, wp.float32, "data.joint_damping"),
    ("write_joint_position_limit_to_sim", "limits", "joint", 2, wp.vec2f, "data.joint_pos_limits"),
    ("write_joint_velocity_limit_to_sim", "limits", "joint", 0, wp.float32, "data.joint_vel_limits"),
    ("write_joint_effort_limit_to_sim", "limits", "joint", 0, wp.float32, "data.joint_effort_limits"),
    ("write_joint_armature_to_sim", "armature", "joint", 0, wp.float32, "data.joint_armature"),
    (
        "write_joint_friction_coefficient_to_sim",
        "joint_friction_coeff",
        "joint",
        0,
        wp.float32,
        "data.joint_friction_coeff",
    ),
    ("set_joint_position_target", "target", "joint", 0, wp.float32, "actuators.target_command.position"),
    ("set_joint_velocity_target", "target", "joint", 0, wp.float32, "actuators.target_command.velocity"),
    ("set_joint_effort_target", "target", "joint", 0, wp.float32, "actuators.target_command.effort"),
    ("set_masses", "masses", "body", 0, wp.float32, "data.body_mass"),
    ("set_coms", "coms", "body", 7, wp.transformf, "data.body_com_pose_b"),
    ("set_inertias", "inertias", "body", 9, wp.float32, "data.body_inertia"),
    *[
        (f"set_fixed_tendon_{name}", name, "fixed_tendon", 0, wp.float32, f"data.fixed_tendon_{name}")
        for name in ("stiffness", "damping", "limit_stiffness", "rest_length", "offset")
    ],
    ("set_fixed_tendon_position_limit", "limit", "fixed_tendon", 2, wp.vec2f, "data.fixed_tendon_pos_limits"),
    *[
        (f"set_spatial_tendon_{name}", name, "spatial_tendon", 0, wp.float32, f"data.spatial_tendon_{name}")
        for name in ("stiffness", "damping", "limit_stiffness", "offset")
    ],
]
# Writers that also accept one scalar for every selected entry; limit pairs reject it.
_SCALAR_WRITERS = {
    "write_joint_stiffness_to_sim",
    "write_joint_damping_to_sim",
    "write_joint_position_limit_to_sim",
    "write_joint_velocity_limit_to_sim",
    "write_joint_effort_limit_to_sim",
    "write_joint_armature_to_sim",
    *[writer for writer, _, axis, *_ in _WRITERS if axis in ("fixed_tendon", "spatial_tendon")],
}
# Tendon setters that Newton implements.
_NEWTON_TENDON_WRITERS = {"set_fixed_tendon_stiffness", "set_fixed_tendon_damping", "set_fixed_tendon_position_limit"}


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("writer, kwarg, axis, trailing, dtype, getter", _WRITERS, ids=[w[0] for w in _WRITERS])
def test_writer(monkeypatch, backend, device, selection, writer, kwarg, axis, trailing, dtype, getter):
    """Writers accept warp and torch data, write only the selected entries, and reject mismatched shapes."""
    is_tendon = axis in ("fixed_tendon", "spatial_tendon")
    if backend == "newton" and is_tendon and writer not in _NEWTON_TENDON_WRITERS:
        pytest.skip("Newton implements only fixed-tendon stiffness, damping, and position limits")
    if backend == "newton" and writer == "set_coms":
        # Newton stores the COM as a position only.
        trailing, dtype, getter = 3, wp.vec3f, "data.body_com_pos_b"
    _ignore_newton_model_changes(backend, monkeypatch)
    art, _ = _articulation(backend, device, monkeypatch)
    art.data.update(dt=0.01)
    method = getattr(art, f"{writer}_{selection}")
    shape = (N,) if axis is None else (N, _AXIS_COUNTS[axis])

    def read() -> ProxyArray:
        if is_tendon:
            # Tendon setters stage values that the property push sends to the simulation.
            getattr(art, f"write_{axis}_properties_to_sim_index")()
        return attrgetter(getter)(art)

    # Full warp data.
    expected = _payload(shape, trailing, dtype, device)
    method(**{kwarg: wp.from_torch(expected.contiguous(), dtype=dtype)})
    _assert_reads_back(read(), expected, writer)

    # Partial torch data for environment 1 and, for per-item writers, items 1 and 0 in that order.
    values = _payload(shape, trailing, dtype, device, offset=100.0)
    items = [1, 0]
    if selection == "index":
        kwargs = {kwarg: values[1:], "env_ids": torch.tensor([1], dtype=torch.int32, device=device)}
        if axis is not None:
            kwargs.update({kwarg: values[1:, items], f"{axis}_ids": items})
    else:
        kwargs = {kwarg: values, "env_mask": _mask(N, [1], device)}
        if axis is not None:
            kwargs[f"{axis}_mask"] = _mask(shape[1], items, device)
    method(**kwargs)
    if axis is None:
        expected[1] = values[1]
    else:
        expected[1, items] = values[1, items]
    _assert_reads_back(read(), expected, f"{writer} partial")

    if writer in _SCALAR_WRITERS:
        if dtype == wp.vec2f:
            with pytest.raises((ValueError, TypeError)):
                method(**{kwarg: 1.0})
        else:
            method(**{kwarg: 1.0})
            _assert_reads_back(read(), torch.ones_like(expected), f"{writer} scalar")

    if backend == "newton" and selection == "mask" and is_tendon:
        return  # pinned by test_newton_fixed_tendon_mask_setters_reject_extra_environments
    extra_env = _payload((N + 1, *shape[1:]), trailing, dtype, device)
    with pytest.raises((AssertionError, RuntimeError)):
        method(**{kwarg: extra_env})
    with pytest.raises((AssertionError, RuntimeError)):
        method(**{kwarg: wp.from_torch(extra_env.contiguous(), dtype=dtype)})


@requires("newton")
def test_newton_fixed_tendon_mask_setters_reject_extra_environments(monkeypatch):
    art, _ = _articulation("newton", "cpu", monkeypatch)
    with pytest.raises((AssertionError, RuntimeError)):
        art.set_fixed_tendon_stiffness_mask(stiffness=torch.ones((N + 1, FT)))


# quantity -> (writer, keyword, trailing size). Joint partial writes are covered by the ordering contract.
_PARTIAL_WRITES = {
    "root_pose": ("write_root_link_pose_to_sim", "root_pose", 7),
    "root_velocity": ("write_root_com_velocity_to_sim", "root_velocity", 6),
    "mass": ("set_masses", "masses", 1),
}


def _read_backend_rows(backend: str, art, raw_backend, quantity: str) -> torch.Tensor:
    """Read one articulation quantity from backend storage as one row per environment."""
    if backend == "ovphysx":
        from isaaclab_ov import tensor_types as TT

        binding = {"root_pose": TT.ROOT_POSE, "root_velocity": TT.ROOT_VELOCITY, "mass": TT.BODY_MASS}[quantity]
        values = torch.as_tensor(raw_backend.bindings[binding]._data)
    elif quantity == "mass":
        values = wp.to_torch(raw_backend.get_masses() if backend == "physx" else art.data._sim_bind_body_mass)
    else:
        getter = raw_backend.get_root_transforms if quantity == "root_pose" else raw_backend.get_root_velocities
        values = wp.to_torch(getter() if backend == "physx" else getter(None))
    return values.reshape(art.num_instances, -1).cpu().clone()


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("quantity", _PARTIAL_WRITES)
def test_partial_write_preserves_unselected_backend_rows(monkeypatch, backend, selection, quantity):
    art, raw_backend = _articulation(backend, "cpu", monkeypatch, num_joints=3, num_bodies=4)
    writer, kwarg, trailing = _PARTIAL_WRITES[quantity]
    width = 4 if quantity == "mass" else trailing
    # Literal, per-element distinct payload; poses keep an identity rotation.
    values = 100.0 * torch.arange(1, N + 1, dtype=torch.float32).unsqueeze(-1) + torch.arange(width)
    if quantity == "root_pose":
        values[:, 3:6] = 0.0
        values[:, 6] = 1.0
    before = _read_backend_rows(backend, art, raw_backend, quantity)
    # Select the second environment (and the second body) so a writer that ignores the selection cannot pass.
    env, body = 1, 1
    if selection == "index":
        kwargs = {kwarg: values[env:], "env_ids": torch.tensor([env], dtype=torch.int32)}
        if quantity == "mass":
            kwargs.update({kwarg: values[env:, body : body + 1], "body_ids": [body]})
    else:
        kwargs = {kwarg: values, "env_mask": _mask(N, [env], "cpu")}
        if quantity == "mass":
            kwargs["body_mask"] = _mask(4, [body], "cpu")

    getattr(art, f"{writer}_{selection}")(**kwargs)

    expected = before.clone()
    if quantity == "mass":
        expected[env, body] = values[env, body]
    else:
        expected[env] = values[env]
    torch.testing.assert_close(_read_backend_rows(backend, art, raw_backend, quantity), expected, rtol=0.0, atol=0.0)


def test_partial_joint_state_write_follows_int64_selectors_and_restarts_acceleration(monkeypatch, backend):
    """A fused joint-state write honors unsorted int64 selectors and restarts the finite-difference baseline."""
    art, raw_backend = _articulation(backend, "cpu", monkeypatch, num_joints=3, num_bodies=2)
    art.data.update(dt=0.01)
    expected_joint_pos = art.data.joint_pos.torch.clone()
    expected_joint_vel = art.data.joint_vel.torch.clone()
    # Newton reads body poses through FK bindings, so only its body velocity is a timestamped cache.
    cached_properties = ("body_link_vel_w",) if backend == "newton" else ("body_link_pose_w", "body_com_vel_w")
    caches = _prime_timestamped_properties(art.data, [f"_{name}" for name in cached_properties])
    # Seed the finite-difference state so an untouched cell shows whether the write leaked into it.
    previous_joint_vel = wp.to_torch(art.data._previous_joint_vel)
    previous_joint_vel.fill_(3.0)
    wp.to_torch(art.data._joint_acc.data).fill_(4.0)
    art.data._joint_acc.timestamp = -1.0
    position = torch.tensor([[0.1, -0.1]])
    velocity = torch.tensor([[0.2, -0.2]])

    # int64 selectors with the joints in reverse order; the payload follows the selector order.
    art.write_joint_state_to_sim_index(
        position=position[:, [1, 0]],
        velocity=velocity[:, [1, 0]],
        env_ids=torch.tensor([1], dtype=torch.int64),
        joint_ids=torch.tensor([2, 0], dtype=torch.int64),
        skip_forward=True,
    )

    expected_joint_pos[1, [0, 2]] = position[0]
    expected_joint_vel[1, [0, 2]] = velocity[0]
    expected_previous_joint_vel = torch.full_like(previous_joint_vel, 3.0)
    expected_previous_joint_vel[1, [0, 2]] = velocity[0]
    expected_joint_acc = torch.full_like(previous_joint_vel, 4.0)
    expected_joint_acc[1, [0, 2]] = 0.0
    torch.testing.assert_close(art.data.joint_pos.torch, expected_joint_pos)
    torch.testing.assert_close(art.data.joint_vel.torch, expected_joint_vel)
    torch.testing.assert_close(previous_joint_vel, expected_previous_joint_vel)
    torch.testing.assert_close(wp.to_torch(art.data._joint_acc.data), expected_joint_acc)
    if backend == "ovphysx":
        # OVPhysX stamps the reset acceleration, so a read before the next step returns it unchanged.
        assert art.data._joint_acc.timestamp == art.data._sim_timestamp
        torch.testing.assert_close(art.data.joint_acc.torch, expected_joint_acc)
    else:
        torch.testing.assert_close(art.data.joint_acc.torch[1, [0, 2]], torch.zeros(2))
    backend_joint_pos, backend_joint_vel = read_backend_joint_state(backend, art, raw_backend)
    torch.testing.assert_close(torch.from_numpy(backend_joint_pos), expected_joint_pos)
    torch.testing.assert_close(torch.from_numpy(backend_joint_vel), expected_joint_vel)
    # ``skip_forward`` leaves the body caches to the caller; a regular write invalidates them.
    for name, buffer in caches:
        assert buffer.timestamp == art.data._sim_timestamp, name
    art.write_joint_state_to_sim_index(position=position, velocity=velocity, env_ids=[1], joint_ids=[0, 2])
    _assert_buffers_stale(art.data, caches)


def test_deprecated_writers_forward_to_public_writers(monkeypatch, backend):
    """Deprecated joint-state, friction, and position-target writers warn and write what the public writers would."""
    art, raw_backend = _articulation(backend, "cpu", monkeypatch, num_joints=4, num_bodies=2)
    art.data.update(dt=0.01)
    expected_joint_pos = art.data.joint_pos.torch.clone()
    expected_joint_vel = art.data.joint_vel.torch.clone()

    with pytest.warns(DeprecationWarning, match="write_joint_state_to_sim"):
        art.write_joint_state_to_sim(
            position=torch.tensor([[0.1]]), velocity=torch.tensor([[0.2]]), joint_ids=[1], env_ids=[1]
        )
    expected_joint_pos[1, 1] = 0.1
    expected_joint_vel[1, 1] = 0.2
    torch.testing.assert_close(art.data.joint_pos.torch, expected_joint_pos)
    torch.testing.assert_close(art.data.joint_vel.torch, expected_joint_vel)
    backend_joint_pos, backend_joint_vel = read_backend_joint_state(backend, art, raw_backend)
    torch.testing.assert_close(torch.from_numpy(backend_joint_pos), expected_joint_pos)
    torch.testing.assert_close(torch.from_numpy(backend_joint_vel), expected_joint_vel)

    for writer_name, value in (("write_joint_friction_coefficient_to_sim", 0.5), ("write_joint_friction_to_sim", 0.25)):
        friction = torch.full((N, 4), value)
        with pytest.warns(DeprecationWarning):
            getattr(art, writer_name)(friction)
        _assert_reads_back(art.data.joint_friction_coeff, friction, writer_name)

    target = torch.tensor([[11.0, 12.0], [21.0, 22.0]])
    with pytest.warns(DeprecationWarning):
        art.set_joint_position_target(target, joint_ids=slice(1, 3))
    expected_target = torch.zeros((N, 4))
    expected_target[:, 1:3] = target
    torch.testing.assert_close(art.actuators.target_command.position.torch, expected_target)


@pytest.mark.parametrize("selection", ["index", "mask"])
@pytest.mark.parametrize("kind", ["fixed", "spatial"])
def test_write_tendon_properties_to_sim_selects_envs(monkeypatch, backend, device, selection, kind):
    """Pushing tendon properties writes the selected environments to the backend."""
    if backend == "newton" and kind == "spatial":
        pytest.skip("Newton does not support spatial tendons")
    _ignore_newton_model_changes(backend, monkeypatch)
    num_instances = 3
    art, raw_backend = _articulation(backend, device, monkeypatch, num_instances=num_instances, num_joints=2)
    art.data.update(dt=0.01)
    env_writes = []
    if backend == "physx":

        def capture(*args, indices=None, **kwargs):
            env_writes.append(indices.numpy().tolist())

        setattr(raw_backend, f"set_{kind}_tendon_properties", MagicMock(side_effect=capture))
    elif backend == "ovphysx":
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
        if backend == "newton":
            # Newton copies staged rows into the solver-model bindings; stage a value no environment holds yet.
            art.set_fixed_tendon_stiffness_index(stiffness=float(10 * len(selected_envs)))
            sim_before = wp.to_torch(art.data._sim_bind_fixed_tendon_stiffness).clone()
        partial = len(selected_envs) < num_instances
        if selection == "index":
            write(env_ids=torch.tensor(selected_envs, device=device) if partial else None)
        else:
            write(env_mask=_mask(num_instances, selected_envs, device) if partial else None)
        if backend == "newton":
            sim_after = wp.to_torch(art.data._sim_bind_fixed_tendon_stiffness)
            env_writes.append([env for env in range(num_instances) if not torch.equal(sim_after[env], sim_before[env])])
        assert env_writes, "no tendon properties were written"
        assert all(envs == selected_envs for envs in env_writes), env_writes


# ---------------------------------------------------------------------------
# Joint and body ordering
#
# A configured ordering exposes joints and bodies under public names while the backend keeps its own order. These
# cases seed backend-order storage, act through the public API, and compare against independently derived
# expectations. Cyclic orderings are preferred because, unlike reversals, they are not their own inverse, so a swapped
# map fails.
# ---------------------------------------------------------------------------


def _names(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}_{index}" for index in range(count))


def _ordering(mode: str, names: tuple[str, ...], keep_first: bool = False) -> tuple[str, ...] | None:
    """Return the public ordering of backend names for a mode, optionally keeping a fixed root first."""
    if keep_first:
        rest = _ordering(mode, names[1:])
        return None if rest is None else (names[0], *rest)
    return {"none": None, "identity": names, "reversed": names[::-1], "cyclic": (*names[-1:], *names[:-1])}[mode]


def _user_to_backend(ordering, count: int) -> np.ndarray:
    """Return public-to-backend indices for an optional ordering map."""
    return np.arange(count) if ordering is None else np.asarray(ordering.user_to_backend_indices, dtype=np.int64)


def _backend_to_user(public_names: tuple[str, ...] | None, backend_names: tuple[str, ...]) -> np.ndarray:
    """Derive backend-to-public indices independently from the ordering map."""
    if public_names is None:
        return np.arange(len(backend_names))
    return np.asarray([public_names.index(name) for name in backend_names])


def _install_cyclic_ordering(art) -> tuple[np.ndarray, np.ndarray]:
    """Install cyclic joint and body orderings on a constructed articulation, keeping a fixed root body first."""
    art.cfg = replace(
        art.cfg,
        joint_ordering=_ordering("cyclic", tuple(art.backend_joint_names)),
        body_ordering=_ordering("cyclic", tuple(art.backend_body_names), keep_first=art.is_fixed_base),
    )
    # Re-resolve and re-stage the maps as backend initialization does after a config change.
    art._resolve_and_install_ordering_maps()
    art._ordering_configure_backend_staging()
    return _user_to_backend(art.joint_ordering, art.num_joints), _user_to_backend(art.body_ordering, art.num_bodies)


def _to_torch(array) -> torch.Tensor:
    """Return a CPU clone of a ProxyArray, Warp array, or NumPy array."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.copy())
    if hasattr(array, "torch"):
        return array.torch.detach().cpu().clone()
    return wp.to_torch(array).detach().cpu().clone()


def _assert_proxy_close(actual, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual.torch.cpu(), expected, rtol=1.0e-5, atol=1.0e-5)


def _make_body_state(num_instances: int, num_bodies: int) -> tuple[np.ndarray, ...]:
    """Return distinct backend-order root pose, root velocity, link pose, COM, COM velocity, and acceleration."""
    root_pose = np.zeros((num_instances, 7), dtype=np.float32)
    root_pose[:, 0] = 10.0 + np.arange(num_instances)
    root_pose[:, 6] = 1.0
    root_vel = np.zeros((num_instances, 6), dtype=np.float32)
    root_vel[:, 0] = 3.0 + np.arange(num_instances)
    root_vel[:, 5] = 7.0 + np.arange(num_instances)
    env, body = np.meshgrid(np.arange(num_instances), np.arange(num_bodies), indexing="ij")
    link_pose = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    link_pose[..., 0], link_pose[..., 1], link_pose[..., 6] = 10.0 * body, env, 1.0
    com_pose_b = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    com_pose_b[..., 0], com_pose_b[..., 6] = body + 1.0, 1.0
    body_com_vel = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)
    body_com_vel[..., 0], body_com_vel[..., 5] = 20.0 + body, 30.0 + body
    body_acc = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)
    body_acc[..., 0], body_acc[..., 3] = 100.0 + body, 200.0 + body
    return root_pose, root_vel, link_pose, com_pose_b, body_com_vel, body_acc


def _set_body_state(backend: str, art, raw_backend, state: tuple[np.ndarray, ...]) -> None:
    """Write backend-order body state from :func:`_make_body_state` into the backend mock."""
    root_pose, root_vel, link_pose, com_pose_b, body_com_vel, body_acc = state
    device = art.device
    if backend == "physx":
        raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32, device=device)
        raw_backend._root_velocities = wp.array(root_vel, dtype=wp.float32, device=device)
        raw_backend._link_transforms = wp.array(link_pose, dtype=wp.float32, device=device)
        raw_backend._link_velocities = wp.array(body_com_vel, dtype=wp.float32, device=device)
        raw_backend._link_accelerations = wp.array(body_acc, dtype=wp.float32, device=device)
        raw_backend._coms = wp.array(com_pose_b, dtype=wp.float32, device="cpu")
    elif backend == "ovphysx":
        from isaaclab_ov import tensor_types as TT

        for binding, value in (
            (TT.ROOT_POSE, root_pose),
            (TT.ROOT_VELOCITY, root_vel),
            (TT.LINK_POSE, link_pose),
            (TT.LINK_VELOCITY, body_com_vel),
            (TT.LINK_ACCELERATION, body_acc),
            (TT.BODY_COM_POSE, com_pose_b),
        ):
            raw_backend.bindings[binding]._data = value.copy()
        _invalidate_com_caches(art)
    else:
        root_pose_wp = wp.array(root_pose[:, None], dtype=wp.transformf, device=device)
        root_vel_wp = wp.array(root_vel[:, None], dtype=wp.spatial_vectorf, device=device)
        link_pose_wp = wp.array(link_pose[:, None], dtype=wp.transformf, device=device)
        body_com_vel_wp = wp.array(body_com_vel[:, None], dtype=wp.spatial_vectorf, device=device)
        body_com_pos_wp = wp.array(com_pose_b[:, None, :, :3], dtype=wp.vec3f, device=device)
        raw_backend.set_mock_root_transforms(root_pose_wp)
        raw_backend.set_mock_root_velocities(root_vel_wp)
        raw_backend.set_mock_link_transforms(link_pose_wp)
        raw_backend.set_mock_link_velocities(body_com_vel_wp)
        raw_backend.set_mock_coms(body_com_pos_wp)
        art.data._sim_bind_root_link_pose_w.assign(root_pose_wp[:, 0])
        art.data._sim_bind_root_com_vel_w.assign(root_vel_wp[:, 0])
        art.data._sim_bind_body_link_pose_w.assign(link_pose_wp[:, 0])
        art.data._sim_bind_body_com_vel_w.assign(body_com_vel_wp[:, 0])
        art.data._sim_bind_body_com_pos_b.assign(body_com_pos_wp[:, 0])
        art.data._previous_body_com_vel.zero_()


def _invalidate_com_caches(art) -> None:
    art.data._body_com_pose_b.timestamp = -1.0
    if art.data._body_com_pose_b_backend is not None:
        art.data._body_com_pose_b_backend.timestamp = -1.0


def _seed_backend_coms(backend: str, art, raw_backend) -> np.ndarray:
    """Seed distinct backend-order PhysX or OVPhysX COM transforms and return them."""
    value = 100.0 * np.arange(art.num_instances)[:, None] + 10.0 * np.arange(art.num_bodies)
    coms = np.zeros((art.num_instances, art.num_bodies, 7), dtype=np.float32)
    coms[..., :3] = value[..., None] + np.arange(1.0, 4.0)
    coms[..., 6] = 1.0
    if backend == "physx":
        raw_backend._coms = wp.array(coms, dtype=wp.float32, device="cpu")
    else:
        from isaaclab_ov import tensor_types as TT

        raw_backend.bindings[TT.BODY_COM_POSE]._data = coms.copy()
    _invalidate_com_caches(art)
    return coms


def _read_backend_coms(backend: str, art, raw_backend) -> np.ndarray:
    if backend == "physx":
        coms = raw_backend.get_coms()
    else:
        from isaaclab_ov import tensor_types as TT

        coms = art.root_view.get_attribute(TT.BODY_COM_POSE)
    return coms.numpy().reshape(art.num_instances, art.num_bodies, 7).copy()


def _com_payload(env_ids: list[int], body_ids: list[int], base: float) -> np.ndarray:
    """Return distinct identity-rotation COM transforms for the selected environments and bodies."""
    value = base + 100.0 * np.asarray(env_ids)[:, None] + 10.0 * np.asarray(body_ids)
    coms = np.zeros((len(env_ids), len(body_ids), 7), dtype=np.float32)
    coms[..., :3] = value[..., None] + np.arange(1.0, 4.0)
    coms[..., 6] = 1.0
    return coms


def _backend_properties(backend: str, art, raw_backend) -> dict[str, torch.Tensor]:
    """Return backend-order joint and body properties."""
    if backend == "physx":
        friction = _to_torch(raw_backend.get_dof_friction_properties())
        properties = {
            "stiffness": raw_backend.get_dof_stiffnesses(),
            "damping": raw_backend.get_dof_dampings(),
            "armature": raw_backend.get_dof_armatures(),
            "position_limits": raw_backend.get_dof_limits(),
            "velocity_limits": raw_backend.get_dof_max_velocities(),
            "effort_limits": raw_backend.get_dof_max_forces(),
            "mass": raw_backend.get_masses(),
            "inertia": raw_backend.get_inertias(),
            "com": raw_backend.get_coms(),
        }
    elif backend == "ovphysx":
        from isaaclab_ov import tensor_types as TT

        friction = _to_torch(raw_backend.bindings[TT.DOF_FRICTION_PROPERTIES]._data)
        bindings = {
            "stiffness": TT.DOF_STIFFNESS,
            "damping": TT.DOF_DAMPING,
            "armature": TT.DOF_ARMATURE,
            "position_limits": TT.DOF_LIMIT,
            "velocity_limits": TT.DOF_MAX_VELOCITY,
            "effort_limits": TT.DOF_MAX_FORCE,
            "mass": TT.BODY_MASS,
            "inertia": TT.BODY_INERTIA,
            "com": TT.BODY_COM_POSE,
        }
        properties = {name: raw_backend.bindings[binding]._data for name, binding in bindings.items()}
    else:
        data = art.data
        position_limits = torch.stack(
            (_to_torch(data._sim_bind_joint_pos_limits_lower), _to_torch(data._sim_bind_joint_pos_limits_upper)), -1
        )
        return {
            "stiffness": _to_torch(data._sim_bind_joint_stiffness_sim),
            "damping": _to_torch(data._sim_bind_joint_damping_sim),
            "armature": _to_torch(data._sim_bind_joint_armature),
            "position_limits": position_limits,
            "velocity_limits": _to_torch(data._sim_bind_joint_vel_limits_sim),
            "effort_limits": _to_torch(data._sim_bind_joint_effort_limits_sim),
            "friction": _to_torch(data._sim_bind_joint_friction_coeff),
            "viscous_friction": _to_torch(data._sim_bind_joint_viscous_friction_coeff),
            "mass": _to_torch(data._sim_bind_body_mass),
            "inertia": _to_torch(data._sim_bind_body_inertia),
            "com": _to_torch(data._sim_bind_body_com_pos_b),
        }
    properties = {name: _to_torch(value) for name, value in properties.items()}
    properties["com"] = properties["com"].reshape(art.num_instances, art.num_bodies, 7)
    properties["inertia"] = properties["inertia"].reshape(art.num_instances, art.num_bodies, 9)
    properties.update(friction=friction[..., 0], dynamic_friction=friction[..., 1], viscous_friction=friction[..., 2])
    return properties


def _write_joint_state(art, selection: str, operation: str, env_ids, joint_ids, position, velocity) -> None:
    """Write public joint state of the selected cells; the payloads have one row per selected environment."""
    values = {"position": position, "velocity": velocity}
    values = {name: value for name, value in values.items() if operation in (name, "state")}
    writer = getattr(art, f"write_joint_{operation}_to_sim_{selection}")
    if selection == "index":
        values = {name: torch.tensor(value, dtype=torch.float32, device=art.device) for name, value in values.items()}
        env_ids_wp = wp.array(env_ids, dtype=wp.int32, device=art.device)
        writer(**values, env_ids=env_ids_wp, joint_ids=wp.array(joint_ids, dtype=wp.int32, device=art.device))
        return
    full = {}
    for name, value in values.items():
        full[name] = torch.zeros((art.num_instances, art.num_joints), device=art.device)
        full[name][np.ix_(env_ids, joint_ids)] = torch.tensor(value, dtype=torch.float32, device=art.device)
    env_mask = _mask(art.num_instances, env_ids, art.device)
    writer(**full, env_mask=env_mask, joint_mask=_mask(art.num_joints, joint_ids, art.device))


def _seed_joint_state(backend: str, art, raw_backend, position: np.ndarray, velocity: np.ndarray) -> None:
    """Seed backend-order joint state and invalidate the public caches."""
    if backend == "physx":
        raw_backend._dof_positions = wp.array(position, dtype=wp.float32, device=art.device)
        raw_backend._dof_velocities = wp.array(velocity, dtype=wp.float32, device=art.device)
        art.data._joint_pos.timestamp = -1.0
        art.data._joint_vel.timestamp = -1.0
    elif backend == "ovphysx":
        _set_ovphysx_joint_state(raw_backend, position, velocity)
        for name in ("_joint_pos_buf", "_joint_vel_buf", "_joint_pos_backend", "_joint_vel_backend"):
            if getattr(art.data, name, None) is not None:
                getattr(art.data, name).timestamp = -1.0
    else:
        art.data._sim_bind_joint_pos.assign(wp.array(position, dtype=wp.float32, device=art.device))
        art.data._sim_bind_joint_vel.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        # Raw binding writes bypass the solver step, so mirror its publish into the public-order shadows.
        art.data._refresh_user_order_joint_state()


def _set_ovphysx_joint_state(raw_backend, position: np.ndarray, velocity: np.ndarray) -> None:
    from isaaclab_ov import tensor_types as TT

    raw_backend.bindings[TT.DOF_POSITION]._data = position.copy()
    raw_backend.bindings[TT.DOF_VELOCITY]._data = velocity.copy()


_INITIAL_POSITION = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
_INITIAL_VELOCITY = _INITIAL_POSITION + 100.0


def _ordered_joint_articulation(backend: str, monkeypatch, mode: str = "cyclic"):
    """Create a two-environment, three-joint articulation with seeded backend joint state."""
    art, raw_backend = get_articulation(
        backend, 2, 3, 2, device="cpu", joint_ordering=_ordering(mode, _names("joint", 3)), monkeypatch=monkeypatch
    )
    _seed_joint_state(backend, art, raw_backend, _INITIAL_POSITION, _INITIAL_VELOCITY)
    return art, raw_backend


# ---------------------------------------------------------------------------
# Ordering resolution and public reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "identity", "cyclic"])
def test_configured_ordering_resolves_public_names(monkeypatch, backend, mode):
    """Default and backend-order configurations normalize to no ordering; others map public names to backend."""
    joint_names, body_names = _names("joint", 3), _names("body", 4)
    joint_ordering, body_ordering = _ordering(mode, joint_names), _ordering(mode, body_names)
    art, _ = get_articulation(
        backend,
        2,
        3,
        4,
        device="cpu",
        joint_ordering=joint_ordering,
        body_ordering=body_ordering,
        monkeypatch=monkeypatch,
    )

    # The asset ordering properties delegate to the single source on data.
    assert art.joint_ordering is art.data.joint_ordering
    assert art.body_ordering is art.data.body_ordering
    if mode != "cyclic":
        assert art.joint_ordering is None and art.body_ordering is None
        assert (art.joint_names, art.body_names) == (list(joint_names), list(body_names))
        return
    assert (tuple(art.joint_names), tuple(art.body_names)) == (joint_ordering, body_ordering)
    np.testing.assert_array_equal(
        art.joint_ordering.backend_to_user_indices, _backend_to_user(joint_ordering, joint_names)
    )
    np.testing.assert_array_equal(
        art.body_ordering.backend_to_user_indices, _backend_to_user(body_ordering, body_names)
    )


def test_body_ordering_reorders_public_body_state(monkeypatch, backend):
    """Body quantities follow the public body order while root quantities are unchanged."""
    identity_art, identity_raw = get_articulation(backend, 2, 1, 3, device="cpu", monkeypatch=monkeypatch)
    ordered_art, ordered_raw = get_articulation(backend, 2, 1, 3, device="cpu", monkeypatch=monkeypatch)
    state = _make_body_state(2, 3)
    _set_body_state(backend, identity_art, identity_raw, state)
    _set_body_state(backend, ordered_art, ordered_raw, state)
    _, user_to_backend = _install_cyclic_ordering(ordered_art)
    identity_art.data.update(dt=0.01)
    ordered_art.data.update(dt=0.01)

    for name in ("body_com_pose_b", "body_com_acc_w", "body_link_vel_w", "body_com_pose_w"):
        _assert_proxy_close(
            getattr(ordered_art.data, name), _to_torch(getattr(identity_art.data, name))[:, user_to_backend]
        )
    for name in ("root_com_pose_w", "root_link_vel_w"):
        _assert_proxy_close(getattr(ordered_art.data, name), _to_torch(getattr(identity_art.data, name)))


@pytest.mark.parametrize("backend", backends("physx", "newton"))
@pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
def test_ordering_reorders_public_dynamics_quantities(monkeypatch, backend, is_fixed_base):
    num_instances, num_joints, num_bodies = 2, 3, 4
    identity_art, identity_raw = get_articulation(
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device="cpu",
        is_fixed_base=is_fixed_base,
        monkeypatch=monkeypatch,
    )
    identity_manager = getattr(identity_art, "_test_simulation_manager", None)
    ordered_art, ordered_raw = get_articulation(
        backend,
        num_instances,
        num_joints,
        num_bodies,
        device="cpu",
        is_fixed_base=is_fixed_base,
        monkeypatch=monkeypatch,
    )
    num_base_dofs = identity_art.num_base_dofs
    num_dofs = num_joints + num_base_dofs
    # A fixed root has no Jacobian row.
    num_jacobi_bodies = num_bodies - (1 if num_base_dofs == 0 else 0)
    jacobian = np.arange(num_instances * num_jacobi_bodies * 6 * num_dofs, dtype=np.float32)
    jacobian = jacobian.reshape(num_instances, num_jacobi_bodies, 6, num_dofs)
    mass_matrix = np.arange(num_instances * num_dofs**2, dtype=np.float32).reshape(num_instances, num_dofs, num_dofs)
    gravity = np.arange(num_instances * num_dofs, dtype=np.float32).reshape(num_instances, num_dofs)
    for art, raw_backend in ((identity_art, identity_raw), (ordered_art, ordered_raw)):
        _set_body_state(backend, art, raw_backend, _make_body_state(num_instances, num_bodies))
        if backend == "physx":
            raw_backend.set_mock_jacobians(wp.array(jacobian, dtype=wp.float32, device="cpu"))
            raw_backend.set_mock_generalized_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device="cpu"))
            raw_backend.set_mock_gravity_compensation_forces(wp.array(gravity, dtype=wp.float32, device="cpu"))
        else:
            model_jacobian = np.zeros((num_instances, num_bodies, 6, num_dofs), dtype=np.float32)
            model_jacobian[:, num_bodies - num_jacobi_bodies :] = jacobian
            raw_backend.set_mock_jacobians(wp.array(model_jacobian, dtype=wp.float32, device="cpu"))
            raw_backend.set_mock_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device="cpu"))
    if backend == "physx":
        # Buffers allocated before the ordering is installed must follow it too.
        for name in ("body_com_jacobian_w", "mass_matrix", "gravity_compensation_forces"):
            getattr(ordered_art.data, name)
    joint_user_to_backend, body_user_to_backend = _install_cyclic_ordering(ordered_art)
    identity_art.data.update(dt=0.01)
    ordered_art.data.update(dt=0.01)
    if backend == "newton":
        # Both mocks patch the same Newton module; read the shared model through the identity manager.
        import isaaclab_newton.assets.articulation.articulation_data as newton_data_module

        monkeypatch.setattr(newton_data_module, "SimulationManager", identity_manager)

    dofs = np.concatenate((np.arange(num_base_dofs), num_base_dofs + joint_user_to_backend))
    bodies = body_user_to_backend if num_base_dofs else body_user_to_backend[body_user_to_backend != 0] - 1
    for name in ("body_com_jacobian_w", "body_link_jacobian_w"):
        expected = _to_torch(getattr(identity_art.data, name))[:, bodies][:, :, :, dofs]
        _assert_proxy_close(getattr(ordered_art.data, name), expected)
    _assert_proxy_close(ordered_art.data.mass_matrix, _to_torch(identity_art.data.mass_matrix)[:, dofs][:, :, dofs])
    if backend == "physx":
        expected = _to_torch(identity_art.data.gravity_compensation_forces)[:, dofs]
        _assert_proxy_close(ordered_art.data.gravity_compensation_forces, expected)


@requires("ovphysx")
def test_ovphysx_ordered_reads_refresh_after_reset(monkeypatch):
    """Ordered OVPhysX pose, velocity, and friction shadows re-read the backend after invalidation."""
    from isaaclab_ov import tensor_types as TT

    art, raw_backend = get_articulation(
        "ovphysx",
        2,
        3,
        3,
        device="cpu",
        joint_ordering=_ordering("cyclic", _names("joint", 3)),
        body_ordering=_ordering("cyclic", _names("body", 3)),
        monkeypatch=monkeypatch,
    )
    state = list(_make_body_state(2, 3))
    # Zero COM offsets make the link and COM velocities equal.
    state[3][..., :3] = 0.0
    _set_body_state("ovphysx", art, raw_backend, state)
    body_user_to_backend = _user_to_backend(art.body_ordering, 3)
    joint_user_to_backend = _user_to_backend(art.joint_ordering, 3)
    art.data.update(dt=0.01)
    for name in ("body_link_pose_w", "body_com_vel_w", "body_link_vel_w", "root_link_vel_w"):
        getattr(art.data, name).torch.clone()

    link_pose = state[2].copy()
    link_pose[..., 0] += 1000.0
    velocity = state[4].copy()
    velocity[..., 0] += 500.0
    raw_backend.bindings[TT.LINK_POSE]._data = link_pose
    raw_backend.bindings[TT.LINK_VELOCITY]._data = velocity
    art.data._reset_pose()
    art.data._reset_velocity()

    _assert_proxy_close(art.data.body_link_pose_w, torch.from_numpy(link_pose[:, body_user_to_backend]))
    _assert_proxy_close(art.data.body_com_vel_w, torch.from_numpy(velocity[:, body_user_to_backend]))
    _assert_proxy_close(art.data.body_link_vel_w, torch.from_numpy(velocity[:, body_user_to_backend]))
    _assert_proxy_close(art.data.root_link_vel_w, torch.from_numpy(velocity[:, 0]))

    # A fresh native read must gather too; initialization and writer caches can hide a missing gather.
    friction = 100.0 + np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3)
    raw_backend.bindings[TT.DOF_FRICTION_PROPERTIES]._data = friction
    art.data._joint_friction_props_buf.timestamp = -1
    art.data._joint_friction_props_backend.timestamp = -1
    for component, name in enumerate(("friction_coeff", "dynamic_friction_coeff", "viscous_friction_coeff")):
        expected = torch.from_numpy(friction[:, joint_user_to_backend, component])
        _assert_proxy_close(getattr(art.data, f"joint_{name}"), expected)


@requires("ovphysx")
def test_ovphysx_ordered_reads_gather_once_per_step(monkeypatch):
    """Ordered root and body velocities share one backend read, and joint state is gathered once per step."""
    from isaaclab_ov import tensor_types as TT

    art, raw_backend = get_articulation(
        "ovphysx",
        2,
        3,
        3,
        device="cpu",
        joint_ordering=("joint_2", "joint_1", "joint_0"),
        body_ordering=("body_2", "body_1", "body_0"),
        monkeypatch=monkeypatch,
    )
    _set_body_state("ovphysx", art, raw_backend, _make_body_state(2, 3))
    art.data.update(dt=0.01)
    art.data._binding_read = MagicMock(wraps=art.data._binding_read)

    for _ in range(2):
        for name in ("body_com_vel_w", "root_link_vel_w", "joint_pos", "joint_vel"):
            getattr(art.data, name).torch.clone()

    reads = [call.args[0] for call in art.data._binding_read.call_args_list]
    assert [reads.count(binding) for binding in (TT.LINK_VELOCITY, TT.DOF_POSITION, TT.DOF_VELOCITY)] == [1, 1, 1]


@requires("ovphysx")
def test_ovphysx_joint_acceleration_differences_public_order_velocities(monkeypatch):
    art, raw_backend = _ordered_joint_articulation("ovphysx", monkeypatch)
    user_to_backend = _user_to_backend(art.joint_ordering, 3)
    first = np.asarray([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]], dtype=np.float32)
    second = first + np.asarray([[3.0, 5.0, 7.0], [11.0, 13.0, 17.0]], dtype=np.float32)
    _set_ovphysx_joint_state(raw_backend, _INITIAL_POSITION, first)
    art.data.update(0.1)
    art.data.joint_acc.torch.clone()
    _set_ovphysx_joint_state(raw_backend, _INITIAL_POSITION, second)
    art.data.update(0.1)

    torch.testing.assert_close(art.data.joint_vel.torch, torch.from_numpy(second[:, user_to_backend]))
    torch.testing.assert_close(art.data.joint_acc.torch, torch.from_numpy((second - first)[:, user_to_backend] / 0.1))


# ---------------------------------------------------------------------------
# Property writes
# ---------------------------------------------------------------------------

# (writer, keyword, axis, backend property, public getter, trailing size)
_PROPERTY_WRITERS = [
    ("write_joint_stiffness_to_sim", "stiffness", "joint", "stiffness", "joint_stiffness", 0),
    ("write_joint_damping_to_sim", "damping", "joint", "damping", "joint_damping", 0),
    ("write_joint_armature_to_sim", "armature", "joint", "armature", "joint_armature", 0),
    ("write_joint_position_limit_to_sim", "limits", "joint", "position_limits", "joint_pos_limits", 2),
    ("write_joint_velocity_limit_to_sim", "limits", "joint", "velocity_limits", "joint_vel_limits", 0),
    ("write_joint_effort_limit_to_sim", "limits", "joint", "effort_limits", "joint_effort_limits", 0),
    ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", "joint", "friction", "joint_friction_coeff", 0),
    *[
        (f"write_joint_{kind}_friction_coefficient_to_sim", f"joint_{kind}_friction_coeff", "joint")
        + (f"{kind}_friction", f"joint_{kind}_friction_coeff", 0)
        for kind in ("dynamic", "viscous")
    ],
    ("set_masses", "masses", "body", "mass", "body_mass", 0),
    ("set_inertias", "inertias", "body", "inertia", "body_inertia", 9),
    ("set_coms", "coms", "body", "com", "body_com_pose_b", 7),
]


def _property_payload(shape: tuple[int, ...], trailing: int, offset: float) -> torch.Tensor:
    """Return distinct property values; limits are ``[-v, v]`` pairs and COMs carry an identity rotation."""
    values = offset + torch.arange(np.prod(shape), dtype=torch.float32).reshape(shape)
    if trailing == 2:
        return torch.stack((-values, values), dim=-1)
    if trailing:
        values = values.unsqueeze(-1) + torch.arange(trailing, dtype=torch.float32) / 10.0
    if trailing == 7:
        values[..., 3:] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    return values


@pytest.mark.parametrize("mode", ["none", "cyclic"])
@pytest.mark.parametrize("selection", ["index", "mask"])
def test_property_writes_route_to_backend_and_preserve_other_rows(monkeypatch, backend, selection, mode):
    """Partial public property writes reach the matching backend joints and bodies and keep every other entry."""
    num_instances, count = 2, 4
    if backend == "newton":
        from isaaclab_newton.physics import NewtonManager

        monkeypatch.setattr(NewtonManager, "add_model_change", MagicMock())
    art, raw_backend = get_articulation(
        backend,
        num_instances,
        count,
        count,
        device="cpu",
        joint_ordering=_ordering(mode, _names("joint", count)),
        body_ordering=_ordering(mode, _names("body", count)),
        monkeypatch=monkeypatch,
    )
    user_to_backend = {"joint": _user_to_backend(art.joint_ordering, count)}
    user_to_backend["body"] = _user_to_backend(art.body_ordering, count)
    expected = _backend_properties(backend, art, raw_backend)

    for case, (writer, kwarg, axis, name, getter, trailing) in enumerate(_PROPERTY_WRITERS):
        if name not in expected:
            continue  # Newton has no dynamic joint friction.
        if backend == "newton" and name == "com":
            # Newton stores the COM as a position only.
            getter, trailing = "body_com_pos_b", 3
        # A write to every environment and two items, then one to a single environment and two unsorted items.
        for env_ids, items in (([0, 1], [0, 2]), ([1], [3, 1])):
            soft_limits_before = art.data.soft_joint_pos_limits.torch.clone()
            values = _property_payload(
                (num_instances, count), trailing, offset=100.0 * (case + 1) + 10.0 * len(env_ids)
            )
            selected = values[np.ix_(env_ids, items)]
            if selection == "index":
                kwargs = {kwarg: selected, "env_ids": env_ids, f"{axis}_ids": items}
            else:
                kwargs = {kwarg: values, "env_mask": _mask(num_instances, env_ids, "cpu")}
                kwargs[f"{axis}_mask"] = _mask(count, items, "cpu")
            getattr(art, f"{writer}_{selection}")(**kwargs)

            expected[name][np.ix_(env_ids, user_to_backend[axis][items])] = selected
            backend_values = _backend_properties(backend, art, raw_backend)[name]
            torch.testing.assert_close(backend_values, expected[name], rtol=0.0, atol=0.0, msg=f"{writer}: {mode}")
            _assert_proxy_close(getattr(art.data, getter), expected[name][:, user_to_backend[axis]])
            if name == "position_limits":
                # The soft limits of the written joints follow (factor 1.0); the others are untouched.
                soft_limits = art.data.soft_joint_pos_limits.torch
                torch.testing.assert_close(soft_limits[np.ix_(env_ids, items)], selected)
                untouched = [joint for joint in range(count) if joint not in items]
                torch.testing.assert_close(soft_limits[:, untouched], soft_limits_before[:, untouched])


@pytest.mark.parametrize("backend", backends("physx", "ovphysx"))
@pytest.mark.parametrize("mode", ["none", "cyclic"])
def test_duplicate_com_selectors_preserve_omitted_rows(monkeypatch, backend, mode):
    """Duplicate body or environment selectors write their last value and keep every omitted backend row."""
    art, raw_backend = get_articulation(
        backend, 2, 1, 4, device="cpu", body_ordering=_ordering(mode, _names("body", 4)), monkeypatch=monkeypatch
    )
    expected = _seed_backend_coms(backend, art, raw_backend)
    user_to_backend = _user_to_backend(art.body_ordering, 4)
    if backend == "physx":
        raw_backend.get_coms = MagicMock(wraps=raw_backend.get_coms)

    # Duplicate body IDs omit body 3.
    body_ids = [0, 1, 1, 2]
    payload = _com_payload([0, 1], body_ids, 3000.0)
    art.set_coms_index(coms=wp.array(payload, dtype=wp.transformf), body_ids=wp.array(body_ids, dtype=wp.int32))
    public = art.data.body_com_pose_b.torch.numpy().copy()
    if backend == "physx":
        # The cold partial write seeds untouched bodies with one backend read; the public read reuses it.
        assert raw_backend.get_coms.call_count == 1
    for offset, body in enumerate(body_ids):
        expected[:, user_to_backend[body]] = payload[:, offset]
    np.testing.assert_array_equal(_read_backend_coms(backend, art, raw_backend), expected)
    np.testing.assert_array_equal(public, expected[:, user_to_backend])

    # From cold caches, duplicate environment IDs omit environment 0 and must not mark the global COM caches valid.
    _invalidate_com_caches(art)
    env_ids = [1, 1]
    payload = _com_payload(env_ids, [0, 1, 2, 3], 4000.0)
    art.set_coms_index(coms=wp.array(payload, dtype=wp.transformf), env_ids=wp.array(env_ids, dtype=wp.int32))
    assert art.data._body_com_pose_b.timestamp < 0.0
    backend_staging = art.data._body_com_pose_b_backend
    assert backend_staging is None or backend_staging.timestamp < 0.0
    expected[1, user_to_backend] = payload[-1]
    np.testing.assert_array_equal(_read_backend_coms(backend, art, raw_backend), expected)
    np.testing.assert_array_equal(art.data.body_com_pose_b.torch.numpy(), expected[:, user_to_backend])


@pytest.mark.parametrize("backend", backends("physx", "ovphysx"))
@pytest.mark.parametrize("mode", ["none", "cyclic"])
def test_static_com_cache_does_not_follow_sim_timestamp(monkeypatch, backend, mode):
    """The COM is read from the backend once; later steps keep serving the cached public and root COM poses."""
    art, raw_backend = get_articulation(
        backend, 2, 1, 4, device="cpu", body_ordering=_ordering(mode, _names("body", 4)), monkeypatch=monkeypatch
    )
    root_pose = np.zeros((2, 7), dtype=np.float32)
    root_pose[:, :3] = [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]]
    root_pose[:, 6] = 1.0
    if backend == "physx":
        raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32)
        raw_backend.get_coms = read_spy = MagicMock(wraps=raw_backend.get_coms)
    else:
        from isaaclab_ov import tensor_types as TT

        raw_backend.bindings[TT.ROOT_POSE]._data = root_pose.copy()
        com_binding = raw_backend.bindings[TT.BODY_COM_POSE]
        com_binding.read = read_spy = MagicMock(wraps=com_binding.read)
    coms = _seed_backend_coms(backend, art, raw_backend)
    expected_root = root_pose.copy()
    expected_root[:, :3] += coms[:, 0, :3]

    for _ in range(3):
        np.testing.assert_array_equal(
            art.data.body_com_pose_b.torch.numpy(), coms[:, _user_to_backend(art.body_ordering, 4)]
        )
        np.testing.assert_array_equal(art.data.root_com_pose_w.torch.numpy(), expected_root)
        assert read_spy.call_count == 1
        art.data.update(dt=0.01)


# ---------------------------------------------------------------------------
# Root writes
# ---------------------------------------------------------------------------


def _read_backend_root_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Read the backend root link pose and COM velocity."""
    if backend == "physx":
        return raw_backend._root_transforms.numpy(), art.data.root_com_vel_w.warp.numpy()
    if backend == "ovphysx":
        from isaaclab_ov import tensor_types as TT

        return raw_backend.bindings[TT.ROOT_POSE]._data, raw_backend.bindings[TT.ROOT_VELOCITY]._data
    return art.data._sim_bind_root_link_pose_w.numpy(), art.data._sim_bind_root_com_vel_w.numpy()


@pytest.mark.parametrize("operation", ["com_pose", "link_velocity"])
@pytest.mark.parametrize("selection", ["index", "mask"])
def test_floating_root_writers_use_the_root_body_com_after_body_reordering(monkeypatch, backend, selection, operation):
    """Root COM conversions use the backend root body, whichever body is first in public order."""
    results = []
    for body_ordering in (_names("body", 4), _ordering("cyclic", _names("body", 4))):
        art, raw_backend = get_articulation(
            backend, 2, 1, 4, device="cpu", body_ordering=body_ordering, monkeypatch=monkeypatch
        )
        state = list(_make_body_state(2, 4))
        state[3][..., :3] = 1.0 + np.arange(4)[:, None] + np.arange(3)
        _set_body_state(backend, art, raw_backend, state)
        pose = wp.array(
            [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], [14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 1.0]], dtype=wp.transformf
        )
        velocity = wp.array(
            [[1.0, 2.0, 3.0, 0.4, 0.5, 0.6], [11.0, 12.0, 13.0, 1.4, 1.5, 1.6]], dtype=wp.spatial_vectorf
        )
        if selection == "index":
            selector, pose, velocity = {"env_ids": wp.array([0], dtype=wp.int32)}, pose[:1], velocity[:1]
        else:
            selector = {"env_mask": _mask(2, [0], "cpu")}
        if operation == "com_pose":
            getattr(art, f"write_root_com_pose_to_sim_{selection}")(root_pose=pose, **selector)
        else:
            getattr(art, f"write_root_link_pose_to_sim_{selection}")(root_pose=pose, **selector)
            getattr(art, f"write_root_link_velocity_to_sim_{selection}")(root_velocity=velocity, **selector)
        results.append(_read_backend_root_state(backend, art, raw_backend)[0 if operation == "com_pose" else 1][0])

    assert art.body_names[0] != "body_0"
    np.testing.assert_allclose(results[1], results[0], atol=1e-6)


# ---------------------------------------------------------------------------
# Joint state writes
# ---------------------------------------------------------------------------


# Each backend selects the joint map separately in every ordered writer, so every ordered (selection, operation) pair
# has one row; the selected environments and joints alternate across rows.
@pytest.mark.parametrize(
    "mode, selection, operation, env_ids, joint_ids",
    [
        ("none", "index", "state", [0, 1], [0]),
        ("none", "mask", "position", [1], [0, 1, 2]),
        ("cyclic", "index", "position", [1], [0, 1, 2]),
        ("cyclic", "index", "velocity", [0, 1], [0]),
        ("cyclic", "index", "state", [1], [2, 0, 1]),
        ("cyclic", "mask", "position", [0, 1], [0]),
        ("cyclic", "mask", "velocity", [1], [0, 1, 2]),
        ("cyclic", "mask", "state", [0, 1], [2]),
    ],
)
def test_partial_joint_write_preserves_backend_rows(
    monkeypatch, backend, mode, selection, operation, env_ids, joint_ids
):
    art, raw_backend = _ordered_joint_articulation(backend, monkeypatch, mode)
    user_to_backend = _user_to_backend(art.joint_ordering, 3)
    expected = {"position": _INITIAL_POSITION.copy(), "velocity": _INITIAL_VELOCITY.copy()}

    for base in (900.0, 1900.0):
        payload = base + 10.0 * np.asarray(env_ids, dtype=np.float32)[:, None] + np.asarray(joint_ids)
        _write_joint_state(art, selection, operation, env_ids, joint_ids, payload, payload + 0.5)
        cells = np.ix_(env_ids, user_to_backend[joint_ids])
        if operation in ("position", "state"):
            expected["position"][cells] = payload
        if operation in ("velocity", "state"):
            expected["velocity"][cells] = payload + 0.5
        position, velocity = read_backend_joint_state(backend, art, raw_backend)
        np.testing.assert_array_equal(position, expected["position"])
        np.testing.assert_array_equal(velocity, expected["velocity"])

    np.testing.assert_array_equal(art.data.joint_pos.torch.numpy(), expected["position"][:, user_to_backend])
    np.testing.assert_array_equal(art.data.joint_vel.torch.numpy(), expected["velocity"][:, user_to_backend])


@pytest.mark.parametrize("backend", backends("physx", "ovphysx"))
@pytest.mark.parametrize("operation", ["position", "velocity", "state"])
def test_partial_joint_write_after_a_step_preserves_newer_backend_rows(monkeypatch, backend, operation):
    """Partial writes restage the backend rows once the simulation advances, also for duplicate full-length selectors.

    Newton writes the selected cells straight into the simulation buffers and has no row staging that could go stale.
    """
    art, raw_backend = _ordered_joint_articulation(backend, monkeypatch)
    user_to_backend = _user_to_backend(art.joint_ordering, 3)
    art.data.joint_pos, art.data.joint_vel
    _write_joint_state(art, "index", operation, [1], [0], np.full((1, 1), 901.0), np.full((1, 1), 902.0))

    # The simulation advances and changes every backend row.
    art.data._sim_timestamp += 0.01
    newer = {"position": _INITIAL_POSITION + 1000.0, "velocity": _INITIAL_VELOCITY + 1000.0}
    if backend == "physx":
        raw_backend._dof_positions.assign(wp.array(newer["position"], dtype=wp.float32))
        raw_backend._dof_velocities.assign(wp.array(newer["velocity"], dtype=wp.float32))
    else:
        _set_ovphysx_joint_state(raw_backend, newer["position"], newer["velocity"])

    # A duplicate full-length selector omits joint 2, whose newer backend value must survive.
    joint_ids = [0, 1, 1]
    payload = np.asarray([[911.0, 912.0, 913.0]], dtype=np.float32)
    _write_joint_state(art, "index", operation, [1], joint_ids, payload, payload + 0.5)
    for name, value in (("position", payload), ("velocity", payload + 0.5)):
        if operation in (name, "state"):
            for offset, joint in enumerate(joint_ids):
                newer[name][1, user_to_backend[joint]] = value[0, offset]
    position, velocity = read_backend_joint_state(backend, art, raw_backend)
    np.testing.assert_array_equal(position, newer["position"])
    np.testing.assert_array_equal(velocity, newer["velocity"])


@requires("physx")
@pytest.mark.parametrize("operation", ["position", "velocity", "state"])
@pytest.mark.parametrize("selector", ["partial_mask", "default_mask", "explicit_full_index"])
def test_physx_joint_writes_read_the_backend_only_for_partial_selectors(monkeypatch, operation, selector):
    """A provably complete write skips the staging read; partial and opaque selectors read each axis once."""
    art, raw_backend = _ordered_joint_articulation("physx", monkeypatch)
    raw_backend.get_dof_positions = MagicMock(wraps=raw_backend.get_dof_positions)
    raw_backend.get_dof_velocities = MagicMock(wraps=raw_backend.get_dof_velocities)

    if selector == "partial_mask":
        _write_joint_state(art, "mask", operation, [1], [0], np.full((1, 1), 901.0), np.full((1, 1), 902.0))
    else:
        values = {"position": torch.tensor(_INITIAL_POSITION), "velocity": torch.tensor(_INITIAL_VELOCITY)}
        values = {name: value for name, value in values.items() if operation in (name, "state")}
        if selector == "default_mask":
            getattr(art, f"write_joint_{operation}_to_sim_mask")(**values)
        else:
            joint_ids = wp.array([0, 1, 2], dtype=wp.int32)
            getattr(art, f"write_joint_{operation}_to_sim_index")(**values, joint_ids=joint_ids)

    reads = int(selector != "default_mask")
    assert raw_backend.get_dof_positions.call_count == (reads if operation in ("position", "state") else 0)
    assert raw_backend.get_dof_velocities.call_count == (reads if operation in ("velocity", "state") else 0)


# ---------------------------------------------------------------------------
# Commands, wrenches, and configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["none", "cyclic"])
@pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
def test_external_wrenches_are_written_in_backend_body_order(monkeypatch, backend, mode, is_fixed_base):
    """Public-order body wrenches reach each backend in backend body order and the requested frame."""
    num_instances, num_bodies = 2, 4
    backend_body_names = _names("body", num_bodies)
    body_ordering = _ordering(mode, backend_body_names, keep_first=is_fixed_base)
    art, raw_backend = get_articulation(
        backend,
        num_instances,
        1,
        num_bodies,
        device="cpu",
        is_fixed_base=is_fixed_base,
        body_ordering=body_ordering,
        monkeypatch=monkeypatch,
    )
    # Distinct quarter-turn rotations about Z and nonzero positions, in backend body order.
    poses = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    poses[..., 0] = np.arange(num_bodies) + 10.0
    poses[..., 5] = np.sin(np.arange(num_bodies) * np.pi / 4)
    poses[..., 6] = np.cos(np.arange(num_bodies) * np.pi / 4)
    captured = {}
    if backend == "physx":

        def capture_wrench(*, force_data, torque_data, position_data, indices, is_global):
            assert position_data is None
            captured.update(is_global=is_global, force=force_data.numpy().copy(), torque=torque_data.numpy().copy())

        raw_backend.apply_forces_and_torques_at_position = capture_wrench
    elif backend == "newton":
        poses_wp = wp.array(poses[:, None], dtype=wp.transformf)
        raw_backend.set_mock_link_transforms(poses_wp)
        art.data._sim_bind_body_link_pose_w.assign(poses_wp[:, 0])
        art.data._refresh_user_order_body_state()
    else:
        from isaaclab_ov import tensor_types as TT

        raw_backend.bindings[TT.LINK_POSE]._data = poses
        art.data._reset_pose()

    forces = np.arange(num_instances * num_bodies * 3, dtype=np.float32).reshape(num_instances, num_bodies, 3)
    torques = forces + 100.0
    backend_to_user = _backend_to_user(body_ordering, backend_body_names)
    composer = art.instantaneous_wrench_composer
    for is_global in (False, True):
        composer.set_forces_and_torques_index(
            forces=wp.array(forces, dtype=wp.vec3f), torques=wp.array(torques, dtype=wp.vec3f), is_global=is_global
        )
        with patch.object(composer, "compose_to_body_frame", wraps=composer.compose_to_body_frame) as compose:
            art.write_data_to_sim()
        assert compose.call_count == int(is_global and backend == "newton")

        expected_force, expected_torque = forces[:, backend_to_user], torques[:, backend_to_user]
        if backend == "physx":
            # PhysX receives the frame flag and rotates the wrench itself.
            assert captured["is_global"] is is_global
            force, torque = captured["force"], captured["torque"]
        else:
            if not is_global:
                # The 0/90/180/270-degree body rotations, independent of the backend quaternion transform.
                rotations = np.asarray(
                    [
                        [[1, 0, 0], [0, 1, 0]],
                        [[0, -1, 0], [1, 0, 0]],
                        [[-1, 0, 0], [0, -1, 0]],
                        [[0, 1, 0], [-1, 0, 0]],
                    ],
                    dtype=np.float32,
                )
                rotations = np.concatenate((rotations, np.tile([[[0, 0, 1]]], (4, 1, 1))), axis=1)
                expected_force = np.einsum("bij,nbj->nbi", rotations, expected_force)
                expected_torque = np.einsum("bij,nbj->nbi", rotations, expected_torque)
            if backend == "newton":
                wrench = art.data._sim_bind_body_external_wrench.numpy()
            else:
                wrench = raw_backend.bindings[TT.LINK_WRENCH]._data
                # OVPhysX packs the link positions after the wrench.
                np.testing.assert_array_equal(wrench[..., 6:9], poses[..., :3])
            force, torque = wrench[..., :3], wrench[..., 3:6]
        np.testing.assert_allclose(force.reshape(expected_force.shape), expected_force, atol=1e-4)
        np.testing.assert_allclose(torque.reshape(expected_torque.shape), expected_torque, atol=1e-4)


@requires("ovphysx")
def test_ovphysx_actuator_commands_are_written_in_backend_order(monkeypatch):
    """Implicit position, velocity, and effort commands reach their matching backend joints."""
    from isaaclab_ov import tensor_types as TT

    art, raw_backend = _ordered_joint_articulation("ovphysx", monkeypatch)
    backend_to_user = _backend_to_user(tuple(art.joint_names), _names("joint", 3))
    position = np.arange(6, dtype=np.float32).reshape(2, 3)
    # Seed the actuator collection's submitted command buffers (no actuator groups recompute them).
    art.actuators._joint_pos_target.assign(wp.array(position, dtype=wp.float32))
    art.actuators._joint_vel_target.assign(wp.array(position + 100.0, dtype=wp.float32))
    art.actuators._joint_effort_target_sim.assign(wp.array(position + 200.0, dtype=wp.float32))
    object.__setattr__(art, "_has_implicit_actuators", True)
    object.__setattr__(art, "_can_write_effort", True)

    art.write_data_to_sim()

    for binding, offset in (
        (TT.DOF_POSITION_TARGET, 0.0),
        (TT.DOF_VELOCITY_TARGET, 100.0),
        (TT.DOF_ACTUATION_FORCE, 200.0),
    ):
        np.testing.assert_array_equal(raw_backend.bindings[binding]._data, (position + offset)[:, backend_to_user])


@requires("ovphysx")
def test_ovphysx_partial_effort_target_write_keeps_commanded_targets(monkeypatch):
    """A partial effort-target write keeps the other joints' commanded targets, not the last applied effort.

    The applied effort that ``write_data_to_sim`` pushes can differ from the commanded target (for example once explicit
    actuators clip it), so it must not share the staging that a partial target write reuses for unselected joints.
    """
    from isaaclab_ov import tensor_types as TT

    art, raw_backend = _ordered_joint_articulation("ovphysx", monkeypatch)
    object.__setattr__(art, "_can_write_effort", True)
    backend_to_user = _backend_to_user(tuple(art.joint_names), _names("joint", 3))
    targets = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    art.set_joint_effort_target_mask(target=torch.tensor(targets))
    # An actuator model applies an effort that differs from the commanded target.
    art.actuators._joint_effort_target_sim.assign(wp.array(targets + 100.0, dtype=wp.float32))
    art.write_data_to_sim()
    np.testing.assert_array_equal(
        raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data, (targets + 100.0)[:, backend_to_user]
    )

    art.set_joint_effort_target_index(target=torch.tensor([[901.0], [902.0]]), joint_ids=[1], env_ids=[0, 1])

    targets[:, 1] = [901.0, 902.0]
    np.testing.assert_array_equal(raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data, targets[:, backend_to_user])


@requires("ovphysx")
@pytest.mark.parametrize("selector_kind", ["torch", "warp"])
def test_ovphysx_int64_effort_target_selector_reaches_binding(monkeypatch, selector_kind):
    """Indexed effort-target environment selectors reach the OVPhysX binding as int32 Warp indices."""
    from isaaclab_ov import tensor_types as TT

    art, raw_backend = get_articulation("ovphysx", 2, 3, 2, device="cpu", monkeypatch=monkeypatch)
    object.__setattr__(art, "_can_write_effort", True)
    expected = raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data.copy()
    set_attribute = art._root_view.set_attribute
    captured = []

    def strict_set_attribute(name, values, *, indices=None, mask=None):
        if name == TT.DOF_ACTUATION_FORCE:
            assert isinstance(indices, wp.array) and indices.dtype == wp.int32 and str(indices.device) == art.device
            captured.append(indices.numpy().tolist())
        set_attribute(name, values, indices=indices, mask=mask)

    monkeypatch.setattr(art._root_view, "set_attribute", strict_set_attribute, raising=False)
    env_ids = torch.tensor([1], dtype=torch.int64)
    if selector_kind == "warp":
        env_ids = wp.from_torch(env_ids, dtype=wp.int64)

    art.set_joint_effort_target_index(target=torch.tensor([[7.0]]), env_ids=env_ids, joint_ids=torch.tensor([2]))

    expected[1] = [0.0, 0.0, 7.0]
    assert captured == [[1]]
    np.testing.assert_array_equal(raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data, expected)


@requires("ovphysx")
def test_ovphysx_configured_defaults_use_public_joint_names(monkeypatch):
    art, _ = get_articulation(
        "ovphysx", 1, 3, 2, device="cpu", joint_ordering=("joint_2", "joint_1", "joint_0"), monkeypatch=monkeypatch
    )
    patterns = {"joint_0": 10.0, "joint_1": 20.0, "joint_2": 30.0}

    art._resolve_joint_values(patterns, art.data._default_joint_pos)

    expected = [[patterns[name] for name in art.joint_names]]
    np.testing.assert_array_equal(art.data.default_joint_pos.warp.numpy(), expected)


@requires("physx")
def test_physx_newton_actuator_forces_are_written_in_backend_order(monkeypatch):
    art, raw_backend = get_articulation(
        "physx",
        2,
        4,
        2,
        device="cpu",
        joint_ordering=_ordering("cyclic", _names("joint", 4)),
        monkeypatch=monkeypatch,
    )
    forces = np.arange(8, dtype=np.float32).reshape(2, 4)
    object.__setattr__(art, "_joint_effort_target_backend", wp.zeros_like(art.data.joint_effort_target.warp))
    object.__setattr__(art, "_physx_actuator_wrapper", MagicMock(joint_f_2d=wp.array(forces, dtype=wp.float32)))
    object.__setattr__(art, "_has_newton_actuators", True)
    object.__setattr__(art, "_has_implicit_actuators", False)
    written = []
    raw_backend.set_dof_actuation_forces = lambda values, indices: written.append(values.numpy().copy())

    art.write_data_to_sim()

    np.testing.assert_allclose(written[-1], forces[:, _backend_to_user(tuple(art.joint_names), _names("joint", 4))])


@requires("physx")
def test_physx_validate_cfg_reports_velocity_limits_in_public_joint_order(monkeypatch):
    """Pair public default velocities with limits for the same named joint."""
    art, raw_backend = get_articulation(
        "physx", 1, 3, 2, device="cpu", joint_ordering=("joint_2", "joint_1", "joint_0"), monkeypatch=monkeypatch
    )
    backend_velocity_limits = np.asarray([[1.0, 100.0, 100.0]], dtype=np.float32)
    raw_backend.set_mock_dof_max_velocities(wp.array(backend_velocity_limits, dtype=wp.float32))
    art.data._joint_vel_limits.assign(wp.array(backend_velocity_limits[:, ::-1].copy(), dtype=wp.float32))
    art.data._joint_pos_limits.assign(wp.array(np.tile((-100.0, 100.0), (1, 3, 1)), dtype=wp.vec2f))
    art.data._default_joint_pos.zero_()
    art.data._default_joint_vel.assign(wp.array([[50.0, 50.0, 2.0]], dtype=wp.float32))

    with pytest.raises(ValueError, match=r"'joint_0': 2\.000 not in \[-1\.000, 1\.000\]"):
        art._validate_cfg()
