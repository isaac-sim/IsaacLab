# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Shared articulation contracts across the production backends.

Checks that every articulation backend provides the data and writer behavior the base articulation class advertises.
The backends run on mocked views, so these cases need neither Isaac Sim nor a GPU simulation. Pure bookkeeping runs on
CPU only; getters and writers also run on CUDA because PhysX stages through CPU-pinned buffers there.
"""

import math
import warnings
from operator import attrgetter
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.assets.articulation.base_articulation_data import BaseArticulationData
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
@pytest.mark.xfail(
    raises=pytest.fail.Exception,
    strict=True,
    reason="Newton fixed-tendon mask setters slice full data without validating its shape",
)
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
