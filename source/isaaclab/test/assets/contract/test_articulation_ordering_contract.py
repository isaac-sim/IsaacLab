# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Cross-backend articulation joint- and body-ordering contracts on mocked backends.

A configured ordering exposes joints and bodies under public names while the backend keeps its own order. These cases
seed backend-order storage, act through the public API, and compare against independently derived expectations.
Cyclic orderings are preferred because, unlike reversals, they are not their own inverse, so a swapped map fails.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.utils import replace

from .articulation_factory import get_articulation
from .backends import backends, read_backend_joint_state, requires

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def _cpu_warp_device():
    """Every ordering case runs on CPU; allocate unqualified Warp arrays there too."""
    with wp.ScopedDevice("cpu"):
        yield


# ---------------------------------------------------------------------------
# Helpers
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


def _mask(total: int, selected, device: str) -> wp.array:
    mask = np.zeros(total, dtype=bool)
    mask[list(selected)] = True
    return wp.array(mask, dtype=wp.bool, device=device)


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
