# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# pyright: reportPrivateUsage=none

"""Mocked cross-backend articulation ordering interface tests."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import warp as wp
from _articulation_iface_test_utils import BACKEND_UNAVAILABLE_REASONS, BACKENDS, get_articulation
from _pytest.mark.structures import ParameterSet

from isaaclab.utils.buffers import TimestampedBufferWarp
from isaaclab.utils.wrench_composer import WrenchComposer


def _make_body_ordering_backend_data(num_instances: int, num_bodies: int) -> tuple[np.ndarray, ...]:
    """Create deterministic backend-order body data with identity rotations."""
    root_pose = np.zeros((num_instances, 7), dtype=np.float32)
    root_pose[:, 6] = 1.0
    root_vel = np.zeros((num_instances, 6), dtype=np.float32)
    link_pose = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    com_pose_b = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    body_com_vel = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)
    body_acc = np.zeros((num_instances, num_bodies, 6), dtype=np.float32)

    for env_index in range(num_instances):
        root_pose[env_index, :3] = (10.0 + env_index, 0.0, 0.0)
        root_vel[env_index, 0] = 3.0 + env_index
        root_vel[env_index, 5] = 7.0 + env_index
        for body_index in range(num_bodies):
            link_pose[env_index, body_index, :3] = (10.0 * body_index, float(env_index), 0.0)
            link_pose[env_index, body_index, 6] = 1.0
            com_pose_b[env_index, body_index, :3] = (float(body_index + 1), 0.0, 0.0)
            com_pose_b[env_index, body_index, 6] = 1.0
            body_com_vel[env_index, body_index, 0] = 20.0 + body_index
            body_com_vel[env_index, body_index, 5] = 30.0 + body_index
            body_acc[env_index, body_index, 0] = 100.0 + body_index
            body_acc[env_index, body_index, 3] = 200.0 + body_index

    return root_pose, root_vel, link_pose, com_pose_b, body_com_vel, body_acc


def _install_test_body_ordering(art) -> np.ndarray:
    """Install a non-identity body ordering that preserves a fixed root body."""
    if art.is_fixed_base:
        body_ordering = (art.backend_body_names[0], *reversed(art.backend_body_names[1:]))
    else:
        body_ordering = tuple(reversed(art.backend_body_names))
    art.cfg = art.cfg.replace(body_ordering=body_ordering)
    # Re-resolve and re-stage the ordering maps exactly as backend initialization
    # does after a config change installs a new ordering on an already-initialized
    # articulation.
    art._resolve_and_install_ordering_maps()
    art._ordering_configure_backend_staging()
    return np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)


def _install_reversed_joint_ordering(art) -> np.ndarray:
    """Install a reversed public joint ordering on an already constructed articulation."""
    art.cfg = art.cfg.replace(joint_ordering=tuple(reversed(art.backend_joint_names)))
    # Re-resolve and re-stage the ordering maps exactly as backend initialization
    # does after a config change installs a new ordering on an already-initialized
    # articulation.
    art._resolve_and_install_ordering_maps()
    art._ordering_configure_backend_staging()
    return np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)


def _joint_ordering_for_mode(mode: str, num_joints: int) -> tuple[str, ...] | None:
    """Return the configured public joint ordering for a parity-test mode."""
    backend_names = tuple(f"joint_{index}" for index in range(num_joints))
    if mode == "none":
        return None
    if mode == "identity":
        return backend_names
    if mode == "reversed":
        return tuple(reversed(backend_names))
    if mode == "cyclic":
        return (backend_names[-1], *backend_names[:-1])
    raise ValueError(f"Unsupported joint ordering mode: {mode}")


def _body_ordering_for_mode(mode: str, num_bodies: int) -> tuple[str, ...] | None:
    """Return the configured public body ordering for a parity-test mode."""
    backend_names = tuple(f"body_{index}" for index in range(num_bodies))
    if mode == "none":
        return None
    if mode == "identity":
        return backend_names
    if mode == "reversed":
        return tuple(reversed(backend_names))
    if mode == "cyclic":
        return (backend_names[-1], *backend_names[:-1])
    raise ValueError(f"Unsupported body ordering mode: {mode}")


def _ordering_shadow_shape(buffer: wp.array | TimestampedBufferWarp) -> tuple[int, ...]:
    """Return the allocation shape for a raw or timestamped ordering shadow."""
    if isinstance(buffer, TimestampedBufferWarp):
        buffer = buffer.data
    return tuple(buffer.shape)


def _ordering_user_to_backend(ordering, item_count: int) -> np.ndarray:
    """Return public-to-backend indices for an optional ordering map."""
    if ordering is None:
        return np.arange(item_count, dtype=np.int64)
    return np.asarray(ordering.user_to_backend_indices, dtype=np.int64)


def _expected_backend_to_user(
    public_names: tuple[str, ...] | None,
    backend_names: tuple[str, ...],
) -> np.ndarray:
    """Derive backend-to-public indices independently from the ordering map."""
    if public_names is None:
        return np.arange(len(backend_names), dtype=np.int64)
    return np.asarray([public_names.index(name) for name in backend_names], dtype=np.int64)


def _make_backend_com_poses(num_instances: int, num_bodies: int) -> np.ndarray:
    """Create distinct deterministic body COM transforms in backend order."""
    coms = np.zeros((num_instances, num_bodies, 7), dtype=np.float32)
    for env_index in range(num_instances):
        for body_index in range(num_bodies):
            value = float(100 * env_index + 10 * body_index)
            coms[env_index, body_index, :3] = (value + 1.0, value + 2.0, value + 3.0)
            coms[env_index, body_index, 6] = 1.0
    return coms


def _seed_backend_com_poses(backend: str, art, raw_backend, coms: np.ndarray) -> None:
    """Seed backend-order COM transforms and invalidate both public and ordered caches."""
    if backend == "physx":
        raw_backend._coms = wp.array(coms, dtype=wp.float32, device="cpu")
    elif backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.BODY_COM_POSE]._data = coms.copy()
    else:
        raise AssertionError(f"Unsupported backend for COM parity test: {backend}")

    art.data._body_com_pose_b.timestamp = -1.0
    backend_staging = art.data._body_com_pose_b_backend
    if backend_staging is not None:
        backend_staging.timestamp = -1.0


def _read_backend_com_poses(backend: str, art, raw_backend) -> np.ndarray:
    """Read complete backend-order body COM transforms through the backend API."""
    if backend == "physx":
        return raw_backend.get_coms().numpy().reshape(art.num_instances, art.num_bodies, 7).copy()
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return (
            art.root_view.get_attribute(TT.BODY_COM_POSE).numpy().reshape(art.num_instances, art.num_bodies, 7).copy()
        )
    raise AssertionError(f"Unsupported backend for COM parity test: {backend}")


def _make_selected_com_poses(env_ids: list[int], body_ids: list[int], base_value: float) -> np.ndarray:
    """Create distinct deterministic public-order COM transforms for a selected rectangle."""
    coms = np.zeros((len(env_ids), len(body_ids), 7), dtype=np.float32)
    for env_offset, env_id in enumerate(env_ids):
        for body_offset, body_id in enumerate(body_ids):
            value = base_value + float(100 * env_id + 10 * body_id)
            coms[env_offset, body_offset, :3] = (value + 1.0, value + 2.0, value + 3.0)
            coms[env_offset, body_offset, 6] = 1.0
    return coms


def _seed_backend_joint_state(
    backend: str,
    art,
    raw_backend,
    position: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Seed deterministic joint state directly in backend-order storage."""
    if backend == "physx":
        raw_backend._noop_setters = False
        raw_backend._dof_positions = wp.array(position, dtype=wp.float32, device=art.device)
        raw_backend._dof_velocities = wp.array(velocity, dtype=wp.float32, device=art.device)
        art.data._joint_pos.timestamp = -1.0
        art.data._joint_vel.timestamp = -1.0
        return
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.DOF_POSITION]._data = position.copy()
        raw_backend.bindings[TT.DOF_VELOCITY]._data = velocity.copy()
        for buffer_name in ("_joint_pos_buf", "_joint_vel_buf", "_joint_pos_backend", "_joint_vel_backend"):
            buffer = getattr(art.data, buffer_name, None)
            if buffer is not None:
                buffer.timestamp = -1.0
        return
    if backend == "newton":
        art.data._sim_bind_joint_pos.assign(wp.array(position, dtype=wp.float32, device=art.device))
        art.data._sim_bind_joint_vel.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        # Raw sim-bind writes bypass the solver step, so mirror the post-step
        # publish that keeps the public-order shadows coherent under ordering.
        art.data._refresh_user_order_joint_state()
        return
    raise AssertionError(f"Unsupported backend for joint-state parity test: {backend}")


def _mutate_backend_joint_state(
    backend: str,
    art,
    raw_backend,
    position: np.ndarray,
    velocity: np.ndarray,
) -> None:
    """Mutate backend-order joint state without invalidating staging timestamps."""
    if backend == "physx":
        raw_backend._dof_positions.assign(wp.array(position, dtype=wp.float32, device=art.device))
        raw_backend._dof_velocities.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        return
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.DOF_POSITION]._data = position.copy()
        raw_backend.bindings[TT.DOF_VELOCITY]._data = velocity.copy()
        return
    raise AssertionError(f"Unsupported backend for stale joint-state parity test: {backend}")


def _read_backend_joint_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Return joint position and velocity from backend-order storage."""
    if backend == "physx":
        return raw_backend._dof_positions.numpy().copy(), raw_backend._dof_velocities.numpy().copy()
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        position = np.asarray(raw_backend.bindings[TT.DOF_POSITION]._data).copy()
        velocity = np.asarray(raw_backend.bindings[TT.DOF_VELOCITY]._data).copy()
        return position, velocity
    if backend == "newton":
        return art.data._sim_bind_joint_pos.numpy().copy(), art.data._sim_bind_joint_vel.numpy().copy()
    raise AssertionError(f"Unsupported backend for joint-state parity test: {backend}")


def _coverage_ids(coverage: str, num_instances: int, num_items: int) -> tuple[list[int], list[int]]:
    """Return environment and item IDs for a write-coverage mode."""
    if coverage == "one_env_one_item":
        return [1], [0]
    if coverage == "all_envs_one_item":
        return list(range(num_instances)), [0]
    if coverage == "one_env_all_items":
        return [1], list(range(num_items))
    raise ValueError(f"Unsupported write coverage: {coverage}")


def _write_selected_joint_state(
    art,
    selection: str,
    operation: str,
    coverage: str,
    position_value: float,
    velocity_value: float,
) -> tuple[list[int], list[int], np.ndarray, np.ndarray]:
    """Write selected public joint state through an index or mask API."""
    if operation not in ("position", "velocity", "state"):
        raise ValueError(f"Unsupported joint-state operation: {operation}")

    env_ids, joint_ids = _coverage_ids(coverage, art.num_instances, art.num_joints)
    payload_offsets = 10.0 * np.asarray(env_ids, dtype=np.float32)[:, None]
    payload_offsets = payload_offsets + np.asarray(joint_ids, dtype=np.float32)[None, :]
    position_payload = np.asarray(position_value + payload_offsets, dtype=np.float32)
    velocity_payload = np.asarray(velocity_value + payload_offsets, dtype=np.float32)
    if selection == "index":
        env_ids_wp = wp.array(env_ids, dtype=wp.int32, device=art.device)
        joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=art.device)
        position = torch.tensor(position_payload, dtype=torch.float32, device=art.device)
        velocity = torch.tensor(velocity_payload, dtype=torch.float32, device=art.device)
        if operation == "position":
            art.write_joint_position_to_sim_index(
                position=position,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        elif operation == "velocity":
            art.write_joint_velocity_to_sim_index(
                velocity=velocity,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        else:
            art.write_joint_state_to_sim_index(
                position=position,
                velocity=velocity,
                env_ids=env_ids_wp,
                joint_ids=joint_ids_wp,
            )
        return env_ids, joint_ids, position_payload, velocity_payload

    if selection == "mask":
        env_mask = wp.array(
            [env_index in env_ids for env_index in range(art.num_instances)],
            dtype=wp.bool,
            device=art.device,
        )
        joint_mask = wp.array(
            [joint_index in joint_ids for joint_index in range(art.num_joints)],
            dtype=wp.bool,
            device=art.device,
        )
        position = torch.zeros((art.num_instances, art.num_joints), dtype=torch.float32, device=art.device)
        velocity = torch.zeros((art.num_instances, art.num_joints), dtype=torch.float32, device=art.device)
        env_index = torch.tensor(env_ids, dtype=torch.long, device=art.device)
        joint_index = torch.tensor(joint_ids, dtype=torch.long, device=art.device)
        position[env_index[:, None], joint_index] = torch.tensor(
            position_payload, dtype=torch.float32, device=art.device
        )
        velocity[env_index[:, None], joint_index] = torch.tensor(
            velocity_payload, dtype=torch.float32, device=art.device
        )
        if operation == "position":
            art.write_joint_position_to_sim_mask(
                position=position,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        elif operation == "velocity":
            art.write_joint_velocity_to_sim_mask(
                velocity=velocity,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        else:
            art.write_joint_state_to_sim_mask(
                position=position,
                velocity=velocity,
                env_mask=env_mask,
                joint_mask=joint_mask,
            )
        return env_ids, joint_ids, position_payload, velocity_payload

    raise ValueError(f"Unsupported joint selection mode: {selection}")


def _exercise_partial_joint_write(
    backend: str,
    ordering_mode: str,
    selection: str,
    operation: str,
    coverage: str,
) -> None:
    """Assert partial public writes preserve complete backend joint rows."""
    num_instances = 2
    num_joints = 3
    joint_ordering = _joint_ordering_for_mode(ordering_mode, num_joints)
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=joint_ordering,
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)

    user_to_backend = _ordering_user_to_backend(art.joint_ordering, num_joints)
    expected_position = initial_position.copy()
    expected_velocity = initial_velocity.copy()

    for position_value, velocity_value in ((901.0, 902.0), (903.0, 904.0)):
        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            selection,
            operation,
            coverage,
            position_value,
            velocity_value,
        )
        backend_joint_ids = user_to_backend[joint_ids]
        selected_cells = np.ix_(env_ids, backend_joint_ids)
        if operation in ("position", "state"):
            expected_position[selected_cells] = position_payload
        if operation in ("velocity", "state"):
            expected_velocity[selected_cells] = velocity_payload

        backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_position, expected_position)
        np.testing.assert_array_equal(backend_velocity, expected_velocity)

    public_position = art.data.joint_pos.torch.detach().cpu().numpy()
    public_velocity = art.data.joint_vel.torch.detach().cpu().numpy()
    np.testing.assert_array_equal(public_position, expected_position[:, user_to_backend])
    np.testing.assert_array_equal(public_velocity, expected_velocity[:, user_to_backend])


def _exercise_stale_partial_joint_write(backend: str, operation: str) -> None:
    """Assert partial writes refresh complete rows after the simulation timestamp advances."""
    num_instances = 2
    num_joints = 3
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=_joint_ordering_for_mode("reversed", num_joints),
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)
    user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)

    expected_position = initial_position.copy()
    expected_velocity = initial_velocity.copy()
    public_position_staging = art.data._joint_pos if backend == "physx" else art.data._joint_pos_buf
    public_velocity_staging = art.data._joint_vel if backend == "physx" else art.data._joint_vel_buf
    public_staging = []
    backend_staging = []
    if operation in ("position", "state"):
        _ = art.data.joint_pos
        public_staging.append(("public position", public_position_staging))
        backend_staging.append(("backend position", art.data._joint_pos_backend))
    if operation in ("velocity", "state"):
        _ = art.data.joint_vel
        public_staging.append(("public velocity", public_velocity_staging))
        backend_staging.append(("backend velocity", art.data._joint_vel_backend))
    # After a read, the public-order buffer is current on every backend. Whether the backend-order
    # staging image is also current depends on the backend: PhysX gathers ordered reads straight
    # from the tensor-API view and stages the backend image only on the write path, so it is not
    # refreshed by a read; OVPhysX still restages it on read. The backend-agnostic guarantee is that
    # the public buffer is current -- the write path's re-staging is exercised by the final assert.
    for staging_name, staging_buffer in public_staging:
        assert staging_buffer is not None
        assert staging_buffer.timestamp == art.data._sim_timestamp, f"{backend} {staging_name} staging is not current"
    for _, staging_buffer in backend_staging:
        assert staging_buffer is not None
    staging_buffers = public_staging + backend_staging

    for write_index, (position_value, velocity_value) in enumerate(((901.0, 902.0), (903.0, 904.0))):
        if write_index == 1:
            for staging_name, staging_buffer in staging_buffers:
                assert staging_buffer is not None
                assert staging_buffer.timestamp >= 0.0, f"{backend} {staging_name} staging was invalidated"
                assert staging_buffer.timestamp < art.data._sim_timestamp
        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            "index",
            operation,
            "one_env_one_item",
            position_value,
            velocity_value,
        )
        backend_joint_ids = user_to_backend[joint_ids]
        selected_cells = np.ix_(env_ids, backend_joint_ids)
        if operation in ("position", "state"):
            expected_position[selected_cells] = position_payload
        if operation in ("velocity", "state"):
            expected_velocity[selected_cells] = velocity_payload

        backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_position, expected_position)
        np.testing.assert_array_equal(backend_velocity, expected_velocity)

        if write_index == 0:
            selected_mask = np.zeros((num_instances, num_joints), dtype=bool)
            selected_mask[selected_cells] = True
            art.data._sim_timestamp += 0.01
            if operation in ("position", "state"):
                newer_position = 1000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(
                    num_instances, num_joints
                )
                expected_position[~selected_mask] = newer_position[~selected_mask]
            if operation in ("velocity", "state"):
                newer_velocity = 2000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(
                    num_instances, num_joints
                )
                expected_velocity[~selected_mask] = newer_velocity[~selected_mask]
            _mutate_backend_joint_state(backend, art, raw_backend, expected_position, expected_velocity)


def _exercise_duplicate_full_length_joint_write(backend: str, operation: str) -> None:
    """Assert duplicate full-length selectors preserve omitted backend joints."""
    num_instances = 2
    num_joints = 3
    art, raw_backend = get_articulation(
        backend,
        num_instances=num_instances,
        num_joints=num_joints,
        num_bodies=2,
        device="cpu",
        joint_ordering=_joint_ordering_for_mode("reversed", num_joints),
    )
    initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
    initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
    _seed_backend_joint_state(backend, art, raw_backend, initial_position, initial_velocity)

    public_staging = []
    backend_staging = []
    if operation in ("position", "state"):
        _ = art.data.joint_pos
        public_staging.append(art.data._joint_pos if backend == "physx" else art.data._joint_pos_buf)
        backend_staging.append(art.data._joint_pos_backend)
    if operation in ("velocity", "state"):
        _ = art.data.joint_vel
        public_staging.append(art.data._joint_vel if backend == "physx" else art.data._joint_vel_buf)
        backend_staging.append(art.data._joint_vel_backend)
    # A read makes the public buffer current on every backend; the backend-order staging image is
    # only guaranteed current on backends that restage on read (OVPhysX), not PhysX, which stages it
    # lazily on the write path. Preservation of the omitted rows is verified by the final assert.
    for staging_buffer in public_staging:
        assert staging_buffer is not None
        assert staging_buffer.timestamp == art.data._sim_timestamp
    for staging_buffer in backend_staging:
        assert staging_buffer is not None

    art.data._sim_timestamp += 0.01
    newer_position = 1000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
    newer_velocity = 2000.0 + np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
    _mutate_backend_joint_state(backend, art, raw_backend, newer_position, newer_velocity)

    env_ids = [1]
    joint_ids = [0, 1, 1]
    position_payload = np.asarray([[901.0, 902.0, 902.0]], dtype=np.float32)
    velocity_payload = np.asarray([[1901.0, 1902.0, 1902.0]], dtype=np.float32)
    env_ids_wp = wp.array(env_ids, dtype=wp.int32, device=art.device)
    joint_ids_wp = wp.array(joint_ids, dtype=wp.int32, device=art.device)
    position = torch.tensor(position_payload, dtype=torch.float32, device=art.device)
    velocity = torch.tensor(velocity_payload, dtype=torch.float32, device=art.device)
    if operation == "position":
        art.write_joint_position_to_sim_index(position=position, env_ids=env_ids_wp, joint_ids=joint_ids_wp)
    elif operation == "velocity":
        art.write_joint_velocity_to_sim_index(velocity=velocity, env_ids=env_ids_wp, joint_ids=joint_ids_wp)
    else:
        art.write_joint_state_to_sim_index(
            position=position,
            velocity=velocity,
            env_ids=env_ids_wp,
            joint_ids=joint_ids_wp,
        )

    user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
    expected_position = newer_position.copy()
    expected_velocity = newer_velocity.copy()
    for payload_index, joint_id in enumerate(joint_ids):
        backend_joint_id = user_to_backend[joint_id]
        if operation in ("position", "state"):
            expected_position[env_ids[0], backend_joint_id] = position_payload[0, payload_index]
        if operation in ("velocity", "state"):
            expected_velocity[env_ids[0], backend_joint_id] = velocity_payload[0, payload_index]

    backend_position, backend_velocity = _read_backend_joint_state(backend, art, raw_backend)
    np.testing.assert_array_equal(backend_position, expected_position)
    np.testing.assert_array_equal(backend_velocity, expected_velocity)
    omitted_backend_joint = user_to_backend[2]
    duplicate_backend_joint = user_to_backend[1]
    if operation in ("position", "state"):
        assert backend_position[env_ids[0], omitted_backend_joint] == newer_position[env_ids[0], omitted_backend_joint]
        assert backend_position[env_ids[0], duplicate_backend_joint] == position_payload[0, 1]
    if operation in ("velocity", "state"):
        assert backend_velocity[env_ids[0], omitted_backend_joint] == newer_velocity[env_ids[0], omitted_backend_joint]
        assert backend_velocity[env_ids[0], duplicate_backend_joint] == velocity_payload[0, 1]


def _make_dynamics_ordering_backend_data(
    num_instances: int, num_joints: int, num_jacobi_bodies: int, num_base_dofs: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create deterministic backend-order dynamics data."""
    num_dofs = num_joints + num_base_dofs
    jacobian = np.arange(num_instances * num_jacobi_bodies * 6 * num_dofs, dtype=np.float32).reshape(
        num_instances, num_jacobi_bodies, 6, num_dofs
    )
    mass_matrix = np.arange(num_instances * num_dofs * num_dofs, dtype=np.float32).reshape(
        num_instances, num_dofs, num_dofs
    )
    gravity = np.arange(num_instances * num_dofs, dtype=np.float32).reshape(num_instances, num_dofs)
    return jacobian, mass_matrix, gravity


def _generalized_dof_user_to_backend(num_base_dofs: int, joint_user_to_backend: np.ndarray) -> np.ndarray:
    """Return generalized DoF user-to-backend indices including the floating base prefix."""
    return np.concatenate(
        (
            np.arange(num_base_dofs, dtype=np.int64),
            num_base_dofs + joint_user_to_backend,
        )
    )


def _jacobian_body_user_to_backend(art, body_user_to_backend: np.ndarray) -> np.ndarray:
    """Return Jacobian body-axis user-to-backend indices with the fixed root excluded."""
    if art.num_base_dofs == 0:
        return np.asarray([backend_id - 1 for backend_id in body_user_to_backend if backend_id != 0], dtype=np.int64)
    return body_user_to_backend


def _set_dynamics_ordering_backend_data(
    backend: str,
    art,
    raw_backend,
    jacobian: np.ndarray,
    mass_matrix: np.ndarray,
    gravity: np.ndarray,
) -> None:
    """Write deterministic backend-order dynamics data into the backend mock."""
    if backend == "physx":
        raw_backend.set_mock_jacobians(wp.array(jacobian, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_generalized_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_gravity_compensation_forces(wp.array(gravity, dtype=wp.float32, device=art.device))
    elif backend == "newton":
        if art.is_fixed_base:
            model_jacobian = np.zeros((art.num_instances, art.num_bodies, 6, art.num_joints), dtype=np.float32)
            model_jacobian[:, 1:] = jacobian
        else:
            model_jacobian = jacobian
        raw_backend.set_mock_jacobians(wp.array(model_jacobian, dtype=wp.float32, device=art.device))
        raw_backend.set_mock_mass_matrices(wp.array(mass_matrix, dtype=wp.float32, device=art.device))
    else:
        raise AssertionError(f"Unsupported backend for dynamics-ordering test: {backend}")


def _set_body_ordering_backend_data(
    backend: str,
    art,
    raw_backend,
    root_pose: np.ndarray,
    root_vel: np.ndarray,
    link_pose: np.ndarray,
    com_pose_b: np.ndarray,
    body_com_vel: np.ndarray,
    body_acc: np.ndarray,
) -> None:
    """Write deterministic backend-order body state into the backend mock."""
    if backend == "physx":
        raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32, device=art.device)
        raw_backend._root_velocities = wp.array(root_vel, dtype=wp.float32, device=art.device)
        raw_backend._link_transforms = wp.array(link_pose, dtype=wp.float32, device=art.device)
        raw_backend._link_velocities = wp.array(body_com_vel, dtype=wp.float32, device=art.device)
        raw_backend._link_accelerations = wp.array(body_acc, dtype=wp.float32, device=art.device)
        raw_backend._coms = wp.array(com_pose_b, dtype=wp.float32, device="cpu")
    elif backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        raw_backend.bindings[TT.ROOT_POSE]._data = root_pose.copy()
        raw_backend.bindings[TT.ROOT_VELOCITY]._data = root_vel.copy()
        raw_backend.bindings[TT.LINK_POSE]._data = link_pose.copy()
        raw_backend.bindings[TT.LINK_VELOCITY]._data = body_com_vel.copy()
        raw_backend.bindings[TT.LINK_ACCELERATION]._data = body_acc.copy()
        raw_backend.bindings[TT.BODY_COM_POSE]._data = com_pose_b.copy()
        art.data._body_com_pose_b.timestamp = -1.0
        if art.data._body_com_pose_b_backend is not None:
            art.data._body_com_pose_b_backend.timestamp = -1.0
    elif backend == "newton":
        root_pose_wp = wp.array(root_pose[:, None, :], dtype=wp.transformf, device=art.device)
        root_vel_wp = wp.array(root_vel[:, None, :], dtype=wp.spatial_vectorf, device=art.device)
        link_pose_wp = wp.array(link_pose[:, None, :, :], dtype=wp.transformf, device=art.device)
        body_com_vel_wp = wp.array(body_com_vel[:, None, :, :], dtype=wp.spatial_vectorf, device=art.device)
        body_com_pos_wp = wp.array(com_pose_b[:, None, :, :3], dtype=wp.vec3f, device=art.device)
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
        art.data._previous_body_com_vel.assign(
            wp.zeros((art.num_instances, art.num_bodies), dtype=wp.spatial_vectorf, device=art.device)
        )
    else:
        raise AssertionError(f"Unsupported backend for body-ordering test: {backend}")


def _set_identity_body_poses(backend: str, art, raw_backend) -> None:
    """Give the OVPhysX wrench transform deterministic identity rotations."""
    if backend != "ovphysx":
        return
    from isaaclab_ovphysx import tensor_types as TT

    poses = np.zeros((art.num_instances, art.num_bodies, 7), dtype=np.float32)
    poses[..., 6] = 1.0
    raw_backend.bindings[TT.LINK_POSE]._data = poses
    art.data._reset_pose()


def _read_backend_wrench(backend: str, art, raw_backend, captured: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return force and torque from the concrete backend write target."""
    if backend == "physx":
        return captured["force"], captured["torque"]
    if backend == "newton":
        wrench = art.data._sim_bind_body_external_wrench.numpy()
        return wrench[..., :3], wrench[..., 3:6]
    from isaaclab_ovphysx import tensor_types as TT

    wrench = raw_backend.bindings[TT.LINK_WRENCH]._data
    return wrench[..., :3], wrench[..., 3:6]


def _seed_root_writer_backend_com(backend: str, art, raw_backend, backend_coms: np.ndarray) -> None:
    """Seed root-writer COM inputs in backend order and verify the public view."""
    if backend in ("physx", "ovphysx"):
        _seed_backend_com_poses(backend, art, raw_backend, backend_coms)
        backend_actual = art.data._backend_body_com_pose_b.numpy()
        public_actual = art.data.body_com_pose_b.warp.numpy()
        expected_public = backend_coms[:, _ordering_user_to_backend(art.body_ordering, art.num_bodies)]
    elif backend == "newton":
        backend_positions = backend_coms[..., :3]
        art.data._sim_bind_body_com_pos_b.assign(wp.array(backend_positions, dtype=wp.vec3f, device=art.device))
        if art.body_ordering is not None:
            art.data._sync_user_ordered_body_property_buffers()
        backend_actual = art.data._sim_bind_body_com_pos_b.numpy()
        public_actual = art.data.body_com_pos_b.warp.numpy()
        expected_public = backend_positions[:, _ordering_user_to_backend(art.body_ordering, art.num_bodies)]
        backend_coms = backend_positions
    else:
        raise AssertionError(f"Unsupported backend for root-writer parity test: {backend}")

    np.testing.assert_array_equal(backend_actual, backend_coms)
    np.testing.assert_array_equal(public_actual, expected_public)


def _write_root_writer_command(
    art,
    selection: str,
    operation: str,
    root_com_pose: np.ndarray,
    root_link_velocity: np.ndarray,
) -> None:
    """Write one root command through an index or mask API."""
    if selection == "index":
        env_ids = wp.array([0], dtype=wp.int32, device=art.device)
        if operation == "com_pose":
            art.write_root_com_pose_to_sim_index(
                root_pose=wp.array(root_com_pose[:1], dtype=wp.transformf, device=art.device),
                env_ids=env_ids,
            )
        else:
            art.write_root_link_pose_to_sim_index(
                root_pose=wp.array(root_com_pose[:1], dtype=wp.transformf, device=art.device),
                env_ids=env_ids,
            )
            art.write_root_link_velocity_to_sim_index(
                root_velocity=wp.array(root_link_velocity[:1], dtype=wp.spatial_vectorf, device=art.device),
                env_ids=env_ids,
            )
    elif selection == "mask":
        env_mask = wp.array([True, False], dtype=wp.bool, device=art.device)
        if operation == "com_pose":
            art.write_root_com_pose_to_sim_mask(
                root_pose=wp.array(root_com_pose, dtype=wp.transformf, device=art.device),
                env_mask=env_mask,
            )
        else:
            art.write_root_link_pose_to_sim_mask(
                root_pose=wp.array(root_com_pose, dtype=wp.transformf, device=art.device),
                env_mask=env_mask,
            )
            art.write_root_link_velocity_to_sim_mask(
                root_velocity=wp.array(root_link_velocity, dtype=wp.spatial_vectorf, device=art.device),
                env_mask=env_mask,
            )
    else:
        raise AssertionError(f"Unsupported root-writer selection: {selection}")


def _read_backend_root_state(backend: str, art, raw_backend) -> tuple[np.ndarray, np.ndarray]:
    """Read backend root link pose and COM velocity after a writer call."""
    if backend == "physx":
        return (
            raw_backend._root_transforms.numpy().copy(),
            art.data.root_com_vel_w.warp.numpy().copy(),
        )
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return (
            raw_backend.bindings[TT.ROOT_POSE]._data.copy(),
            raw_backend.bindings[TT.ROOT_VELOCITY]._data.copy(),
        )
    if backend == "newton":
        return (
            art.data._sim_bind_root_link_pose_w.numpy().copy(),
            art.data._sim_bind_root_com_vel_w.numpy().copy(),
        )
    raise AssertionError(f"Unsupported backend for root-writer parity test: {backend}")


def _clone_proxy_tensor(arr) -> torch.Tensor:
    """Return a CPU clone of a proxy array's torch view."""
    return arr.torch.detach().cpu().clone()


def _assert_proxy_close(actual, expected: torch.Tensor) -> None:
    """Assert a proxy array is close to a CPU tensor."""
    torch.testing.assert_close(actual.torch.cpu(), expected, rtol=1.0e-5, atol=1.0e-5)


def _clone_backend_tensor(array) -> torch.Tensor:
    """Return a CPU tensor clone from a backend Warp or NumPy array."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array.copy())
    return wp.to_torch(array).detach().cpu().clone()


def _get_backend_joint_property_tensors(backend: str, art, raw_backend) -> dict[str, torch.Tensor]:
    """Return backend-order joint property values for ordering parity checks."""
    if backend == "physx":
        friction_properties = _clone_backend_tensor(raw_backend.get_dof_friction_properties())
        return {
            "stiffness": _clone_backend_tensor(raw_backend.get_dof_stiffnesses()),
            "damping": _clone_backend_tensor(raw_backend.get_dof_dampings()),
            "armature": _clone_backend_tensor(raw_backend.get_dof_armatures()),
            "position_limits": _clone_backend_tensor(raw_backend.get_dof_limits()),
            "velocity_limits": _clone_backend_tensor(raw_backend.get_dof_max_velocities()),
            "effort_limits": _clone_backend_tensor(raw_backend.get_dof_max_forces()),
            "friction": friction_properties[..., 0],
            "dynamic_friction": friction_properties[..., 1],
            "viscous_friction": friction_properties[..., 2],
        }
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        friction_properties = _clone_backend_tensor(raw_backend.bindings[TT.DOF_FRICTION_PROPERTIES]._data)
        return {
            "stiffness": _clone_backend_tensor(raw_backend.bindings[TT.DOF_STIFFNESS]._data),
            "damping": _clone_backend_tensor(raw_backend.bindings[TT.DOF_DAMPING]._data),
            "armature": _clone_backend_tensor(raw_backend.bindings[TT.DOF_ARMATURE]._data),
            "position_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_LIMIT]._data),
            "velocity_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_MAX_VELOCITY]._data),
            "effort_limits": _clone_backend_tensor(raw_backend.bindings[TT.DOF_MAX_FORCE]._data),
            "friction": friction_properties[..., 0],
            "dynamic_friction": friction_properties[..., 1],
            "viscous_friction": friction_properties[..., 2],
        }
    if backend == "newton":
        return {
            "stiffness": _clone_backend_tensor(art.data._sim_bind_joint_stiffness_sim),
            "damping": _clone_backend_tensor(art.data._sim_bind_joint_damping_sim),
            "armature": _clone_backend_tensor(art.data._sim_bind_joint_armature),
            "position_limits": torch.stack(
                (
                    _clone_backend_tensor(art.data._sim_bind_joint_pos_limits_lower),
                    _clone_backend_tensor(art.data._sim_bind_joint_pos_limits_upper),
                ),
                dim=-1,
            ),
            "velocity_limits": _clone_backend_tensor(art.data._sim_bind_joint_vel_limits_sim),
            "effort_limits": _clone_backend_tensor(art.data._sim_bind_joint_effort_limits_sim),
            "friction": _clone_backend_tensor(art.data._sim_bind_joint_friction_coeff),
        }
    raise AssertionError(f"Unsupported backend for joint-property ordering test: {backend}")


def _get_backend_body_property_tensors(backend: str, art, raw_backend) -> dict[str, torch.Tensor]:
    """Return backend-order body property values for ordering parity checks."""
    if backend == "physx":
        return {
            "mass": _clone_backend_tensor(raw_backend.get_masses()),
            "inertia": _clone_backend_tensor(raw_backend.get_inertias()),
        }
    if backend == "ovphysx":
        from isaaclab_ovphysx import tensor_types as TT

        return {
            "mass": _clone_backend_tensor(raw_backend.bindings[TT.BODY_MASS]._data),
            "inertia": _clone_backend_tensor(raw_backend.bindings[TT.BODY_INERTIA]._data),
        }
    if backend == "newton":
        return {
            "mass": _clone_backend_tensor(art.data._sim_bind_body_mass),
            "inertia": _clone_backend_tensor(art.data._sim_bind_body_inertia),
        }
    raise AssertionError(f"Unsupported backend for body-property ordering test: {backend}")


def _backend_param(backend: str, *values, **kwargs) -> ParameterSet:
    """Build a backend parameter that skips unavailable plugins at collection time."""
    marks = list(kwargs.pop("marks", ()))
    marks.append(pytest.mark.skipif(backend not in BACKENDS, reason=_backend_unavailable_reason(backend)))
    return pytest.param(backend, *values, marks=marks, **kwargs)


def _backend_unavailable_reason(backend: str) -> str:
    """Describe why an optional articulation backend could not be imported."""
    label = {"physx": "PhysX", "ovphysx": "OVPhysX", "newton": "Newton"}.get(backend, backend)
    reason = f"{label} backend is not available"
    if detail := BACKEND_UNAVAILABLE_REASONS.get(backend):
        reason = f"{reason}: {detail}"
    return reason


_requires_physx = pytest.mark.skipif("physx" not in BACKENDS, reason=_backend_unavailable_reason("physx"))
_requires_ovphysx = pytest.mark.skipif("ovphysx" not in BACKENDS, reason=_backend_unavailable_reason("ovphysx"))
_requires_newton = pytest.mark.skipif("newton" not in BACKENDS, reason=_backend_unavailable_reason("newton"))
_all_backends = pytest.mark.parametrize(
    "backend", [_backend_param(backend) for backend in ("physx", "ovphysx", "newton")], indirect=False
)
_physx_ovphysx_backends = pytest.mark.parametrize(
    "backend", [_backend_param(backend) for backend in ("physx", "ovphysx")], indirect=False
)
_dynamics_ordering_backends = pytest.mark.parametrize(
    "backend", [_backend_param(backend) for backend in ("physx", "newton")], indirect=False
)
_non_mock_backends = _all_backends


class TestArticulationDataBodyState:
    """Test data properties for all body states."""

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 1, 3)])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_reversed_body_ordering_reorders_public_body_quantities(
        self, backend, num_instances, num_joints, num_bodies, device
    ):
        identity_art, identity_raw = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        ordered_art, ordered_raw = get_articulation(backend, num_instances, num_joints, num_bodies, device=device)
        raw_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data(backend, identity_art, identity_raw, *raw_data)
        _set_body_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_data)
        user_to_backend = _install_test_body_ordering(ordered_art)

        identity_art.data.update(dt=0.01)
        ordered_art.data.update(dt=0.01)

        identity_body_com_pose_b = _clone_proxy_tensor(identity_art.data.body_com_pose_b)
        identity_body_com_acc_w = _clone_proxy_tensor(identity_art.data.body_com_acc_w)
        identity_body_link_vel_w = _clone_proxy_tensor(identity_art.data.body_link_vel_w)
        identity_body_com_pose_w = _clone_proxy_tensor(identity_art.data.body_com_pose_w)
        identity_root_com_pose_w = _clone_proxy_tensor(identity_art.data.root_com_pose_w)
        identity_root_link_vel_w = _clone_proxy_tensor(identity_art.data.root_link_vel_w)

        _assert_proxy_close(ordered_art.data.body_com_pose_b, identity_body_com_pose_b[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_com_acc_w, identity_body_com_acc_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_link_vel_w, identity_body_link_vel_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.body_com_pose_w, identity_body_com_pose_w[:, user_to_backend])
        _assert_proxy_close(ordered_art.data.root_com_pose_w, identity_root_com_pose_w)
        _assert_proxy_close(ordered_art.data.root_link_vel_w, identity_root_link_vel_w)

    @_non_mock_backends
    @pytest.mark.parametrize("ordering_mode", ["reversed", "cyclic"])
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 1, 3)])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_body_ordering_reorders_public_body_properties(
        self, backend, ordering_mode, num_instances, num_joints, num_bodies, device
    ):
        """Expose mass and every inertia component under the matching public body name."""
        body_ordering = _body_ordering_for_mode(ordering_mode, num_bodies)
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device=device,
            body_ordering=body_ordering,
        )
        user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        backend_properties = _get_backend_body_property_tensors(backend, art, raw_backend)

        _assert_proxy_close(art.data.body_mass, backend_properties["mass"][:, user_to_backend])
        _assert_proxy_close(
            art.data.body_inertia,
            backend_properties["inertia"][:, user_to_backend],
        )

    @_requires_ovphysx
    def test_ovphysx_reversed_body_ordering_rereads_backend_shadows_after_reset(self):
        """Refresh OVPhysX backend shadow buffers after same-step pose/velocity invalidation."""
        num_instances = 2
        num_joints = 1
        num_bodies = 3
        art, raw_backend = get_articulation("ovphysx", num_instances, num_joints, num_bodies, device="cpu")
        raw_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data("ovphysx", art, raw_backend, *raw_data)
        user_to_backend = _install_test_body_ordering(art)
        art.data.update(dt=0.01)

        _ = _clone_proxy_tensor(art.data.body_link_pose_w)
        _ = _clone_proxy_tensor(art.data.body_com_vel_w)

        from isaaclab_ovphysx import tensor_types as TT

        next_link_pose = raw_data[2].copy()
        next_link_pose[..., 0] += 1000.0
        next_body_com_vel = raw_data[4].copy()
        next_body_com_vel[..., 0] += 500.0
        raw_backend.bindings[TT.LINK_POSE]._data = next_link_pose
        raw_backend.bindings[TT.LINK_VELOCITY]._data = next_body_com_vel

        art.data._reset_pose()
        art.data._reset_velocity()

        expected_link_pose = torch.from_numpy(next_link_pose[:, user_to_backend])
        expected_body_com_vel = torch.from_numpy(next_body_com_vel[:, user_to_backend])
        _assert_proxy_close(art.data.body_link_pose_w, expected_link_pose)
        _assert_proxy_close(art.data.body_com_vel_w, expected_body_com_vel)

    @_requires_ovphysx
    def test_ovphysx_reversed_body_ordering_rereads_all_velocity_shadows_after_reset(self):
        """Refresh every OVPhysX velocity shadow after a same-step reset."""
        art, raw_backend = get_articulation(
            "ovphysx", 2, 1, 3, device="cpu", body_ordering=("body_2", "body_1", "body_0")
        )
        from isaaclab_ovphysx import tensor_types as TT

        raw_data = list(_make_body_ordering_backend_data(2, 3))
        raw_data[3][..., :3] = 0.0
        _set_body_ordering_backend_data("ovphysx", art, raw_backend, *raw_data)
        user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        art.data.update(0.01)
        art.data.body_com_vel_w.torch.clone()
        art.data.body_link_vel_w.torch.clone()
        art.data.root_link_vel_w.torch.clone()

        next_velocity = raw_data[4].copy()
        next_velocity[..., 0] += 500.0
        raw_backend.bindings[TT.LINK_VELOCITY]._data = next_velocity
        art.data._reset_velocity()

        expected = torch.from_numpy(next_velocity[:, user_to_backend])
        _assert_proxy_close(art.data.body_com_vel_w, expected)
        _assert_proxy_close(art.data.body_link_vel_w, expected)
        _assert_proxy_close(art.data.root_link_vel_w, torch.from_numpy(next_velocity[:, 0]))

    @_requires_ovphysx
    def test_ovphysx_ordered_root_and_body_velocity_share_one_backend_read(self) -> None:
        """Read the shared backend link-velocity binding once per timestamp."""
        art, raw_backend = get_articulation(
            "ovphysx",
            2,
            1,
            3,
            device="cpu",
            body_ordering=("body_2", "body_1", "body_0"),
        )
        raw_data = _make_body_ordering_backend_data(2, 3)
        _set_body_ordering_backend_data("ovphysx", art, raw_backend, *raw_data)
        art.data.update(0.01)
        binding_read = MagicMock(wraps=art.data._binding_read)
        art.data._binding_read = binding_read

        _ = art.data.body_com_vel_w
        _ = art.data.root_link_vel_w

        from isaaclab_ovphysx import tensor_types as TT

        link_velocity_reads = [call for call in binding_read.call_args_list if call.args[0] == TT.LINK_VELOCITY]
        assert len(link_velocity_reads) == 1

    @_requires_ovphysx
    def test_ovphysx_com_write_invalidates_all_dependent_caches_under_ordering(self):
        """Invalidate every public and backend cache derived from OVPhysX COM poses."""
        art, _ = get_articulation("ovphysx", 2, 1, 3, device="cpu", body_ordering=("body_2", "body_1", "body_0"))
        cache_names = (
            "_root_com_pose_w",
            "_root_com_vel_w",
            "_root_link_vel_w",
            "_body_com_pose_w",
            "_body_com_vel_w",
            "_body_com_vel_w_backend",
            "_body_link_vel_w",
            "_root_link_lin_vel_b",
            "_root_link_ang_vel_b",
            "_root_com_lin_vel_b",
            "_root_com_ang_vel_b",
            "_root_state_w_buf",
            "_root_link_state_w_buf",
            "_root_com_state_w_buf",
            "_body_state_w_buf",
            "_body_link_state_w_buf",
            "_body_com_state_w_buf",
        )
        for cache_name in cache_names:
            getattr(art.data, cache_name).timestamp = art.data._sim_timestamp

        coms = np.zeros((art.num_instances, art.num_bodies, 7), dtype=np.float32)
        coms[..., 6] = 1.0
        art.set_coms_index(coms=wp.array(coms, dtype=wp.transformf, device=art.device))

        for cache_name in cache_names:
            assert getattr(art.data, cache_name).timestamp == -1.0, cache_name

    @_dynamics_ordering_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 3, 4)])
    @pytest.mark.parametrize("device", ["cpu"])
    @pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
    def test_ordering_reorders_public_dynamics_quantities(
        self, backend, num_instances, num_joints, num_bodies, device, is_fixed_base
    ):
        identity_art, identity_raw = get_articulation(
            backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
        )
        ordered_art, ordered_raw = get_articulation(
            backend, num_instances, num_joints, num_bodies, device=device, is_fixed_base=is_fixed_base
        )
        raw_body_data = _make_body_ordering_backend_data(num_instances, num_bodies)
        _set_body_ordering_backend_data(backend, identity_art, identity_raw, *raw_body_data)
        _set_body_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_body_data)

        num_base_dofs = identity_art.num_base_dofs
        num_jacobi_bodies = num_bodies - (1 if num_base_dofs == 0 else 0)
        raw_dynamics_data = _make_dynamics_ordering_backend_data(
            num_instances, num_joints, num_jacobi_bodies, num_base_dofs
        )
        _set_dynamics_ordering_backend_data(backend, identity_art, identity_raw, *raw_dynamics_data)
        _set_dynamics_ordering_backend_data(backend, ordered_art, ordered_raw, *raw_dynamics_data)
        body_user_to_backend = _install_test_body_ordering(ordered_art)
        joint_user_to_backend = _install_reversed_joint_ordering(ordered_art)

        identity_art.data.update(dt=0.01)
        ordered_art.data.update(dt=0.01)

        if backend == "newton":
            import isaaclab_newton.assets.articulation.articulation_data as newton_data_module

            original_simulation_manager = newton_data_module.SimulationManager
            newton_data_module.SimulationManager = identity_art._test_simulation_manager
        else:
            newton_data_module = None
            original_simulation_manager = None

        try:
            jacobian_body_indices = _jacobian_body_user_to_backend(ordered_art, body_user_to_backend)
            dof_indices = _generalized_dof_user_to_backend(num_base_dofs, joint_user_to_backend)
            identity_body_com_jacobian_w = _clone_proxy_tensor(identity_art.data.body_com_jacobian_w)
            identity_body_link_jacobian_w = _clone_proxy_tensor(identity_art.data.body_link_jacobian_w)
            identity_mass_matrix = _clone_proxy_tensor(identity_art.data.mass_matrix)

            expected_body_com_jacobian_w = identity_body_com_jacobian_w[:, jacobian_body_indices][:, :, :, dof_indices]
            expected_body_link_jacobian_w = identity_body_link_jacobian_w[:, jacobian_body_indices][
                :, :, :, dof_indices
            ]
            expected_mass_matrix = identity_mass_matrix[:, dof_indices][:, :, dof_indices]

            _assert_proxy_close(ordered_art.data.body_com_jacobian_w, expected_body_com_jacobian_w)
            _assert_proxy_close(ordered_art.data.body_link_jacobian_w, expected_body_link_jacobian_w)
            _assert_proxy_close(ordered_art.data.mass_matrix, expected_mass_matrix)

            if backend == "physx":
                identity_gravity_compensation_forces = _clone_proxy_tensor(
                    identity_art.data.gravity_compensation_forces
                )
                expected_gravity_compensation_forces = identity_gravity_compensation_forces[:, dof_indices]
                _assert_proxy_close(
                    ordered_art.data.gravity_compensation_forces,
                    expected_gravity_compensation_forces,
                )
        finally:
            if newton_data_module is not None:
                newton_data_module.SimulationManager = original_simulation_manager


class TestArticulationOrderingAllocation:
    """Test that ordering-only shadows exist only for nonidentity mappings."""

    _DATA_SHADOWS = {
        "physx": (
            "_joint_pos_backend",
            "_joint_vel_backend",
            "_joint_stiffness_backend",
            "_joint_damping_backend",
            "_joint_armature_backend",
            "_joint_pos_limits_backend",
            "_joint_vel_limits_backend",
            "_joint_effort_limits_backend",
            "_joint_friction_props_backend",
            "_joint_friction_props_user",
            "_body_com_pose_b_backend",
            "_body_mass_backend",
            "_body_inertia_backend",
        ),
        "ovphysx": (
            "_body_link_pose_w_backend",
            "_body_com_pose_b_backend",
            "_body_com_vel_w_backend",
            "_body_com_acc_w_backend",
            "_joint_pos_backend",
            "_joint_vel_backend",
            "_joint_stiffness_backend",
            "_joint_damping_backend",
            "_joint_armature_backend",
            "_joint_pos_limits_backend",
            "_joint_vel_limits_backend",
            "_joint_effort_limits_backend",
            "_joint_friction_props_backend",
            "_body_mass_backend",
            "_body_inertia_backend",
        ),
        "newton": (
            "_joint_pos_user",
            "_joint_vel_user",
            "_joint_stiffness_user",
            "_joint_damping_user",
            "_joint_armature_user",
            "_joint_friction_coeff_user",
            "_joint_pos_limits_lower_user",
            "_joint_pos_limits_upper_user",
            "_joint_vel_limits_user",
            "_joint_effort_limits_user",
            "_body_link_pose_w_user",
            "_body_com_vel_w_user",
            "_body_mass_user",
            "_body_inertia_user",
            "_body_com_pos_b_user",
        ),
    }
    _ARTICULATION_SHADOWS = {
        "physx": (
            "_joint_pos_target_backend",
            "_joint_vel_target_backend",
            "_joint_effort_target_backend",
            "_body_wrench_force_backend",
            "_body_wrench_torque_backend",
        ),
        "ovphysx": (
            "_joint_pos_target_backend",
            "_joint_vel_target_backend",
            "_joint_effort_target_backend",
        ),
        "newton": (),
    }

    @_all_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "identity", "reversed"])
    def test_backends_allocate_shadows_only_for_nonidentity_ordering(self, backend: str, ordering_mode: str) -> None:
        num_instances = 2
        num_joints = 3
        num_bodies = 4
        art, _ = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode(ordering_mode, num_joints),
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )

        expected_ordering = ordering_mode == "reversed"
        assert (art.joint_ordering is not None) is expected_ordering
        assert (art.body_ordering is not None) is expected_ordering
        # The asset ordering properties delegate to the single source on data.
        assert art.joint_ordering is art.data.joint_ordering
        assert art.body_ordering is art.data.body_ordering

        for owner, field_names in (
            (art.data, self._DATA_SHADOWS[backend]),
            (art, self._ARTICULATION_SHADOWS[backend]),
        ):
            for field_name in field_names:
                shadow = getattr(owner, field_name)
                if not expected_ordering:
                    assert shadow is None, f"{backend} {ordering_mode} unexpectedly allocated {field_name}"
                    continue
                assert shadow is not None, f"{backend} reversed ordering did not allocate {field_name}"
                item_count = num_bodies if "_body_" in field_name else num_joints
                assert _ordering_shadow_shape(shadow)[:2] == (num_instances, item_count)


class TestArticulationOrderingComWrites:
    """Test coherent partial COM writes and step-independent static COM caches."""

    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    # ``one_env_one_item`` is dropped: env selection is always explicit for both index and mask
    # writers, so it exercises the same branch as ``all_envs_one_item`` (both partial-body). The
    # retained pair keeps one partial-body and one full-body write.
    @pytest.mark.parametrize(
        "coverage",
        ["all_envs_one_item", "one_env_all_items"],
    )
    @_requires_ovphysx
    def test_ovphysx_partial_com_write_preserves_backend_rows(
        self,
        ordering_mode: str,
        selection: str,
        coverage: str,
    ) -> None:
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            "ovphysx",
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses("ovphysx", art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses("ovphysx", art, raw_backend), backend_seed)

        user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)

        first_env_ids, first_body_ids = _coverage_ids(coverage, num_instances, num_bodies)
        if len(first_body_ids) == num_bodies:
            second_env_ids = [0]
            second_body_ids = first_body_ids
        else:
            second_env_ids = first_env_ids
            second_body_ids = [1]
        write_selections = (
            (first_env_ids, first_body_ids, 1000.0),
            (second_env_ids, second_body_ids, 2000.0),
        )
        if ordering_mode == "reversed":
            for _, body_ids, _ in write_selections:
                if len(body_ids) < num_bodies:
                    assert all(user_to_backend[body_id] != body_id for body_id in body_ids)

        expected_backend = backend_seed.copy()
        first_backend_cells = None
        first_payload = None
        backend_after = None
        for write_index, (env_ids, body_ids, base_value) in enumerate(write_selections):
            payload = _make_selected_com_poses(env_ids, body_ids, base_value)
            expected_before = expected_backend.copy()
            if selection == "index":
                art.set_coms_index(
                    coms=wp.array(payload, dtype=wp.transformf, device=art.device),
                    env_ids=wp.array(env_ids, dtype=wp.int32, device=art.device),
                    body_ids=wp.array(body_ids, dtype=wp.int32, device=art.device),
                )
            else:
                full_coms = np.zeros_like(backend_seed)
                full_coms[np.ix_(env_ids, body_ids)] = payload
                env_mask = wp.array(
                    [env_index in env_ids for env_index in range(num_instances)],
                    dtype=wp.bool,
                    device=art.device,
                )
                body_mask = None
                if len(body_ids) < num_bodies:
                    body_mask = wp.array(
                        [body_index in body_ids for body_index in range(num_bodies)],
                        dtype=wp.bool,
                        device=art.device,
                    )
                art.set_coms_mask(
                    coms=wp.array(full_coms, dtype=wp.transformf, device=art.device),
                    env_mask=env_mask,
                    body_mask=body_mask,
                )

            backend_body_ids = user_to_backend[np.asarray(body_ids, dtype=np.int64)]
            selected_cells = np.ix_(env_ids, backend_body_ids)
            selected_mask = np.zeros((num_instances, num_bodies), dtype=bool)
            selected_mask[selected_cells] = True
            expected_backend[selected_cells] = payload

            backend_after = _read_backend_com_poses("ovphysx", art, raw_backend)
            np.testing.assert_array_equal(backend_after, expected_backend)
            np.testing.assert_array_equal(backend_after[~selected_mask], expected_before[~selected_mask])
            public_coms = art.data.body_com_pose_b.torch.detach().cpu().numpy()
            np.testing.assert_array_equal(public_coms, expected_backend[:, user_to_backend])

            if write_index == 0:
                first_backend_cells = selected_cells
                first_payload = payload.copy()

        assert backend_after is not None
        assert first_backend_cells is not None
        assert first_payload is not None
        np.testing.assert_array_equal(backend_after[first_backend_cells], first_payload)

    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @_requires_newton
    def test_newton_partial_com_write_routes_to_backend_bodies(self, ordering_mode: str, selection: str) -> None:
        """Route partial Newton COM writes to backend bodies and preserve the untouched rows.

        Newton stores the body COM as a position-only ``vec3`` and its ``set_coms`` deliberately
        cannot set an orientation, so the 7-component COM parity harness used for PhysX/OVPhysX does
        not apply. This test exercises the same reorder-routing and preservation contract natively in
        ``vec3`` against ``_sim_bind_body_com_pos_b``.
        """
        num_instances = 2
        num_bodies = 4
        art, _ = get_articulation(
            "newton",
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)[..., :3].copy()
        art.data._sim_bind_body_com_pos_b.assign(wp.array(backend_seed, dtype=wp.vec3f, device=art.device))
        if art.body_ordering is not None:
            art.data._sync_user_ordered_body_property_buffers()

        user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)
        env_ids = list(range(num_instances))
        public_body_ids = [0, 2]
        omitted_public_body_ids = [1, 3]
        payload = _make_selected_com_poses(env_ids, public_body_ids, 5000.0)[..., :3].copy()
        if ordering_mode == "reversed":
            assert all(user_to_backend[body_id] != body_id for body_id in public_body_ids)

        if selection == "index":
            art.set_coms_index(
                coms=wp.array(payload, dtype=wp.vec3f, device=art.device),
                env_ids=wp.array(env_ids, dtype=wp.int32, device=art.device),
                body_ids=wp.array(public_body_ids, dtype=wp.int32, device=art.device),
            )
        else:
            full_coms = np.zeros((num_instances, num_bodies, 3), dtype=np.float32)
            full_coms[:, public_body_ids] = payload
            art.set_coms_mask(
                coms=wp.array(full_coms, dtype=wp.vec3f, device=art.device),
                env_mask=wp.ones(num_instances, dtype=wp.bool, device=art.device),
                body_mask=_make_item_mask(num_bodies, public_body_ids, art.device),
            )

        backend_body_ids = user_to_backend[np.asarray(public_body_ids, dtype=np.int64)]
        expected_backend = backend_seed.copy()
        expected_backend[np.ix_(env_ids, backend_body_ids)] = payload

        backend_after = art.data._sim_bind_body_com_pos_b.numpy()
        np.testing.assert_array_equal(backend_after, expected_backend)
        omitted_backend_body_ids = user_to_backend[np.asarray(omitted_public_body_ids, dtype=np.int64)]
        np.testing.assert_array_equal(
            backend_after[:, omitted_backend_body_ids],
            backend_seed[:, omitted_backend_body_ids],
        )
        public_after = art.data.body_com_pos_b.torch.detach().cpu().numpy()
        np.testing.assert_array_equal(public_after, expected_backend[:, user_to_backend])

    @_physx_ovphysx_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    def test_duplicate_body_ids_preserve_omitted_backend_com(self, backend: str, ordering_mode: str) -> None:
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        if backend == "physx":
            raw_backend._noop_setters = False
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)

        user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)
        duplicate_body_ids = [0, 1, 1, 2]
        omitted_public_body_id = 3
        env_ids = list(range(num_instances))
        payload = _make_selected_com_poses(env_ids, duplicate_body_ids, 3000.0)

        art.set_coms_index(
            coms=wp.array(payload, dtype=wp.transformf, device=art.device),
            body_ids=wp.array(duplicate_body_ids, dtype=wp.int32, device=art.device),
        )

        expected_backend = backend_seed.copy()
        for env_offset, env_id in enumerate(env_ids):
            for body_offset, public_body_id in enumerate(duplicate_body_ids):
                expected_backend[env_id, user_to_backend[public_body_id]] = payload[env_offset, body_offset]
        backend_after = _read_backend_com_poses(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_after, expected_backend)
        np.testing.assert_array_equal(
            backend_after[:, user_to_backend[omitted_public_body_id]],
            backend_seed[:, user_to_backend[omitted_public_body_id]],
        )
        public_after = art.data.body_com_pose_b.torch.detach().cpu().numpy()
        np.testing.assert_array_equal(public_after, expected_backend[:, user_to_backend])

    @_physx_ovphysx_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    def test_duplicate_env_ids_leave_global_com_cache_invalid(self, backend: str, ordering_mode: str) -> None:
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        if backend == "physx":
            raw_backend._noop_setters = False
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)

        user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)
        duplicate_env_ids = [1, 1]
        omitted_env_id = 0
        public_body_ids = list(range(num_bodies))
        payload = _make_selected_com_poses(duplicate_env_ids, public_body_ids, 4000.0)

        art.set_coms_index(
            coms=wp.array(payload, dtype=wp.transformf, device=art.device),
            env_ids=wp.array(duplicate_env_ids, dtype=wp.int32, device=art.device),
        )

        expected_backend = backend_seed.copy()
        for env_offset, env_id in enumerate(duplicate_env_ids):
            expected_backend[env_id, user_to_backend] = payload[env_offset]
        backend_after = _read_backend_com_poses(backend, art, raw_backend)
        np.testing.assert_array_equal(backend_after, expected_backend)
        np.testing.assert_array_equal(backend_after[omitted_env_id], backend_seed[omitted_env_id])

        public_cache_was_invalid = art.data._body_com_pose_b.timestamp < 0.0
        backend_staging = art.data._body_com_pose_b_backend
        backend_cache_was_invalid = backend_staging is None or backend_staging.timestamp < 0.0
        public_after = art.data.body_com_pose_b.torch.detach().cpu().numpy()
        np.testing.assert_array_equal(public_after, expected_backend[:, user_to_backend])
        assert public_cache_was_invalid
        assert backend_cache_was_invalid

    @_physx_ovphysx_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    def test_static_com_cache_does_not_follow_sim_timestamp(self, backend: str, ordering_mode: str) -> None:
        num_instances = 2
        num_bodies = 4
        art, raw_backend = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            body_ordering=_body_ordering_for_mode(ordering_mode, num_bodies),
        )
        root_pose = np.zeros((num_instances, 7), dtype=np.float32)
        root_pose[:, :3] = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        root_pose[:, 6] = 1.0
        backend_seed = _make_backend_com_poses(num_instances, num_bodies)
        if backend == "physx":
            raw_backend._root_transforms = wp.array(root_pose, dtype=wp.float32, device=art.device)
        else:
            from isaaclab_ovphysx import tensor_types as TT

            raw_backend.bindings[TT.ROOT_POSE]._data = root_pose.copy()
        _seed_backend_com_poses(backend, art, raw_backend, backend_seed)
        np.testing.assert_array_equal(_read_backend_com_poses(backend, art, raw_backend), backend_seed)
        art.data._body_com_pose_b.timestamp = -1.0
        backend_staging = art.data._body_com_pose_b_backend
        if backend_staging is not None:
            backend_staging.timestamp = -1.0

        user_to_backend = _ordering_user_to_backend(art.body_ordering, num_bodies)
        expected_public = backend_seed[:, user_to_backend]
        expected_root = root_pose.copy()
        expected_root[:, :3] += backend_seed[:, 0, :3]

        if backend == "physx":
            raw_backend.get_coms = MagicMock(wraps=raw_backend.get_coms)
            read_spy = raw_backend.get_coms
        else:
            com_binding = raw_backend.bindings[TT.BODY_COM_POSE]
            com_binding.read = MagicMock(wraps=com_binding.read)
            read_spy = com_binding.read

        np.testing.assert_array_equal(art.data.body_com_pose_b.torch.detach().cpu().numpy(), expected_public)
        np.testing.assert_array_equal(art.data.root_com_pose_w.torch.detach().cpu().numpy(), expected_root)
        assert read_spy.call_count == 1

        for _ in range(3):
            art.data.update(dt=0.01)
            np.testing.assert_array_equal(art.data.body_com_pose_b.torch.detach().cpu().numpy(), expected_public)
            np.testing.assert_array_equal(art.data.root_com_pose_w.torch.detach().cpu().numpy(), expected_root)
            assert read_spy.call_count == 1


# ---------------------------------------------------------------------------
# Tests: ArticulationData joint state and properties
# ---------------------------------------------------------------------------


class TestArticulationOrderingRootWriteParity:
    """Test root writers against backend-order COM offsets."""

    @pytest.mark.parametrize("operation", ["com_pose", "link_velocity"])
    @pytest.mark.parametrize(
        "backend, selection",
        [
            _backend_param("physx", "index", id="physx-index"),
            _backend_param("physx", "mask", id="physx-mask"),
            _backend_param("ovphysx", "index", id="ovphysx-index"),
            _backend_param("ovphysx", "mask", id="ovphysx-mask"),
            _backend_param("newton", "index", id="newton-index"),
            _backend_param("newton", "mask", id="newton-mask"),
        ],
    )
    def test_floating_root_writers_match_identity_after_body_reordering(
        self,
        backend: str,
        selection: str,
        operation: str,
    ) -> None:
        num_instances = 2
        num_bodies = 4
        backend_body_names = tuple(f"body_{index}" for index in range(num_bodies))
        ordered_body_names = _body_ordering_for_mode("cyclic", num_bodies)
        identity_art, identity_raw = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            is_fixed_base=False,
            body_ordering=backend_body_names,
        )
        ordered_art, ordered_raw = get_articulation(
            backend,
            num_instances=num_instances,
            num_joints=1,
            num_bodies=num_bodies,
            device="cpu",
            is_fixed_base=False,
            body_ordering=ordered_body_names,
        )
        # The identity-configured reference normalizes to ``None`` at install time.
        assert identity_art.body_ordering is None
        assert ordered_art.body_ordering is not None
        assert tuple(ordered_art.body_names) == ordered_body_names
        np.testing.assert_array_equal(
            ordered_art.body_ordering.backend_to_user_indices,
            _expected_backend_to_user(ordered_body_names, backend_body_names),
        )
        if backend == "physx":
            identity_raw._noop_setters = False
            ordered_raw._noop_setters = False

        backend_coms = _make_backend_com_poses(num_instances, num_bodies)
        _seed_root_writer_backend_com(backend, identity_art, identity_raw, backend_coms)
        _seed_root_writer_backend_com(backend, ordered_art, ordered_raw, backend_coms)

        root_com_pose = np.asarray(
            [[4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 1.0], [14.0, 15.0, 16.0, 0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        root_link_velocity = np.asarray(
            [[1.0, 2.0, 3.0, 0.4, 0.5, 0.6], [11.0, 12.0, 13.0, 1.4, 1.5, 1.6]],
            dtype=np.float32,
        )
        _write_root_writer_command(identity_art, selection, operation, root_com_pose, root_link_velocity)
        _write_root_writer_command(ordered_art, selection, operation, root_com_pose, root_link_velocity)

        identity_root_link_pose, identity_root_com_velocity = _read_backend_root_state(
            backend, identity_art, identity_raw
        )
        ordered_root_link_pose, ordered_root_com_velocity = _read_backend_root_state(backend, ordered_art, ordered_raw)
        if operation == "com_pose":
            np.testing.assert_allclose(ordered_root_link_pose[0], identity_root_link_pose[0], atol=1e-6)
        else:
            np.testing.assert_allclose(ordered_root_com_velocity[0], identity_root_com_velocity[0], atol=1e-6)


class TestArticulationOrderingWriteParity:
    """Test that partial joint writes preserve unselected backend state."""

    @_all_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "reversed"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    @pytest.mark.parametrize(
        "coverage",
        ["all_envs_one_item", "one_env_all_items"],
    )
    def test_partial_joint_write_preserves_backend_rows(
        self,
        backend: str,
        ordering_mode: str,
        selection: str,
        operation: str,
        coverage: str,
    ) -> None:
        _exercise_partial_joint_write(backend, ordering_mode, selection, operation, coverage)

    @pytest.mark.parametrize(
        "operation, expected_position_reads, expected_velocity_reads",
        [("position", 1, 0), ("velocity", 0, 1), ("state", 1, 1)],
    )
    @_requires_physx
    def test_physx_mask_partial_joint_write_reads_each_staging_axis_once(
        self,
        operation: str,
        expected_position_reads: int,
        expected_velocity_reads: int,
    ) -> None:
        art, raw_backend = get_articulation(
            "physx",
            num_instances=2,
            num_joints=3,
            num_bodies=2,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode("reversed", 3),
        )
        initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
        _seed_backend_joint_state("physx", art, raw_backend, initial_position, initial_velocity)
        # Spy on the public backend tensor-API seam rather than a private refresh helper: a
        # partial write must stage each touched axis from the backend view exactly once so the
        # unselected rows survive, independent of how the data class names its internals.
        position_read = MagicMock(wraps=raw_backend.get_dof_positions)
        velocity_read = MagicMock(wraps=raw_backend.get_dof_velocities)
        raw_backend.get_dof_positions = position_read
        raw_backend.get_dof_velocities = velocity_read

        env_ids, joint_ids, position_payload, velocity_payload = _write_selected_joint_state(
            art,
            "mask",
            operation,
            "one_env_one_item",
            901.0,
            902.0,
        )

        assert position_read.call_count == expected_position_reads
        assert velocity_read.call_count == expected_velocity_reads
        # One representative preservation check (exhaustive coverage lives in
        # test_partial_joint_write_preserves_backend_rows): the untouched rows keep
        # their seeded values while the selected cell takes the payload.
        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
        selected_cells = np.ix_(env_ids, user_to_backend[joint_ids])
        if operation in ("position", "state"):
            initial_position[selected_cells] = position_payload
        backend_position, _ = _read_backend_joint_state("physx", art, raw_backend)
        np.testing.assert_array_equal(backend_position, initial_position)

    @_physx_ovphysx_backends
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_duplicate_full_length_joint_selector_preserves_newer_backend_rows(
        self, backend: str, operation: str
    ) -> None:
        _exercise_duplicate_full_length_joint_write(backend, operation)

    @pytest.mark.parametrize("selector", ["default_mask", "explicit_full_index"])
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    @_requires_physx
    def test_physx_joint_selector_provenance_controls_staging_read(self, operation: str, selector: str) -> None:
        art, raw_backend = get_articulation(
            "physx",
            num_instances=2,
            num_joints=3,
            num_bodies=2,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode("reversed", 3),
        )
        initial_position = np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        initial_velocity = np.asarray([[110.0, 120.0, 130.0], [140.0, 150.0, 160.0]], dtype=np.float32)
        _seed_backend_joint_state("physx", art, raw_backend, initial_position, initial_velocity)
        # Spy on the public backend tensor-API seam. A full-length write expressed as the default
        # mask (joint_ids is None) is provably complete, so it must not read the backend to stage
        # unselected rows; an explicit full index selector is opaque to the writer and
        # conservatively triggers exactly one staging read per touched axis.
        position_read = MagicMock(wraps=raw_backend.get_dof_positions)
        velocity_read = MagicMock(wraps=raw_backend.get_dof_velocities)
        raw_backend.get_dof_positions = position_read
        raw_backend.get_dof_velocities = velocity_read

        position = torch.tensor(initial_position[:, ::-1].copy(), dtype=torch.float32, device=art.device)
        velocity = torch.tensor(initial_velocity[:, ::-1].copy(), dtype=torch.float32, device=art.device)
        if selector == "default_mask":
            if operation == "position":
                art.write_joint_position_to_sim_mask(position=position)
            elif operation == "velocity":
                art.write_joint_velocity_to_sim_mask(velocity=velocity)
            else:
                art.write_joint_state_to_sim_mask(position=position, velocity=velocity)
        else:
            joint_ids = wp.array([0, 1, 2], dtype=wp.int32, device=art.device)
            if operation == "position":
                art.write_joint_position_to_sim_index(position=position, joint_ids=joint_ids)
            elif operation == "velocity":
                art.write_joint_velocity_to_sim_index(velocity=velocity, joint_ids=joint_ids)
            else:
                art.write_joint_state_to_sim_index(position=position, velocity=velocity, joint_ids=joint_ids)

        should_read = selector == "explicit_full_index"
        assert position_read.call_count == int(should_read and operation in ("position", "state"))
        assert velocity_read.call_count == int(should_read and operation in ("velocity", "state"))

    # Newton is intentionally absent from the backend list. Its partial joint writers (e.g.
    # write_joint_position_to_sim_index) scatter the selected cells straight into the live sim-bound
    # buffer instead of staging and rewriting a full backend row, so there is no stale full-row
    # staging image that a later partial write could resurrect. The hazard this test guards against
    # is structural to the PhysX/OVPhysX read-modify-write staging path and cannot occur on Newton.
    @_physx_ovphysx_backends
    @pytest.mark.parametrize("operation", ["position", "velocity", "state"])
    def test_stale_partial_joint_write_preserves_newer_backend_rows(self, backend: str, operation: str) -> None:
        _exercise_stale_partial_joint_write(backend, operation)

    @_requires_ovphysx
    def test_ovphysx_partial_effort_target_write_preserves_unselected_backend_rows(self) -> None:
        """A partial effort-target write must not leak the last applied torque onto unselected joints.

        ``write_data_to_sim`` pushes the actuator-computed ``_applied_torque`` (which can differ
        from the raw commanded target, e.g. once explicit actuators clip or otherwise transform it)
        to the ``DOF_ACTUATION_FORCE`` binding. That push must use its own backend-order scratch
        buffer rather than ``_joint_effort_target_backend``, because the latter is also the staging
        that :meth:`~isaaclab_ovphysx.assets.articulation.Articulation.set_joint_effort_target_index`
        reuses for unselected joints on a partial write. If the two were the same buffer, a partial
        write would resurrect stale applied torque for every joint it did not touch.
        """
        from isaaclab_ovphysx import tensor_types as TT

        num_instances = 2
        num_joints = 3
        art, raw_backend = get_articulation(
            "ovphysx",
            num_instances=num_instances,
            num_joints=num_joints,
            num_bodies=2,
            device="cpu",
            joint_ordering=_joint_ordering_for_mode("reversed", num_joints),
        )
        art._effort_write_view = object()
        user_to_backend = _ordering_user_to_backend(art.joint_ordering, num_joints)

        # Persist raw effort targets through the public setter, in backend order via the
        # ``_joint_effort_target_backend`` mirror.
        raw_targets = np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        art.set_joint_effort_target_mask(target=torch.tensor(raw_targets, dtype=torch.float32, device=art.device))

        # Simulate an actuator model computing an applied torque that differs from the raw
        # target, then run one full simulation step.
        applied_torque = raw_targets + 100.0
        art.data._applied_torque.assign(wp.array(applied_torque, dtype=wp.float32, device=art.device))
        art.write_data_to_sim()

        pushed_after_step = raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data.copy()
        np.testing.assert_array_equal(pushed_after_step, applied_torque[:, user_to_backend])

        # A partial write on a single joint must preserve the persisted raw targets -- not the
        # applied torque from the last step -- for every joint it does not touch.
        joint_ids = wp.array([1], dtype=wp.int32, device=art.device)
        env_ids = wp.array([0, 1], dtype=wp.int32, device=art.device)
        new_target = torch.tensor([[901.0], [902.0]], dtype=torch.float32, device=art.device)
        art.set_joint_effort_target_index(target=new_target, joint_ids=joint_ids, env_ids=env_ids)

        expected_targets = raw_targets.copy()
        expected_targets[:, 1] = new_target.cpu().numpy()[:, 0]
        expected_backend = expected_targets[:, user_to_backend]

        pushed_after_partial_write = raw_backend.bindings[TT.DOF_ACTUATION_FORCE]._data.copy()
        np.testing.assert_array_equal(pushed_after_partial_write, expected_backend)


class TestArticulationDataJointState:
    """Test data properties for joint state and joint properties."""

    @_requires_ovphysx
    def test_ovphysx_joint_acceleration_differences_public_order_velocities(self):
        """Finite-difference OVPhysX joint velocity entirely in public joint order."""
        art, raw_backend = get_articulation("ovphysx", 2, 3, 2, device="cpu")
        user_to_backend = _install_reversed_joint_ordering(art)
        from isaaclab_ovphysx import tensor_types as TT

        first = np.asarray([[1.0, 2.0, 4.0], [10.0, 20.0, 40.0]], dtype=np.float32)
        second = first + np.asarray([[3.0, 5.0, 7.0], [11.0, 13.0, 17.0]], dtype=np.float32)
        raw_backend.bindings[TT.DOF_VELOCITY]._data = first
        art.data.update(0.1)
        art.data.joint_acc.torch.clone()
        raw_backend.bindings[TT.DOF_VELOCITY]._data = second
        art.data.update(0.1)

        torch.testing.assert_close(art.data.joint_vel.torch, torch.from_numpy(second[:, user_to_backend]))
        torch.testing.assert_close(
            art.data.joint_acc.torch,
            torch.from_numpy((second - first)[:, user_to_backend] / 0.1),
        )

    @_requires_ovphysx
    def test_ovphysx_ordered_joint_state_is_cached_per_sim_timestamp(self):
        """Gather ordered OVPhysX joint position and velocity at most once per timestamp."""
        art, _ = get_articulation(
            "ovphysx",
            2,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        art.data.update(0.01)

        art.data.joint_pos.torch.clone()
        art.data.joint_vel.torch.clone()

        assert art.data._joint_pos_buf.timestamp == art.data._sim_timestamp
        assert art.data._joint_vel_buf.timestamp == art.data._sim_timestamp

    @_non_mock_backends
    @pytest.mark.parametrize("num_instances, num_joints, num_bodies", [(2, 3, 2)])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_reversed_joint_ordering_reorders_public_joint_properties(
        self, backend, num_instances, num_joints, num_bodies, device
    ):
        """Expose every backend joint property under the matching public joint name."""
        joint_ordering = tuple(f"joint_{index}" for index in reversed(range(num_joints)))
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device=device,
            joint_ordering=joint_ordering,
        )
        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
        backend_properties = _get_backend_joint_property_tensors(backend, art, raw_backend)
        public_properties = {
            "stiffness": art.data.joint_stiffness,
            "damping": art.data.joint_damping,
            "armature": art.data.joint_armature,
            "position_limits": art.data.joint_pos_limits,
            "velocity_limits": art.data.joint_vel_limits,
            "effort_limits": art.data.joint_effort_limits,
            "friction": art.data.joint_friction_coeff,
        }
        if backend in ("physx", "ovphysx"):
            public_properties.update(
                {
                    "dynamic_friction": art.data.joint_dynamic_friction_coeff,
                    "viscous_friction": art.data.joint_viscous_friction_coeff,
                }
            )

        for property_name, public_property in public_properties.items():
            _assert_proxy_close(public_property, backend_properties[property_name][:, user_to_backend])


def _make_item_mask(total: int, selected: list[int], device: str) -> wp.array:
    """Create a bool Warp mask with the selected indices set."""
    mask = np.zeros(total, dtype=bool)
    mask[selected] = True
    return wp.array(mask, dtype=wp.bool, device=device)


# ---------------------------------------------------------------------------
# Tests: Articulation operations
# ---------------------------------------------------------------------------


class TestArticulationOperations:
    """Test cross-cutting articulation operations."""

    @_non_mock_backends
    @pytest.mark.parametrize("ordering_mode", ["none", "reversed", "cyclic"])
    @pytest.mark.parametrize("is_fixed_base", [False, True], ids=["floating", "fixed"])
    @pytest.mark.parametrize("device", ["cpu"])
    def test_external_wrenches_are_written_in_backend_body_order(self, backend, ordering_mode, is_fixed_base, device):
        """Write public-order body wrenches to each backend in backend body order."""
        num_instances, num_joints, num_bodies = 2, 1, 4
        backend_body_names = tuple(f"body_{index}" for index in range(num_bodies))
        if ordering_mode == "none":
            body_ordering = None
        elif is_fixed_base:
            if ordering_mode == "reversed":
                body_ordering = (backend_body_names[0], *reversed(backend_body_names[1:]))
            else:
                body_ordering = (backend_body_names[0], backend_body_names[-1], *backend_body_names[1:-1])
        else:
            body_ordering = _body_ordering_for_mode(ordering_mode, num_bodies)
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device=device,
            is_fixed_base=is_fixed_base,
            body_ordering=body_ordering,
        )
        _set_identity_body_poses(backend, art, raw_backend)
        object.__setattr__(art, "_instantaneous_wrench_composer", WrenchComposer(art))
        object.__setattr__(art, "_permanent_wrench_composer", WrenchComposer(art))
        captured = {}
        if backend == "physx":

            def capture_wrench(*, force_data, torque_data, position_data, indices, is_global):
                captured["force"] = force_data.numpy().reshape(num_instances, num_bodies, 3).copy()
                captured["torque"] = torque_data.numpy().reshape(num_instances, num_bodies, 3).copy()

            raw_backend.apply_forces_and_torques_at_position = capture_wrench

        forces = np.arange(num_instances * num_bodies * 3, dtype=np.float32).reshape(num_instances, num_bodies, 3)
        torques = forces + 100.0
        art.instantaneous_wrench_composer.set_forces_and_torques_index(
            forces=wp.array(forces, dtype=wp.vec3f, device=device),
            torques=wp.array(torques, dtype=wp.vec3f, device=device),
        )

        art.write_data_to_sim()

        backend_to_user = _expected_backend_to_user(body_ordering, backend_body_names)
        backend_force, backend_torque = _read_backend_wrench(backend, art, raw_backend, captured)
        np.testing.assert_allclose(backend_force, forces[:, backend_to_user])
        np.testing.assert_allclose(backend_torque, torques[:, backend_to_user])

    @_requires_ovphysx
    def test_ovphysx_configured_defaults_use_public_joint_names(self):
        """Resolve configured OVPhysX defaults against public joint names."""
        art, _ = get_articulation(
            "ovphysx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        zeros = wp.zeros((1, 3), dtype=wp.float32, device="cpu")
        art.data._default_joint_pos.assign(zeros)
        art.data._default_joint_vel.assign(zeros)
        patterns = {"joint_0": 10.0, "joint_1": 20.0, "joint_2": 30.0}

        art._resolve_joint_values(patterns, art.data._default_joint_pos)

        expected = np.asarray([[patterns[name] for name in art.joint_names]], dtype=np.float32)
        np.testing.assert_array_equal(art.data.default_joint_pos.warp.numpy(), expected)

    @_requires_ovphysx
    @pytest.mark.parametrize("ordering_mode", ["reversed", "cyclic"])
    def test_ovphysx_implicit_targets_are_written_in_backend_order(self, ordering_mode: str) -> None:
        """Write implicit position and velocity targets under their matching backend joint names."""
        from isaaclab_ovphysx import tensor_types as TT

        num_instances, num_joints = 2, 3
        backend_joint_names = tuple(f"joint_{index}" for index in range(num_joints))
        joint_ordering = _joint_ordering_for_mode(ordering_mode, num_joints)
        art, raw_backend = get_articulation(
            "ovphysx",
            num_instances=num_instances,
            num_joints=num_joints,
            device="cpu",
            joint_ordering=joint_ordering,
        )
        position = np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
        velocity = position + 100.0
        art.data._joint_pos_target.assign(wp.array(position, dtype=wp.float32, device=art.device))
        art.data._joint_vel_target.assign(wp.array(velocity, dtype=wp.float32, device=art.device))
        object.__setattr__(art, "_has_implicit_actuators", True)

        art.write_data_to_sim()

        backend_to_user = _expected_backend_to_user(joint_ordering, backend_joint_names)
        np.testing.assert_array_equal(
            raw_backend.bindings[TT.DOF_POSITION_TARGET]._data,
            position[:, backend_to_user],
        )
        np.testing.assert_array_equal(
            raw_backend.bindings[TT.DOF_VELOCITY_TARGET]._data,
            velocity[:, backend_to_user],
        )

    @_requires_physx
    @pytest.mark.parametrize("ordering_mode", ["reversed", "cyclic"])
    def test_physx_newton_actuator_forces_are_written_in_backend_order(self, ordering_mode: str):
        """Write Newton-actuator PhysX forces in backend joint order."""
        num_instances = 2
        num_joints = 4
        num_bodies = 2
        backend_joint_names = tuple(f"joint_{index}" for index in range(num_joints))
        joint_ordering = _joint_ordering_for_mode(ordering_mode, num_joints)
        art, raw_backend = get_articulation(
            "physx",
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            joint_ordering=joint_ordering,
        )
        user_forces_np = np.arange(num_instances * num_joints, dtype=np.float32).reshape(num_instances, num_joints)
        user_forces = wp.array(user_forces_np, dtype=wp.float32, device=art.device)
        object.__setattr__(art, "_joint_effort_target_backend", wp.zeros_like(art.data.joint_effort_target.warp))
        wrapper = MagicMock()
        wrapper.joint_f_2d = user_forces
        object.__setattr__(art, "_physx_actuator_wrapper", wrapper)
        object.__setattr__(art, "_has_newton_actuators", True)
        object.__setattr__(art, "_has_implicit_actuators", False)
        art._apply_actuator_model_newton = MagicMock()
        captured = {}

        def _capture_forces(forces, indices):
            captured["forces"] = wp.clone(forces, device="cpu").numpy()
            captured["indices"] = wp.clone(indices, device="cpu").numpy()

        raw_backend.set_dof_actuation_forces = _capture_forces

        art.write_data_to_sim()

        backend_to_user = _expected_backend_to_user(joint_ordering, backend_joint_names)
        np.testing.assert_allclose(captured["forces"], user_forces_np[:, backend_to_user])

    @pytest.mark.parametrize(
        ("method_name", "value_name", "controller_attr"),
        [
            ("write_actuator_stiffness_to_sim", "stiffness", "kp"),
            ("write_actuator_damping_to_sim", "damping", "kd"),
        ],
    )
    @_requires_physx
    def test_physx_newton_actuator_gain_updates_use_public_joint_ids(
        self, method_name: str, value_name: str, controller_attr: str
    ):
        """Route PhysX Newton-actuator gain updates by public joint ID."""
        art, _ = get_articulation(
            "physx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        controller = MagicMock(
            kp=wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"),
            kd=wp.array([1.0, 2.0, 3.0], dtype=wp.float32, device="cpu"),
        )
        actuator = MagicMock(
            controller=controller,
            indices=wp.array([0, 1, 2], dtype=wp.uint32, device="cpu"),
        )
        adapter = MagicMock(actuators=[actuator])
        object.__setattr__(art, "newton_actuator_adapter", adapter)

        getattr(art, method_name)(
            **{
                value_name: torch.tensor([[99.0]], dtype=torch.float32),
                "env_ids": torch.tensor([0], dtype=torch.int32),
                "joint_ids": torch.tensor([0], dtype=torch.int32),
            }
        )

        np.testing.assert_array_equal(
            getattr(controller, controller_attr).numpy(),
            np.asarray([99.0, 2.0, 3.0], dtype=np.float32),
        )

    @_requires_physx
    def test_physx_validate_cfg_reports_velocity_limits_in_public_joint_order(self):
        """Pair public default velocities with limits for the same named joint."""
        art, raw_backend = get_articulation(
            "physx",
            1,
            3,
            2,
            device="cpu",
            joint_ordering=("joint_2", "joint_1", "joint_0"),
        )
        backend_velocity_limits = np.asarray([[1.0, 100.0, 100.0]], dtype=np.float32)
        public_velocity_limits = backend_velocity_limits[:, [2, 1, 0]]
        raw_backend.set_mock_dof_max_velocities(wp.array(backend_velocity_limits, dtype=wp.float32, device="cpu"))
        art.data._joint_vel_limits.assign(wp.array(public_velocity_limits, dtype=wp.float32, device="cpu"))
        art.data._joint_pos_limits.assign(wp.array(np.tile((-100.0, 100.0), (1, 3, 1)), dtype=wp.vec2f, device="cpu"))
        art.data._default_joint_pos.assign(wp.zeros((1, 3), dtype=wp.float32, device="cpu"))
        art.data._default_joint_vel.assign(
            wp.array(np.asarray([[50.0, 50.0, 2.0]], dtype=np.float32), dtype=wp.float32, device="cpu")
        )

        with pytest.raises(ValueError, match=r"'joint_0': 2\.000 not in \[-1\.000, 1\.000\]"):
            art._validate_cfg()


class TestArticulationWritersJoint:
    """Test joint writers/setters with all input combinations."""

    @_non_mock_backends
    @pytest.mark.parametrize("selection", ["index", "mask"])
    def test_reversed_joint_ordering_routes_property_writes_to_backend(self, backend: str, selection: str):
        """Route partial public property writes to matching backend joints."""
        num_instances, num_joints, num_bodies = 2, 4, 2
        joint_ordering = tuple(f"joint_{index}" for index in reversed(range(num_joints)))
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            joint_ordering=joint_ordering,
        )
        if backend == "physx":
            raw_backend._noop_setters = False

        user_to_backend = np.asarray(art.joint_ordering.user_to_backend_indices, dtype=np.int64)
        selected_envs = np.arange(num_instances, dtype=np.int32)
        selected_joints = np.asarray([0, 2], dtype=np.int32)
        property_cases = [
            ("write_joint_stiffness_to_sim", "stiffness", "stiffness"),
            ("write_joint_damping_to_sim", "damping", "damping"),
            ("write_joint_armature_to_sim", "armature", "armature"),
            ("write_joint_position_limit_to_sim", "limits", "position_limits"),
            ("write_joint_velocity_limit_to_sim", "limits", "velocity_limits"),
            ("write_joint_effort_limit_to_sim", "limits", "effort_limits"),
            ("write_joint_friction_coefficient_to_sim", "joint_friction_coeff", "friction"),
        ]
        if backend in ("physx", "ovphysx"):
            property_cases.extend(
                [
                    (
                        "write_joint_dynamic_friction_coefficient_to_sim",
                        "joint_dynamic_friction_coeff",
                        "dynamic_friction",
                    ),
                    (
                        "write_joint_viscous_friction_coefficient_to_sim",
                        "joint_viscous_friction_coeff",
                        "viscous_friction",
                    ),
                ]
            )

        for case_index, (method_name, value_name, property_name) in enumerate(property_cases):
            before = _get_backend_joint_property_tensors(backend, art, raw_backend)[property_name]
            leading_shape = (
                (num_instances, selected_joints.size) if selection == "index" else (num_instances, num_joints)
            )
            values = torch.arange(int(np.prod(leading_shape)), dtype=torch.float32).reshape(leading_shape)
            values = values + 100.0 * (case_index + 1)
            if property_name == "position_limits":
                values = torch.stack((-values, values), dim=-1)
                soft_limits_before = art.data.soft_joint_pos_limits.torch.clone()

            kwargs = {value_name: values}
            if selection == "index":
                kwargs.update(
                    {
                        "env_ids": wp.array(selected_envs, dtype=wp.int32, device=art.device),
                        "joint_ids": wp.array(selected_joints, dtype=wp.int32, device=art.device),
                    }
                )
            else:
                kwargs.update(
                    {
                        "env_mask": wp.ones(num_instances, dtype=wp.bool, device=art.device),
                        "joint_mask": _make_item_mask(num_joints, selected_joints.tolist(), art.device),
                    }
                )

            getattr(art, f"{method_name}_{selection}")(**kwargs)

            after = _get_backend_joint_property_tensors(backend, art, raw_backend)[property_name]
            expected = before.clone()
            backend_joint_ids = torch.as_tensor(user_to_backend[selected_joints], dtype=torch.long)
            if selection == "index":
                expected[:, backend_joint_ids] = values
            else:
                public_joint_ids = torch.as_tensor(selected_joints, dtype=torch.long)
                expected[:, backend_joint_ids] = values[:, public_joint_ids]
            torch.testing.assert_close(after, expected, rtol=0.0, atol=0.0)

            if property_name == "position_limits":
                soft_limits_after = art.data.soft_joint_pos_limits.torch
                public_joint_ids = torch.as_tensor(selected_joints, dtype=torch.long)
                selected_limits = expected[:, backend_joint_ids]
                center = selected_limits.mean(dim=-1)
                half_range = 0.5 * (selected_limits[..., 1] - selected_limits[..., 0])
                factor = art.cfg.soft_joint_pos_limit_factor
                expected_soft_limits = torch.stack(
                    (center - factor * half_range, center + factor * half_range),
                    dim=-1,
                )
                torch.testing.assert_close(
                    soft_limits_after[:, public_joint_ids],
                    expected_soft_limits,
                )
                unselected_joint_ids = torch.tensor([1, 3], dtype=torch.long)
                torch.testing.assert_close(
                    soft_limits_after[:, unselected_joint_ids],
                    soft_limits_before[:, unselected_joint_ids],
                )


class TestArticulationWritersBody:
    """Test body property writers/setters with all input combinations."""

    @_non_mock_backends
    @pytest.mark.parametrize("ordering_mode", ["reversed", "cyclic"])
    @pytest.mark.parametrize("selection", ["index", "mask"])
    def test_body_ordering_routes_property_writes_to_backend(self, backend: str, ordering_mode: str, selection: str):
        """Route partial public property writes to matching backend bodies."""
        num_instances, num_joints, num_bodies = 2, 1, 4
        body_ordering = _body_ordering_for_mode(ordering_mode, num_bodies)
        art, raw_backend = get_articulation(
            backend,
            num_instances,
            num_joints,
            num_bodies,
            device="cpu",
            body_ordering=body_ordering,
        )
        if backend == "physx":
            raw_backend._noop_setters = False

        user_to_backend = np.asarray(art.body_ordering.user_to_backend_indices, dtype=np.int64)
        selected_envs = np.arange(num_instances, dtype=np.int32)
        selected_bodies = np.asarray([0, 2], dtype=np.int32)
        property_cases = [
            ("set_masses", "masses", "mass"),
            ("set_inertias", "inertias", "inertia"),
        ]

        for case_index, (method_name, value_name, property_name) in enumerate(property_cases):
            before = _get_backend_body_property_tensors(backend, art, raw_backend)[property_name]
            leading_shape = (
                (num_instances, selected_bodies.size) if selection == "index" else (num_instances, num_bodies)
            )
            values = torch.arange(int(np.prod(leading_shape)), dtype=torch.float32).reshape(leading_shape)
            values = values + 100.0 * (case_index + 1)
            if property_name == "inertia":
                component_offsets = torch.arange(9, dtype=torch.float32) / 10.0
                values = values.unsqueeze(-1) + component_offsets

            kwargs = {value_name: values}
            if selection == "index":
                kwargs.update(
                    {
                        "env_ids": wp.array(selected_envs, dtype=wp.int32, device=art.device),
                        "body_ids": wp.array(selected_bodies, dtype=wp.int32, device=art.device),
                    }
                )
            else:
                kwargs.update(
                    {
                        "env_mask": wp.ones(num_instances, dtype=wp.bool, device=art.device),
                        "body_mask": _make_item_mask(num_bodies, selected_bodies.tolist(), art.device),
                    }
                )

            getattr(art, f"{method_name}_{selection}")(**kwargs)

            after = _get_backend_body_property_tensors(backend, art, raw_backend)[property_name]
            expected = before.clone()
            backend_body_ids = torch.as_tensor(user_to_backend[selected_bodies], dtype=torch.long)
            if selection == "index":
                expected[:, backend_body_ids] = values
            else:
                public_body_ids = torch.as_tensor(selected_bodies, dtype=torch.long)
                expected[:, backend_body_ids] = values[:, public_body_ids]
            torch.testing.assert_close(after, expected, rtol=0.0, atol=0.0)
