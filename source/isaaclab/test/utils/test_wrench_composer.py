# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.test.utils import test_devices
from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.wrench_composer import WrenchComposer

pytestmark = pytest.mark.unit

IDENTITY_QUAT = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def make_asset(
    num_envs: int,
    num_bodies: int,
    device: str,
    link_quat: np.ndarray | None = None,
    com_pos: np.ndarray | None = None,
) -> SimpleNamespace:
    """Build the minimal asset state read by :class:`WrenchComposer` (CoM positions and link quaternions)."""
    if link_quat is None:
        link_quat = np.broadcast_to(IDENTITY_QUAT, (num_envs, num_bodies, 4))
    if com_pos is None:
        com_pos = np.zeros((num_envs, num_bodies, 3), dtype=np.float32)
    data = SimpleNamespace(
        body_com_pos_w=ProxyArray(wp.array(np.ascontiguousarray(com_pos), dtype=wp.vec3f, device=device)),
        body_link_quat_w=ProxyArray(wp.array(np.ascontiguousarray(link_quat), dtype=wp.quatf, device=device)),
    )
    return SimpleNamespace(num_instances=num_envs, num_bodies=num_bodies, device=device, data=data)


def random_unit_quaternion(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    q = rng.standard_normal((*shape, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rotate_inverse(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate ``vec`` by the inverse of unit quaternion ``quat_xyzw`` (broadcasting over leading dims)."""
    xyz = -quat_xyzw[..., :3]
    w = quat_xyzw[..., 3:4]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def to_wp(array: np.ndarray, device: str, dtype=wp.vec3f) -> wp.array:
    return wp.array(np.ascontiguousarray(array), dtype=dtype, device=device)


class Reference:
    """Numpy model of the composer: accumulate contributions on a full (num_envs, num_bodies) grid, then compose."""

    def __init__(self, link_quat: np.ndarray, com_pos: np.ndarray):
        self.link_quat = link_quat
        self.com_pos = com_pos
        shape = link_quat.shape[:2] + (3,)
        self.force_w = np.zeros(shape, dtype=np.float32)
        self.torque_w = np.zeros(shape, dtype=np.float32)
        self.force_b = np.zeros(shape, dtype=np.float32)
        self.torque_b = np.zeros(shape, dtype=np.float32)

    def add(self, forces=None, torques=None, positions=None, env_ids=None, body_ids=None, is_global=False):
        num_envs, num_bodies = self.force_w.shape[:2]
        env_ids = np.arange(num_envs) if env_ids is None else np.asarray(env_ids)
        body_ids = np.arange(num_bodies) if body_ids is None else np.asarray(body_ids)
        cell = np.ix_(env_ids, body_ids)
        force, torque = (self.force_w, self.torque_w) if is_global else (self.force_b, self.torque_b)
        if forces is not None:
            force[cell] += forces
            if positions is not None:
                lever = positions - self.com_pos[cell] if is_global else positions
                torque[cell] += np.cross(lever, forces)
        if torques is not None:
            torque[cell] += torques

    @property
    def out_force_b(self) -> np.ndarray:
        return rotate_inverse(self.link_quat, self.force_w) + self.force_b

    @property
    def out_torque_b(self) -> np.ndarray:
        return rotate_inverse(self.link_quat, self.torque_w) + self.torque_b


def assert_composed(composer: WrenchComposer, reference: Reference, atol: float = 1e-3):
    np.testing.assert_allclose(composer.out_force_b.warp.numpy(), reference.out_force_b, atol=atol, rtol=1e-4)
    np.testing.assert_allclose(composer.out_torque_b.warp.numpy(), reference.out_torque_b, atol=atol, rtol=1e-4)


def make_scene(rng: np.random.Generator, num_envs: int, num_bodies: int, device: str):
    """Asset with random link orientations and CoM positions, plus a matching numpy reference."""
    link_quat = random_unit_quaternion(rng, (num_envs, num_bodies))
    com_pos = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
    composer = WrenchComposer(make_asset(num_envs, num_bodies, device, link_quat, com_pos))
    return composer, Reference(link_quat, com_pos)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize(("num_envs", "num_bodies"), [(1, 1), (7, 3)])
@pytest.mark.parametrize("is_global", [False, True], ids=["local", "global"])
@pytest.mark.parametrize(
    ("use_forces", "use_torques", "use_positions"),
    [(True, False, False), (False, True, False), (True, False, True), (True, True, True)],
    ids=["forces", "torques", "forces_at_positions", "forces_torques_at_positions"],
)
def test_add_index_accumulates_over_random_subsets(
    device, num_envs, num_bodies, is_global, use_forces, use_torques, use_positions
):
    rng = np.random.default_rng(0)
    composer, reference = make_scene(rng, num_envs, num_bodies, device)

    for _ in range(3):
        env_ids = np.sort(rng.choice(num_envs, size=rng.integers(1, num_envs, endpoint=True), replace=False))
        body_ids = np.sort(rng.choice(num_bodies, size=rng.integers(1, num_bodies, endpoint=True), replace=False))
        shape = (len(env_ids), len(body_ids), 3)
        forces = rng.uniform(-100.0, 100.0, shape).astype(np.float32) if use_forces else None
        torques = rng.uniform(-100.0, 100.0, shape).astype(np.float32) if use_torques else None
        positions = rng.uniform(-10.0, 10.0, shape).astype(np.float32) if use_positions else None

        composer.add_forces_and_torques_index(
            forces=None if forces is None else to_wp(forces, device),
            torques=None if torques is None else to_wp(torques, device),
            positions=None if positions is None else to_wp(positions, device),
            env_ids=to_wp(env_ids.astype(np.int32), device, wp.int32),
            body_ids=to_wp(body_ids.astype(np.int32), device, wp.int32),
            is_global=is_global,
        )
        reference.add(forces, torques, positions, env_ids, body_ids, is_global)

    assert composer.active
    assert_composed(composer, reference)


@pytest.mark.parametrize("device", test_devices())
def test_global_force_rotation_and_com_lever_arm(device):
    """Hand-checkable cases: a +X world force under a 90-degree yaw and a lever arm about an offset CoM."""
    yaw_90 = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)], dtype=np.float32).reshape(1, 1, 4)
    composer = WrenchComposer(make_asset(1, 1, device, link_quat=yaw_90))
    composer.add_forces_and_torques_index(forces=to_wp(np.array([[[1.0, 0.0, 0.0]]]), device), is_global=True)
    np.testing.assert_allclose(composer.out_force_b.warp.numpy(), [[[0.0, -1.0, 0.0]]], atol=1e-6)

    com_pos = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
    composer = WrenchComposer(make_asset(1, 1, device, com_pos=com_pos))
    force = np.array([[[0.0, 0.0, 10.0]]], dtype=np.float32)
    # Force at the world origin acts through a lever arm of (0 - com) = (-1, 0, 0): torque = cross(-x, 10z) = +10y.
    composer.add_forces_and_torques_index(
        forces=to_wp(force, device), positions=to_wp(0 * force, device), is_global=True
    )
    np.testing.assert_allclose(composer.out_torque_b.warp.numpy(), [[[0.0, 10.0, 0.0]]], atol=1e-6)
    # Force applied at the CoM produces no torque.
    composer.reset()
    composer.add_forces_and_torques_index(forces=to_wp(force, device), positions=to_wp(com_pos, device), is_global=True)
    np.testing.assert_allclose(composer.out_torque_b.warp.numpy(), 0.0, atol=1e-6)
    np.testing.assert_allclose(composer.out_force_b.warp.numpy(), force, atol=1e-6)


@pytest.mark.parametrize("device", test_devices())
def test_mixed_local_and_global_contributions(device):
    rng = np.random.default_rng(1)
    composer, reference = make_scene(rng, 5, 3, device)
    local = rng.uniform(-100.0, 100.0, (5, 3, 3)).astype(np.float32)
    global_ = rng.uniform(-100.0, 100.0, (5, 3, 3)).astype(np.float32)

    composer.add_forces_and_torques_index(forces=to_wp(local, device), torques=to_wp(local, device))
    composer.add_forces_and_torques_index(forces=to_wp(global_, device), torques=to_wp(global_, device), is_global=True)
    reference.add(forces=local, torques=local)
    reference.add(forces=global_, torques=global_, is_global=True)

    np.testing.assert_allclose(composer.local_force_b.numpy(), local, atol=1e-5)
    np.testing.assert_allclose(composer.global_force_at_com_w.numpy(), global_, atol=1e-5)
    np.testing.assert_allclose(composer.global_torque_w.numpy(), global_, atol=1e-5)
    assert_composed(composer, reference)


@pytest.mark.parametrize("device", test_devices())
def test_mask_matches_index_for_add_and_set(device):
    rng = np.random.default_rng(2)
    num_envs, num_bodies = 6, 3
    env_mask = np.array([True, False, True, True, False, True])
    body_mask = np.array([True, False, True])
    env_ids = np.flatnonzero(env_mask).astype(np.int32)
    body_ids = np.flatnonzero(body_mask).astype(np.int32)
    link_quat = random_unit_quaternion(rng, (num_envs, num_bodies))

    for method in ("add", "set"):
        forces = rng.uniform(-100.0, 100.0, (num_envs, num_bodies, 3)).astype(np.float32)
        positions = rng.uniform(-10.0, 10.0, (num_envs, num_bodies, 3)).astype(np.float32)
        subset = np.ix_(env_ids, body_ids)

        by_index = WrenchComposer(make_asset(num_envs, num_bodies, device, link_quat))
        getattr(by_index, f"{method}_forces_and_torques_index")(
            forces=to_wp(forces[subset], device),
            positions=to_wp(positions[subset], device),
            env_ids=to_wp(env_ids, device, wp.int32),
            body_ids=to_wp(body_ids, device, wp.int32),
            is_global=True,
        )
        by_mask = WrenchComposer(make_asset(num_envs, num_bodies, device, link_quat))
        getattr(by_mask, f"{method}_forces_and_torques_mask")(
            forces=to_wp(forces, device),
            positions=to_wp(positions, device),
            env_mask=to_wp(env_mask, device, wp.bool),
            body_mask=to_wp(body_mask, device, wp.bool),
            is_global=True,
        )
        np.testing.assert_allclose(
            by_index.out_force_b.warp.numpy(), by_mask.out_force_b.warp.numpy(), atol=1e-4, rtol=1e-5
        )
        np.testing.assert_allclose(
            by_index.out_torque_b.warp.numpy(), by_mask.out_torque_b.warp.numpy(), atol=1e-4, rtol=1e-5
        )


@pytest.mark.parametrize("device", test_devices())
def test_add_raw_buffers_from(device):
    rng = np.random.default_rng(3)
    num_envs, num_bodies = 4, 3
    link_quat = random_unit_quaternion(rng, (num_envs, num_bodies))
    kwargs_a = dict(forces=rng.uniform(-50, 50, (num_envs, num_bodies, 3)).astype(np.float32), is_global=False)
    kwargs_b = dict(
        forces=rng.uniform(-50, 50, (num_envs, num_bodies, 3)).astype(np.float32),
        positions=rng.uniform(-5, 5, (num_envs, num_bodies, 3)).astype(np.float32),
        is_global=True,
    )
    composers = [WrenchComposer(make_asset(num_envs, num_bodies, device, link_quat)) for _ in range(3)]
    for composer, kwargs in zip(composers, (kwargs_a, kwargs_b, kwargs_a)):
        composer.add_forces_and_torques_index(
            **{k: (v if k == "is_global" else to_wp(v, device)) for k, v in kwargs.items()}
        )
    target, source, reference = composers
    reference.add_forces_and_torques_index(
        **{k: (v if k == "is_global" else to_wp(v, device)) for k, v in kwargs_b.items()}
    )

    target.add_raw_buffers_from(source)
    np.testing.assert_allclose(target.out_force_b.warp.numpy(), reference.out_force_b.warp.numpy(), atol=1e-4)
    np.testing.assert_allclose(target.out_torque_b.warp.numpy(), reference.out_torque_b.warp.numpy(), atol=1e-4)

    # Merging an inactive composer is a no-op.
    before = target.local_force_b.numpy().copy()
    inactive = WrenchComposer(make_asset(num_envs, num_bodies, device, link_quat))
    assert not inactive.active
    target.add_raw_buffers_from(inactive)
    np.testing.assert_array_equal(target.local_force_b.numpy(), before)


@pytest.mark.parametrize("device", test_devices())
@pytest.mark.parametrize("selector", ["index", "mask"])
def test_set_replaces_only_targeted_envs(device, selector):
    rng = np.random.default_rng(4)
    num_envs, num_bodies = 4, 3
    composer = WrenchComposer(make_asset(num_envs, num_bodies, device))
    global_forces = rng.uniform(-50, 50, (num_envs, num_bodies, 3)).astype(np.float32)
    positions = rng.uniform(-5, 5, (num_envs, num_bodies, 3)).astype(np.float32)
    local_torques = rng.uniform(-50, 50, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(
        forces=to_wp(global_forces, device), positions=to_wp(positions, device), is_global=True
    )
    composer.add_forces_and_torques_index(torques=to_wp(local_torques, device))

    targeted = np.array([0, 2])
    kept = np.array([1, 3])
    new_forces = rng.uniform(-50, 50, (num_envs, num_bodies, 3)).astype(np.float32)
    if selector == "index":
        composer.set_forces_and_torques_index(
            forces=to_wp(new_forces[targeted], device), env_ids=to_wp(targeted.astype(np.int32), device, wp.int32)
        )
    else:
        env_mask = np.isin(np.arange(num_envs), targeted)
        composer.set_forces_and_torques_mask(
            forces=to_wp(new_forces, device), env_mask=to_wp(env_mask, device, wp.bool)
        )

    expected_local_force = np.zeros_like(new_forces)
    expected_local_force[targeted] = new_forces[targeted]
    np.testing.assert_allclose(composer.local_force_b.numpy(), expected_local_force, atol=1e-5)
    np.testing.assert_array_equal(composer.global_force_w.numpy()[targeted], 0.0)
    np.testing.assert_array_equal(composer.global_torque_w.numpy()[targeted], 0.0)
    np.testing.assert_array_equal(composer.local_torque_b.numpy()[targeted], 0.0)
    np.testing.assert_allclose(composer.global_force_w.numpy()[kept], global_forces[kept], atol=1e-5)
    np.testing.assert_allclose(composer.local_torque_b.numpy()[kept], local_torques[kept], atol=1e-5)


@pytest.mark.parametrize("env_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("body_dtype", [torch.int32, torch.int64])
def test_torch_index_dtypes_select_cells(env_dtype, body_dtype):
    composer = WrenchComposer(make_asset(3, 3, "cpu"))
    env_ids = torch.tensor([2, 0], dtype=env_dtype)
    body_ids = torch.tensor([1, 2], dtype=body_dtype)
    set_forces = np.arange(1, 13, dtype=np.float32).reshape(2, 2, 3)
    add_forces = np.full((2, 2, 3), 100.0, dtype=np.float32)

    composer.set_forces_and_torques_index(forces=to_wp(set_forces, "cpu"), env_ids=env_ids, body_ids=body_ids)
    composer.add_forces_and_torques_index(forces=to_wp(add_forces, "cpu"), env_ids=env_ids, body_ids=body_ids)
    expected = np.zeros((3, 3, 3), dtype=np.float32)
    expected[np.ix_([2, 0], [1, 2])] = set_forces + add_forces
    np.testing.assert_array_equal(composer.local_force_b.numpy(), expected)

    composer.reset(env_ids=env_ids[:1])
    expected[2] = 0.0
    np.testing.assert_array_equal(composer.local_force_b.numpy(), expected)


@pytest.mark.parametrize("device", test_devices())
def test_reset_full_and_partial(device):
    rng = np.random.default_rng(5)
    num_envs, num_bodies = 6, 2
    composer = WrenchComposer(make_asset(num_envs, num_bodies, device))
    local = rng.uniform(-100, 100, (num_envs, num_bodies, 3)).astype(np.float32)
    global_ = rng.uniform(-100, 100, (num_envs, num_bodies, 3)).astype(np.float32)
    composer.add_forces_and_torques_index(forces=to_wp(local, device), torques=to_wp(local, device))
    composer.add_forces_and_torques_index(forces=to_wp(global_, device), is_global=True)
    composer.compose_to_body_frame()

    reset_ids = np.array([1, 3], dtype=np.int32)
    kept = np.array([0, 2, 4, 5])
    composer.reset(env_ids=to_wp(reset_ids, device, wp.int32))
    np.testing.assert_array_equal(composer.local_force_b.numpy()[reset_ids], 0.0)
    np.testing.assert_array_equal(composer.global_force_at_com_w.numpy()[reset_ids], 0.0)
    np.testing.assert_allclose(composer.local_force_b.numpy()[kept], local[kept], atol=1e-5)
    np.testing.assert_allclose(composer.global_force_at_com_w.numpy()[kept], global_[kept], atol=1e-5)
    assert composer.active and composer._dirty

    for env_ids in (None, slice(None)):
        composer.add_forces_and_torques_index(forces=to_wp(local, device))
        composer.reset(env_ids=env_ids)
        for buffer in (
            composer.global_force_w,
            composer.global_torque_w,
            composer.global_force_at_com_w,
            composer.local_force_b,
            composer.local_torque_b,
            composer.out_force_b.warp,
            composer.out_torque_b.warp,
        ):
            np.testing.assert_array_equal(buffer.numpy(), 0.0)
        assert not composer.active and not composer._dirty


@pytest.mark.parametrize("device", test_devices())
def test_lazy_composition_and_dirty_flag(device):
    rng = np.random.default_rng(6)
    composer, reference = make_scene(rng, 4, 2, device)
    forces = rng.uniform(-100, 100, (4, 2, 3)).astype(np.float32)
    positions = rng.uniform(-1, 1, (4, 2, 3)).astype(np.float32)
    assert not composer._dirty

    composer.add_forces_and_torques_index(
        forces=to_wp(forces, device), positions=to_wp(positions, device), is_global=True
    )
    reference.add(forces=forces, positions=positions, is_global=True)
    assert composer._dirty
    # Output properties compose on demand without an explicit compose_to_body_frame().
    assert_composed(composer, reference)
    assert not composer._dirty

    composer.add_forces_and_torques_index(torques=to_wp(forces, device), is_global=True)
    reference.add(torques=forces, is_global=True)
    assert composer._dirty
    first = composer.out_torque_b.warp.numpy().copy()
    assert not composer._dirty
    composer.compose_to_body_frame()
    np.testing.assert_array_equal(composer.out_torque_b.warp.numpy(), first)
    assert_composed(composer, reference)


@pytest.mark.parametrize("device", test_devices())
def test_deprecated_api_warns_and_forwards(device):
    rng = np.random.default_rng(7)
    composer = WrenchComposer(make_asset(4, 2, device))
    first = rng.uniform(-50, 50, (4, 2, 3)).astype(np.float32)
    second = rng.uniform(-50, 50, (4, 2, 3)).astype(np.float32)

    with pytest.warns(DeprecationWarning, match="add_forces_and_torques.*is deprecated"):
        composer.add_forces_and_torques(forces=to_wp(first, device), torques=to_wp(first, device))
    with pytest.warns(DeprecationWarning, match="composed_force.*is deprecated"):
        np.testing.assert_allclose(composer.composed_force.warp.numpy(), first, atol=1e-5)
    with pytest.warns(DeprecationWarning, match="composed_torque.*is deprecated"):
        np.testing.assert_allclose(composer.composed_torque.warp.numpy(), first, atol=1e-5)

    with pytest.warns(DeprecationWarning, match="set_forces_and_torques.*is deprecated"):
        composer.set_forces_and_torques(forces=to_wp(second, device))
    np.testing.assert_allclose(composer.out_force_b.warp.numpy(), second, atol=1e-5)
    np.testing.assert_array_equal(composer.out_torque_b.warp.numpy(), 0.0)
