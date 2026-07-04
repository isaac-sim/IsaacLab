# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for task-authored Newton particle-state synchronization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.assets.mpm_object import mpm_object as mpm_object_module
from isaaclab_newton.assets.mpm_object.mpm_object import MPMObject
from isaaclab_newton.physics import NewtonCfg, NewtonManager
from isaaclab_newton.physics._authored_state_transaction import AuthoredStateTransaction

from isaaclab.physics import PhysicsManager


class _MPMObjectUnderTest(MPMObject):
    """MPM object with teardown disabled for lightweight method tests."""

    def __del__(self) -> None:
        pass


@pytest.mark.parametrize(
    ("selector_name", "selector", "expected_worlds"),
    [
        ("env_ids", [2, 0], [True, False, True]),
        ("env_mask", [False, True, False], [False, True, False]),
    ],
)
def test_particle_invalidation_marks_exact_worlds_without_fk(monkeypatch, selector_name, selector, expected_worlds):
    """Particle writes select solver-transaction worlds without selecting rigid articulations."""
    transaction = AuthoredStateTransaction(3, 2, "cpu", lambda worlds, articulations: None)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)
    monkeypatch.setattr(NewtonManager, "_particles_dirty", False, raising=False)

    if selector_name == "env_ids":
        value = wp.array(selector, dtype=wp.int32, device="cpu")
    else:
        value = wp.array(selector, dtype=wp.bool, device="cpu")
    NewtonManager.invalidate_particles(**{selector_name: value})

    assert transaction.world_mask.numpy().tolist() == expected_worlds
    assert transaction.fk_mask.numpy().tolist() == [False, False]
    assert transaction._pending.numpy().tolist() == [1]
    assert NewtonManager._particles_dirty is True


@pytest.mark.parametrize("selector_name", ["env_ids", "env_mask"])
def test_empty_particle_selection_does_not_publish_state_work(monkeypatch, selector_name):
    """Present-but-empty particle selections remain no-ops rather than selecting every world."""
    calls: list[str] = []
    transaction = AuthoredStateTransaction(3, 2, "cpu", lambda worlds, articulations: calls.append("apply"))
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    if selector_name == "env_ids":
        value = wp.empty(0, dtype=wp.int32, device="cpu")
    else:
        value = wp.zeros(3, dtype=wp.bool, device="cpu")
    NewtonManager.invalidate_particles(**{selector_name: value})
    transaction.flush()

    assert calls == []
    assert transaction.world_mask.numpy().tolist() == [False, False, False]
    assert transaction.fk_mask.numpy().tolist() == [False, False]
    assert transaction._pending.numpy().tolist() == [0]


def test_particle_and_rigid_writes_coalesce_before_owned_solver_resets(monkeypatch):
    """One transaction preserves both selection domains before reset fan-out."""

    class _ResetRecorder:
        def __init__(self) -> None:
            self.calls: list[tuple[object, list[bool], int]] = []

        def reset(self, state, *, world_mask, flags):
            self.calls.append((state, world_mask.numpy().tolist(), flags))

    first = _ResetRecorder()
    second = _ResetRecorder()
    state = object()
    observed_fk: list[list[bool]] = []
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(solver_reset=True), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_0", state, raising=False)
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_owned_solvers", classmethod(lambda cls: (first, second)))
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk_impl",
        classmethod(lambda cls, worlds, articulations: observed_fk.append(articulations.numpy().tolist())),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_synchronize_solver_state",
        classmethod(lambda cls, solver, worlds, articulations: None),
    )

    transaction = AuthoredStateTransaction(3, 3, "cpu", NewtonManager._apply_state_writes)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)
    particle_ids = wp.array([2], dtype=wp.int32, device="cpu")
    rigid_mask = wp.array([False, True, False], dtype=wp.bool, device="cpu")
    articulation_ids = wp.array([[0], [1], [2]], dtype=wp.int32, device="cpu")

    NewtonManager.invalidate_particles(env_ids=particle_ids)
    NewtonManager.invalidate_fk(env_mask=rigid_mask, articulation_ids=articulation_ids)
    NewtonManager._flush_pending_changes()

    assert observed_fk == [[False, True, False]]
    expected_reset = [(state, [False, True, True], 0)]
    assert first.calls == expected_reset
    assert second.calls == expected_reset


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_captured_particle_writer_rearms_only_particle_rendering(monkeypatch):
    """Particle writes replayed from a CUDA graph remain visible at every render boundary."""
    device = "cuda:0"
    env_mask = wp.zeros(1, dtype=wp.bool, device=device)
    transaction = AuthoredStateTransaction(1, 0, device, lambda worlds, articulations: None)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    with wp.ScopedCapture(device=device) as capture:
        NewtonManager.invalidate_particles(env_mask=env_mask)

    assert transaction.replay_render_domains == transaction.RENDER_PARTICLES

    observed: list[tuple[bool, bool]] = []

    def sync_particles(cls):
        observed.append((NewtonManager._transforms_dirty, NewtonManager._particles_dirty))
        NewtonManager._particles_dirty = False

    monkeypatch.setattr(NewtonManager, "_flush_pending_changes", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "sync_transforms_to_usd", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "sync_particles_to_usd", classmethod(sync_particles))
    monkeypatch.setattr(NewtonManager, "_transforms_dirty", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_particles_dirty", False, raising=False)

    env_mask.fill_(True)
    for _ in range(2):
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        NewtonManager.pre_render()

    assert observed == [(False, True), (False, True)]


@pytest.mark.parametrize(
    ("method_name", "selector_name"),
    [
        ("write_nodal_state_to_sim_index", "env_ids"),
        ("write_nodal_pos_to_sim_index", "env_ids"),
        ("write_nodal_velocity_to_sim_index", "env_ids"),
        ("write_nodal_state_to_sim_mask", "env_mask"),
        ("write_nodal_pos_to_sim_mask", "env_mask"),
        ("write_nodal_velocity_to_sim_mask", "env_mask"),
    ],
)
def test_mpm_writers_publish_the_resolved_selection(monkeypatch, method_name, selector_name):
    """Every MPM state writer publishes the selector used by its scatter kernel."""
    asset = object.__new__(_MPMObjectUnderTest)
    asset._device = "cpu"
    asset._num_instances = 3
    asset._particles_per_object = 1
    asset._particle_offsets = object()
    asset._data = SimpleNamespace(
        _particle_pos_w=SimpleNamespace(timestamp=1.0),
        _particle_vel_w=SimpleNamespace(timestamp=1.0),
        _particle_state_w=SimpleNamespace(timestamp=1.0),
        _root_pos_w=SimpleNamespace(timestamp=1.0),
        _root_vel_w=SimpleNamespace(timestamp=1.0),
    )
    caller_selection = object()
    resolved_selection = SimpleNamespace(shape=(3,))
    calls: list[dict[str, object]] = []
    state = SimpleNamespace(particle_q=object(), particle_qd=object())

    monkeypatch.setattr(asset, "_resolve_env_ids", lambda value: resolved_selection)
    monkeypatch.setattr(asset, "_resolve_mask", lambda value: resolved_selection)
    monkeypatch.setattr(asset, "_as_warp", lambda *args, **kwargs: object())
    monkeypatch.setattr(asset, "_iter_particle_states", lambda: iter((state,)))
    monkeypatch.setattr(mpm_object_module.wp, "launch", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        NewtonManager,
        "invalidate_particles",
        classmethod(lambda cls, **kwargs: calls.append(kwargs)),
    )

    getattr(asset, method_name)(object(), **{selector_name: caller_selection})

    expected = {"env_ids": None, "env_mask": None}
    expected[selector_name] = resolved_selection
    assert calls == [expected]
