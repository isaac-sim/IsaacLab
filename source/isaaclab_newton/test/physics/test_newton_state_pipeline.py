# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for consuming task-authored Newton state as one transaction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.physics import (
    KaminoSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonKaminoManager,
    NewtonManager,
    NewtonMJWarpManager,
)
from isaaclab_newton.physics import kamino_manager as kamino_manager_module
from isaaclab_newton.physics import mjwarp_manager as mjwarp_manager_module
from isaaclab_newton.physics import newton_manager as newton_manager_module
from isaaclab_newton.physics._authored_state_transaction import AuthoredStateTransaction

from isaaclab.physics import PhysicsManager


class _SolverRecorder:
    """Minimal solver that records synchronization calls."""

    def __init__(self, events: list[str] | None = None) -> None:
        self.events = [] if events is None else events
        self.notifications: list[int] = []

    def notify_model_changed(self, flags: int) -> None:
        self.notifications.append(flags)
        self.events.append(f"notify:{flags}")


@wp.kernel
def _increment_counter(counter: wp.array(dtype=wp.int32)):
    wp.atomic_add(counter, 0, 1)


def test_apply_state_writes_reconciles_derived_and_solver_state(monkeypatch):
    """Every transaction updates FK before backend-specific state handoff."""
    events: list[str] = []
    solver = _SolverRecorder(events)
    monkeypatch.setattr(NewtonManager, "_solver", solver, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_eval_fk_impl",
        classmethod(lambda cls, worlds, articulations: events.append("fk")),
    )
    monkeypatch.setattr(
        NewtonManager,
        "_synchronize_solver_state",
        classmethod(lambda cls, owned_solver, worlds, articulations: events.append("sync")),
    )

    NewtonManager._apply_state_writes(object(), object())

    assert events == ["fk", "sync"]


def test_model_changes_are_delivered_to_every_owned_solver(monkeypatch):
    """Pending flags reach every owned solver without changing their representation."""
    first = _SolverRecorder()
    second = _SolverRecorder()
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_owned_solvers",
        classmethod(lambda cls: (first, second)),
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)

    NewtonManager.add_model_change(1)
    NewtonManager.add_model_change(4)
    NewtonManager._notify_solver_model_changes()

    assert sorted(first.notifications) == [1, 4]
    assert sorted(second.notifications) == [1, 4]
    assert NewtonManager._model_changes == set()


def test_model_changes_wait_until_a_solver_exists(monkeypatch):
    """Pre-initialization notifications are retained for the first real solver."""
    monkeypatch.setattr(NewtonManager, "_model_changes", {3}, raising=False)
    monkeypatch.setattr(NewtonManager, "_owned_solvers", classmethod(lambda cls: ()))

    NewtonManager._notify_solver_model_changes()

    assert NewtonManager._model_changes == {3}


def test_invalidate_fk_sets_masks_and_device_transaction_gate(monkeypatch):
    """The same kernel launch records both dirty topology and pending work."""
    transaction = AuthoredStateTransaction(2, 2, "cpu", lambda worlds, articulations: None)
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    articulation_ids = wp.array([[0], [1]], dtype=wp.int32, device="cpu")
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager.invalidate_fk(env_mask=env_mask, articulation_ids=articulation_ids)

    assert transaction.world_mask.numpy().tolist() == [False, True]
    assert transaction.fk_mask.numpy().tolist() == [False, True]
    assert transaction._pending.numpy().tolist() == [1]


def test_invalidate_fk_preserves_selected_articulation_columns(monkeypatch):
    """Partial collection writes do not dirty history for untouched objects."""
    transaction = AuthoredStateTransaction(2, 4, "cpu", lambda worlds, articulations: None)
    env_ids = wp.array([1], dtype=wp.int32, device="cpu")
    articulation_ids = wp.array([[0, 1], [2, 3]], dtype=wp.int32, device="cpu")
    articulation_selection = wp.array([0], dtype=wp.int32, device="cpu")
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager.invalidate_fk(
        env_ids=env_ids,
        articulation_ids=articulation_ids,
        articulation_selection=articulation_selection,
    )

    assert transaction.world_mask.numpy().tolist() == [False, True]
    assert transaction.fk_mask.numpy().tolist() == [False, False, True, False]


def test_invalidate_fk_marks_world_without_articulations(monkeypatch):
    """Rigid-only worlds still request solver-owned state clearing."""
    transaction = AuthoredStateTransaction(1, 0, "cpu", lambda worlds, articulations: None)
    env_mask = wp.array([True], dtype=wp.bool, device="cpu")
    articulation_ids = wp.empty((1, 0), dtype=wp.int32, device="cpu")
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager.invalidate_fk(env_mask=env_mask, articulation_ids=articulation_ids)

    assert transaction.world_mask.numpy().tolist() == [True]
    assert transaction._pending.numpy().tolist() == [1]


def test_invalidate_fk_without_topology_conservatively_marks_all_articulations(monkeypatch):
    """Topology-free rigid invalidation preserves the conservative FK fallback."""
    transaction = AuthoredStateTransaction(2, 3, "cpu", lambda worlds, articulations: None)
    env_mask = wp.array([False, True], dtype=wp.bool, device="cpu")
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager.invalidate_fk(env_mask=env_mask)

    assert transaction.world_mask.numpy().tolist() == [True, True]
    assert transaction.fk_mask.numpy().tolist() == [True, True, True]
    assert transaction._pending.numpy().tolist() == [1]


@pytest.mark.skipif(
    not wp.get_cuda_device_count() or not wp.is_conditional_graph_supported(),
    reason="CUDA conditional graphs are unavailable",
)
def test_captured_state_writer_replays_one_device_gated_transaction(monkeypatch):
    """Replayed reset writers do not depend on a capture-time Python flag."""
    device = "cuda:0"
    env_mask = wp.zeros(1, dtype=wp.bool, device=device)
    articulation_ids = wp.array([[0]], dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)

    def apply(worlds, articulations):
        wp.launch(_increment_counter, 1, inputs=[counter], device=device)

    transaction = AuthoredStateTransaction(1, 1, device, apply)
    transaction.configure_capture(graph_safe=True, defer=True)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    with wp.ScopedCapture(device=device) as capture:
        NewtonManager.invalidate_fk(env_mask=env_mask, articulation_ids=articulation_ids)
        NewtonManager._flush_pending_changes()

    assert transaction.writes_may_replay
    observed_dirty: list[bool] = []

    def sync_transforms(cls):
        observed_dirty.append(NewtonManager._transforms_dirty)
        NewtonManager._transforms_dirty = False

    monkeypatch.setattr(NewtonManager, "_flush_pending_changes", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "sync_transforms_to_usd", classmethod(sync_transforms))
    monkeypatch.setattr(NewtonManager, "sync_particles_to_usd", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_transforms_dirty", False, raising=False)

    env_mask.fill_(True)
    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    NewtonManager.pre_render()

    wp.capture_launch(capture.graph)
    wp.synchronize_device(device)
    NewtonManager.pre_render()

    assert counter.numpy().tolist() == [2]
    assert observed_dirty == [True, True]


@pytest.mark.skipif(
    not wp.get_cuda_device_count() or not wp.is_conditional_graph_supported(),
    reason="CUDA conditional graphs are unavailable",
)
def test_clean_boundary_captured_before_writer_graph_still_consumes_replay():
    """Boundary capture always records the conditional, even before any writer is captured."""
    device = "cuda:0"
    env_mask = wp.array([True], dtype=wp.bool, device=device)
    articulation_ids = wp.array([[0]], dtype=wp.int32, device=device)
    counter = wp.zeros(1, dtype=wp.int32, device=device)

    def apply(worlds, articulations):
        wp.launch(_increment_counter, 1, inputs=[counter], device=device)

    transaction = AuthoredStateTransaction(1, 1, device, apply)
    transaction.configure_capture(graph_safe=True, enabled=False)

    with wp.ScopedCapture(device=device) as boundary_capture:
        transaction.flush()
    with wp.ScopedCapture(device=device) as writer_capture:
        transaction.mark_rigid(env_mask=env_mask, articulation_ids=articulation_ids)

    transaction._world_mask.zero_()
    transaction._fk_mask.zero_()
    transaction._pending.zero_()
    transaction._host_pending = False

    wp.capture_launch(writer_capture.graph)
    wp.capture_launch(boundary_capture.graph)
    wp.synchronize_device(device)

    assert counter.numpy().tolist() == [1]


@pytest.mark.skipif(not wp.get_cuda_device_count(), reason="CUDA is unavailable")
def test_captured_step_rearms_transform_render_domain(monkeypatch):
    """A replayed physics step cannot lose host-only transform dirtiness."""
    device = "cuda:0"
    counter = wp.zeros(1, dtype=wp.int32, device=device)
    transaction = AuthoredStateTransaction(1, 0, device, lambda worlds, articulations: None)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    with wp.ScopedCapture(device=device) as capture:
        wp.launch(_increment_counter, 1, inputs=[counter], device=device)
        NewtonManager._mark_transforms_dirty()

    observed_dirty: list[bool] = []

    def sync_transforms(cls):
        observed_dirty.append(NewtonManager._transforms_dirty)
        NewtonManager._transforms_dirty = False

    monkeypatch.setattr(NewtonManager, "_flush_pending_changes", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "sync_transforms_to_usd", classmethod(sync_transforms))
    monkeypatch.setattr(NewtonManager, "sync_particles_to_usd", classmethod(lambda cls: None))
    monkeypatch.setattr(NewtonManager, "_transforms_dirty", False, raising=False)

    for _ in range(2):
        wp.capture_launch(capture.graph)
        wp.synchronize_device(device)
        NewtonManager.pre_render()

    assert counter.numpy().tolist() == [2]
    assert observed_dirty == [True, True]


def test_graph_unsafe_state_handoff_is_rejected_during_outer_capture():
    """A backend that performs host transfers cannot enter an outer CUDA graph."""
    transaction = AuthoredStateTransaction(1, 1, "cpu", lambda worlds, articulations: None)
    transaction.configure_capture(graph_safe=False)
    transaction._device = SimpleNamespace(is_cuda=True, stream=SimpleNamespace(is_capturing=True))

    with pytest.raises(RuntimeError, match="not CUDA-graph-safe"):
        transaction.flush()


def test_outer_capture_requires_cuda_conditional_nodes(monkeypatch):
    """Unsupported conditional capture fails explicitly instead of during callback work."""
    transaction = AuthoredStateTransaction(1, 1, "cpu", lambda worlds, articulations: None)
    transaction.configure_capture(graph_safe=True)
    transaction._device = SimpleNamespace(is_cuda=True, stream=SimpleNamespace(is_capturing=True))
    monkeypatch.setattr(wp, "is_conditional_graph_supported", lambda: False)

    with pytest.raises(RuntimeError, match="requires CUDA conditional graph nodes"):
        transaction.flush()


def test_flush_applies_and_consumes_state_transaction(monkeypatch):
    """A flush clears masks only after state reconciliation succeeds."""
    events: list[str] = []
    transaction = AuthoredStateTransaction(1, 1, "cpu", lambda worlds, articulations: events.append("apply"))
    transaction.mark_rigid()

    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager._flush_pending_changes()

    assert events == ["apply"]
    assert transaction.world_mask.numpy().tolist() == [False]
    assert transaction.fk_mask.numpy().tolist() == [False]
    assert transaction._pending.numpy().tolist() == [0]


def test_clean_second_flush_skips_state_transaction(monkeypatch):
    """A consumed transaction does not repeat reset work at later boundaries."""
    calls: list[str] = []
    transaction = AuthoredStateTransaction(1, 1, "cpu", lambda worlds, articulations: calls.append("consume"))
    transaction.mark_rigid()
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    NewtonManager._flush_pending_changes()
    NewtonManager._flush_pending_changes()

    assert calls == ["consume"]


def test_flush_retains_masks_when_state_sync_fails(monkeypatch):
    """A failed reconciliation leaves device masks available for diagnosis/retry."""

    def fail(worlds, articulations):
        raise RuntimeError("state sync failed")

    transaction = AuthoredStateTransaction(1, 1, "cpu", fail)
    transaction.mark_rigid(articulation_ids=wp.array([[0]], dtype=wp.int32, device="cpu"))
    monkeypatch.setattr(NewtonManager, "_model_changes", set(), raising=False)
    monkeypatch.setattr(NewtonManager, "_state_writes", transaction, raising=False)

    with pytest.raises(RuntimeError, match="state sync failed"):
        NewtonManager._flush_pending_changes()

    assert transaction.world_mask.numpy().tolist() == [True]
    assert transaction.fk_mask.numpy().tolist() == [True]
    assert transaction._pending.numpy().tolist() == [1]


def test_forward_resolves_concrete_manager(monkeypatch):
    """Data-layer calls through NewtonManager still reach solver-specific policy."""
    calls: list[str] = []

    class _ActiveManager:
        @classmethod
        def _flush_pending_changes(cls) -> None:
            calls.append("flush")

    monkeypatch.setattr(
        PhysicsManager,
        "_sim",
        SimpleNamespace(physics_manager=_ActiveManager),
        raising=False,
    )

    NewtonManager.forward()

    assert calls == ["flush"]


def test_step_flushes_before_deferred_graph_warmup(monkeypatch):
    """No graph warmup may advance state before authored writes are coherent."""
    events: list[str] = []

    def capture(cls, device):
        events.append("capture")
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(use_cuda_graph=True),
        raising=False,
    )
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_flush_pending_changes",
        classmethod(lambda cls: events.append("flush")),
    )
    monkeypatch.setattr(NewtonManager, "_capture_relaxed_graph", classmethod(capture))

    with pytest.raises(RuntimeError, match="ordering check"):
        NewtonManager.step()

    assert events == ["flush", "capture"]


def test_step_captures_deferred_state_graph_before_flush(monkeypatch):
    """RTX uses one safe relaxed window before consuming the first transaction."""
    events: list[str] = []

    class _DeferredTransaction:
        needs_capture = True

        def capture_deferred(self, capture_fn):
            capture_fn(lambda: None)

    def capture(cls, device, capture_fn=None, *, warmup=True):
        if capture_fn is not None:
            events.append(f"state_capture:{warmup}")
            return object()
        events.append("physics_capture")
        raise RuntimeError("stop after ordering check")

    monkeypatch.setattr(PhysicsManager, "_sim", SimpleNamespace(is_playing=lambda: True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(use_cuda_graph=True), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cuda:0", raising=False)
    monkeypatch.setattr(NewtonManager, "_state_writes", _DeferredTransaction(), raising=False)
    monkeypatch.setattr(NewtonManager, "_graph_capture_pending", True, raising=False)
    monkeypatch.setattr(
        NewtonManager,
        "_flush_pending_changes",
        classmethod(lambda cls: events.append("flush")),
    )
    monkeypatch.setattr(NewtonManager, "_capture_relaxed_graph", classmethod(capture))

    with pytest.raises(RuntimeError, match="ordering check"):
        NewtonManager.step()

    assert events == ["state_capture:False", "flush", "physics_capture"]


def test_pre_render_flushes_before_fabric_sync(monkeypatch):
    """Fabric cannot consume stale body transforms after an authored write."""
    events: list[str] = []
    monkeypatch.setattr(
        NewtonManager,
        "_flush_pending_changes",
        classmethod(lambda cls: events.append("flush")),
    )
    monkeypatch.setattr(
        NewtonManager,
        "sync_transforms_to_usd",
        classmethod(lambda cls: events.append("transforms")),
    )
    monkeypatch.setattr(
        NewtonManager,
        "sync_particles_to_usd",
        classmethod(lambda cls: events.append("particles")),
    )

    NewtonManager.pre_render()

    assert events == ["flush", "transforms", "particles"]


def test_get_state_flushes_newton_backend(monkeypatch):
    """Direct renderer/visualizer state access receives coherent Newton state."""
    events: list[str] = []
    state = object()
    monkeypatch.setattr(
        NewtonManager,
        "_backend_is_newton",
        classmethod(lambda cls, provider=None: True),
    )
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(
        NewtonManager,
        "update_visualization_state",
        classmethod(lambda cls, provider=None: events.append("visualization")),
    )
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))

    assert NewtonManager.get_state() is state
    assert events == ["forward", "visualization"]


def test_scene_data_backend_state_flushes_before_transform_read(monkeypatch):
    """SceneDataProvider's direct state boundary cannot expose stale body_q."""
    events: list[str] = []
    state = object()
    backend = newton_manager_module.NewtonSceneDataBackend()
    monkeypatch.setattr(NewtonManager, "forward", classmethod(lambda cls: events.append("forward")))
    monkeypatch.setattr(NewtonManager, "get_state_0", classmethod(lambda cls: state))

    assert backend.state is state
    assert events == ["forward"]


def test_kamino_registers_state_buffers_before_finalize(monkeypatch):
    """Kamino state arrays are model-owned rather than allocated during capture."""
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=KaminoSolverCfg()),
        raising=False,
    )

    builder = NewtonKaminoManager.create_builder()

    assert builder.has_custom_attribute("body_f_total")
    assert builder.has_custom_attribute("joint_q_prev")
    assert builder.has_custom_attribute("joint_lambdas")

    # Registration is intentionally unconditional and idempotent so a custom
    # builder cannot leave the three state buffers only partially declared.
    NewtonKaminoManager._register_builder_attributes(builder)


def test_kamino_state_write_uses_one_mandatory_reset(monkeypatch):
    """Kamino reconciles authored joint state exactly once."""

    class _KaminoResetRecorder:
        class ResetConfig:
            @staticmethod
            def from_joints():
                return "from_joints"

            @staticmethod
            def preserve():
                return "preserve"

        def __init__(self) -> None:
            self.calls: list[tuple[object, object, object, object]] = []

        def reset(self, state, *, world_mask, flags=None, config=None):
            self.calls.append((state, world_mask, flags, config))

    solver = _KaminoResetRecorder()
    state = object()
    world_mask = object()
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=KaminoSolverCfg(use_fk_solver=True)),
        raising=False,
    )
    monkeypatch.setattr(kamino_manager_module, "SolverKamino", _KaminoResetRecorder)
    monkeypatch.setattr(NewtonKaminoManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonKaminoManager, "_state_0", state, raising=False)

    NewtonKaminoManager._apply_state_writes(world_mask, object())

    assert solver.calls == [(state, world_mask, None, "from_joints")]


def test_cpu_mujoco_rejects_multiworld(monkeypatch):
    """The CPU MuJoCo backend itself is limited to one Newton world."""
    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_cpu=True))
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)

    with pytest.raises(ValueError, match="supports only one Newton world"):
        NewtonMJWarpManager._build_solver(SimpleNamespace(world_count=2), cfg.solver_cfg)


def test_cpu_mujoco_allows_single_world(monkeypatch):
    """CPU MuJoCo accepts its supported single-world topology."""

    class _CpuMuJoCo:
        def __init__(self, model, **kwargs):
            self.model = model
            self.use_mujoco_cpu = True

    cfg = NewtonCfg(solver_cfg=MJWarpSolverCfg(use_mujoco_cpu=True))
    monkeypatch.setattr(PhysicsManager, "_cfg", cfg, raising=False)
    monkeypatch.setattr(mjwarp_manager_module, "SolverMuJoCo", _CpuMuJoCo)

    NewtonMJWarpManager._build_solver(SimpleNamespace(world_count=1), cfg.solver_cfg)

    assert isinstance(NewtonManager._solver, _CpuMuJoCo)


@pytest.mark.parametrize(
    ("use_mujoco_cpu", "update_data_interval", "expected"),
    [(True, 1, False), (False, 2, False), (False, 1, True), (False, 0, True)],
)
def test_mujoco_main_graph_requires_device_per_step_handoff(
    monkeypatch, use_mujoco_cpu, update_data_interval, expected
):
    """Host execution and Python sparse cadence stay outside the main CUDA graph."""
    solver = SimpleNamespace(use_mujoco_cpu=use_mujoco_cpu, update_data_interval=update_data_interval)
    monkeypatch.setattr(NewtonMJWarpManager, "_solver", solver, raising=False)

    assert NewtonMJWarpManager._supports_cuda_graph_capture() is expected


def test_mujoco_sparse_update_interval_receives_authored_state(monkeypatch):
    """Sparse MuJoCo cadence still receives task-authored state immediately."""

    class _MuJoCoRecorder:
        use_mujoco_cpu = False
        update_data_interval = 2
        mj_data = None
        mjw_data = object()

        def __init__(self):
            self.events: list[tuple] = []

        def reset(self, state, *, world_mask, flags):
            self.events.append(("reset", state, world_mask, flags))

        def _update_mjc_data(self, data, model, state):
            self.events.append(("sync", data, model, state))

    solver = _MuJoCoRecorder()
    state = object()
    model = object()
    mask = object()
    monkeypatch.setattr(
        PhysicsManager,
        "_cfg",
        NewtonCfg(solver_cfg=MJWarpSolverCfg()),
        raising=False,
    )
    monkeypatch.setattr(NewtonMJWarpManager, "_solver", solver, raising=False)
    monkeypatch.setattr(NewtonMJWarpManager, "_state_0", state, raising=False)
    monkeypatch.setattr(NewtonMJWarpManager, "_model", model, raising=False)
    monkeypatch.setattr(
        NewtonMJWarpManager,
        "_eval_fk_impl",
        classmethod(lambda cls, worlds, articulations: solver.events.append(("fk", worlds, articulations))),
    )

    fk_mask = object()
    NewtonMJWarpManager._apply_state_writes(mask, fk_mask)

    assert solver.events == [
        ("fk", mask, fk_mask),
        ("sync", solver.mjw_data, model, state),
    ]
