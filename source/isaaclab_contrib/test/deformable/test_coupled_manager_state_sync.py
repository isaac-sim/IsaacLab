# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for coupled Newton manager solver ownership and state synchronization."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import warp as wp
from isaaclab_newton.physics import NewtonCfg, NewtonManager

from isaaclab.physics import PhysicsManager

from isaaclab_contrib.deformable import (
    CoupledFeatherstoneVBDSolverCfg,
    CoupledMJWarpVBDSolverCfg,
)
from isaaclab_contrib.deformable import coupled_featherstone_vbd_manager as featherstone_module
from isaaclab_contrib.deformable import coupled_mjwarp_vbd_manager as mjwarp_module
from isaaclab_contrib.deformable._state_sync import rebase_rigid_body_history
from isaaclab_contrib.deformable.coupled_featherstone_vbd_manager import NewtonCoupledFeatherstoneVBDManager
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager


class _SolverRecorder:
    """Minimal solver that records model notifications."""

    def __init__(self, model=None, **kwargs) -> None:
        self.model = model
        self.update_data_interval = 1
        self.use_mujoco_cpu = False
        self.notifications: list[int] = []

    def notify_model_changed(self, flags: int) -> None:
        self.notifications.append(flags)


@pytest.mark.parametrize(
    "manager",
    [NewtonCoupledMJWarpVBDManager, NewtonCoupledFeatherstoneVBDManager],
)
def test_coupled_state_sync_fans_out_to_both_real_solvers(monkeypatch, manager):
    """The placeholder/primary slot never swallows coupled notifications or state sync."""
    rigid = _SolverRecorder()
    soft = _SolverRecorder()
    state = object()
    world_mask = object()
    module = mjwarp_module if manager is NewtonCoupledMJWarpVBDManager else featherstone_module
    monkeypatch.setattr(module, "rebase_rigid_body_history", lambda model, current, previous, fk_mask: None)
    monkeypatch.setattr(manager, "_rigid_solver", rigid, raising=False)
    monkeypatch.setattr(manager, "_soft_solver", soft, raising=False)
    monkeypatch.setattr(manager, "_state_0", state, raising=False)
    monkeypatch.setattr(manager, "_state_1", object(), raising=False)
    monkeypatch.setattr(manager, "_model", object(), raising=False)
    monkeypatch.setattr(manager, "_eval_fk_impl", classmethod(lambda cls, worlds, articulations: None))
    monkeypatch.setattr(NewtonManager, "_model_changes", {1, 4}, raising=False)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(), raising=False)
    monkeypatch.setattr(PhysicsManager, "_device", "cpu", raising=False)

    manager._notify_solver_model_changes()
    manager._apply_state_writes(world_mask, object())

    assert sorted(rigid.notifications) == [1, 4]
    assert sorted(soft.notifications) == [1, 4]


def test_coupled_history_rebases_only_worlds_with_rigid_writes():
    """A reset teleport cannot become finite-difference velocity on the next substep."""
    model = SimpleNamespace(
        joint_count=2,
        joint_articulation=wp.array([0, 1], dtype=wp.int32, device="cpu"),
        joint_child=wp.array([0, 1], dtype=wp.int32, device="cpu"),
    )
    state = SimpleNamespace(
        body_q=wp.array(
            [(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
            dtype=wp.transformf,
            device="cpu",
        ),
        body_qd=wp.array(
            [(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), (6.0, 5.0, 4.0, 3.0, 2.0, 1.0)],
            dtype=wp.spatial_vectorf,
            device="cpu",
        ),
    )
    state_prev = SimpleNamespace(
        body_q=wp.array(
            [(-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0), (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)],
            dtype=wp.transformf,
            device="cpu",
        ),
        body_qd=wp.zeros(2, dtype=wp.spatial_vectorf, device="cpu"),
    )
    fk_mask = wp.array([False, True], dtype=wp.bool, device="cpu")

    rebase_rigid_body_history(model, state, state_prev, fk_mask)

    assert state_prev.body_q.numpy()[:, 0].tolist() == [-1.0, 2.0]
    assert state_prev.body_qd.numpy().tolist() == [[0.0] * 6, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]]


def test_coupled_history_rebases_global_free_body():
    """Global rigid bodies are selected by articulation rather than world ownership."""
    model = SimpleNamespace(
        joint_count=1,
        joint_articulation=wp.array([0], dtype=wp.int32, device="cpu"),
        joint_child=wp.array([0], dtype=wp.int32, device="cpu"),
    )
    state = SimpleNamespace(
        body_q=wp.array([(3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.transformf, device="cpu"),
        body_qd=wp.array([(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)], dtype=wp.spatial_vectorf, device="cpu"),
    )
    state_prev = SimpleNamespace(
        body_q=wp.array([(-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)], dtype=wp.transformf, device="cpu"),
        body_qd=wp.zeros(1, dtype=wp.spatial_vectorf, device="cpu"),
    )

    rebase_rigid_body_history(model, state, state_prev, wp.array([True], dtype=wp.bool, device="cpu"))

    assert state_prev.body_q.numpy()[0, 0] == 3.0
    assert state_prev.body_qd.numpy()[0].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    ("manager", "module", "cfg"),
    [
        (
            NewtonCoupledMJWarpVBDManager,
            mjwarp_module,
            CoupledMJWarpVBDSolverCfg(coupling_mode="one_way"),
        ),
        (
            NewtonCoupledFeatherstoneVBDManager,
            featherstone_module,
            CoupledFeatherstoneVBDSolverCfg(coupling_mode="one_way"),
        ),
    ],
)
def test_coupled_build_keeps_contact_placeholder_out_of_owned_solvers(monkeypatch, manager, module, cfg):
    """Contact plumbing keeps its placeholder while state work reaches real solvers."""
    rigid_name = "SolverMuJoCo" if manager is NewtonCoupledMJWarpVBDManager else "SolverFeatherstone"
    monkeypatch.setattr(module, rigid_name, _SolverRecorder)
    monkeypatch.setattr(module, "SolverVBD", _SolverRecorder)
    monkeypatch.setattr(module, "SolverBase", _SolverRecorder)
    monkeypatch.setattr(PhysicsManager, "_cfg", NewtonCfg(), raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", None, raising=False)

    manager._build_solver(SimpleNamespace(world_count=1), cfg)

    assert NewtonManager._solver is not manager._rigid_solver
    assert NewtonManager._solver not in manager._owned_solvers()
    assert manager._owned_solvers() == (manager._rigid_solver, manager._soft_solver)


def test_coupled_featherstone_clear_releases_kinematic_buffers(monkeypatch):
    """Full teardown releases every coupled Featherstone device-buffer reference."""
    sentinel = object()
    monkeypatch.setattr(featherstone_module, "clear_deformable_builder_hooks", lambda: None)
    for name in ("_gravity_zero", "_gravity_saved", "_ke_saved", "_kd_saved", "_ke_zero", "_kd_zero"):
        monkeypatch.setattr(NewtonCoupledFeatherstoneVBDManager, name, sentinel, raising=False)

    NewtonCoupledFeatherstoneVBDManager._solver_specific_clear()

    for name in ("_gravity_zero", "_gravity_saved", "_ke_saved", "_kd_saved", "_ke_zero", "_kd_zero"):
        assert getattr(NewtonCoupledFeatherstoneVBDManager, name) is None


@pytest.mark.parametrize(
    ("use_mujoco_cpu", "update_data_interval", "expected"),
    [(True, 1, False), (False, 2, False), (False, 1, True), (False, 0, True)],
)
def test_coupled_mujoco_main_graph_requires_device_per_step_handoff(
    monkeypatch, use_mujoco_cpu, update_data_interval, expected
):
    """Coupled MuJoCo preserves Python cadence by staying eager when required."""
    solver = SimpleNamespace(use_mujoco_cpu=use_mujoco_cpu, update_data_interval=update_data_interval)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", solver, raising=False)

    assert NewtonCoupledMJWarpVBDManager._supports_cuda_graph_capture() is expected
