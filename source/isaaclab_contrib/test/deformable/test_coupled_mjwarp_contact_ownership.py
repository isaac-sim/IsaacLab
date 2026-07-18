# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for coupled MJWarp-VBD contact ownership."""

from types import SimpleNamespace

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonManager

import isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager as coupled_manager_module
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.deformable.newton_manager_cfg import CoupledMJWarpVBDSolverCfg


class _State:
    def __init__(self):
        self.particle_f = SimpleNamespace(zero_=lambda: None)

    def clear_forces(self) -> None:
        pass


class _RecordingRigidSolver:
    def __init__(self, model, *, use_mujoco_contacts=True, use_mujoco_cpu=False):
        self.model = model
        self.use_mujoco_cpu = use_mujoco_cpu


class _SoftSolver:
    def __init__(self, model):
        self.model = model


class _BaseSolver:
    def __init__(self, model):
        self.model = model


def _patch_solver_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(coupled_manager_module, "SolverMuJoCo", _RecordingRigidSolver)
    monkeypatch.setattr(coupled_manager_module, "SolverVBD", _SoftSolver)
    monkeypatch.setattr(coupled_manager_module, "SolverBase", _BaseSolver)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", None, raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", None, raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_coupling_mode", None)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_uses_mujoco_contacts", False, raising=False)
    monkeypatch.setattr(NewtonManager, "_solver", NewtonManager._solver)
    monkeypatch.setattr(NewtonManager, "_use_single_state", NewtonManager._use_single_state)
    monkeypatch.setattr(NewtonManager, "_needs_collision_pipeline", NewtonManager._needs_collision_pipeline)


def _run_one_way_step(monkeypatch: pytest.MonkeyPatch, model, contacts) -> list:
    observed_contacts = []

    class _CollisionPipeline:
        def collide(self, state, stage_contacts) -> None:
            observed_contacts.append(stage_contacts)

    class _RigidSolver:
        def step(self, state_in, state_out, control, stage_contacts, dt) -> None:
            assert model.particle_count == 0
            observed_contacts.append(stage_contacts)

    class _SoftSolver:
        def step(self, state_in, state_out, control, stage_contacts, dt) -> None:
            observed_contacts.append(stage_contacts)

    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_contacts", contacts)
    monkeypatch.setattr(NewtonManager, "_collision_pipeline", _CollisionPipeline())
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", _RigidSolver())
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", _SoftSolver())

    NewtonCoupledMJWarpVBDManager._step_one_way(_State(), _State(), object(), 0.01)  # type: ignore[arg-type]
    return observed_contacts


def test_coupled_mjwarp_public_default_preserves_legacy_contacts():
    """The coupled public default must preserve legacy contacts throughout the deprecation window."""
    assert CoupledMJWarpVBDSolverCfg().rigid_solver_cfg.use_mujoco_contacts is True


def test_coupled_mjwarp_legacy_contacts_warn(monkeypatch):
    """Legacy mode warns and keeps internal contacts isolated to the rigid sub-solver."""
    contacts = object()
    model = SimpleNamespace(particle_count=7)
    _patch_solver_constructors(monkeypatch)
    cfg = CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=True))

    with pytest.warns(DeprecationWarning, match="internal contact pipeline"):
        NewtonCoupledMJWarpVBDManager._build_solver(model, cfg)

    observed_contacts = _run_one_way_step(monkeypatch, model, contacts)

    assert observed_contacts == [contacts, None, contacts]
    assert NewtonCoupledMJWarpVBDManager._rigid_uses_mujoco_contacts is True
    assert model.particle_count == 7


def test_coupled_mjwarp_cpu_with_newton_contacts_raises(monkeypatch):
    """The coupled manager must reject a CPU sub-solver that cannot consume Newton contacts."""
    monkeypatch.setattr(coupled_manager_module, "SolverMuJoCo", None)
    cfg = CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_cpu=True, use_mujoco_contacts=False))

    with pytest.raises(ValueError, match="cannot consume Newton CollisionPipeline contacts"):
        NewtonCoupledMJWarpVBDManager._build_solver(None, cfg)  # type: ignore[arg-type]


def test_coupled_subsolvers_share_manager_contacts(monkeypatch):
    """External mode must pass the same manager-owned contacts through every stage."""
    contacts = object()
    model = SimpleNamespace(particle_count=7)
    _patch_solver_constructors(monkeypatch)
    cfg = CoupledMJWarpVBDSolverCfg(
        rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False),
    )
    NewtonCoupledMJWarpVBDManager._build_solver(model, cfg)

    observed_contacts = _run_one_way_step(monkeypatch, model, contacts)

    assert NewtonCoupledMJWarpVBDManager._rigid_uses_mujoco_contacts is False
    assert observed_contacts == [contacts, contacts, contacts]
    assert model.particle_count == 7


def test_coupled_rigid_step_restores_particle_count_after_failure(monkeypatch):
    """The temporary particle-count override must be restored when the rigid solver raises."""
    model = SimpleNamespace(particle_count=7)

    class _FailingRigidSolver:
        def step(self, state_in, state_out, control, contacts, dt) -> None:
            assert model.particle_count == 0
            raise RuntimeError("rigid solver failed")

    monkeypatch.setattr(NewtonManager, "_model", model)
    monkeypatch.setattr(NewtonManager, "_contacts", object())
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", _FailingRigidSolver(), raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_uses_mujoco_contacts", False, raising=False)

    with pytest.raises(RuntimeError, match="rigid solver failed"):
        NewtonCoupledMJWarpVBDManager._rigid_step(object(), object(), object(), 0.01)  # type: ignore[arg-type]

    assert model.particle_count == 7
