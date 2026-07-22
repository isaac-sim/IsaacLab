# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling manager."""

from unittest.mock import MagicMock

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg

import isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager as manager_module
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg


def test_lifecycle_solver_forwards_model_changes() -> None:
    rigid_solver = MagicMock()
    soft_solver = MagicMock()
    solver = manager_module._CoupledLifecycleSolver(MagicMock(), rigid_solver, soft_solver)

    solver.notify_model_changed(3)

    rigid_solver.notify_model_changed.assert_called_once_with(3)
    soft_solver.notify_model_changed.assert_called_once_with(3)


def test_lifecycle_solver_resets_mjwarp() -> None:
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = False
    soft_solver = MagicMock()
    state = object()
    world_mask = object()
    solver = manager_module._CoupledLifecycleSolver(MagicMock(), rigid_solver, soft_solver)

    solver.reset(state, world_mask=world_mask, flags=0)

    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_not_called()


def test_lifecycle_solver_skips_all_false_cpu_mask() -> None:
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = True
    world_mask = MagicMock()
    world_mask.numpy.return_value.any.return_value = False
    solver = manager_module._CoupledLifecycleSolver(MagicMock(), rigid_solver, MagicMock())

    solver.reset(object(), world_mask=world_mask, flags=0)

    rigid_solver.reset.assert_not_called()


@pytest.mark.parametrize(
    ("solver_cfg", "match"),
    [
        (CoupledMJWarpVBDSolverCfg(coupling_mode="invalid"), "coupling_mode"),
        (
            CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=MJWarpSolverCfg(use_mujoco_contacts=False)),
            "MJWarp internal contacts",
        ),
        (
            CoupledMJWarpVBDSolverCfg(soft_solver_cfg=VBDSolverCfg()),
            "VBD external rigid-body integration",
        ),
    ],
)
def test_build_solver_rejects_invalid_configuration(solver_cfg: CoupledMJWarpVBDSolverCfg, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        NewtonCoupledMJWarpVBDManager._build_solver(MagicMock(), solver_cfg)


def test_build_solver_rejects_contact_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_module.NewtonManager, "_report_contacts", True)

    with pytest.raises(NotImplementedError, match="contact sensors are not supported"):
        NewtonCoupledMJWarpVBDManager._build_solver(MagicMock(), CoupledMJWarpVBDSolverCfg())


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
def test_step_preserves_input_forces(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    state_in = MagicMock()
    state_in.body_f = object()
    state_in.particle_f = MagicMock()
    state_out = MagicMock()
    control = object()
    contacts = object()
    collision_pipeline = MagicMock()
    rigid_solver = MagicMock()
    soft_solver = MagicMock()
    reactions = MagicMock()

    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_contacts", contacts)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_collision_pipeline", collision_pipeline)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", rigid_solver)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", soft_solver)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_apply_reactions", reactions)

    getattr(NewtonCoupledMJWarpVBDManager, f"_step_{mode}")(state_in, state_out, control, 0.01)

    state_in.clear_forces.assert_not_called()
    state_in.particle_f.zero_.assert_not_called()
    state_out.clear_forces.assert_called_once_with()
    collision_pipeline.collide.assert_called_once_with(state_in, contacts)
    rigid_solver.step.assert_called_once_with(state_in, state_out, control, None, 0.01)
    soft_solver.step.assert_called_once_with(state_in, state_out, control, contacts, 0.01)
    if mode == "two_way":
        reactions.assert_called_once_with(state_in, state_out, 0.01)
    else:
        reactions.assert_not_called()


def test_solver_specific_clear_releases_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    base_clear = MagicMock()
    monkeypatch.setattr(
        manager_module.NewtonVBDManager,
        "_solver_specific_clear",
        classmethod(lambda cls: base_clear()),
    )
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", object())
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", object())
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_coupling_mode", "two_way")

    NewtonCoupledMJWarpVBDManager._solver_specific_clear()

    base_clear.assert_called_once_with()
    assert NewtonCoupledMJWarpVBDManager._rigid_solver is None
    assert NewtonCoupledMJWarpVBDManager._soft_solver is None
    assert NewtonCoupledMJWarpVBDManager._coupling_mode is None
