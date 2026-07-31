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
from isaaclab_contrib.deformable.newton_manager_cfg import (
    CoupledMJWarpVBDSolverCfg as LegacyMJWarpCfg,
)
from isaaclab_contrib.deformable.newton_manager_cfg import (
    VBDSolverCfg,
)


def test_reset_forwards_to_both_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the real sub-solvers instead of the dummy solver slot."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = False
    soft_solver = MagicMock()
    state = object()
    world_mask = object()

    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", rigid_solver, raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", soft_solver, raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_state_0", state)

    NewtonCoupledMJWarpVBDManager._reset_solver_internals(world_mask)

    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)


def test_reset_skips_all_false_cpu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU warm-start state when no world needs reset."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = True
    soft_solver = MagicMock()
    world_mask = MagicMock()
    world_mask.numpy.return_value.any.return_value = False

    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", rigid_solver, raising=False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", soft_solver, raising=False)

    NewtonCoupledMJWarpVBDManager._reset_solver_internals(world_mask)

    rigid_solver.reset.assert_not_called()
    soft_solver.reset.assert_not_called()


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


def test_build_solver_disables_contact_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    monkeypatch.setattr(manager_module.NewtonManager, "_report_contacts", False)
    monkeypatch.setattr(manager_module.NewtonManager, "_supports_contact_sensors", True)
    monkeypatch.setattr(manager_module.NewtonManager, "_solver", None)
    monkeypatch.setattr(manager_module.NewtonManager, "_use_single_state", True)
    monkeypatch.setattr(manager_module.NewtonManager, "_needs_collision_pipeline", False)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_rigid_solver", None)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_soft_solver", None)
    monkeypatch.setattr(NewtonCoupledMJWarpVBDManager, "_coupling_mode", None)
    rigid_manager = MagicMock()
    soft_manager = MagicMock()
    monkeypatch.setattr(solver_cfg.rigid_solver_cfg, "class_type", rigid_manager)
    monkeypatch.setattr(solver_cfg.soft_solver_cfg, "class_type", soft_manager)
    monkeypatch.setattr(manager_module, "SolverBase", MagicMock())

    NewtonCoupledMJWarpVBDManager._build_solver(MagicMock(), solver_cfg)

    assert manager_module.NewtonManager._supports_contact_sensors is False


def test_legacy_mjwarp_solver_config_warns() -> None:
    with pytest.warns(DeprecationWarning, match="custom_coupling.CoupledMJWarpVBDSolverCfg"):
        solver_cfg = LegacyMJWarpCfg()

    assert solver_cfg.class_type == (
        "isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager:NewtonCoupledMJWarpVBDManager"
    )


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
