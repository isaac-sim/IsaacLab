# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling manager."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from isaaclab_newton.physics import MJWarpSolverCfg

import isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager as manager_module
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import NewtonCoupledMJWarpVBDManager
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg
from isaaclab_contrib.deformable.coupled_featherstone_vbd_manager import (
    NewtonCoupledFeatherstoneVBDManager as LegacyFeatherstoneManager,
)
from isaaclab_contrib.deformable.coupled_mjwarp_vbd_manager import (
    NewtonCoupledMJWarpVBDManager as LegacyMJWarpManager,
)
from isaaclab_contrib.deformable.newton_manager_cfg import VBDSolverCfg


@pytest.mark.parametrize("manager_cls", [NewtonCoupledMJWarpVBDManager, LegacyMJWarpManager, LegacyFeatherstoneManager])
def test_reset_forwards_to_both_subsolvers(manager_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the real sub-solvers instead of the dummy solver slot."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = False
    soft_solver = MagicMock()
    state = object()
    world_mask = object()

    monkeypatch.setattr(manager_cls, "_rigid_solver", rigid_solver, raising=False)
    monkeypatch.setattr(manager_cls, "_soft_solver", soft_solver, raising=False)
    monkeypatch.setattr(manager_cls, "_state_0", state)

    manager_cls._reset_solver_internals(world_mask)

    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)


def test_reset_skips_all_false_cpu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU warm-start state when no world needs reset."""
    rigid_solver = MagicMock()
    rigid_solver.use_mujoco_cpu = True
    soft_solver = MagicMock()
    world_mask = MagicMock()
    world_mask.numpy.return_value = np.zeros(2, dtype=bool)

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


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
@pytest.mark.parametrize("manager_cls", [NewtonCoupledMJWarpVBDManager, LegacyMJWarpManager, LegacyFeatherstoneManager])
def test_step_preserves_input_forces(manager_cls: type, mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
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
    model = MagicMock()
    model.particle_count = 0

    monkeypatch.setattr(manager_cls, "_model", model)
    monkeypatch.setattr(manager_cls, "_contacts", contacts)
    monkeypatch.setattr(manager_cls, "_collision_pipeline", collision_pipeline)
    monkeypatch.setattr(manager_cls, "_rigid_solver", rigid_solver)
    monkeypatch.setattr(manager_cls, "_soft_solver", soft_solver)
    monkeypatch.setattr(manager_cls, "_apply_reactions", reactions)

    getattr(manager_cls, f"_step_{mode}")(state_in, state_out, control, 0.01)

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


def test_featherstone_rigid_step_isolates_particles(monkeypatch: pytest.MonkeyPatch) -> None:
    model = MagicMock()
    model.particle_count = 3
    rigid_solver = MagicMock()
    state_in = MagicMock()
    input_particle_f = object()
    state_in.particle_f = input_particle_f
    state_out = MagicMock()
    scratch_particle_f = object()
    state_out.particle_f = scratch_particle_f

    def check_isolation(*_args) -> None:
        assert model.particle_count == 0
        assert state_in.particle_f is scratch_particle_f

    rigid_solver.step.side_effect = check_isolation
    monkeypatch.setattr(LegacyFeatherstoneManager, "_model", model)
    monkeypatch.setattr(LegacyFeatherstoneManager, "_rigid_solver", rigid_solver)

    LegacyFeatherstoneManager._rigid_step(state_in, state_out, object(), 0.01)

    assert model.particle_count == 3
    assert state_in.particle_f is input_particle_f


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


@pytest.mark.parametrize("manager_cls", [LegacyMJWarpManager, LegacyFeatherstoneManager])
def test_legacy_solver_specific_clear_releases_subsolvers(manager_cls: type, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(manager_cls, "_rigid_solver", object())
    monkeypatch.setattr(manager_cls, "_soft_solver", object())
    monkeypatch.setattr(manager_cls, "_coupling_mode", "two_way")

    manager_cls._solver_specific_clear()

    assert manager_cls._rigid_solver is None
    assert manager_cls._soft_solver is None
    assert manager_cls._coupling_mode is None
