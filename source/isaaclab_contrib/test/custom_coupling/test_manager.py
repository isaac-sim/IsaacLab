# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg
from isaaclab_newton.physics import newton_backend as nb
from newton import ModelBuilder

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import (
    CoupledMJWarpVBDSolver,
    CoupledMJWarpVBDSolverAdapter,
)
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg


def _stub_subsolvers(monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
    """Replace the MJWarp and VBD sub-solver managers with mocks."""
    monkeypatch.setattr(solver_cfg.rigid_solver_cfg, "class_type", MagicMock())
    monkeypatch.setattr(solver_cfg.soft_solver_cfg, "class_type", MagicMock())


def _create(
    monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg | None = None
) -> CoupledMJWarpVBDSolver:
    """Construct the coupled solver over mocked sub-solvers."""
    solver_cfg = CoupledMJWarpVBDSolverCfg() if solver_cfg is None else solver_cfg
    _stub_subsolvers(monkeypatch, solver_cfg)
    return CoupledMJWarpVBDSolverAdapter.create_solver(MagicMock(), solver_cfg)


def test_registered_mujoco_solver_imports_mujoco_joint_properties() -> None:
    """The coupled manager imports joint properties consumed by its MuJoCo solver."""
    stage = Usd.Stage.CreateInMemory()
    root_path = "/World/robot"
    root = UsdGeom.Cube.Define(stage, root_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    child_path = f"{root_path}/child"
    child = UsdGeom.Cube.Define(stage, child_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(child)
    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{child_path}/joint")
    joint.CreateAxisAttr().Set("Z")
    joint.CreateBody0Rel().SetTargets([root_path])
    joint.CreateBody1Rel().SetTargets([child_path])
    joint.GetPrim().CreateAttribute("mjc:frictionloss", Sdf.ValueTypeNames.Double, True).Set(0.11)
    joint.GetPrim().CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Double, True).Set(0.23)

    builder = ModelBuilder()
    CoupledMJWarpVBDSolverAdapter.register_builder_attributes(builder, CoupledMJWarpVBDSolverCfg())
    builder.add_usd(
        stage,
        schema_resolvers=CoupledMJWarpVBDSolverAdapter.get_usd_import_schema_resolvers(CoupledMJWarpVBDSolverCfg()),
    )
    model = builder.finalize(device="cpu")

    assert model.joint_friction.numpy()[-1] == pytest.approx(0.11)
    assert model.joint_damping.numpy()[-1] == pytest.approx(0.23)


def test_reset_forwards_to_both_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The manager's reset reaches both sub-solvers."""
    solver = _create(monkeypatch)
    rigid_solver = solver.rigid_solver
    rigid_solver.use_mujoco_cpu = False
    soft_solver = solver.soft_solver
    state = object()
    world_mask = object()

    CoupledMJWarpVBDSolverAdapter.reset_solver(SimpleNamespace(solver=solver), state, world_mask)

    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)


def test_reset_skips_all_false_cpu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU warm-start state when no world needs reset."""
    solver = _create(monkeypatch)
    rigid_solver = solver.rigid_solver
    rigid_solver.use_mujoco_cpu = True
    soft_solver = solver.soft_solver
    world_mask = MagicMock()
    world_mask.numpy.return_value.any.return_value = False

    solver.reset(object(), world_mask=world_mask)

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
        CoupledMJWarpVBDSolverAdapter.create_solver(MagicMock(), solver_cfg)


def test_add_contact_sensor_rejects_coupled_solver() -> None:
    backend = SimpleNamespace(manager=CoupledMJWarpVBDSolverAdapter, contact_sensors={})

    with pytest.raises(NotImplementedError, match="contact sensors are not yet supported"):
        nb.add_contact_sensor(backend, body_names_expr="body")
    assert backend.contact_sensors == {}


def test_build_solver_sets_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    solver = _create(monkeypatch, solver_cfg)
    backend = SimpleNamespace(solver=solver)
    manager = CoupledMJWarpVBDSolverAdapter

    assert solver.rigid_solver is solver_cfg.rigid_solver_cfg.class_type.create_solver.return_value
    assert solver.soft_solver is solver_cfg.soft_solver_cfg.class_type.create_solver.return_value
    assert manager.single_state is False
    assert manager.uses_collision_pipeline(backend) is True
    assert manager.supports_contact_sensors is False
    assert manager.supports_body_forces(backend) is True
    assert manager.prepares_step is True


@pytest.mark.parametrize("mode", ["one_way", "two_way"])
def test_step_preserves_input_forces(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    state_in = MagicMock()
    state_in.body_f = object()
    state_in.particle_f = MagicMock()
    state_out = MagicMock()
    control = object()
    contacts = object()
    collision_pipeline = MagicMock()
    reactions = MagicMock()
    solver = _create(monkeypatch, CoupledMJWarpVBDSolverCfg(coupling_mode=mode))
    rigid_solver = solver.rigid_solver
    soft_solver = solver.soft_solver
    backend = SimpleNamespace(solver=solver, contacts=contacts, collision_pipeline=collision_pipeline)
    CoupledMJWarpVBDSolverAdapter.prepare_contacts(backend)
    monkeypatch.setattr(solver, "_apply_reactions", reactions)

    solver.step(state_in, state_out, control, contacts, 0.01)

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
