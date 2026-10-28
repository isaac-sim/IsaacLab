# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupled solver."""

from unittest.mock import MagicMock

import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg
from isaaclab_newton.physics.newton_builder import register_builder_attributes, usd_schema_resolvers
from newton import ModelBuilder

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_solver import (
    CoupledMJWarpVBDSolver,
    SolverCoupledMJWarpVBD,
)
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_solver_cfg import CoupledMJWarpVBDSolverCfg

NOT_GUARANTEED = wp.DeterministicMode.NOT_GUARANTEED


def _stub_subsolvers(monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
    """Make the MJWarp and VBD sub-solver configurations build mocks."""
    monkeypatch.setattr(solver_cfg.rigid_solver_cfg, "class_type", MagicMock())
    monkeypatch.setattr(solver_cfg.soft_solver_cfg, "class_type", MagicMock())


def _create(
    monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg | None = None
) -> SolverCoupledMJWarpVBD:
    """Construct the coupled solver over mocked sub-solvers."""
    solver_cfg = CoupledMJWarpVBDSolverCfg() if solver_cfg is None else solver_cfg
    _stub_subsolvers(monkeypatch, solver_cfg)
    return CoupledMJWarpVBDSolver.build(solver_cfg, MagicMock(), NOT_GUARANTEED)


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
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    register_builder_attributes(builder, solver_cfg)
    builder.add_usd(stage, schema_resolvers=usd_schema_resolvers(solver_cfg))
    model = builder.finalize(device="cpu")

    assert model.joint_friction.numpy()[-1] == pytest.approx(0.11)
    assert model.joint_damping.numpy()[-1] == pytest.approx(0.23)


def test_reset_forwards_to_both_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """The coupled solver's reset reaches both sub-solvers."""
    solver = _create(monkeypatch)
    rigid_solver = solver.rigid_solver
    rigid_solver.use_mujoco_cpu = False
    soft_solver = solver.soft_solver
    state = object()
    world_mask = object()

    solver.reset(state, world_mask=world_mask)

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
        CoupledMJWarpVBDSolver.build(solver_cfg, MagicMock(), NOT_GUARANTEED)


def test_build_solver_builds_each_sub_solver(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    solver = _create(monkeypatch, solver_cfg)

    assert solver.rigid_solver is solver_cfg.rigid_solver_cfg.class_type.build.return_value
    assert solver.soft_solver is solver_cfg.soft_solver_cfg.class_type.build.return_value


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
    solver_cfg = CoupledMJWarpVBDSolverCfg(coupling_mode=mode)
    solver = _create(monkeypatch, solver_cfg)
    rigid_solver = solver.rigid_solver
    soft_solver = solver.soft_solver
    CoupledMJWarpVBDSolver.prepare_contacts(solver_cfg, solver, contacts, collision_pipeline)
    monkeypatch.setattr(solver, "_apply_reactions", reactions)

    solver.step(state_in, state_out, control, contacts, 0.01)

    state_in.clear_forces.assert_not_called()
    state_in.particle_f.zero_.assert_not_called()
    state_out.clear_forces.assert_called_once_with()
    collision_pipeline.collide.assert_called_once_with(state_in, contacts)
    rigid_solver.step.assert_called_once_with(state_in, state_out, control, None, 0.01)
    soft_solver.step.assert_called_once_with(state_in, state_out, control, contacts, 0.01)
    if mode == "two_way":
        reactions.assert_called_once_with(state_in, state_out, contacts, 0.01)
    else:
        reactions.assert_not_called()
