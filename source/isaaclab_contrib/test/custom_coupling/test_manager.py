# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the custom coupling manager."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, VBDSolverCfg
from isaaclab_newton.physics.runtime import NewtonRuntime
from newton import ModelBuilder

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager as manager_module
from isaaclab_contrib.custom_coupling.coupled_mjwarp_vbd_manager import (
    CoupledMJWarpVBDSolverBinding,
    NewtonCoupledMJWarpVBDManager,
)
from isaaclab_contrib.custom_coupling.newton_manager_cfg import CoupledMJWarpVBDSolverCfg


def _stub_subsolvers(monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg) -> None:
    """Replace the MJWarp and VBD solvers, and the placeholder solver, with mocks."""
    monkeypatch.setattr(solver_cfg.rigid_solver_cfg, "class_type", MagicMock())
    monkeypatch.setattr(solver_cfg.soft_solver_cfg, "class_type", MagicMock())
    monkeypatch.setattr(manager_module, "SolverBase", MagicMock())


def _bind(
    monkeypatch: pytest.MonkeyPatch, solver_cfg: CoupledMJWarpVBDSolverCfg | None = None
) -> CoupledMJWarpVBDSolverBinding:
    """Construct the coupled binding over mocked sub-solvers."""
    solver_cfg = CoupledMJWarpVBDSolverCfg() if solver_cfg is None else solver_cfg
    _stub_subsolvers(monkeypatch, solver_cfg)
    return CoupledMJWarpVBDSolverBinding(MagicMock(), solver_cfg, wp.DeterministicMode.NOT_GUARANTEED)


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
    NewtonCoupledMJWarpVBDManager.solver_binding.register_builder_attributes(builder)
    builder.add_usd(stage, schema_resolvers=NewtonCoupledMJWarpVBDManager.get_usd_import_schema_resolvers())
    model = builder.finalize(device="cpu")

    assert model.joint_friction.numpy()[-1] == pytest.approx(0.11)
    assert model.joint_damping.numpy()[-1] == pytest.approx(0.23)


def test_reset_forwards_to_both_subsolvers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the real sub-solvers instead of the dummy solver slot."""
    binding = _bind(monkeypatch)
    rigid_solver = binding.rigid_solver
    rigid_solver.use_mujoco_cpu = False
    soft_solver = binding.soft_solver
    state = object()
    world_mask = object()

    binding.reset(state, world_mask)

    binding.solver.reset.assert_not_called()
    rigid_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)
    soft_solver.reset.assert_called_once_with(state, world_mask=world_mask, flags=0)


def test_reset_skips_all_false_cpu_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU warm-start state when no world needs reset."""
    binding = _bind(monkeypatch)
    rigid_solver = binding.rigid_solver
    rigid_solver.use_mujoco_cpu = True
    soft_solver = binding.soft_solver
    world_mask = MagicMock()
    world_mask.numpy.return_value.any.return_value = False

    binding.reset(object(), world_mask)

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
        CoupledMJWarpVBDSolverBinding(MagicMock(), solver_cfg, wp.DeterministicMode.NOT_GUARANTEED)


def test_build_solver_rejects_contact_sensors(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    _stub_subsolvers(monkeypatch, solver_cfg)
    runtime = NewtonRuntime(
        SimpleNamespace(model=SimpleNamespace(world_count=1, articulation_count=0)), SimpleNamespace(device="cpu")
    )
    runtime.sensors.contact[("body", None, None, None)] = object()

    with pytest.raises(NotImplementedError, match="contact sensors are not yet supported"):
        runtime.bind_solver(
            NewtonCoupledMJWarpVBDManager.solver_binding,
            NewtonCfg(solver_cfg=solver_cfg),
            wp.DeterministicMode.NOT_GUARANTEED,
        )


def test_build_solver_sets_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    solver_cfg = CoupledMJWarpVBDSolverCfg()
    binding = _bind(monkeypatch, solver_cfg)

    assert binding.solver is manager_module.SolverBase.return_value
    assert binding.rigid_solver is solver_cfg.rigid_solver_cfg.class_type.solver_binding.create.return_value
    assert binding.soft_solver is solver_cfg.soft_solver_cfg.class_type.solver_binding.create.return_value
    assert binding.single_state is False
    assert binding.needs_collision_pipeline is True
    assert binding.supports_contact_sensors is False
    assert binding.supports_body_forces is True


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
    binding = _bind(monkeypatch, CoupledMJWarpVBDSolverCfg(coupling_mode=mode))
    rigid_solver = binding.rigid_solver
    soft_solver = binding.soft_solver
    binding.prepare_contacts(contacts, collision_pipeline)
    monkeypatch.setattr(binding, "_apply_reactions", reactions)

    binding.step(state_in, state_out, control, contacts, 0.01)

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
