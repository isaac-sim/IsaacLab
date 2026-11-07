# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for the deprecated coupled-solver configuration."""

import pytest
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, VBDSolverCfg
from isaaclab_newton.physics.coupling import CouplerProxyCfg, CouplerSolver
from isaaclab_newton.physics.newton_builder import register_builder_attributes, usd_schema_resolvers
from newton import ModelBuilder

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_contrib.custom_coupling import CoupledMJWarpVBDSolver, CoupledMJWarpVBDSolverCfg

pytestmark = pytest.mark.kitless


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


@pytest.mark.parametrize(("mode", "relaxation"), [("one_way", 0.0), ("two_way", 1.0)])
def test_legacy_config_translates_to_proxy_coupling(mode, relaxation, monkeypatch):
    """Legacy fields keep their settings and select feedback without mutating caller-owned configs."""
    rigid = MJWarpSolverCfg(iterations=17)
    soft = VBDSolverCfg(iterations=7, integrate_with_external_rigid_solver=True)
    with pytest.warns(FutureWarning, match="proxy feedback"):
        cfg = CoupledMJWarpVBDSolverCfg(rigid_solver_cfg=rigid, soft_solver_cfg=soft, coupling_mode=mode)
    # Native construction is covered by the core coupler runtime tests; this boundary owns legacy translation.
    monkeypatch.setattr(CouplerSolver, "build", lambda *args: None)
    driver = CoupledMJWarpVBDSolver(cfg, None, wp.DeterministicMode.NOT_GUARANTEED)
    proxy_cfg = driver.cfg

    assert isinstance(proxy_cfg, CouplerProxyCfg)
    rigid_entry, soft_entry = proxy_cfg.entries
    assert rigid_entry.solver_cfg.iterations == 17
    assert rigid_entry.bodies == [".*"] and rigid_entry.include_static_shapes
    assert soft_entry.solver_cfg.iterations == 7
    assert soft_entry.solver_cfg.integrate_with_external_rigid_solver is False
    assert soft_entry.all_particles and not soft_entry.include_static_shapes
    assert soft.integrate_with_external_rigid_solver is True
    assert cfg.soft_solver_cfg.integrate_with_external_rigid_solver is True
    (mapping,) = proxy_cfg.proxies
    assert (mapping.source, mapping.destination) == (rigid_entry.name, soft_entry.name)
    assert mapping.bodies == [".*"]
    assert mapping.collision_pipeline is None
    assert mapping.proxy_relaxation == relaxation
    assert mapping.proxy_relaxation_mode == "fixed"


def test_legacy_config_rejects_invalid_mode():
    with pytest.warns(FutureWarning):
        cfg = CoupledMJWarpVBDSolverCfg(coupling_mode="invalid")
    with pytest.raises(ValueError, match="coupling_mode"):
        CoupledMJWarpVBDSolver(cfg, None, wp.DeterministicMode.NOT_GUARANTEED)
