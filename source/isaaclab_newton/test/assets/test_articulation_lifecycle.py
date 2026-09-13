# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Kit-less native articulation lifecycle regressions with local USD fixtures."""

import numpy as np
import pytest
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.utils import configclass


@pytest.mark.parametrize("device, use_cuda_graph", [("cpu", False), ("cuda:0", True)])
def test_native_actuators_rebind_after_repeated_hard_model_reset(tmp_path, device, use_cuda_graph):
    """Hard reset preserves folded stepping and replaces native actuator owners and hooks."""
    from isaaclab_newton.physics import FeatherstoneSolverCfg, NewtonCfg

    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(tmp_path / "native_joint.usda"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    body = UsdGeom.Sphere.Define(stage, "/Robot/link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(1.0)
    mass.CreateDiagonalInertiaAttr((0.4, 0.4, 0.4))
    base = UsdGeom.Sphere.Define(stage, "/Robot/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.ArticulationRootAPI.Apply(base)
    base_mass = UsdPhysics.MassAPI.Apply(base)
    base_mass.CreateMassAttr(1.0)
    base_mass.CreateDiagonalInertiaAttr((0.4, 0.4, 0.4))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/fixed")
    fixed.CreateBody1Rel().SetTargets(["/Robot/base"])
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint")
    joint.CreateBody0Rel().SetTargets(["/Robot/base"])
    joint.CreateBody1Rel().SetTargets(["/Robot/link"])
    joint.CreateAxisAttr().Set("Z")
    stage.GetRootLayer().Save()
    cfg = ArticulationCfg(
        prim_path="/World/Robot",
        articulation_root_prim_path="/base",
        spawn=sim_utils.UsdFileCfg(usd_path=str(tmp_path / "native_joint.usda")),
        actuators={"joint": IdealPDActuatorCfg(joint_names_expr=["joint"], stiffness=0.0, damping=0.0)},
    )

    @configclass
    class SceneCfg(InteractiveSceneCfg):
        robot = cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")

    sim_cfg = SimulationCfg(
        dt=0.005,
        device=device,
        gravity=(0.0, 0.0, 0.0),
        use_newton_actuators=True,
        physics=NewtonCfg(solver_cfg=FeatherstoneSolverCfg(), num_substeps=5, use_cuda_graph=use_cuda_graph),
    )
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=3.0))
        articulation = scene["robot"]
        sim.reset()
        retired_telemetry = []
        for reset_index in range(3):
            if reset_index:
                old_model = sim.physics_manager.get_model()
                retired_telemetry.append(articulation.actuators.computed_effort.warp)
                sim.reset(soft=False)
                assert sim.physics_manager.get_model() is not old_model
                for telemetry in retired_telemetry:
                    telemetry.fill_(123.0)
            assert sim.physics_manager.handles_decimation()
            sim.physics_manager.set_decimation(4)
            articulation.actuators.target_command.set_effort_index(value=torch.full((1, 1), 2.0, device=device))
            articulation.write_data_to_sim()
            sim.step(render=False)
            # Constant 2 N m for 20 ms around a 0.4 kg m^2 axis gives 0.1 rad/s.
            np.testing.assert_allclose(sim.physics_manager.get_state_0().joint_qd.numpy(), [0.1], atol=1e-6)
            np.testing.assert_allclose(articulation.actuators.applied_effort.warp.numpy(), [[2.0]], atol=1e-6)
            for telemetry in retired_telemetry:
                np.testing.assert_array_equal(telemetry.numpy(), [[123.0]])
