# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


"""Kit-less native articulation lifecycle regressions with local USD fixtures."""

import gc
import weakref

import numpy as np
import pytest
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.physics import PhysicsEvent
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
        marker = wp.zeros(1, device=device)
        sim.physics_manager.register_callback(
            lambda _: sim.physics_manager.register_post_step_callback(lambda: marker.fill_(1.0)),
            PhysicsEvent.PHYSICS_READY,
        )
        sim.reset()
        sim.physics_manager.set_decimation(4)
        retired_telemetry = []
        retired_collections = []

        def check_retired_collections(_event):
            gc.collect()
            assert all(collection() is None for collection in retired_collections)

        sim.physics_manager.register_callback(check_retired_collections, PhysicsEvent.MODEL_INIT, order=100)
        for reset_index in range(3):
            if reset_index:
                old_model = sim.physics_manager.get_model()
                retired_telemetry.append(articulation.actuators.computed_effort.warp)
                retired_collections.append(weakref.ref(articulation.actuators))
                sim.reset(soft=False)
                assert sim.physics_manager.get_model() is not old_model
                for telemetry in retired_telemetry:
                    telemetry.fill_(123.0)
            assert sim.physics_manager.handles_decimation()
            articulation.actuators.target_command.set_effort_index(value=torch.full((1, 1), 2.0, device=device))
            articulation.write_data_to_sim()
            marker.zero_()
            sim.step(render=False)
            assert (sim.physics_manager._graph is not None) is use_cuda_graph
            np.testing.assert_array_equal(marker.numpy(), [1.0])
            # Constant 2 N m for 20 ms around a 0.4 kg m^2 axis gives 0.1 rad/s.
            np.testing.assert_allclose(sim.physics_manager.get_state_0().joint_qd.numpy(), [0.1], atol=1e-6)
            np.testing.assert_allclose(articulation.actuators.applied_effort.warp.numpy(), [[2.0]], atol=1e-6)
            for telemetry in retired_telemetry:
                np.testing.assert_array_equal(telemetry.numpy(), [[123.0]])
        final_collection = weakref.ref(articulation.actuators)
    gc.collect()
    assert final_collection() is None
