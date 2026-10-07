# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kitless real-Newton coverage of an articulation composed into two scenes of different content.

Both scenes hold the same arm; the first also holds a free cube. Folded with :func:`isaaclab.scene.add`, a shared arm
spans worlds whose joint rows are not regularly spaced, which Newton assets reject; per-scene arms are regular.
"""

from isaaclab_newton.physics import NewtonCfg

from isaaclab.sim import SimulationCfg
from isaaclab.test.utils import launch_test_simulation

launch_test_simulation(SimulationCfg(physics=NewtonCfg()))

from pathlib import Path

import numpy as np
import pytest
import torch
from isaaclab_newton.physics import FeatherPGSSolverCfg, NewtonManager
from newton_test_utils import newton_sim_cfg

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.scene import add as scene_add
from isaaclab.sim import build_simulation_context

pytestmark = [pytest.mark.integration, pytest.mark.kitless]

NUM_ENVS = 4


def _write_arm(path: Path) -> str:
    """Author a fixed-base arm with one revolute joint and return the file path."""
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/Arm").GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name, z in (("base", 0.1), ("link", 0.4)):
        body = UsdGeom.Cube.Define(stage, f"/Arm/{name}")
        body.CreateSizeAttr(0.1)
        body.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Arm/root_joint")
    fixed.CreateBody1Rel().SetTargets(["/Arm/base"])
    hinge = UsdPhysics.RevoluteJoint.Define(stage, "/Arm/hinge")
    hinge.CreateBody0Rel().SetTargets(["/Arm/base"])
    hinge.CreateBody1Rel().SetTargets(["/Arm/link"])
    hinge.CreateAxisAttr("Y")
    hinge.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.15))
    hinge.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, -0.15))
    stage.SetDefaultPrim(root)
    stage.Save()
    return str(path)


def _scene_cfg(arm_usd: str, share_equal_assets: bool) -> InteractiveSceneCfg:
    """Fold a scene of arm and cube with a scene of the same arm only."""

    def arm() -> ArticulationCfg:
        return ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Arm",
            spawn=sim_utils.UsdFileCfg(usd_path=arm_usd),
            actuators={"hinge": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=10.0, damping=1.0)},
        )

    with_cube = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=2.0)
    with_cube.arm = arm()
    with_cube.cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    arm_only = InteractiveSceneCfg(num_envs=NUM_ENVS, env_spacing=2.0)
    arm_only.arm = arm()
    return scene_add(with_cube, arm_only, share_equal_assets=share_equal_assets)


def test_shared_arm_is_rejected(tmp_path: Path):
    """An arm shared by both scenes spans irregularly spaced joint rows; initialization names the remedy."""
    scene_cfg = _scene_cfg(_write_arm(tmp_path / "arm.usda"), share_equal_assets=True)
    assert [name for name in ("arm", "arm_1") if getattr(scene_cfg, name, None) is not None] == ["arm"]
    sim_cfg = newton_sim_cfg("cpu", solver_cfg=FeatherPGSSolverCfg(pgs_mode="split"))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        with pytest.raises(ValueError, match="share_equal_assets=False"):
            sim.reset()
        assert not scene["arm"].is_initialized


def test_per_scene_arms_write_their_own_worlds(tmp_path: Path):
    """Each scene's arm covers only its worlds, and its joint writes reach exactly those worlds' rows."""
    scene_cfg = _scene_cfg(_write_arm(tmp_path / "arm.usda"), share_equal_assets=False)
    sim_cfg = newton_sim_cfg("cpu", solver_cfg=FeatherPGSSolverCfg(pgs_mode="split"))
    with build_simulation_context(sim_cfg=sim_cfg) as sim:
        scene = InteractiveScene(scene_cfg)
        sim.reset()
        model = NewtonManager.get_model()
        arms = (scene["arm"], scene["arm_1"])
        worlds = [arm.root_view.world_ids.numpy().tolist() for arm in arms]
        assert sorted(worlds[0] + worlds[1]) == list(range(NUM_ENVS)) and all(len(w) == NUM_ENVS // 2 for w in worlds)

        for offset, arm in zip((1.0, 3.0), arms):
            position = torch.tensor([[offset], [offset + 1.0]])
            arm.write_joint_position_to_sim_index(position=position, env_ids=torch.arange(2))
            arm.set_joint_position_target_index(target=-position, env_ids=torch.arange(2))
        scene.write_data_to_sim()

        joint_q = NewtonManager.get_state_0().joint_q.numpy()
        target_q = NewtonManager.get_control().joint_target_q.numpy()
        qd_start = model.joint_q_start.numpy()
        hinge_joints = [index for index, label in enumerate(model.joint_label) if label.endswith("/hinge")]
        hinge_world = model.joint_world.numpy()[hinge_joints]
        expected = {}
        for offset, world_ids in zip((1.0, 3.0), worlds):
            for row, world in enumerate(world_ids):
                expected[world] = offset + row
        for joint, world in zip(hinge_joints, hinge_world):
            np.testing.assert_allclose(joint_q[qd_start[joint]], expected[int(world)], rtol=1e-6)
            np.testing.assert_allclose(target_q[qd_start[joint]], -expected[int(world)], rtol=1e-6)
