# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Joint-wrench contract tests imported by every backend's sensor test module.

Backend modules supply the ``sim`` fixture; this test owns the physical scene and oracle.
"""

from pathlib import Path

import pytest
import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path


@pytest.mark.integration
def test_joint_wrench_frame(sim, tmp_path: Path) -> None:
    """A rotated, offset joint reports the analytic gravity reaction at its anchor."""
    source = retrieve_file_path(f"{ISAAC_NUCLEUS_DIR}/Robots/IsaacSim/SimpleArticulation/revolute_articulation.usd")
    usd_path = str(tmp_path / "joint_wrench.usda")
    stage = Usd.Stage.CreateNew(usd_path)
    root = stage.DefinePrim("/Articulation", "Xform")
    stage.SetDefaultPrim(root)
    root.GetReferences().AddReference(source)
    joint = next(UsdPhysics.Joint(prim) for prim in stage.Traverse() if prim.IsA(UsdPhysics.RevoluteJoint))
    UsdPhysics.RevoluteJoint(joint).GetAxisAttr().Set("Z")
    joint.GetLocalPos1Attr().Set(Gf.Vec3f(0.25, -0.15, 0.1))
    joint.GetLocalRot1Attr().Set(Gf.Quatf(2.0**-0.5, Gf.Vec3f(2.0**-0.5, 0.0, 0.0)))
    # Unit scale makes the authored joint offset a metric offset. Align the joint frames initially.
    arm_prim = stage.GetPrimAtPath(joint.GetBody1Rel().GetTargets()[0])
    pose = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), -90.0))
    pose.SetTranslateOnly(Gf.Vec3d(-0.25, -0.1, -0.15))
    UsdGeom.Xformable(arm_prim).MakeMatrixXform().Set(pose)
    mass = UsdPhysics.MassAPI.Apply(arm_prim)
    mass.CreateMassAttr(2.0)
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0))
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
    stage.GetRootLayer().Save()

    cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(0.0, 0.0, 2.0**-0.5, 2.0**-0.5)),
    )
    cfg.wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(cfg)
    sim.reset()
    for _ in range(20):
        sim.step()
        scene.update(sim.get_physics_dt())

    robot, sensor = scene["robot"], scene["wrench"]
    arm = robot.body_names.index("Arm")
    sensor_arm = sensor.find_bodies("Arm")[0][0]
    assert robot.data.body_com_vel_w.torch[:, arm].norm() < 1e-3
    # The joint frame is rotated 90 degrees about world Z, so gravity still points along its -Z.
    # Its 2 kg load is offset by (-0.25, -0.1, -0.15) m in joint coordinates.
    # Reaction force is (0, 0, mg), and r x F gives torque (-0.1 mg, 0.25 mg, 0).
    weight = -2.0 * sim.cfg.gravity[2]
    expected_force = torch.tensor([[0.0, 0.0, weight]], device=sim.device)
    expected_torque = torch.tensor([[-0.1 * weight, 0.25 * weight, 0.0]], device=sim.device)
    torch.testing.assert_close(sensor.data.force.torch[:, sensor_arm], expected_force, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(sensor.data.torque.torch[:, sensor_arm], expected_torque, atol=1e-2, rtol=1e-3)


@pytest.mark.integration
def test_joint_wrench_body_ordering(sim, tmp_path: Path) -> None:
    """Sensor names and distinct gravity loads follow the owning articulation's public order."""
    usd_path = str(tmp_path / "ordered_wrenches.usda")
    stage = Usd.Stage.CreateNew(usd_path)
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    stage.SetDefaultPrim(root)
    UsdPhysics.ArticulationRootAPI.Apply(root)
    for name, mass in (("base", 1.0), ("left", 2.0), ("right", 3.0)):
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass_api = UsdPhysics.MassAPI.Apply(body)
        mass_api.CreateMassAttr(mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.1, 0.0, 0.0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0))
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/root_joint")
    fixed.CreateBody1Rel().SetTargets(["/Robot/base"])
    for name in ("left", "right"):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"/Robot/{name}_joint")
        joint.CreateBody0Rel().SetTargets(["/Robot/base"])
        joint.CreateBody1Rel().SetTargets([f"/Robot/{name}"])
        joint.CreateAxisAttr("X" if name == "left" else "Z")
        if name == "left":
            rotation = Gf.Quatf(2.0**-0.5, Gf.Vec3f(2.0**-0.5, 0.0, 0.0))
            joint.CreateLocalRot0Attr(rotation)
            joint.CreateLocalRot1Attr(rotation)
    stage.GetRootLayer().Save()

    cfg = InteractiveSceneCfg(num_envs=2, env_spacing=2.0)
    cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=usd_path),
        body_ordering=("base", "right", "left"),
        actuators={"joints": ImplicitActuatorCfg(joint_names_expr=[".*"], stiffness=0.0, damping=0.0)},
    )
    # A different prefix verifies association by articulation root rather than matching config paths.
    cfg.wrench = JointWrenchSensorCfg(prim_path="{ENV_REGEX_NS}")
    scene = InteractiveScene(cfg)
    sim.reset()
    robot, sensor = scene["robot"], scene["wrench"]
    assert sensor.body_names == [name for name in robot.body_names if name in sensor.body_names]
    assert sensor.find_bodies(["left", "right"])[1] == ["right", "left"]
    for _ in range(5):
        sim.step()
        scene.update(sim.get_physics_dt())
    for name, force_direction, torque_per_weight, mass in (
        ("left", (0.0, 1.0, 0.0), (0.0, 0.0, 0.1), 2.0),
        ("right", (0.0, 0.0, 1.0), (0.0, -0.1, 0.0), 3.0),
    ):
        index = sensor.find_bodies(name)[0][0]
        weight = -mass * sim.cfg.gravity[2]
        expected_force = (weight * torch.tensor([force_direction], device=sim.device)).expand(2, -1)
        expected_torque = (weight * torch.tensor([torque_per_weight], device=sim.device)).expand(2, -1)
        torch.testing.assert_close(sensor.data.force.torch[:, index], expected_force, atol=1e-2, rtol=1e-3)
        torch.testing.assert_close(sensor.data.torque.torch[:, index], expected_torque, atol=1e-2, rtol=1e-3)
