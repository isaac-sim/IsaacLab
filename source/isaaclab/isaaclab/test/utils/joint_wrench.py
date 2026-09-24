# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared physical contract for joint-wrench sensor backends."""

from pathlib import Path

import torch

from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import JointWrenchSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path


def check_joint_wrench_frame(physics: PhysicsCfg, tmp_path: Path, device: str | None = None) -> None:
    """Compare a loaded joint's wrench with gravity equilibrium using any physics backend.

    Args:
        physics: Physics backend configuration.
        tmp_path: Directory for the generated USD file.
        device: Simulation device. Defaults to the :class:`~isaaclab.sim.SimulationCfg` default.
    """
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

    sim_cfg = SimulationCfg(dt=1.0 / 200.0, physics=physics)
    with sim_utils.build_simulation_context(device=device, sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
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
