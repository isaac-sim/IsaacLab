# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local real-PhysX integration coverage for surface grippers."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True, device="cpu").app

import pytest
import torch
import warp as wp
from isaaclab_physx.assets import SurfaceGripper, SurfaceGripperCfg

from isaaclab.sim.utils import enable_extension

enable_extension("isaacsim.robot.surface_gripper")

from usd.schema.isaac import robot_schema

from isaacsim.robot.surface_gripper import create_surface_gripper
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import build_simulation_context

pytestmark = pytest.mark.integration


def _create_rigid_cube(path: str, position: tuple[float, float, float]) -> None:
    """Author one local rigid collision cube for the gripper attachment."""
    stage = sim_utils.get_current_stage()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(0.1)
    cube.AddTranslateOp().Set(Gf.Vec3f(*position))
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)


def _author_local_surface_gripper() -> SurfaceGripper:
    """Build the minimal schema, rigid bodies, and attachment point locally."""
    stage = sim_utils.get_current_stage()
    env_path = "/World/Env_0"
    UsdGeom.Xform.Define(stage, env_path)
    _create_rigid_cube(f"{env_path}/box0", (0.0, 0.0, 0.05))
    _create_rigid_cube(f"{env_path}/box1", (0.0, 0.0, 0.15))
    create_surface_gripper(stage, env_path)

    gripper_prim = stage.GetPrimAtPath(f"{env_path}/SurfaceGripper")
    gripper_prim.GetAttribute(robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name).Set(100.0)
    gripper_prim.GetAttribute(robot_schema.Attributes.SHEAR_FORCE_LIMIT.name).Set(100.0)
    gripper_prim.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.1)

    joint_path = Sdf.Path(f"{env_path}/box1/attachment")
    joint = UsdPhysics.Joint.Define(stage, joint_path)
    robot_schema.ApplyAttachmentPointAPI(joint.GetPrim())
    joint.GetPrim().CreateAttribute(
        robot_schema.Attributes.FORWARD_AXIS.name, robot_schema.Attributes.FORWARD_AXIS.type
    ).Set(UsdPhysics.Tokens.x)
    joint.GetPrim().CreateAttribute(
        robot_schema.Attributes.CLEARANCE_OFFSET.name, robot_schema.Attributes.CLEARANCE_OFFSET.type
    ).Set(0.0)
    for limit in ["rotX", "rotY", "rotZ", "transX", "transY", "transZ"]:
        limit_api = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), limit)
        limit_api.CreateHighAttr().Set(-1.0)
        limit_api.CreateLowAttr().Set(1.0)
    joint.CreateBody0Rel().SetTargets([f"{env_path}/box1"])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, -0.0499))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(0.5, -0.5, 0.5, 0.5))
    gripper_prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([joint_path])

    return SurfaceGripper(
        SurfaceGripperCfg(
            prim_path="/World/Env_[^/]*/SurfaceGripper",
            max_grip_distance=0.1,
            coaxial_force_limit=100.0,
            shear_force_limit=100.0,
            retry_interval=0.1,
        )
    )


def test_initialization_and_open_close_commands() -> None:
    """Initialize the local view and prove close/open commands reach the real plugin."""
    with build_simulation_context(device="cpu", gravity_enabled=False) as sim:
        gripper = _author_local_surface_gripper()
        sim.reset()

        assert gripper.is_initialized
        assert gripper.command.shape == (1,)
        assert gripper.state.shape == (1,)
        assert wp.to_torch(gripper.command).item() == 0.0
        assert wp.to_torch(gripper.state).item() == -1.0

        gripper.set_grippers_command_index(wp.array([1.0], dtype=wp.float32, device="cpu"))
        gripper.write_data_to_sim()
        for _ in range(3):
            sim.step()
            gripper.update(sim.cfg.dt)
        assert torch.all(wp.to_torch(gripper.state) >= 0.0)

        gripper.set_grippers_command_index(wp.array([-1.0], dtype=wp.float32, device="cpu"))
        gripper.write_data_to_sim()
        for _ in range(3):
            sim.step()
            gripper.update(sim.cfg.dt)
        assert torch.all(wp.to_torch(gripper.state) == -1.0)
