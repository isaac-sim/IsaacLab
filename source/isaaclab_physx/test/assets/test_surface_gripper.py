# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real PhysX surface-gripper coverage on a locally authored gripper."""

import os

from isaaclab.test.utils import DeviceScope, launch_test_simulation, test_devices

launch_test_simulation(physics="isaacsim_physx")

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.assets import SurfaceGripper, SurfaceGripperCfg

from isaaclab.sim.utils import enable_extension

enable_extension("isaacsim.robot.surface_gripper")

from usd.schema.isaac import robot_schema

from isaacsim.robot.surface_gripper import create_surface_gripper
from pxr import Gf, Sdf, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, build_simulation_context

_RUNNING_CI = bool(
    os.environ.get("CI") == "true" or os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("GITLAB_CI")
)


def _create_rigid_cube(path: str, position: tuple[float, float, float]) -> None:
    """Author one local 1 kg rigid collision cube."""
    stage = sim_utils.get_current_stage()
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(0.1)
    cube.AddTranslateOp().Set(Gf.Vec3f(*position))
    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(1.0)


def _author_surface_gripper() -> SurfaceGripper:
    """Author a gripper on the upper of two stacked cubes, with its attachment point facing the lower cube."""
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


@pytest.mark.isaacsim_ci
@pytest.mark.skipif(
    _RUNNING_CI,
    reason="Isaac Sim SurfaceGripperView initialization can deadlock in CI; keep CUDA fail-fast coverage only.",
)
def test_close_and_open_command() -> None:
    """Test that the close/open commands actually drive the surface gripper status.

    This is a regression test for the command plumbing: a single ``close`` command must move the
    gripper out of the *open* state into a *closing* (or *closed*) state, and a subsequent ``open``
    command must bring it back to the *open* state.
    """
    with build_simulation_context(device="cpu", sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, gravity=(0.0, 0.0, 0.0))) as sim:
        sim._app_control_on_stop_handle = None
        surface_gripper = _author_surface_gripper()

        sim.reset()

        assert surface_gripper.is_initialized
        assert surface_gripper.command.shape == (1,)
        assert surface_gripper.state.shape == (1,)
        # after a reset the gripper is idle (0.0) and open (-1.0)
        assert torch.all(wp.to_torch(surface_gripper.command) == 0.0)
        assert torch.all(wp.to_torch(surface_gripper.state) == -1.0)

        # send a single close command (the action term is edge-triggered, so commands are sent once)
        surface_gripper.set_grippers_command_index(wp.array([1.0], dtype=wp.float32, device="cpu"))
        surface_gripper.write_data_to_sim()
        for _ in range(3):
            sim.step()
            surface_gripper.update(sim.cfg.dt)
        # the close command must take effect: status is "closing" (0.0) or "closed" (1.0), never "open" (-1.0)
        state_after_close = wp.to_torch(surface_gripper.state)
        assert torch.all(state_after_close >= 0.0), f"close command had no effect, state={state_after_close.tolist()}"

        # send a single open command; the gripper must return to the open state
        surface_gripper.set_grippers_command_index(wp.array([-1.0], dtype=wp.float32, device="cpu"))
        surface_gripper.write_data_to_sim()
        for _ in range(3):
            sim.step()
            surface_gripper.update(sim.cfg.dt)
        # the open command must take effect: status is back to "open" (-1.0)
        state_after_open = wp.to_torch(surface_gripper.state)
        assert torch.all(state_after_open == -1.0), f"open command had no effect, state={state_after_open.tolist()}"


@pytest.mark.parametrize("device", test_devices(DeviceScope.CUDA))
@pytest.mark.isaacsim_ci
def test_raise_error_if_not_cpu(device) -> None:
    """Test that the SurfaceGripper raises an error if the device is not CPU."""
    with build_simulation_context(device=device, sim_cfg=SimulationCfg(physics=PhysxCfg(), dt=0.01, gravity=(0.0, 0.0, 0.0))) as sim:
        sim._app_control_on_stop_handle = None
        surface_gripper = _author_surface_gripper()
        assert not surface_gripper.is_initialized

        with pytest.raises(Exception, match="only supported on CPU"):
            sim.reset()


##
# View payloads, without a simulation.
##


def test_command_filter_and_partial_property_update_use_literal_view_payloads() -> None:
    """Submit only open/close commands and preserve property selector ordering."""
    # Skip USD initialization: a three-environment CPU gripper around a recording view.
    gripper = object.__new__(SurfaceGripper)
    gripper._device = "cpu"
    gripper._num_envs = 3
    gripper._ALL_INDICES = wp.array([0, 1, 2], dtype=wp.int32, device="cpu")
    gripper._gripper_command = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._max_grip_distance = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._coaxial_force_limit = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._shear_force_limit = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._retry_interval = wp.zeros(3, dtype=wp.float32, device="cpu")
    gripper._gripper_view = SimpleNamespace(
        apply_gripper_action=Mock(),
        set_surface_gripper_properties=Mock(),
    )

    gripper.set_grippers_command_index(wp.array([0.5, 0.0, -0.5], dtype=wp.float32, device="cpu"))

    gripper.write_data_to_sim()

    gripper.gripper_view.apply_gripper_action.assert_called_once_with([0.5, 0.0, -0.5], [[0], [2]])

    env_ids = wp.array([2, 0], dtype=wp.int32, device="cpu")
    gripper.update_gripper_properties_index(
        max_grip_distance=wp.array([0.2, 0.4], dtype=wp.float32, device="cpu"),
        env_ids=env_ids,
    )

    np.testing.assert_array_equal(gripper._max_grip_distance.numpy(), np.asarray([0.4, 0.0, 0.2], dtype=np.float32))
    properties = gripper.gripper_view.set_surface_gripper_properties.call_args.kwargs
    np.testing.assert_array_equal(properties.pop("max_grip_distance"), np.asarray([0.4, 0.0, 0.2], dtype=np.float32))
    assert properties == {
        "coaxial_force_limit": [0.0, 0.0, 0.0],
        "shear_force_limit": [0.0, 0.0, 0.0],
        "retry_interval": [0.0, 0.0, 0.0],
        "indices": [2, 0],
    }
