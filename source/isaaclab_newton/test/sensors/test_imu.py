# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify IMU sensor functionality using Newton physics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import warp as wp
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.imu import Imu, ImuCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class ImuTestSceneCfg(InteractiveSceneCfg):
    """Scene with a rigid cube and an IMU sensor."""

    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
    )


@pytest.fixture
def sim():
    """Create a simulation context with Newton physics."""
    sim_cfg = SimulationCfg(
        dt=1.0 / 200.0,
        physics=NewtonCfg(
            solver_cfg=MJWarpSolverCfg(),
            num_substeps=1,
        ),
    )
    with sim_utils.build_simulation_context(sim_cfg=sim_cfg) as sim:
        sim._app_control_on_stop_handle = None
        sim.set_camera_view(eye=(5.0, 5.0, 5.0), target=(0.0, 0.0, 0.0))
        yield sim


def test_sensor_initialization(sim):
    """Test that the Newton IMU sensor initializes correctly."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    imu: Imu = scene["imu"]
    assert imu.num_instances == 2
    assert imu.data.ang_vel_b is not None
    assert imu.data.lin_acc_b is not None


def test_data_shapes(sim):
    """Test that IMU output tensors have correct shapes."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim.step()
    scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]
    ang_vel = wp.to_torch(imu.data.ang_vel_b)
    lin_acc = wp.to_torch(imu.data.lin_acc_b)

    assert ang_vel.shape == (2, 3)
    assert lin_acc.shape == (2, 3)


def test_gravity_at_rest(sim):
    """Test that a resting IMU measures gravity (~9.81 m/s^2 upward)."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Step enough for the cube to settle on the ground
    for _ in range(500):
        sim.step()
        scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]
    lin_acc = wp.to_torch(imu.data.lin_acc_b)

    # At rest, accelerometer should read ~9.81 in the up direction (Z body frame)
    torch.testing.assert_close(
        lin_acc[:, 2],
        torch.full((lin_acc.shape[0],), 9.81, dtype=lin_acc.dtype, device=lin_acc.device),
        atol=0.5,
        rtol=0.0,
    )
    # X and Y components should be near zero
    torch.testing.assert_close(
        lin_acc[:, :2],
        torch.zeros(lin_acc.shape[0], 2, dtype=lin_acc.dtype, device=lin_acc.device),
        atol=0.5,
        rtol=0.0,
    )


def test_angular_velocity_at_rest(sim):
    """Test that a resting IMU reports near-zero angular velocity."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    for _ in range(500):
        sim.step()
        scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]
    ang_vel = wp.to_torch(imu.data.ang_vel_b)

    torch.testing.assert_close(
        ang_vel,
        torch.zeros_like(ang_vel),
        atol=0.1,
        rtol=0.0,
    )


def test_reset(sim):
    """Test that reset zeroes out IMU data."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Step to get non-zero data
    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]

    ang_vel = wp.to_torch(imu.data.ang_vel_b)
    lin_acc = wp.to_torch(imu.data.lin_acc_b)
    assert torch.any(lin_acc != 0), "Expected non-zero data before reset"

    imu.reset()

    ang_vel = wp.to_torch(imu.data.ang_vel_b)
    lin_acc = wp.to_torch(imu.data.lin_acc_b)

    torch.testing.assert_close(ang_vel, torch.zeros_like(ang_vel))
    torch.testing.assert_close(lin_acc, torch.zeros_like(lin_acc))


@configclass
class FreefallSceneCfg(InteractiveSceneCfg):
    """Scene with a rigid cube and IMU but no ground plane (freefall)."""

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
    )


def test_freefall_acceleration(sim):
    """Test that a freefalling IMU measures near-zero acceleration."""
    scene_cfg = FreefallSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Step a few times while the cube is in freefall (no ground contact)
    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]
    lin_acc = wp.to_torch(imu.data.lin_acc_b)

    # In freefall, accelerometer should read near zero (gravity and inertial acceleration cancel)
    acc_magnitude = torch.norm(lin_acc, dim=-1)
    torch.testing.assert_close(
        acc_magnitude,
        torch.zeros_like(acc_magnitude),
        atol=0.5,
        rtol=0.0,
    )


def test_sensor_print(sim):
    """Test that the sensor string representation works."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    imu: Imu = scene["imu"]
    sensor_str = str(imu)
    assert "newton" in sensor_str
    assert "IMU sensor" in sensor_str
