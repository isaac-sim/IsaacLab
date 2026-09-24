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

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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


def test_initialization_and_data_shapes(sim):
    """The Newton IMU sensor initializes and exposes correctly shaped buffers after one step."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    imu: Imu = scene["imu"]
    assert imu.num_instances == 2

    sim.step()
    scene.update(sim.get_physics_dt())

    assert imu.data.ang_vel_b.torch.shape == (2, 3)
    assert imu.data.lin_acc_b.torch.shape == (2, 3)


def test_at_rest_measures_gravity_and_zero_angular_velocity(sim):
    """A settled IMU measures gravity (~9.81 m/s^2 upward) and near-zero angular velocity."""
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Step enough for the cube to settle on the ground
    for _ in range(500):
        sim.step()
        scene.update(sim.get_physics_dt())

    imu: Imu = scene["imu"]
    lin_acc = imu.data.lin_acc_b.torch
    ang_vel = imu.data.ang_vel_b.torch

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
    torch.testing.assert_close(ang_vel, torch.zeros_like(ang_vel), atol=0.1, rtol=0.0)


@configclass
class FreefallSceneCfg(InteractiveSceneCfg):
    """Scene with a rigid cube and IMU but no ground plane (freefall)."""

    env_spacing = 2.0
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
            mass_props=sim_utils.MassCfg(mass=1.0),
            collision_props=sim_utils.UsdPhysicsCollisionCfg(),
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
    lin_acc = imu.data.lin_acc_b.torch

    # In freefall, accelerometer should read near zero (gravity and inertial acceleration cancel)
    acc_magnitude = torch.norm(lin_acc, dim=-1)
    torch.testing.assert_close(
        acc_magnitude,
        torch.zeros_like(acc_magnitude),
        atol=0.5,
        rtol=0.0,
    )


def test_no_stale_data_after_scene_reset(sim):
    """Regression for #4970: resets must not surface pre-reset IMU values (Newton).

    Reproduces the ``ManagerBasedRLEnv._reset_idx`` flow, where reset runs inside a step without a
    subsequent physics step: Newton's accelerometer still holds the pre-reset reading, so the public
    ``data`` accessor must return the zeroed buffers instead of refetching it. The cube rests on the
    ground first so the stale reading (gravity, ~9.81 m/s^2) is distinguishable from a reset one.
    """
    scene_cfg = ImuTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    scene.reset()

    imu: Imu = scene["imu"]

    # The cube falls from z=1.0 (bottom at z=0.9) and lands in ~86 steps at 200 Hz; 200 steps let it settle.
    for _ in range(200):
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt=sim.get_physics_dt())

    pre_reset_lin_acc = imu.data.lin_acc_b.torch.clone()
    assert (pre_reset_lin_acc[:, 2] > 5.0).all(), f"Expected a settled gravity reading, got {pre_reset_lin_acc}"

    # Partial reset: env 0 reads zeros, env 1 keeps its measurement.
    scene.reset(env_ids=torch.tensor([0], device=imu.device))
    lin_acc = imu.data.lin_acc_b.torch
    ang_vel = imu.data.ang_vel_b.torch
    torch.testing.assert_close(lin_acc[0], torch.zeros_like(lin_acc[0]))
    torch.testing.assert_close(ang_vel[0], torch.zeros_like(ang_vel[0]))
    torch.testing.assert_close(lin_acc[1], pre_reset_lin_acc[1])

    # Full reset zeroes every environment.
    imu.reset()
    lin_acc = imu.data.lin_acc_b.torch
    ang_vel = imu.data.ang_vel_b.torch
    torch.testing.assert_close(lin_acc, torch.zeros_like(lin_acc))
    torch.testing.assert_close(ang_vel, torch.zeros_like(ang_vel))
