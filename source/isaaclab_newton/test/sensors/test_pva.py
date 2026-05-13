# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests to verify PVA sensor functionality using Newton physics."""

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
from isaaclab.sensors.pva import Pva, PvaCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass


@configclass
class PvaTestSceneCfg(InteractiveSceneCfg):
    """Scene with a rigid cube and a PVA sensor."""

    env_spacing = 2.0
    terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    pva = PvaCfg(
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
    """Test that the Newton PVA sensor initializes correctly."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    pva: Pva = scene["pva"]
    assert pva.num_instances == 2
    assert pva.data.pos_w is not None
    assert pva.data.quat_w is not None
    assert pva.data.lin_vel_b is not None
    assert pva.data.ang_vel_b is not None
    assert pva.data.lin_acc_b is not None
    assert pva.data.ang_acc_b is not None
    assert pva.data.projected_gravity_b is not None
    assert pva.data.pose_w is not None


def test_data_shapes(sim):
    """Test that PVA output tensors have correct shapes."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim.step()
    scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    assert pva.data.pos_w.torch.shape == (2, 3)
    assert pva.data.quat_w.torch.shape == (2, 4)
    assert pva.data.pose_w.torch.shape == (2, 7)
    assert pva.data.lin_vel_b.torch.shape == (2, 3)
    assert pva.data.ang_vel_b.torch.shape == (2, 3)
    assert pva.data.lin_acc_b.torch.shape == (2, 3)
    assert pva.data.ang_acc_b.torch.shape == (2, 3)
    assert pva.data.projected_gravity_b.torch.shape == (2, 3)


def test_gravity_at_rest(sim):
    """Test that a resting PVA sensor reports correct projected gravity."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Cube falls from z=1.0 (bottom at z=0.9), reaches ground in ~86 steps at 200 Hz.
    for _ in range(200):
        sim.step()
        scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    proj_grav = pva.data.projected_gravity_b.torch

    expected = torch.tensor([[0.0, 0.0, -1.0]], dtype=proj_grav.dtype, device=proj_grav.device).repeat(2, 1)
    torch.testing.assert_close(proj_grav, expected, atol=0.05, rtol=0.0)


def test_velocity_at_rest(sim):
    """Test that a resting PVA sensor reports near-zero velocity."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    for _ in range(200):
        sim.step()
        scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    lin_vel = pva.data.lin_vel_b.torch
    ang_vel = pva.data.ang_vel_b.torch

    torch.testing.assert_close(lin_vel, torch.zeros_like(lin_vel), atol=0.05, rtol=0.0)
    torch.testing.assert_close(ang_vel, torch.zeros_like(ang_vel), atol=0.05, rtol=0.0)


def test_position_nonzero(sim):
    """Test that the PVA sensor reports a non-zero world-frame position."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim.step()
    scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    pos = pva.data.pos_w.torch

    assert torch.all(pos[:, 2] > 0.0), f"Expected positive z position, got {pos[:, 2]}"


def test_reset(sim):
    """Test that reset zeroes out PVA data."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]

    pos = pva.data.pos_w.torch
    assert torch.any(pos != 0), "Expected non-zero data before reset"

    pva.reset()

    # Access internal buffers directly to avoid lazy re-evaluation via pva.data
    # (the data property triggers _update_buffers_impl which would overwrite reset values).
    pos = wp.to_torch(pva._data._pos_w)
    lin_vel = wp.to_torch(pva._data._lin_vel_b)
    ang_vel = wp.to_torch(pva._data._ang_vel_b)
    lin_acc = wp.to_torch(pva._data._lin_acc_b)
    ang_acc = wp.to_torch(pva._data._ang_acc_b)
    quat = wp.to_torch(pva._data._quat_w)

    torch.testing.assert_close(pos, torch.zeros_like(pos))
    torch.testing.assert_close(lin_vel, torch.zeros_like(lin_vel))
    torch.testing.assert_close(ang_vel, torch.zeros_like(ang_vel))
    torch.testing.assert_close(lin_acc, torch.zeros_like(lin_acc))
    torch.testing.assert_close(ang_acc, torch.zeros_like(ang_acc))
    expected_quat = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=quat.dtype, device=quat.device).repeat(2, 1)
    torch.testing.assert_close(quat, expected_quat)


@configclass
class FreefallSceneCfg(InteractiveSceneCfg):
    """Scene with a rigid cube and PVA but no ground plane (freefall)."""

    env_spacing = 2.0
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0)),
    )

    pva = PvaCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
    )


def test_freefall_velocity_increases(sim):
    """Test that a freefalling body's downward velocity increases over time."""
    scene_cfg = FreefallSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    for _ in range(50):
        sim.step()
        scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    lin_vel = pva.data.lin_vel_b.torch

    speed = torch.norm(lin_vel, dim=-1)
    assert torch.all(speed > 0.1), f"Expected non-zero velocity in freefall, got {speed}"


def test_freefall_acceleration(sim):
    """Test that a freefalling body reports coordinate acceleration equal to gravity.

    PVA reports coordinate acceleration (from body_qdd), not proper acceleration.
    In freefall, coordinate acceleration equals gravitational acceleration (~9.81 m/s^2
    downward). For an upright body, this is (0, 0, -9.81) in the body frame.
    """
    scene_cfg = FreefallSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    for _ in range(10):
        sim.step()
        scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    lin_acc = pva.data.lin_acc_b.torch
    ang_acc = pva.data.ang_acc_b.torch

    # Coordinate acceleration in freefall should be ~(0, 0, -9.81) in body frame.
    expected_acc = torch.tensor([[0.0, 0.0, -9.81]], dtype=lin_acc.dtype, device=lin_acc.device).repeat(2, 1)
    torch.testing.assert_close(lin_acc, expected_acc, atol=0.5, rtol=0.0)

    # Angular acceleration should be near zero (no torques in freefall).
    torch.testing.assert_close(ang_acc, torch.zeros_like(ang_acc), atol=0.05, rtol=0.0)


def test_sensor_print(sim):
    """Test that the sensor string representation works."""
    scene_cfg = PvaTestSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    pva: Pva = scene["pva"]
    sensor_str = str(pva)
    assert "newton" in sensor_str
    assert "Pva sensor" in sensor_str


@configclass
class OffsetRotatedSceneCfg(InteractiveSceneCfg):
    """Scene with a tilted cube and offset PVA sensor in freefall.

    The cube is rotated 90 degrees about the X axis and the sensor has a
    +Z offset of 0.5 m in the body frame. This exercises the lever-arm
    velocity/acceleration corrections and body-frame gravity projection.
    """

    env_spacing = 2.0
    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 5.0),
            rot=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 deg about X (x, y, z, w)
        ),
    )

    pva = PvaCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        offset=PvaCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
    )


def test_offset_and_rotated_body(sim):
    """Test position offset and body-frame gravity for a 90-degree-tilted body.

    With a 90-deg rotation about X, the body-frame +Z offset of 0.5 m maps
    to world-frame (0, -0.5, 0). Projected gravity in the tilted sensor
    frame should be approximately (0, -1, 0) instead of (0, 0, -1).
    """
    scene_cfg = OffsetRotatedSceneCfg(num_envs=2)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim.step()
    scene.update(sim.get_physics_dt())

    pva: Pva = scene["pva"]
    pos = pva.data.pos_w.torch
    proj_grav = pva.data.projected_gravity_b.torch

    # pos_w is in absolute world frame; subtract env origins to get env-relative position.
    # Expected env-relative: body at (0,0,5) + R_x(90) * (0,0,0.5) = (0, -0.5, 5)
    env_pos = pos - scene.env_origins.to(pos.device)
    torch.testing.assert_close(env_pos[:, 0], torch.zeros(2, dtype=pos.dtype, device=pos.device), atol=0.01, rtol=0.0)
    torch.testing.assert_close(
        env_pos[:, 1], torch.full((2,), -0.5, dtype=pos.dtype, device=pos.device), atol=0.01, rtol=0.0
    )
    assert torch.all(env_pos[:, 2] > 4.5), f"Expected z near 5.0, got {env_pos[:, 2]}"

    # Projected gravity: R_x(-90) * (0, 0, -1) = (0, -1, 0)
    expected_grav = torch.tensor([[0.0, -1.0, 0.0]], dtype=proj_grav.dtype, device=proj_grav.device).repeat(2, 1)
    torch.testing.assert_close(proj_grav, expected_grav, atol=0.05, rtol=0.0)
