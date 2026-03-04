# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import math

import pytest
import torch
import warp as wp

import omni.replicator.core as rep

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import TiledCameraCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensor, VisuoTactileSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor_cfg import GelSightRenderCfg

# Sample sensor poses

TEST_RENDER_CFG = GelSightRenderCfg(
    sensor_data_dir_name="gelsight_r15_data",
    image_height=320,
    image_width=240,
    mm_per_pixel=0.0877,
)


def get_sensor_cfg_by_type(sensor_type: str) -> VisuoTactileSensorCfg:
    """Return a sensor configuration based on the input type.

    Args:
        sensor_type: Type of sensor configuration. Options: "minimum_config", "tactile_cam", "nut_rgb_ff".

    Returns:
        The sensor configuration for the specified type.

    Raises:
        ValueError: If the sensor_type is not supported.
    """

    if sensor_type == "minimum_config":
        return VisuoTactileSensorCfg(
            prim_path="/World/Robot/elastomer/sensor_minimum_config",
            enable_camera_tactile=False,
            enable_force_field=False,
            render_cfg=TEST_RENDER_CFG,
            tactile_array_size=(10, 10),
            tactile_margin=0.003,
        )
    elif sensor_type == "tactile_cam":
        return VisuoTactileSensorCfg(
            prim_path="/World/Robot/elastomer/tactile_cam",
            enable_force_field=False,
            camera_cfg=TiledCameraCfg(
                height=320,
                width=240,
                prim_path="/World/Robot/elastomer_tip/cam",
                update_period=0,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
            render_cfg=TEST_RENDER_CFG,
            tactile_array_size=(10, 10),
            tactile_margin=0.003,
        )

    elif sensor_type == "nut_rgb_ff":
        return VisuoTactileSensorCfg(
            prim_path="/World/Robot/elastomer/sensor_nut",
            update_period=0,
            debug_vis=False,
            enable_camera_tactile=True,
            enable_force_field=True,
            camera_cfg=TiledCameraCfg(
                height=320,
                width=240,
                prim_path="/World/Robot/elastomer_tip/cam",
                update_period=0,
                data_types=["distance_to_image_plane"],
                spawn=None,
            ),
            render_cfg=TEST_RENDER_CFG,
            tactile_array_size=(5, 10),
            tactile_margin=0.003,
            contact_object_prim_path_expr="/World/Nut",
        )

    else:
        raise ValueError(
            f"Unsupported sensor type: {sensor_type}. Supported types: 'minimum_config', 'tactile_cam', 'nut_rgb_ff'"
        )


def setup(sensor_type: str = "cube"):
    """Create a new stage and setup simulation environment with robot, objects, and sensor.

    Args:
        sensor_type: Type of sensor configuration. Options: "minimum_config", "tactile_cam", "nut_rgb_ff".

    Returns:
        Tuple containing simulation context, sensor config, timestep, robot config, cube config, and nut config.
    """
    # Create a new stage
    sim_utils.create_new_stage()

    # Simulation time-step
    dt = 0.01

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)

    # Ground-plane
    mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
    create_prim_from_mesh("/World/defaultGroundPlane", mesh)

    # gelsightr15 filter
    usd_file_path = f"{ISAACLAB_NUCLEUS_DIR}/TacSL/gelsight_r15_finger/gelsight_r15_finger.usd"

    # robot
    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileWithCompliantContactCfg(
            usd_path=usd_file_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            compliant_contact_stiffness=10.0,
            compliant_contact_damping=1.0,
            physics_material_prim_path="elastomer",
        ),
        actuators={},
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5),
            rot=(-math.sqrt(2) / 2, 0.0, 0.0, math.sqrt(2) / 2),  # 90° rotation
            joint_pos={},
            joint_vel={},
        ),
    )
    # Cube
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )
    # Nut
    nut_cfg = RigidObjectCfg(
        prim_path="/World/Nut",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Factory/factory_nut_m16.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0 + 0.06776, 0.52),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    # Get the requested sensor configuration using the factory function
    sensor_cfg = get_sensor_cfg_by_type(sensor_type)

    # load stage
    sim_utils.update_stage()
    return sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg


def teardown(sim):
    """Teardown simulation environment."""
    # close all the opened viewport from before.
    rep.vp_manager.destroy_hydra_textures("Replicator")
    # stop simulation
    sim.stop()
    # clear the stage
    sim.clear_instance()


@pytest.fixture
def setup_minimum_config():
    """Create simulation context with minimum config sensor."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup("minimum_config")
    yield sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg
    teardown(sim)


@pytest.fixture
def setup_tactile_cam():
    """Create simulation context with tactile camera sensor."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup("tactile_cam")
    yield sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg
    teardown(sim)


@pytest.fixture
def setup_nut_rgb_ff():
    """Create simulation context with nut RGB force field sensor."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg = setup("nut_rgb_ff")
    yield sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg
    teardown(sim)


@pytest.mark.isaacsim_ci
def test_sensor_minimum_config(setup_minimum_config):
    """Test sensor with minimal configuration (no camera, no force field)."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_minimum_config
    _ = Articulation(cfg=robot_cfg)
    sensor_minimum = VisuoTactileSensor(cfg=sensor_cfg)
    sim.reset()
    # Simulate physics
    for _ in range(10):
        sim.step()
        sensor_minimum.update(dt)

    # check data should be None, since both camera and force field are disabled
    assert sensor_minimum.data.tactile_depth_image is None
    assert sensor_minimum.data.tactile_rgb_image is None
    assert sensor_minimum.data.tactile_points_pos_w is None
    assert sensor_minimum.data.tactile_points_quat_w is None
    assert sensor_minimum.data.penetration_depth is None
    assert sensor_minimum.data.tactile_normal_force is None
    assert sensor_minimum.data.tactile_shear_force is None

    # Check reset functionality
    sensor_minimum.reset()

    for i in range(10):
        sim.step()
        sensor_minimum.update(dt)
    sensor_minimum.reset(env_ids=[0])


@pytest.mark.isaacsim_ci
def test_sensor_cam_size_false(setup_tactile_cam):
    """Test sensor initialization fails with incorrect camera image size."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_tactile_cam
    sensor_cfg.camera_cfg.height = 80
    _ = VisuoTactileSensor(cfg=sensor_cfg)
    with pytest.raises(ValueError) as excinfo:
        sim.reset()
    assert "Camera configuration image size is not consistent with the render config" in str(excinfo.value)


@pytest.mark.isaacsim_ci
def test_sensor_cam_type_false(setup_tactile_cam):
    """Test sensor initialization fails with unsupported camera data types."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_tactile_cam
    sensor_cfg.camera_cfg.data_types = ["rgb"]
    _ = VisuoTactileSensor(cfg=sensor_cfg)
    with pytest.raises(ValueError) as excinfo:
        sim.reset()
    assert "Camera configuration data types are not supported" in str(excinfo.value)


@pytest.mark.isaacsim_ci
def test_sensor_cam_set(setup_tactile_cam):
    """Test sensor with camera configuration using existing camera prim."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_tactile_cam
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    sim.reset()
    sensor.get_initial_render()
    for _ in range(10):
        sim.step()
        sensor.update(dt, force_recompute=True)
        robot.update(dt)
    assert sensor.is_initialized
    assert sensor.data.tactile_depth_image.shape == (1, 320, 240, 1)
    assert sensor.data.tactile_rgb_image.shape == (1, 320, 240, 3)
    assert sensor.data.tactile_points_pos_w is None

    sensor.reset()
    for _ in range(10):
        sim.step()
        sensor.update(dt, force_recompute=True)
        robot.update(dt)
    sensor.reset(env_ids=[0])


@pytest.mark.isaacsim_ci
def test_sensor_cam_set_wrong_prim(setup_tactile_cam):
    """Test sensor initialization fails with invalid camera prim path."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_tactile_cam
    sensor_cfg.camera_cfg.prim_path = "/World/Robot/elastomer_tip/cam_wrong"
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    with pytest.raises(RuntimeError) as excinfo:
        sim.reset()
        robot.update(dt)
        sensor.update(dt)
    assert "Could not find prim with path" in str(excinfo.value)


@pytest.mark.isaacsim_ci
def test_sensor_cam_new_spawn(setup_tactile_cam):
    """Test sensor with camera configuration that spawns a new camera."""
    sim, sensor_cfg, dt, robot_cfg, object_cfg, nut_cfg = setup_tactile_cam
    sensor_cfg.camera_cfg.prim_path = "/World/Robot/elastomer_tip/cam_new"
    sensor_cfg.camera_cfg.spawn = sim_utils.PinholeCameraCfg(
        focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.01, 1.0e5)
    )
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    sim.reset()
    sensor.get_initial_render()
    for _ in range(10):
        sim.step()
        sensor.update(dt)
        robot.update(dt)
        # test lazy sensor update
        data = sensor.data
        assert data is not None
        assert data.tactile_depth_image.shape == (1, 320, 240, 1)
        assert data.tactile_rgb_image.shape == (1, 320, 240, 3)
        assert data.tactile_points_pos_w is None

    assert sensor.is_initialized


@pytest.mark.isaacsim_ci
def test_sensor_rgb_forcefield(setup_nut_rgb_ff):
    """Test sensor with both camera and force field enabled, detecting contact forces."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg = setup_nut_rgb_ff
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    nut = RigidObject(cfg=nut_cfg)
    sim.reset()
    sensor.get_initial_render()
    for _ in range(10):
        sim.step()
        sensor.update(dt, force_recompute=True)
        robot.update(dt)
        nut.update(dt)
    # check str
    print(sensor)
    assert sensor.is_initialized
    assert sensor.data.tactile_depth_image.shape == (1, 320, 240, 1)
    assert sensor.data.tactile_rgb_image.shape == (1, 320, 240, 3)
    assert sensor.data.tactile_points_pos_w.shape == (1, 50, 3)
    assert sensor.data.penetration_depth.shape == (1, 50)
    assert sensor.data.tactile_normal_force.shape == (1, 50)
    assert sensor.data.tactile_shear_force.shape == (1, 50, 2)
    sum_depth = torch.sum(sensor.data.penetration_depth)  # 0.020887471735477448
    normal_force_sum = torch.sum(sensor.data.tactile_normal_force.abs())
    shear_force_sum = torch.sum(sensor.data.tactile_shear_force.abs())
    assert normal_force_sum > 0.0
    assert sum_depth > 0.0
    assert shear_force_sum > 0.0


@pytest.mark.isaacsim_ci
def test_sensor_no_contact_object(setup_nut_rgb_ff):
    """Test sensor with force field but no contact object specified."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg = setup_nut_rgb_ff
    sensor_cfg.contact_object_prim_path_expr = None
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    nut = RigidObject(cfg=nut_cfg)
    sim.reset()
    sensor.get_initial_render()
    for _ in range(10):
        sim.step()
        sensor.update(dt, force_recompute=True)
        robot.update(dt)
        nut.update(dt)

    assert sensor.is_initialized
    assert sensor.data.tactile_depth_image.shape == (1, 320, 240, 1)
    assert sensor.data.tactile_rgb_image.shape == (1, 320, 240, 3)
    assert sensor.data.tactile_points_pos_w.shape == (1, 50, 3)
    # check no forces are detected
    assert torch.all(torch.abs(sensor.data.penetration_depth) < 1e-9)
    assert torch.all(torch.abs(sensor.data.tactile_normal_force) < 1e-9)
    assert torch.all(torch.abs(sensor.data.tactile_shear_force) < 1e-9)


@pytest.mark.isaacsim_ci
def test_sensor_force_field_contact_object_not_found(setup_nut_rgb_ff):
    """Test sensor initialization fails when contact object prim path is not found."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, NutCfg = setup_nut_rgb_ff

    sensor_cfg.enable_camera_tactile = False
    sensor_cfg.contact_object_prim_path_expr = "/World/Nut/wrong_prim"
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    with pytest.raises(RuntimeError) as excinfo:
        sim.reset()
        robot.update(dt)
        sensor.update(dt)
    assert "No contact object prim found matching pattern" in str(excinfo.value)


@pytest.mark.isaacsim_ci
def test_sensor_force_field_contact_object_no_sdf(setup_nut_rgb_ff):
    """Test sensor initialization fails when contact object has no SDF mesh."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, NutCfg = setup_nut_rgb_ff
    sensor_cfg.enable_camera_tactile = False
    sensor_cfg.contact_object_prim_path_expr = "/World/Cube"
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    cube = RigidObject(cfg=cube_cfg)
    with pytest.raises(RuntimeError) as excinfo:
        sim.reset()
        robot.update(dt)
        sensor.update(dt)
        cube.update(dt)
    assert "No SDF mesh found under contact object at path" in str(excinfo.value)


@pytest.mark.isaacsim_ci
def test_sensor_update_period_mismatch(setup_nut_rgb_ff):
    """Test sensor with both camera and force field enabled, detecting contact forces."""
    sim, sensor_cfg, dt, robot_cfg, cube_cfg, nut_cfg = setup_nut_rgb_ff
    sensor_cfg.update_period = dt
    sensor_cfg.camera_cfg.update_period = dt * 2
    robot = Articulation(cfg=robot_cfg)
    sensor = VisuoTactileSensor(cfg=sensor_cfg)
    nut = RigidObject(cfg=nut_cfg)
    sim.reset()
    sensor.get_initial_render()
    assert sensor.cfg.camera_cfg.update_period == sensor.cfg.update_period
    for i in range(10):
        sim.step()
        sensor.update(dt, force_recompute=True)
        robot.update(dt)
        nut.update(dt)
        assert torch.allclose(
            wp.to_torch(sensor._timestamp_last_update), torch.tensor((i + 1) * dt, device=sensor.device)
        )
        assert torch.allclose(
            wp.to_torch(sensor._camera_sensor._timestamp_last_update), torch.tensor((i + 1) * dt, device=sensor.device)
        )
