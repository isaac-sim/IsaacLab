# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import copy
import numpy as np
import random
import torch

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
import pytest
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from pxr import Gf, UsdGeom

# from Isaac Sim 4.2 onwards, pxr.Semantics is deprecated
try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.timer import Timer


@pytest.fixture(scope="function")
def setup_camera() -> tuple[sim_utils.SimulationContext, TiledCameraCfg, float]:
    """Fixture to set up and tear down the camera simulation environment."""
    camera_cfg = TiledCameraCfg(
        height=128,
        width=256,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 0.0, 1.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create a new stage
    stage_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    stage_utils.update_stage()
    yield sim, camera_cfg, dt
    # Teardown
    rep.vp_manager.destroy_hydra_textures("Replicator")
    sim._timeline.stop()
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_single_camera_init(setup_camera):
    """Test single camera initialization."""
    sim, camera_cfg, dt = setup_camera
    # Create camera
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.quat_w_ros.shape == (1, 4)
    assert camera.data.quat_w_world.shape == (1, 4)
    assert camera.data.quat_w_opengl.shape == (1, 4)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 3)
                assert (im_data / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)
                assert im_data.mean() > 0.0
    del camera


@pytest.mark.isaacsim_ci
def test_depth_clipping_max(setup_camera):
    """Test depth max clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="max",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0
    assert camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0]
    assert camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1]

    del camera


@pytest.mark.isaacsim_ci
def test_depth_clipping_none(setup_camera):
    """Test depth none clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="none",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) > 0
    assert camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0]
    if len(camera.data.output["depth"][~torch.isinf(camera.data.output["depth"])]) > 0:
        assert (
            camera.data.output["depth"][~torch.isinf(camera.data.output["depth"])].max()
            <= camera_cfg.spawn.clipping_range[1]
        )

    del camera


@pytest.mark.isaacsim_ci
def test_depth_clipping_zero(setup_camera):
    """Test depth zero clipping."""
    sim, _, dt = setup_camera
    # get camera cfgs
    camera_cfg = TiledCameraCfg(
        prim_path="/World/Camera",
        offset=TiledCameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(-0.125, 0.362, 0.873, -0.302), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(4.9, 5.0),
        ),
        height=540,
        width=960,
        data_types=["depth"],
        depth_clipping_behavior="zero",
    )
    camera = TiledCamera(camera_cfg)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    camera.update(dt)

    assert len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0
    assert camera.data.output["depth"].min() == 0.0
    assert camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1]

    del camera


@pytest.mark.isaacsim_ci
def test_multi_camera_init(setup_camera):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_camera

    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
                for i in range(4):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(4):
                    assert im_data[i].mean() > 0.0
    del camera


@pytest.mark.isaacsim_ci
def test_rgb_only_camera(setup_camera):
    """Test initialization with only RGB data type."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["rgb"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["rgba", "rgb"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        im_data = camera.data.output["rgb"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
        for i in range(4):
            assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["rgb"].dtype == torch.uint8
    del camera


@pytest.mark.isaacsim_ci
def test_data_types(setup_camera):
    """Test different data types for camera initialization."""
    sim, camera_cfg, dt = setup_camera
    # Create camera
    camera_cfg_distance = copy.deepcopy(camera_cfg)
    camera_cfg_distance.data_types = ["distance_to_camera"]
    camera_cfg_distance.prim_path = "/World/CameraDistance"
    camera_distance = TiledCamera(camera_cfg_distance)
    camera_cfg_depth = copy.deepcopy(camera_cfg)
    camera_cfg_depth.data_types = ["depth"]
    camera_cfg_depth.prim_path = "/World/CameraDepth"
    camera_depth = TiledCamera(camera_cfg_depth)
    camera_cfg_both = copy.deepcopy(camera_cfg)
    camera_cfg_both.data_types = ["distance_to_camera", "depth"]
    camera_cfg_both.prim_path = "/World/CameraBoth"
    camera_both = TiledCamera(camera_cfg_both)

    # Play sim
    sim.reset()

    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check if cameras are initialized
    assert camera_distance.is_initialized
    assert camera_depth.is_initialized
    assert camera_both.is_initialized

    # Check if camera prims are set correctly and that they are camera prims
    assert camera_distance._sensor_prims[0].GetPath().pathString == "/World/CameraDistance"
    assert isinstance(camera_distance._sensor_prims[0], UsdGeom.Camera)
    assert camera_depth._sensor_prims[0].GetPath().pathString == "/World/CameraDepth"
    assert isinstance(camera_depth._sensor_prims[0], UsdGeom.Camera)
    assert camera_both._sensor_prims[0].GetPath().pathString == "/World/CameraBoth"
    assert isinstance(camera_both._sensor_prims[0], UsdGeom.Camera)
    assert list(camera_distance.data.output.keys()) == ["distance_to_camera"]
    assert list(camera_depth.data.output.keys()) == ["depth"]
    assert list(camera_both.data.output.keys()) == ["depth", "distance_to_camera"]

    del camera_distance
    del camera_depth
    del camera_both


@pytest.mark.isaacsim_ci
def test_depth_only_camera(setup_camera):
    """Test initialization with only depth."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_camera"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_camera"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        im_data = camera.data.output["distance_to_camera"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
        for i in range(4):
            assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_camera"].dtype == torch.float
    del camera


@pytest.mark.isaacsim_ci
def test_rgba_only_camera(setup_camera):
    """Test initialization with only RGBA."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["rgba"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["rgba"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(4):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["rgba"].dtype == torch.uint8
    del camera


@pytest.mark.isaacsim_ci
def test_distance_to_camera_only_camera(setup_camera):
    """Test initialization with only distance_to_camera."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_camera"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_camera"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_camera"].dtype == torch.float
    del camera


@pytest.mark.isaacsim_ci
def test_distance_to_image_plane_only_camera(setup_camera):
    """Test initialization with only distance_to_image_plane."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["distance_to_image_plane"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["distance_to_image_plane"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["distance_to_image_plane"].dtype == torch.float
    del camera


@pytest.mark.isaacsim_ci
def test_normals_only_camera(setup_camera):
    """Test initialization with only normals."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["normals"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["normals"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            for i in range(4):
                assert im_data[i].mean() > 0.0
    assert camera.data.output["normals"].dtype == torch.float
    del camera


@pytest.mark.isaacsim_ci
def test_motion_vectors_only_camera(setup_camera):
    """Test initialization with only motion_vectors."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["motion_vectors"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["motion_vectors"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
            for i in range(4):
                assert im_data[i].mean() != 0.0
    assert camera.data.output["motion_vectors"].dtype == torch.float
    del camera


@pytest.mark.isaacsim_ci
def test_semantic_segmentation_colorize_only_camera(setup_camera):
    """Test initialization with only semantic_segmentation."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["semantic_segmentation"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(4):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["semantic_segmentation"].dtype == torch.uint8
    assert isinstance(camera.data.info["semantic_segmentation"], dict)
    del camera


@pytest.mark.isaacsim_ci
def test_instance_segmentation_fast_colorize_only_camera(setup_camera):
    """Test initialization with only instance_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(num_cameras):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["instance_segmentation_fast"].dtype == torch.uint8
    assert isinstance(camera.data.info["instance_segmentation_fast"], dict)
    del camera


@pytest.mark.isaacsim_ci
def test_instance_id_segmentation_fast_colorize_only_camera(setup_camera):
    """Test initialization with only instance_id_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_id_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
            for i in range(num_cameras):
                assert (im_data[i] / 255.0).mean() > 0.0
    assert camera.data.output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(camera.data.info["instance_id_segmentation_fast"], dict)
    del camera


@pytest.mark.isaacsim_ci
def test_semantic_segmentation_non_colorize_only_camera(setup_camera):
    """Test initialization with only semantic_segmentation."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_semantic_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["semantic_segmentation"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["semantic_segmentation"].dtype == torch.int32
    assert isinstance(camera.data.info["semantic_segmentation"], dict)

    del camera


@pytest.mark.isaacsim_ci
def test_instance_segmentation_fast_non_colorize_only_camera(setup_camera):
    """Test initialization with only instance_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_instance_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["instance_segmentation_fast"].dtype == torch.int32
    assert isinstance(camera.data.info["instance_segmentation_fast"], dict)
    del camera


def test_instance_id_segmentation_fast_non_colorize_only_camera(setup_camera):
    """Test initialization with only instance_id_segmentation_fast."""
    sim, camera_cfg, dt = setup_camera
    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.colorize_instance_id_segmentation = False
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert list(camera.data.output.keys()) == ["instance_id_segmentation_fast"]

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for _, im_data in camera.data.output.items():
            assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
            for i in range(num_cameras):
                assert im_data[i].to(dtype=float).mean() > 0.0
    assert camera.data.output["instance_id_segmentation_fast"].dtype == torch.int32
    assert isinstance(camera.data.info["instance_id_segmentation_fast"], dict)
    del camera


@pytest.mark.isaacsim_ci
def test_all_annotators_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 9
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


@pytest.mark.isaacsim_ci
def test_all_annotators_low_resolution_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 2
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 40
    camera_cfg.width = 40
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


@pytest.mark.isaacsim_ci
def test_all_annotators_non_perfect_square_number_camera(setup_camera):
    """Test initialization with all supported annotators."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 11
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.0
            elif data_type in ["motion_vectors"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert im_data[i].mean() != 0.0
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 0.0

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


@pytest.mark.isaacsim_ci
def test_all_annotators_instanceable(setup_camera):
    """Test initialization with all supported annotators on instanceable assets."""
    sim, camera_cfg, dt = setup_camera
    all_annotator_types = [
        "rgb",
        "rgba",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]

    num_cameras = 10
    for i in range(num_cameras):
        prim_utils.create_prim(f"/World/Origin_{i}", "Xform", translation=(0.0, i, 0.0))

    # Create a stage with 10 instanceable cubes, where each camera points to one cube
    stage = stage_utils.get_current_stage()
    for i in range(10):
        # Remove objects added to stage by default
        stage.RemovePrim(f"/World/Objects/Obj_{i:02d}")
        # Add instanceable cubes
        prim_utils.create_prim(
            f"/World/Cube_{i}",
            "Xform",
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            translation=(0.0, i, 5.0),
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=(5.0, 5.0, 5.0),
        )
        prim = stage.GetPrimAtPath(f"/World/Cube_{i}")
        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr()
        sem.CreateSemanticDataAttr()
        sem.GetSemanticTypeAttr().Set("class")
        sem.GetSemanticDataAttr().Set("cube")

    # Create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 120
    camera_cfg.width = 80
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.offset.pos = (0.0, 0.0, 5.5)
    camera = TiledCamera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    # Check buffers that exists and have correct shapes
    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(2):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # check image data
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "semantic_segmentation",
                "instance_segmentation_fast",
                "instance_id_segmentation_fast",
            ]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 4)
                # semantic_segmentation has mean 0.43
                # rgba has mean 0.38
                # instance_segmentation_fast has mean 0.42
                # instance_id_segmentation_fast has mean 0.55-0.62
                for i in range(num_cameras):
                    assert (im_data[i] / 255.0).mean() > 0.2
            elif data_type in ["motion_vectors"]:
                # motion vectors have mean 0.2
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 2)
                for i in range(num_cameras):
                    assert (im_data[i].abs().mean()) > 0.15
            elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                # depth has mean 2.7
                # distance_to_image_plane has mean 3.1
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
                for i in range(num_cameras):
                    assert im_data[i].mean() > 2.5

    # access image data and compare dtype
    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8
    assert isinstance(info["semantic_segmentation"], dict)
    assert isinstance(info["instance_segmentation_fast"], dict)
    assert isinstance(info["instance_id_segmentation_fast"], dict)

    del camera


@pytest.mark.isaacsim_ci
def test_throughput(setup_camera):
    """Test tiled camera throughput."""
    sim, camera_cfg, dt = setup_camera
    # create camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 640
    camera = TiledCamera(camera_cfg)

    # Play simulator
    sim.reset()

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(5):
        # perform rendering
        sim.step()
        # update camera
        with Timer(f"Time taken for updating camera with shape {camera.image_shape}"):
            camera.update(dt)
        # Check image data
        for im_type, im_data in camera.data.output.items():
            if im_type == "rgb":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 3)
                assert (im_data / 255.0).mean() > 0.0
            elif im_type == "distance_to_camera":
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)
                assert im_data.mean() > 0.0
    del camera


@pytest.mark.isaacsim_ci
def test_output_equal_to_usd_camera_intrinsics(setup_camera):
    """
    Test that the output of the ray caster camera and the usd camera are the same when both are
    initialized with the same intrinsic matrix.
    """
    sim, _, dt = setup_camera
    # create cameras
    offset_rot = (-0.1251, 0.3617, 0.8731, -0.3020)
    offset_pos = (2.5, 2.5, 4.0)
    intrinsics = [380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0]
    # get camera cfgs
    # TODO: add clipping range back, once correctly supported by tiled camera
    camera_tiled_cfg = TiledCameraCfg(
        prim_path="/World/Camera_tiled",
        offset=TiledCameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
        ),
        height=540,
        width=960,
        data_types=["depth"],
    )
    camera_usd_cfg = CameraCfg(
        prim_path="/World/Camera_usd",
        offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsics,
            height=540,
            width=960,
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane"],
    )

    # set aperture offsets to 0, as currently not supported for usd/ tiled camera
    camera_tiled_cfg.spawn.horizontal_aperture_offset = 0
    camera_tiled_cfg.spawn.vertical_aperture_offset = 0
    camera_usd_cfg.spawn.horizontal_aperture_offset = 0
    camera_usd_cfg.spawn.vertical_aperture_offset = 0
    # init cameras
    camera_tiled = TiledCamera(camera_tiled_cfg)
    camera_usd = Camera(camera_usd_cfg)

    # play sim
    sim.reset()
    sim.play()

    # perform steps
    for _ in range(5):
        sim.step()

    # update camera
    camera_usd.update(dt)
    camera_tiled.update(dt)

    # filter nan and inf from output
    cam_tiled_output = camera_tiled.data.output["depth"].clone()
    cam_usd_output = camera_usd.data.output["distance_to_image_plane"].clone()
    cam_tiled_output[torch.isnan(cam_tiled_output)] = 0
    cam_tiled_output[torch.isinf(cam_tiled_output)] = 0
    cam_usd_output[torch.isnan(cam_usd_output)] = 0
    cam_usd_output[torch.isinf(cam_usd_output)] = 0

    # check that both have the same intrinsic matrices
    torch.testing.assert_close(camera_tiled.data.intrinsic_matrices[0], camera_usd.data.intrinsic_matrices[0])

    # check the apertures
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
        camera_tiled._sensor_prims[0].GetHorizontalApertureAttr().Get(),
    )
    torch.testing.assert_close(
        camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
        camera_tiled._sensor_prims[0].GetVerticalApertureAttr().Get(),
    )

    # check image data
    torch.testing.assert_close(
        cam_tiled_output[..., 0],
        cam_usd_output[..., 0],
        atol=5e-5,
        rtol=5e-6,
    )

    del camera_tiled
    del camera_usd


@pytest.mark.isaacsim_ci
def test_sensor_print(setup_camera):
    """Test sensor print is working correctly."""
    sim, camera_cfg, _ = setup_camera
    # Create sensor
    sensor = TiledCamera(cfg=camera_cfg)
    # Play sim
    sim.reset()
    # print info
    print(sensor)


@pytest.mark.isaacsim_ci
def test_frame_offset_small_resolution(setup_camera):
    """Test frame offset issue with small resolution camera."""
    sim, camera_cfg, dt = setup_camera
    # Create sensor
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 80
    camera_cfg.width = 80
    camera_cfg.offset.pos = (0.0, 0.0, 0.5)
    tiled_camera = TiledCamera(camera_cfg)
    # play sim
    sim.reset()
    # simulate some steps first to make sure objects are settled
    stage = stage_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        UsdGeom.Gprim(prim).GetOrderedXformOps()[2].Set(Gf.Vec3d(1.0, 1.0, 1.0))
    for i in range(100):
        # step simulation
        sim.step()
        # update camera
        tiled_camera.update(dt)
    # collect image data
    image_before = tiled_camera.data.output["rgb"].clone() / 255.0

    # update scene
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # update rendering
    sim.step()
    # update camera
    tiled_camera.update(dt)

    # make sure the image is different
    image_after = tiled_camera.data.output["rgb"].clone() / 255.0

    # check difference is above threshold
    assert torch.abs(image_after - image_before).mean() > 0.1  # images of same color should be below 0.01


@pytest.mark.isaacsim_ci
def test_frame_offset_large_resolution(setup_camera):
    """Test frame offset issue with large resolution camera."""
    sim, camera_cfg, dt = setup_camera
    # Create sensor
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 480
    tiled_camera = TiledCamera(camera_cfg)

    # modify scene to be less stochastic
    stage = stage_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(1, 1, 1)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # play sim
    sim.reset()
    # simulate some steps first to make sure objects are settled
    for i in range(100):
        # step simulation
        sim.step()
        # update camera
        tiled_camera.update(dt)
    # collect image data
    image_before = tiled_camera.data.output["rgb"].clone() / 255.0

    # update scene
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    # update rendering
    sim.step()
    # update camera
    tiled_camera.update(dt)

    # make sure the image is different
    image_after = tiled_camera.data.output["rgb"].clone() / 255.0

    # check difference is above threshold
    assert torch.abs(image_after - image_before).mean() > 0.01  # images of same color should be below 0.001


"""
Helper functions.
"""


@staticmethod
def _populate_scene():
    """Add prims to the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.SphereLightCfg()
    cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
    cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    # Random objects
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    for i in range(10):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = prim_utils.create_prim(
            f"/World/Objects/Obj_{i:02d}",
            prim_type,
            translation=position,
            scale=(0.25, 0.25, 0.25),
            semantic_label=prim_type,
        )
        # cast to geom prim
        geom_prim = getattr(UsdGeom, prim_type)(prim)
        # set random color
        color = Gf.Vec3f(random.random(), random.random(), random.random())
        geom_prim.CreateDisplayColorAttr()
        geom_prim.GetDisplayColorAttr().Set([color])
        # add rigid properties
        SingleGeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
        SingleRigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
