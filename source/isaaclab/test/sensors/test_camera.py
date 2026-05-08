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

import copy
import random

import numpy as np
import pytest
import scipy.spatial.transform as tf
import torch

import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg

pytestmark = pytest.mark.isaacsim_ci

# sample camera poses
POSITION = (2.5, 2.5, 2.5)
# Quaternions in xyzw format
QUAT_ROS = (0.33985114, 0.82047325, -0.42470819, -0.17591989)
QUAT_OPENGL = (0.17591988, 0.42470818, 0.82047324, 0.33985113)
QUAT_WORLD = (-0.27984815, -0.1159169, 0.88047623, -0.3647052)

# NOTE: setup and teardown are own function to allow calling them in the tests

# resolutions
HEIGHT = 240
WIDTH = 320


def setup() -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=dt)
    sim = sim_utils.SimulationContext(sim_cfg)
    # populate scene
    _populate_scene()
    # load stage
    sim_utils.update_stage()
    return sim, camera_cfg, dt


def teardown(sim: sim_utils.SimulationContext):
    # Cleanup
    # close all the opened viewport from before.
    rep.vp_manager.destroy_hydra_textures("Replicator")
    # stop simulation
    sim.stop()
    # clear the stage
    sim.clear_instance()


@pytest.fixture
def setup_sim_camera():
    """Create a simulation context."""
    sim, camera_cfg, dt = setup()
    yield sim, camera_cfg, dt
    teardown(sim)


def test_camera_init(setup_sim_camera):
    """Test camera initialization."""
    # Create camera configuration
    sim, camera_cfg, dt = setup_sim_camera
    # Create camera
    camera = Camera(camera_cfg)
    # Check simulation parameter is set correctly
    assert sim.get_setting("/isaaclab/render/rtx_sensors")
    # Play sim
    sim.reset()
    # Check if camera is initialized
    assert camera.is_initialized
    # Check if camera prim is set correctly and that it is a camera prim
    assert camera._sensor_prims[0].GetPath().pathString == camera_cfg.prim_path
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    # Check buffers that exist and have correct shapes
    assert camera.data.pos_w.shape == (1, 3)
    assert camera.data.quat_w_ros.shape == (1, 4)
    assert camera.data.quat_w_world.shape == (1, 4)
    assert camera.data.quat_w_opengl.shape == (1, 4)
    assert camera.data.intrinsic_matrices.shape == (1, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)
    assert camera.data.info == {camera_cfg.data_types[0]: None}

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(sim.cfg.dt)
        # check image data
        for im_data in camera.data.output.values():
            assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_camera_init_offset(setup_sim_camera):
    """Test camera initialization with offset using different conventions."""
    sim, camera_cfg, dt = setup_sim_camera
    # define the same offset in all conventions
    # -- ROS convention
    cam_cfg_offset_ros = copy.deepcopy(camera_cfg)
    cam_cfg_offset_ros.update_latest_camera_pose = True
    cam_cfg_offset_ros.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_ROS,
        convention="ros",
    )
    cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
    camera_ros = Camera(cam_cfg_offset_ros)
    # -- OpenGL convention
    cam_cfg_offset_opengl = copy.deepcopy(camera_cfg)
    cam_cfg_offset_opengl.update_latest_camera_pose = True
    cam_cfg_offset_opengl.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_OPENGL,
        convention="opengl",
    )
    cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
    camera_opengl = Camera(cam_cfg_offset_opengl)
    # -- World convention
    cam_cfg_offset_world = copy.deepcopy(camera_cfg)
    cam_cfg_offset_world.update_latest_camera_pose = True
    cam_cfg_offset_world.offset = CameraCfg.OffsetCfg(
        pos=POSITION,
        rot=QUAT_WORLD,
        convention="world",
    )
    cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
    camera_world = Camera(cam_cfg_offset_world)

    # play sim
    sim.reset()

    # retrieve camera pose using USD API
    prim_tf_ros = camera_ros._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf_opengl = camera_opengl._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf_world = camera_world._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # convert them from column-major to row-major
    prim_tf_ros = np.transpose(prim_tf_ros)
    prim_tf_opengl = np.transpose(prim_tf_opengl)
    prim_tf_world = np.transpose(prim_tf_world)

    # check that all transforms are set correctly
    np.testing.assert_allclose(prim_tf_ros[0:3, 3], cam_cfg_offset_ros.offset.pos)
    np.testing.assert_allclose(prim_tf_opengl[0:3, 3], cam_cfg_offset_opengl.offset.pos)
    np.testing.assert_allclose(prim_tf_world[0:3, 3], cam_cfg_offset_world.offset.pos)
    # scipy's as_quat() returns xyzw format, which matches our config format
    np.testing.assert_allclose(
        tf.Rotation.from_matrix(prim_tf_ros[:3, :3]).as_quat(),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        tf.Rotation.from_matrix(prim_tf_opengl[:3, :3]).as_quat(),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        tf.Rotation.from_matrix(prim_tf_world[:3, :3]).as_quat(),
        cam_cfg_offset_opengl.offset.rot,
        rtol=1e-5,
    )

    # check if transform correctly set in output
    np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_ros[0].cpu().numpy(), QUAT_ROS, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_opengl[0].cpu().numpy(), QUAT_OPENGL, rtol=1e-5)
    np.testing.assert_allclose(camera_ros.data.quat_w_world[0].cpu().numpy(), QUAT_WORLD, rtol=1e-5)


def test_multi_camera_init(setup_sim_camera):
    """Test multi-camera initialization."""
    sim, camera_cfg, dt = setup_sim_camera
    # create two cameras with different prim paths
    # -- camera 1
    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    cam_1 = Camera(cam_cfg_1)
    # -- camera 2
    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_2 = Camera(cam_cfg_2)

    # play sim
    sim.reset()

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        cam_1.update(dt)
        cam_2.update(dt)
        # check image data
        for cam in [cam_1, cam_2]:
            for im_data in cam.data.output.values():
                assert im_data.shape == (1, camera_cfg.height, camera_cfg.width, 1)


def test_multi_camera_with_different_resolution(setup_sim_camera):
    """Test multi-camera initialization with cameras having different image resolutions."""
    sim, camera_cfg, dt = setup_sim_camera
    # create two cameras with different prim paths
    # -- camera 1
    cam_cfg_1 = copy.deepcopy(camera_cfg)
    cam_cfg_1.prim_path = "/World/Camera_1"
    cam_1 = Camera(cam_cfg_1)
    # -- camera 2
    cam_cfg_2 = copy.deepcopy(camera_cfg)
    cam_cfg_2.prim_path = "/World/Camera_2"
    cam_cfg_2.height = 240
    cam_cfg_2.width = 320
    cam_2 = Camera(cam_cfg_2)

    # play sim
    sim.reset()

    # perform rendering
    sim.step()
    # update camera
    cam_1.update(dt)
    cam_2.update(dt)
    # check image sizes
    assert cam_1.data.output["distance_to_image_plane"].shape == (1, camera_cfg.height, camera_cfg.width, 1)
    assert cam_2.data.output["distance_to_image_plane"].shape == (1, cam_cfg_2.height, cam_cfg_2.width, 1)


def test_camera_init_intrinsic_matrix(setup_sim_camera):
    """Test camera initialization from intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera
    # get the first camera
    camera_1 = Camera(cfg=camera_cfg)
    # get intrinsic matrix
    sim.reset()
    intrinsic_matrix = camera_1.data.intrinsic_matrices[0].cpu().flatten().tolist()
    teardown(sim)
    # reinit the first camera
    sim, camera_cfg, dt = setup()
    camera_1 = Camera(cfg=camera_cfg)
    # initialize from intrinsic matrix
    intrinsic_camera_cfg = CameraCfg(
        height=HEIGHT,
        width=WIDTH,
        prim_path="/World/Camera_2",
        update_period=0,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix,
            width=WIDTH,
            height=HEIGHT,
            focal_length=24.0,
            focus_distance=400.0,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera_2 = Camera(cfg=intrinsic_camera_cfg)

    # play sim
    sim.reset()

    # update cameras
    camera_1.update(dt)
    camera_2.update(dt)

    # check image data
    torch.testing.assert_close(
        camera_1.data.output["distance_to_image_plane"],
        camera_2.data.output["distance_to_image_plane"],
        rtol=5e-3,
        atol=1e-4,
    )
    # check that both intrinsic matrices are the same
    torch.testing.assert_close(
        camera_1.data.intrinsic_matrices[0],
        camera_2.data.intrinsic_matrices[0],
        rtol=5e-3,
        atol=1e-4,
    )


def test_camera_set_world_poses(setup_sim_camera):
    """Test camera function to set specific world pose."""
    sim, camera_cfg, dt = setup_sim_camera
    # enable update latest camera pose
    camera_cfg.update_latest_camera_pose = True
    # init camera
    camera = Camera(camera_cfg)
    # play sim
    sim.reset()

    # convert to torch tensors
    position = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    orientation = torch.tensor([QUAT_WORLD], dtype=torch.float32, device=camera.device)
    # set new pose
    camera.set_world_poses(position.clone(), orientation.clone(), convention="world")

    # check if transform correctly set in output
    torch.testing.assert_close(camera.data.pos_w, position)
    torch.testing.assert_close(camera.data.quat_w_world, orientation)


def test_camera_set_world_poses_from_view(setup_sim_camera):
    """Test camera function to set specific world pose from view."""
    sim, camera_cfg, dt = setup_sim_camera
    # enable update latest camera pose
    camera_cfg.update_latest_camera_pose = True
    # init camera
    camera = Camera(camera_cfg)
    # play sim
    sim.reset()

    # convert to torch tensors
    eyes = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
    targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
    quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)
    # set new pose
    camera.set_world_poses_from_view(eyes.clone(), targets.clone())

    # check if transform correctly set in output
    torch.testing.assert_close(camera.data.pos_w, eyes)
    torch.testing.assert_close(camera.data.quat_w_ros, quat_ros_gt)


def test_intrinsic_matrix(setup_sim_camera):
    """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
    sim, camera_cfg, dt = setup_sim_camera
    # enable update latest camera pose
    camera_cfg.update_latest_camera_pose = True
    # init camera
    camera = Camera(camera_cfg)
    # play sim
    sim.reset()
    # Desired properties (obtained from realsense camera at 320x240 resolution)
    rs_intrinsic_matrix = [229.8, 0.0, 160.0, 0.0, 229.8, 120.0, 0.0, 0.0, 1.0]
    rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=camera.device).reshape(3, 3).unsqueeze(0)
    # Set matrix into simulator
    camera.set_intrinsic_matrices(rs_intrinsic_matrix.clone())

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        camera.update(dt)
        # Check that matrix is correct
        torch.testing.assert_close(rs_intrinsic_matrix[0, 0, 0], camera.data.intrinsic_matrices[0, 0, 0])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 1, 1], camera.data.intrinsic_matrices[0, 1, 1])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 0, 2], camera.data.intrinsic_matrices[0, 0, 2])
        torch.testing.assert_close(rs_intrinsic_matrix[0, 1, 2], camera.data.intrinsic_matrices[0, 1, 2])


def test_depth_clipping(setup_sim_camera):
    """Test depth clipping.

    .. note::

        This test is the same for all camera models to enforce the same clipping behavior.
    """
    # get camera cfgs
    sim, _, dt = setup_sim_camera
    camera_cfg_zero = CameraCfg(
        prim_path="/World/CameraZero",
        offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 6.0), rot=(0.362, 0.873, -0.302, -0.125), convention="ros"),
        spawn=sim_utils.PinholeCameraCfg().from_intrinsic_matrix(
            focal_length=38.0,
            intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
            height=540,
            width=960,
            clipping_range=(0.1, 10),
        ),
        height=540,
        width=960,
        data_types=["distance_to_image_plane", "distance_to_camera"],
        depth_clipping_behavior="zero",
    )
    camera_zero = Camera(camera_cfg_zero)

    camera_cfg_none = copy.deepcopy(camera_cfg_zero)
    camera_cfg_none.prim_path = "/World/CameraNone"
    camera_cfg_none.renderer_cfg.depth_clipping_behavior = "none"
    camera_none = Camera(camera_cfg_none)

    camera_cfg_max = copy.deepcopy(camera_cfg_zero)
    camera_cfg_max.prim_path = "/World/CameraMax"
    camera_cfg_max.renderer_cfg.depth_clipping_behavior = "max"
    camera_max = Camera(camera_cfg_max)

    # Play sim
    sim.reset()

    camera_zero.update(dt)
    camera_none.update(dt)
    camera_max.update(dt)

    # none clipping should contain inf values
    assert torch.isinf(camera_none.data.output["distance_to_camera"]).any()
    assert torch.isinf(camera_none.data.output["distance_to_image_plane"]).any()
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_camera"][~torch.isinf(camera_none.data.output["distance_to_camera"])].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert (
        camera_none.data.output["distance_to_image_plane"][
            ~torch.isinf(camera_none.data.output["distance_to_camera"])
        ].max()
        <= camera_cfg_zero.spawn.clipping_range[1]
    )

    # zero clipping should result in zero values
    assert torch.all(
        camera_zero.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])] == 0.0
    )
    assert torch.all(
        camera_zero.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == 0.0
    )
    assert (
        camera_zero.data.output["distance_to_camera"][camera_zero.data.output["distance_to_camera"] != 0.0].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert (
        camera_zero.data.output["distance_to_image_plane"][
            camera_zero.data.output["distance_to_image_plane"] != 0.0
        ].min()
        >= camera_cfg_zero.spawn.clipping_range[0]
    )
    assert camera_zero.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]

    # max clipping should result in max values
    assert torch.all(
        camera_max.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert torch.all(
        camera_max.data.output["distance_to_image_plane"][
            torch.isinf(camera_none.data.output["distance_to_image_plane"])
        ]
        == camera_cfg_zero.spawn.clipping_range[1]
    )
    assert camera_max.data.output["distance_to_camera"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_camera"].max() <= camera_cfg_zero.spawn.clipping_range[1]
    assert camera_max.data.output["distance_to_image_plane"].min() >= camera_cfg_zero.spawn.clipping_range[0]
    assert camera_max.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.spawn.clipping_range[1]


def test_camera_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = True
    camera_cfg.renderer_cfg.colorize_instance_segmentation = True
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = True
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation_fast"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["albedo"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8


def test_camera_resolution_no_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with no colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_segmentation = False
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = False
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()
    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_1c_shape
    assert output["instance_segmentation_fast"].shape == hw_1c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_1c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["albedo"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.int32
    assert output["instance_segmentation_fast"].dtype == torch.int32
    assert output["instance_id_segmentation_fast"].dtype == torch.int32


def test_camera_large_resolution_all_colorize(setup_sim_camera):
    """Test camera resolution is correctly set for all types with colorization enabled."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [
        "rgb",
        "rgba",
        "albedo",
        "depth",
        "distance_to_camera",
        "distance_to_image_plane",
        "normals",
        "motion_vectors",
        "semantic_segmentation",
        "instance_segmentation_fast",
        "instance_id_segmentation_fast",
    ]
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = True
    camera_cfg.renderer_cfg.colorize_instance_segmentation = True
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = True
    camera_cfg.width = 512
    camera_cfg.height = 512
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    hw_2c_shape = (1, camera_cfg.height, camera_cfg.width, 2)
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    assert output["rgba"].shape == hw_4c_shape
    assert output["albedo"].shape == hw_4c_shape
    assert output["depth"].shape == hw_1c_shape
    assert output["distance_to_camera"].shape == hw_1c_shape
    assert output["distance_to_image_plane"].shape == hw_1c_shape
    assert output["normals"].shape == hw_3c_shape
    assert output["motion_vectors"].shape == hw_2c_shape
    assert output["semantic_segmentation"].shape == hw_4c_shape
    assert output["instance_segmentation_fast"].shape == hw_4c_shape
    assert output["instance_id_segmentation_fast"].shape == hw_4c_shape

    # access image data and compare dtype
    output = camera.data.output
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["albedo"].dtype == torch.uint8
    assert output["depth"].dtype == torch.float
    assert output["distance_to_camera"].dtype == torch.float
    assert output["distance_to_image_plane"].dtype == torch.float
    assert output["normals"].dtype == torch.float
    assert output["motion_vectors"].dtype == torch.float
    assert output["semantic_segmentation"].dtype == torch.uint8
    assert output["instance_segmentation_fast"].dtype == torch.uint8
    assert output["instance_id_segmentation_fast"].dtype == torch.uint8


def test_camera_resolution_rgb_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGB only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgb"]
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgb"].shape == hw_3c_shape
    # access image data and compare dtype
    assert output["rgb"].dtype == torch.uint8


def test_camera_resolution_rgba_only(setup_sim_camera):
    """Test camera resolution is correctly set for RGBA only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["rgba"]
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["rgba"].shape == hw_4c_shape
    # access image data and compare dtype
    assert output["rgba"].dtype == torch.uint8


def test_camera_resolution_albedo_only(setup_sim_camera):
    """Test camera resolution is correctly set for albedo only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["albedo"]
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_4c_shape = (1, camera_cfg.height, camera_cfg.width, 4)
    # access image data and compare shapes
    output = camera.data.output
    assert output["albedo"].shape == hw_4c_shape
    # access image data and compare dtype
    assert output["albedo"].dtype == torch.uint8


@pytest.mark.parametrize(
    "data_type",
    ["simple_shading_constant_diffuse", "simple_shading_diffuse_mdl", "simple_shading_full_mdl"],
)
def test_camera_resolution_simple_shading_only(setup_sim_camera, data_type):
    """Test camera resolution is correctly set for simple shading only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = [data_type]
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_3c_shape = (1, camera_cfg.height, camera_cfg.width, 3)
    # access image data and compare shapes
    output = camera.data.output
    assert output[data_type].shape == hw_3c_shape
    # access image data and compare dtype
    assert output[data_type].dtype == torch.uint8


def test_camera_resolution_depth_only(setup_sim_camera):
    """Test camera resolution is correctly set for depth only."""
    # Add all types
    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg.data_types = ["depth"]
    # Create camera
    camera = Camera(camera_cfg)

    # Play sim
    sim.reset()

    camera.update(dt)

    # expected sizes
    hw_1c_shape = (1, camera_cfg.height, camera_cfg.width, 1)
    # access image data and compare shapes
    output = camera.data.output
    assert output["depth"].shape == hw_1c_shape
    # access image data and compare dtype
    assert output["depth"].dtype == torch.float


def test_sensor_print(setup_sim_camera):
    """Test sensor print is working correctly."""
    # Create sensor
    sim, camera_cfg, dt = setup_sim_camera
    sensor = Camera(cfg=camera_cfg)
    # Play sim
    sim.reset()
    # print info
    print(sensor)


def setup_with_device(device) -> tuple[sim_utils.SimulationContext, CameraCfg, float]:
    camera_cfg = CameraCfg(
        height=128,
        width=256,
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        prim_path="/World/Camera",
        update_period=0,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
    )
    sim_utils.create_new_stage()
    dt = 0.01
    sim_cfg = sim_utils.SimulationCfg(dt=dt, device=device)
    sim = sim_utils.SimulationContext(sim_cfg)
    _populate_scene()
    sim_utils.update_stage()
    return sim, camera_cfg, dt


@pytest.fixture(scope="function")
def setup_camera_device(device):
    """Fixture with explicit device parametrization for GPU/CPU testing."""
    sim, camera_cfg, dt = setup_with_device(device)
    yield sim, camera_cfg, dt
    teardown(sim)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_multi_regex_init(setup_camera_device, device):
    """Test multi-camera initialization with regex prim paths and content validation."""
    sim, camera_cfg, dt = setup_camera_device

    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = Camera(camera_cfg)

    sim.reset()

    assert camera.is_initialized
    assert camera._sensor_prims[1].GetPath().pathString == "/World/Origin_1/CameraSensor"
    assert isinstance(camera._sensor_prims[0], UsdGeom.Camera)

    assert camera.data.pos_w.shape == (num_cameras, 3)
    assert camera.data.quat_w_ros.shape == (num_cameras, 4)
    assert camera.data.quat_w_world.shape == (num_cameras, 4)
    assert camera.data.quat_w_opengl.shape == (num_cameras, 4)
    assert camera.data.intrinsic_matrices.shape == (num_cameras, 3, 3)
    assert camera.data.image_shape == (camera_cfg.height, camera_cfg.width)

    for _ in range(10):
        sim.step()
        camera.update(dt)
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_all_annotators(setup_camera_device, device):
    """Test all supported annotators produce correct shapes, dtypes, content, and info."""
    sim, camera_cfg, dt = setup_camera_device
    all_annotator_types = [
        "rgb",
        "rgba",
        "albedo",
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
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = all_annotator_types
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = Camera(camera_cfg)

    sim.reset()

    assert camera.is_initialized
    assert sorted(camera.data.output.keys()) == sorted(all_annotator_types)

    for _ in range(10):
        sim.step()
        camera.update(dt)
        for data_type, im_data in camera.data.output.items():
            if data_type in ["rgb", "normals"]:
                assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
            elif data_type in [
                "rgba",
                "albedo",
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

    output = camera.data.output
    info = camera.data.info
    assert output["rgb"].dtype == torch.uint8
    assert output["rgba"].dtype == torch.uint8
    assert output["albedo"].dtype == torch.uint8
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_segmentation_non_colorize(setup_camera_device, device):
    """Test segmentation outputs with colorization disabled produce correct dtypes and info."""
    sim, camera_cfg, dt = setup_camera_device
    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["semantic_segmentation", "instance_segmentation_fast", "instance_id_segmentation_fast"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera_cfg.renderer_cfg.colorize_semantic_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_segmentation = False
    camera_cfg.renderer_cfg.colorize_instance_id_segmentation = False
    camera = Camera(camera_cfg)

    sim.reset()

    for _ in range(5):
        sim.step()
        camera.update(dt)

    for seg_type in camera_cfg.data_types:
        assert camera.data.output[seg_type].shape == (num_cameras, camera_cfg.height, camera_cfg.width, 1)
        assert camera.data.output[seg_type].dtype == torch.int32
        assert isinstance(camera.data.info[seg_type], dict)

    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_normals_unit_length(setup_camera_device, device):
    """Test that normals output vectors have approximately unit length."""
    sim, camera_cfg, dt = setup_camera_device
    num_cameras = 9
    for i in range(num_cameras):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform")

    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["normals"]
    camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
    camera = Camera(camera_cfg)

    sim.reset()

    for _ in range(10):
        sim.step()
        camera.update(dt)
        im_data = camera.data.output["normals"]
        assert im_data.shape == (num_cameras, camera_cfg.height, camera_cfg.width, 3)
        for i in range(4):
            assert im_data[i].mean() > 0.0
        norms = torch.linalg.norm(im_data, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-9)

    assert camera.data.output["normals"].dtype == torch.float
    del camera


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_camera_data_types_ordering(setup_camera_device, device):
    """Test that requesting specific data types produces the expected output keys."""
    sim, camera_cfg, dt = setup_camera_device
    camera_cfg_distance = copy.deepcopy(camera_cfg)
    camera_cfg_distance.data_types = ["distance_to_camera"]
    camera_cfg_distance.prim_path = "/World/CameraDistance"
    camera_distance = Camera(camera_cfg_distance)

    camera_cfg_depth = copy.deepcopy(camera_cfg)
    camera_cfg_depth.data_types = ["depth"]
    camera_cfg_depth.prim_path = "/World/CameraDepth"
    camera_depth = Camera(camera_cfg_depth)

    camera_cfg_both = copy.deepcopy(camera_cfg)
    camera_cfg_both.data_types = ["distance_to_camera", "depth"]
    camera_cfg_both.prim_path = "/World/CameraBoth"
    camera_both = Camera(camera_cfg_both)

    sim.reset()

    assert camera_distance.is_initialized
    assert camera_depth.is_initialized
    assert camera_both.is_initialized
    assert list(camera_distance.data.output.keys()) == ["distance_to_camera"]
    assert list(camera_depth.data.output.keys()) == ["depth"]
    assert list(camera_both.data.output.keys()) == ["depth", "distance_to_camera"]

    del camera_distance
    del camera_depth
    del camera_both


@pytest.mark.parametrize("device", ["cuda:0"])
def test_camera_frame_offset(setup_camera_device, device):
    """Test that camera reflects scene color changes without frame-offset lag."""
    sim, camera_cfg, dt = setup_camera_device
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.height = 480
    camera_cfg.width = 480
    camera = Camera(camera_cfg)

    stage = sim_utils.get_current_stage()
    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(1, 1, 1)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    sim.reset()

    for _ in range(100):
        sim.step()
        camera.update(dt)

    image_before = camera.data.output["rgb"].clone() / 255.0

    for i in range(10):
        prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
        color = Gf.Vec3f(0, 0, 0)
        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

    sim.step()
    camera.update(dt)

    image_after = camera.data.output["rgb"].clone() / 255.0

    assert torch.abs(image_after - image_before).mean() > 0.01

    del camera


def test_camera_warns_once_on_unsupported_data_types(setup_sim_camera, caplog):
    """Test Camera warns once and drops data types its renderer cannot produce."""
    import logging

    from isaaclab.renderers import Renderer
    from isaaclab.renderers.base_renderer import BaseRenderer

    sim, camera_cfg, dt = setup_sim_camera
    camera_cfg = copy.deepcopy(camera_cfg)
    camera_cfg.data_types = ["rgba", "depth", "normals"]

    from isaaclab.sensors.camera.camera_data import RenderBufferKind, RenderBufferSpec

    class _PartialRenderer(BaseRenderer):
        """Publishes only ``rgba`` in its supported-output contract."""

        def __init__(self, cfg=None):
            self.cfg = cfg

        def supported_output_types(self):
            return {RenderBufferKind.RGBA: RenderBufferSpec(4, torch.uint8)}

        def prepare_stage(self, stage, num_envs):
            pass

        def create_render_data(self, sensor):
            return object()

        def set_outputs(self, render_data, output_data):
            pass

        def update_transforms(self):
            pass

        def update_camera(self, render_data, positions, orientations, intrinsics):
            pass

        def render(self, render_data):
            pass

        def read_output(self, render_data, camera_data):
            pass

        def cleanup(self, render_data):
            pass

    backend = Renderer._get_backend(camera_cfg.renderer_cfg)
    original = Renderer._registry.get(backend)
    Renderer._registry[backend] = _PartialRenderer
    try:
        camera = Camera(camera_cfg)
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="isaaclab.sensors.camera.camera"):
            sim.reset()
            # Step a few frames and confirm the warning is emitted once at init.
            for _ in range(3):
                sim.step()
                camera.update(dt)

        warning_records = [
            r for r in caplog.records if r.levelno == logging.WARNING and "does not support" in r.getMessage()
        ]
        assert len(warning_records) == 1, (
            f"Expected exactly one 'does not support' warning, got {len(warning_records)}:"
            f" {[r.getMessage() for r in warning_records]}"
        )
        msg = warning_records[0].getMessage()
        assert "_PartialRenderer" in msg
        assert "depth" in msg
        assert "normals" in msg
        assert "rgba" not in msg

        # Only the supported subset is in ``data.output``; the rest were dropped.
        assert set(camera.data.output.keys()) == {"rgba"}
        # ``data.info`` mirrors the ``data.output`` keys.
        assert set(camera.data.info.keys()) == {"rgba"}

        del camera
    finally:
        if original is not None:
            Renderer._registry[backend] = original
        else:
            Renderer._registry.pop(backend, None)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.isaacsim_ci
def test_camera_pose_update_reflected_in_render(setup_camera_device, device):
    """Camera pose changes via FrameView should be visible in rendered depth.

    Moves the camera close then far, renders depth, and verifies that the mean
    valid depth from the far position is significantly larger (>1.5×) than the
    close position.  This validates that Fabric-side pose writes (via
    PrepareForReuse) and USD writes are correctly propagated to the RTX
    renderer.
    """
    sim, _unused_cam_cfg, dt = setup_camera_device

    cam_cfg = CameraCfg(
        prim_path="/World/PoseTestCam",
        height=128,
        width=256,
        update_period=0,
        update_latest_camera_pose=True,
        data_types=["distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(cam_cfg)
    try:
        sim.reset()

        target = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
        max_range = cam_cfg.spawn.clipping_range[1]

        # -- close position --
        eyes_close = torch.tensor([[2.0, 2.0, 2.0]], dtype=torch.float32, device=camera.device)
        camera.set_world_poses_from_view(eyes_close, target)
        sim.step()
        camera.update(dt)
        depth_close = camera.data.output["distance_to_camera"].clone()

        # -- far position --
        eyes_far = torch.tensor([[8.0, 8.0, 8.0]], dtype=torch.float32, device=camera.device)
        camera.set_world_poses_from_view(eyes_far, target)
        sim.step()
        camera.update(dt)
        depth_far = camera.data.output["distance_to_camera"].clone()

        # -- validate --
        valid_close = depth_close[depth_close < max_range]
        valid_far = depth_far[depth_far < max_range]

        assert valid_close.numel() > 0, "No valid close-range depth pixels"
        assert valid_far.numel() > 0, "No valid far-range depth pixels"

        mean_close = valid_close.mean().item()
        mean_far = valid_far.mean().item()

        assert mean_far > mean_close * 1.5, (
            f"Far depth ({mean_far:.2f}) should be > 1.5× close depth ({mean_close:.2f}). "
            "Camera pose change may not be reaching the renderer."
        )
    finally:
        del camera


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
    for i in range(10):
        # sample random position
        position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
        position *= np.asarray([1.5, 1.5, 0.5])
        # create prim
        prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
        prim = sim_utils.create_prim(
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
        # add rigid body and collision properties using Isaac Lab schemas
        prim_path = f"/World/Objects/Obj_{i:02d}"
        sim_utils.define_rigid_body_properties(prim_path, sim_utils.RigidBodyPropertiesCfg())
        sim_utils.define_mass_properties(prim_path, sim_utils.MassPropertiesCfg(mass=5.0))
        sim_utils.define_collision_properties(prim_path, sim_utils.CollisionPropertiesCfg())
