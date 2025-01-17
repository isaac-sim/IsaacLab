# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import numpy as np
import random
import torch
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
from isaacsim.core.prims import SingleGeometryPrim, SingleRigidPrim
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg, TiledCamera, TiledCameraCfg
from isaaclab.utils.timer import Timer


class TestTiledCamera(unittest.TestCase):
    """Test for USD tiled Camera sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        self.camera_cfg = TiledCameraCfg(
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
        self.dt = 0.01
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt)
        self.sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
        # populate scene
        self._populate_scene()
        # load stage
        stage_utils.update_stage()

    def tearDown(self):
        """Stops simulator after each test."""
        # close all the opened viewport from before.
        rep.vp_manager.destroy_hydra_textures("Replicator")
        # stop simulation
        # note: cannot use self.sim.stop() since it does one render step after stopping!! This doesn't make sense :(
        self.sim._timeline.stop()
        # clear the stage
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_single_camera_init(self):
        """Test single camera initialization."""
        # Create camera
        camera = TiledCamera(self.camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[0].GetPath().pathString, self.camera_cfg.prim_path)
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (1, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (1, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (1, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (1, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (1, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_type, im_data in camera.data.output.items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (1, self.camera_cfg.height, self.camera_cfg.width, 3))
                    self.assertGreater((im_data / 255.0).mean().item(), 0.0)
                elif im_type == "distance_to_camera":
                    self.assertEqual(im_data.shape, (1, self.camera_cfg.height, self.camera_cfg.width, 1))
                    self.assertGreater(im_data.mean().item(), 0.0)
        del camera

    def test_depth_clipping_max(self):
        """Test depth max clipping."""
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
        self.sim.reset()

        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        camera.update(self.dt)

        self.assertTrue(len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0)
        self.assertTrue(camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0])
        self.assertTrue(camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1])

        del camera

    def test_depth_clipping_none(self):
        """Test depth none clipping."""
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
        self.sim.reset()

        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        camera.update(self.dt)

        self.assertTrue(len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) > 0)
        self.assertTrue(camera.data.output["depth"].min() >= camera_cfg.spawn.clipping_range[0])
        self.assertTrue(
            camera.data.output["depth"][~torch.isinf(camera.data.output["depth"])].max()
            <= camera_cfg.spawn.clipping_range[1]
        )

        del camera

    def test_depth_clipping_zero(self):
        """Test depth zero clipping."""
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
        self.sim.reset()

        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        camera.update(self.dt)

        self.assertTrue(len(camera.data.output["depth"][torch.isinf(camera.data.output["depth"])]) == 0)
        self.assertTrue(camera.data.output["depth"].min() == 0.0)
        self.assertTrue(camera.data.output["depth"].max() <= camera_cfg.spawn.clipping_range[1])

        del camera

    def test_multi_camera_init(self):
        """Test multi-camera initialization."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_type, im_data in camera.data.output.items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 3))
                    for i in range(4):
                        self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
                elif im_type == "distance_to_camera":
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                    for i in range(4):
                        self.assertGreater(im_data[i].mean().item(), 0.0)
        del camera

    def test_rgb_only_camera(self):
        """Test initialization with only RGB."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["rgb"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(sorted(camera.data.output.keys()), sorted(["rgb", "rgba"]))
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            im_data = camera.data.output["rgb"]
            self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 3))
            for i in range(4):
                self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["rgb"].dtype, torch.uint8)
        del camera

    def test_data_types(self):
        """Test single camera initialization."""
        # Create camera
        camera_cfg_distance = copy.deepcopy(self.camera_cfg)
        camera_cfg_distance.data_types = ["distance_to_camera"]
        camera_cfg_distance.prim_path = "/World/CameraDistance"
        camera_distance = TiledCamera(camera_cfg_distance)
        camera_cfg_depth = copy.deepcopy(self.camera_cfg)
        camera_cfg_depth.data_types = ["depth"]
        camera_cfg_depth.prim_path = "/World/CameraDepth"
        camera_depth = TiledCamera(camera_cfg_depth)
        camera_cfg_both = copy.deepcopy(self.camera_cfg)
        camera_cfg_both.data_types = ["distance_to_camera", "depth"]
        camera_cfg_both.prim_path = "/World/CameraBoth"
        camera_both = TiledCamera(camera_cfg_both)
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera_distance.is_initialized)
        self.assertTrue(camera_depth.is_initialized)
        self.assertTrue(camera_both.is_initialized)
        self.assertListEqual(list(camera_distance.data.output.keys()), ["distance_to_camera"])
        self.assertListEqual(list(camera_depth.data.output.keys()), ["depth"])
        self.assertListEqual(list(camera_both.data.output.keys()), ["depth", "distance_to_camera"])

        del camera_distance, camera_depth, camera_both

    def test_depth_only_camera(self):
        """Test initialization with only depth."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["distance_to_camera"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(sorted(camera.data.output.keys()), ["distance_to_camera"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            im_data = camera.data.output["distance_to_camera"]
            self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
            for i in range(4):
                self.assertGreater((im_data[i]).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["distance_to_camera"].dtype, torch.float)
        del camera

    def test_rgba_only_camera(self):
        """Test initialization with only RGBA."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["rgba"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["rgba"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                for i in range(4):
                    self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["rgba"].dtype, torch.uint8)
        del camera

    def test_distance_to_camera_only_camera(self):
        """Test initialization with only distance_to_camera."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["distance_to_camera"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["distance_to_camera"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                for i in range(4):
                    self.assertGreater((im_data[i]).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["distance_to_camera"].dtype, torch.float)
        del camera

    def test_distance_to_image_plane_only_camera(self):
        """Test initialization with only distance_to_image_plane."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["distance_to_image_plane"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["distance_to_image_plane"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                for i in range(4):
                    self.assertGreater((im_data[i]).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["distance_to_image_plane"].dtype, torch.float)
        del camera

    def test_normals_only_camera(self):
        """Test initialization with only normals."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["normals"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["normals"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 3))
                for i in range(4):
                    self.assertGreater((im_data[i]).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["normals"].dtype, torch.float)
        del camera

    def test_motion_vectors_only_camera(self):
        """Test initialization with only motion_vectors."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["motion_vectors"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["motion_vectors"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 2))
                for i in range(4):
                    self.assertNotEqual((im_data[i]).mean().item(), 0.0)
        # Check data type of image
        self.assertEqual(camera.data.output["motion_vectors"].dtype, torch.float)
        del camera

    def test_semantic_segmentation_colorize_only_camera(self):
        """Test initialization with only semantic_segmentation."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["semantic_segmentation"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["semantic_segmentation"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                for i in range(4):
                    self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["semantic_segmentation"].dtype, torch.uint8)
        self.assertEqual(type(camera.data.info["semantic_segmentation"]), dict)
        del camera

    def test_instance_segmentation_fast_colorize_only_camera(self):
        """Test initialization with only instance_segmentation_fast."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["instance_segmentation_fast"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["instance_segmentation_fast"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                for i in range(num_cameras):
                    self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["instance_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(type(camera.data.info["instance_segmentation_fast"]), dict)
        del camera

    def test_instance_id_segmentation_fast_colorize_only_camera(self):
        """Test initialization with only instance_id_segmentation_fast."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["instance_id_segmentation_fast"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["instance_id_segmentation_fast"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                for i in range(num_cameras):
                    self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["instance_id_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(type(camera.data.info["instance_id_segmentation_fast"]), dict)
        del camera

    def test_semantic_segmentation_non_colorize_only_camera(self):
        """Test initialization with only semantic_segmentation."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["semantic_segmentation"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera_cfg.colorize_semantic_segmentation = False
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["semantic_segmentation"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                for i in range(num_cameras):
                    self.assertGreater(im_data[i].to(dtype=float).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["semantic_segmentation"].dtype, torch.int32)
        self.assertEqual(type(camera.data.info["semantic_segmentation"]), dict)

        del camera

    def test_instance_segmentation_fast_non_colorize_only_camera(self):
        """Test initialization with only instance_segmentation_fast."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["instance_segmentation_fast"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera_cfg.colorize_instance_segmentation = False
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["instance_segmentation_fast"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                for i in range(num_cameras):
                    self.assertGreater(im_data[i].to(dtype=float).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["instance_segmentation_fast"].dtype, torch.int32)
        self.assertEqual(type(camera.data.info["instance_segmentation_fast"]), dict)
        del camera

    def test_instance_id_segmentation_fast_non_colorize_only_camera(self):
        """Test initialization with only instance_id_segmentation_fast."""

        num_cameras = 9
        for i in range(num_cameras):
            prim_utils.create_prim(f"/World/Origin_{i}", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["instance_id_segmentation_fast"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera_cfg.colorize_instance_id_segmentation = False
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["instance_id_segmentation_fast"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.items():
                self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                for i in range(num_cameras):
                    self.assertGreater(im_data[i].to(dtype=float).mean().item(), 0.0)
        # Check data type of image and info
        self.assertEqual(camera.data.output["instance_id_segmentation_fast"].dtype, torch.int32)
        self.assertEqual(type(camera.data.info["instance_id_segmentation_fast"]), dict)
        del camera

    def test_all_annotators_camera(self):
        """Test initialization with all supported annotators."""
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
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = all_annotator_types
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(sorted(camera.data.output.keys()), sorted(all_annotator_types))

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type in ["rgb", "normals"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 3))
                elif data_type in [
                    "rgba",
                    "semantic_segmentation",
                    "instance_segmentation_fast",
                    "instance_id_segmentation_fast",
                ]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                    for i in range(num_cameras):
                        self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
                elif data_type in ["motion_vectors"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 2))
                    for i in range(num_cameras):
                        self.assertNotEqual(im_data[i].mean().item(), 0.0)
                elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                    for i in range(num_cameras):
                        self.assertGreater(im_data[i].mean().item(), 0.0)

        # access image data and compare dtype
        output = camera.data.output
        info = camera.data.info
        self.assertEqual(output["rgb"].dtype, torch.uint8)
        self.assertEqual(output["rgba"].dtype, torch.uint8)
        self.assertEqual(output["depth"].dtype, torch.float)
        self.assertEqual(output["distance_to_camera"].dtype, torch.float)
        self.assertEqual(output["distance_to_image_plane"].dtype, torch.float)
        self.assertEqual(output["normals"].dtype, torch.float)
        self.assertEqual(output["motion_vectors"].dtype, torch.float)
        self.assertEqual(output["semantic_segmentation"].dtype, torch.uint8)
        self.assertEqual(output["instance_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(output["instance_id_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(type(info["semantic_segmentation"]), dict)
        self.assertEqual(type(info["instance_segmentation_fast"]), dict)
        self.assertEqual(type(info["instance_id_segmentation_fast"]), dict)

        del camera

    def test_all_annotators_low_resolution_camera(self):
        """Test initialization with all supported annotators."""
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
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.height = 40
        camera_cfg.width = 40
        camera_cfg.data_types = all_annotator_types
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(sorted(camera.data.output.keys()), sorted(all_annotator_types))

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (camera_cfg.height, camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type in ["rgb", "normals"]:
                    self.assertEqual(im_data.shape, (num_cameras, camera_cfg.height, camera_cfg.width, 3))
                elif data_type in [
                    "rgba",
                    "semantic_segmentation",
                    "instance_segmentation_fast",
                    "instance_id_segmentation_fast",
                ]:
                    self.assertEqual(im_data.shape, (num_cameras, camera_cfg.height, camera_cfg.width, 4))
                    for i in range(num_cameras):
                        self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
                elif data_type in ["motion_vectors"]:
                    self.assertEqual(im_data.shape, (num_cameras, camera_cfg.height, camera_cfg.width, 2))
                    for i in range(num_cameras):
                        self.assertGreater(im_data[i].mean().item(), 0.0)
                elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                    self.assertEqual(im_data.shape, (num_cameras, camera_cfg.height, camera_cfg.width, 1))
                    for i in range(num_cameras):
                        self.assertGreater(im_data[i].mean().item(), 0.0)

        # access image data and compare dtype
        output = camera.data.output
        info = camera.data.info
        self.assertEqual(output["rgb"].dtype, torch.uint8)
        self.assertEqual(output["rgba"].dtype, torch.uint8)
        self.assertEqual(output["depth"].dtype, torch.float)
        self.assertEqual(output["distance_to_camera"].dtype, torch.float)
        self.assertEqual(output["distance_to_image_plane"].dtype, torch.float)
        self.assertEqual(output["normals"].dtype, torch.float)
        self.assertEqual(output["motion_vectors"].dtype, torch.float)
        self.assertEqual(output["semantic_segmentation"].dtype, torch.uint8)
        self.assertEqual(output["instance_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(output["instance_id_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(type(info["semantic_segmentation"]), dict)
        self.assertEqual(type(info["instance_segmentation_fast"]), dict)
        self.assertEqual(type(info["instance_id_segmentation_fast"]), dict)

        del camera

    def test_all_annotators_non_perfect_square_number_camera(self):
        """Test initialization with all supported annotators."""
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
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = all_annotator_types
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_1/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(sorted(camera.data.output.keys()), sorted(all_annotator_types))

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (num_cameras, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (num_cameras, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for data_type, im_data in camera.data.output.items():
                if data_type in ["rgb", "normals"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 3))
                elif data_type in [
                    "rgba",
                    "semantic_segmentation",
                    "instance_segmentation_fast",
                    "instance_id_segmentation_fast",
                ]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 4))
                    for i in range(num_cameras):
                        self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
                elif data_type in ["motion_vectors"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 2))
                    for i in range(num_cameras):
                        self.assertNotEqual(im_data[i].mean().item(), 0.0)
                elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                    self.assertEqual(im_data.shape, (num_cameras, self.camera_cfg.height, self.camera_cfg.width, 1))
                    for i in range(num_cameras):
                        self.assertGreater(im_data[i].mean().item(), 0.0)

        # access image data and compare dtype
        output = camera.data.output
        info = camera.data.info
        self.assertEqual(output["rgb"].dtype, torch.uint8)
        self.assertEqual(output["rgba"].dtype, torch.uint8)
        self.assertEqual(output["depth"].dtype, torch.float)
        self.assertEqual(output["distance_to_camera"].dtype, torch.float)
        self.assertEqual(output["distance_to_image_plane"].dtype, torch.float)
        self.assertEqual(output["normals"].dtype, torch.float)
        self.assertEqual(output["motion_vectors"].dtype, torch.float)
        self.assertEqual(output["semantic_segmentation"].dtype, torch.uint8)
        self.assertEqual(output["instance_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(output["instance_id_segmentation_fast"].dtype, torch.uint8)
        self.assertEqual(type(info["semantic_segmentation"]), dict)
        self.assertEqual(type(info["instance_segmentation_fast"]), dict)
        self.assertEqual(type(info["instance_id_segmentation_fast"]), dict)

        del camera

    def test_throughput(self):
        """Test tiled camera throughput."""

        # create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.height = 480
        camera_cfg.width = 640
        camera = TiledCamera(camera_cfg)

        # Play simulator
        self.sim.reset()

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Simulate physics
        for _ in range(5):
            # perform rendering
            self.sim.step()
            # update camera
            with Timer(f"Time taken for updating camera with shape {camera.image_shape}"):
                camera.update(self.dt)
            # Check image data
            for im_type, im_data in camera.data.output.items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (1, camera_cfg.height, camera_cfg.width, 3))
                    self.assertGreater((im_data / 255.0).mean().item(), 0.0)
                elif im_type == "distance_to_camera":
                    self.assertEqual(im_data.shape, (1, camera_cfg.height, camera_cfg.width, 1))
                    self.assertGreater(im_data.mean().item(), 0.0)
        del camera

    def test_output_equal_to_usd_camera_intrinsics(self):
        """
        Test that the output of the ray caster camera and the usd camera are the same when both are
        initialized with the same intrinsic matrix.
        """

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
                focal_length=38.0,
                # clipping_range=(0.01, 20),
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
                focal_length=38.0,
                # clipping_range=(0.01, 20),
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
        self.sim.reset()
        self.sim.play()

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_tiled.update(self.dt)

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

    def test_sensor_print(self):
        """Test sensor print is working correctly."""
        # Create sensor
        sensor = TiledCamera(cfg=self.camera_cfg)
        # Play sim
        self.sim.reset()
        # print info
        print(sensor)

    def test_frame_offset_small_resolution(self):
        """Test frame offset issue with small resolution camera."""
        # Create sensor
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.height = 80
        camera_cfg.width = 80
        tiled_camera = TiledCamera(camera_cfg)
        # play sim
        self.sim.reset()
        # simulate some steps first to make sure objects are settled
        for i in range(100):
            # step simulation
            self.sim.step()
            # update camera
            tiled_camera.update(self.dt)
        # collect image data
        image_before = tiled_camera.data.output["rgb"].clone() / 255.0

        # update scene
        stage = stage_utils.get_current_stage()
        for i in range(10):
            prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
            color = Gf.Vec3f(0, 0, 0)
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

        # update rendering
        self.sim.step()
        # update camera
        tiled_camera.update(self.dt)

        # make sure the image is different
        image_after = tiled_camera.data.output["rgb"].clone() / 255.0

        # check difference is above threshold
        self.assertGreater(
            torch.abs(image_after - image_before).mean(), 0.05
        )  # images of same color should be below 0.001

    def test_frame_offset_large_resolution(self):
        """Test frame offset issue with large resolution camera."""
        # Create sensor
        camera_cfg = copy.deepcopy(self.camera_cfg)
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
        self.sim.reset()
        # simulate some steps first to make sure objects are settled
        for i in range(100):
            # step simulation
            self.sim.step()
            # update camera
            tiled_camera.update(self.dt)
        # collect image data
        image_before = tiled_camera.data.output["rgb"].clone() / 255.0

        # update scene
        for i in range(10):
            prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
            color = Gf.Vec3f(0, 0, 0)
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

        # update rendering
        self.sim.step()
        # update camera
        tiled_camera.update(self.dt)

        # make sure the image is different
        image_after = tiled_camera.data.output["rgb"].clone() / 255.0

        # check difference is above threshold
        self.assertGreater(
            torch.abs(image_after - image_before).mean(), 0.05
        )  # images of same color should be below 0.001

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


if __name__ == "__main__":
    run_tests()
