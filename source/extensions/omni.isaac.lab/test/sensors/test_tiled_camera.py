# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import numpy as np
import random
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from pxr import Gf, UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors.camera import TiledCamera, TiledCameraCfg
from omni.isaac.lab.utils.timer import Timer


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
            data_types=["rgb", "depth"],
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
            for im_type, im_data in camera.data.output.to_dict().items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (1, self.camera_cfg.height, self.camera_cfg.width, 3))
                else:
                    self.assertEqual(im_data.shape, (1, self.camera_cfg.height, self.camera_cfg.width, 1))
                self.assertGreater(im_data.mean().item(), 0.0)
        del camera

    def test_multi_camera_init(self):
        """Test multi-camera initialization."""

        prim_utils.create_prim("/World/Origin_00", "Xform")
        prim_utils.create_prim("/World/Origin_01", "Xform")

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
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_01/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (2, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (2, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (2, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_type, im_data in camera.data.output.to_dict().items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (2, self.camera_cfg.height, self.camera_cfg.width, 3))
                else:
                    self.assertEqual(im_data.shape, (2, self.camera_cfg.height, self.camera_cfg.width, 1))
                self.assertGreater(im_data[0].mean().item(), 0.0)
                self.assertGreater(im_data[1].mean().item(), 0.0)
        del camera

    def test_rgb_only_camera(self):
        """Test initialization with only RGB."""

        prim_utils.create_prim("/World/Origin_00", "Xform")
        prim_utils.create_prim("/World/Origin_01", "Xform")

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
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_01/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["rgb"])
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (2, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (2, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (2, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.to_dict().items():
                self.assertEqual(im_data.shape, (2, self.camera_cfg.height, self.camera_cfg.width, 3))
                self.assertGreater(im_data[0].mean().item(), 0.0)
                self.assertGreater(im_data[1].mean().item(), 0.0)
        del camera

    def test_depth_only_camera(self):
        """Test initialization with only depth."""

        prim_utils.create_prim("/World/Origin_00", "Xform")
        prim_utils.create_prim("/World/Origin_01", "Xform")

        # Create camera
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.data_types = ["depth"]
        camera_cfg.prim_path = "/World/Origin_.*/CameraSensor"
        camera = TiledCamera(camera_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera.is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertEqual(camera._sensor_prims[1].GetPath().pathString, "/World/Origin_01/CameraSensor")
        self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
        self.assertListEqual(list(camera.data.output.keys()), ["depth"])

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Check buffers that exists and have correct shapes
        self.assertEqual(camera.data.pos_w.shape, (2, 3))
        self.assertEqual(camera.data.quat_w_ros.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_world.shape, (2, 4))
        self.assertEqual(camera.data.quat_w_opengl.shape, (2, 4))
        self.assertEqual(camera.data.intrinsic_matrices.shape, (2, 3, 3))
        self.assertEqual(camera.data.image_shape, (self.camera_cfg.height, self.camera_cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for _, im_data in camera.data.output.to_dict().items():
                self.assertEqual(im_data.shape, (2, self.camera_cfg.height, self.camera_cfg.width, 1))
                self.assertGreater(im_data[0].mean().item(), 0.0)
                self.assertGreater(im_data[1].mean().item(), 0.0)
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
            for im_type, im_data in camera.data.output.to_dict().items():
                if im_type == "rgb":
                    self.assertEqual(im_data.shape, (1, camera_cfg.height, camera_cfg.width, 3))
                else:
                    self.assertEqual(im_data.shape, (1, camera_cfg.height, camera_cfg.width, 1))
                self.assertGreater(im_data.mean().item(), 0.0)
        del camera

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
            GeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
            RigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)


if __name__ == "__main__":
    run_tests()
