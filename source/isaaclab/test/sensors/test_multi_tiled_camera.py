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
from isaaclab.sensors.camera import TiledCamera, TiledCameraCfg


class TestMultiTiledCamera(unittest.TestCase):
    """Test for USD multi tiled Camera sensor."""

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

    def test_multi_tiled_camera_init(self):
        """Test initialization of multiple tiled cameras."""

        num_tiled_cameras = 3
        num_cameras_per_tiled_camera = 7

        tiled_cameras = []
        for i in range(num_tiled_cameras):
            for j in range(num_cameras_per_tiled_camera):
                prim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")

            # Create camera
            camera_cfg = copy.deepcopy(self.camera_cfg)
            camera_cfg.prim_path = f"/World/Origin_{i}.*/CameraSensor"
            camera = TiledCamera(camera_cfg)
            tiled_cameras.append(camera)

            # Check simulation parameter is set correctly
            self.assertTrue(self.sim.has_rtx_sensors())

        # Play sim
        self.sim.reset()

        for i, camera in enumerate(tiled_cameras):
            # Check if camera is initialized
            self.assertTrue(camera.is_initialized)
            # Check if camera prim is set correctly and that it is a camera prim
            self.assertEqual(camera._sensor_prims[1].GetPath().pathString, f"/World/Origin_{i}_1/CameraSensor")
            self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        for camera in tiled_cameras:
            # Check buffers that exists and have correct shapes
            self.assertEqual(camera.data.pos_w.shape, (num_cameras_per_tiled_camera, 3))
            self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_world.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras_per_tiled_camera, 3, 3))
            self.assertEqual(camera.data.image_shape, (camera.cfg.height, camera.cfg.width))

        # Simulate physics
        for _ in range(10):
            # Initialize data arrays
            rgbs = []
            distances = []

            # perform rendering
            self.sim.step()
            for i, camera in enumerate(tiled_cameras):
                # update camera
                camera.update(self.dt)
                # check image data
                for data_type, im_data in camera.data.output.items():
                    if data_type == "rgb":
                        im_data = im_data.clone() / 255.0
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 3)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater((im_data[j]).mean().item(), 0.0)
                        rgbs.append(im_data)
                    elif data_type == "distance_to_camera":
                        im_data = im_data.clone()
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 1)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater(im_data[j].mean().item(), 0.0)
                        distances.append(im_data)

            # Check data from tiled cameras are consistent, assumes >1 tiled cameras
            for i in range(1, num_tiled_cameras):
                self.assertLess(torch.abs(rgbs[0] - rgbs[i]).mean(), 0.05)  # images of same color should be below 0.001
                self.assertLess(
                    torch.abs(distances[0] - distances[i]).mean(), 0.01
                )  # distances of same scene should be 0

        for camera in tiled_cameras:
            del camera

    def test_all_annotators_multi_tiled_camera(self):
        """Test initialization of multiple tiled cameras with all supported annotators."""

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

        num_tiled_cameras = 2
        num_cameras_per_tiled_camera = 9

        tiled_cameras = []
        for i in range(num_tiled_cameras):
            for j in range(num_cameras_per_tiled_camera):
                prim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")

            # Create camera
            camera_cfg = copy.deepcopy(self.camera_cfg)
            camera_cfg.data_types = all_annotator_types
            camera_cfg.prim_path = f"/World/Origin_{i}.*/CameraSensor"
            camera = TiledCamera(camera_cfg)
            tiled_cameras.append(camera)

            # Check simulation parameter is set correctly
            self.assertTrue(self.sim.has_rtx_sensors())

        # Play sim
        self.sim.reset()

        for i, camera in enumerate(tiled_cameras):
            # Check if camera is initialized
            self.assertTrue(camera.is_initialized)
            # Check if camera prim is set correctly and that it is a camera prim
            self.assertEqual(camera._sensor_prims[1].GetPath().pathString, f"/World/Origin_{i}_1/CameraSensor")
            self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)
            self.assertListEqual(sorted(camera.data.output.keys()), sorted(all_annotator_types))

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        for camera in tiled_cameras:
            # Check buffers that exists and have correct shapes
            self.assertEqual(camera.data.pos_w.shape, (num_cameras_per_tiled_camera, 3))
            self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_world.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras_per_tiled_camera, 3, 3))
            self.assertEqual(camera.data.image_shape, (camera.cfg.height, camera.cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            for i, camera in enumerate(tiled_cameras):
                # update camera
                camera.update(self.dt)
                # check image data
                for data_type, im_data in camera.data.output.items():
                    if data_type in ["rgb", "normals"]:
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 3)
                        )
                    elif data_type in [
                        "rgba",
                        "semantic_segmentation",
                        "instance_segmentation_fast",
                        "instance_id_segmentation_fast",
                    ]:
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 4)
                        )
                        for i in range(num_cameras_per_tiled_camera):
                            self.assertGreater((im_data[i] / 255.0).mean().item(), 0.0)
                    elif data_type in ["motion_vectors"]:
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 2)
                        )
                        for i in range(num_cameras_per_tiled_camera):
                            self.assertNotEqual(im_data[i].mean().item(), 0.0)
                    elif data_type in ["depth", "distance_to_camera", "distance_to_image_plane"]:
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 1)
                        )
                        for i in range(num_cameras_per_tiled_camera):
                            self.assertGreater(im_data[i].mean().item(), 0.0)

        for camera in tiled_cameras:
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

        for camera in tiled_cameras:
            del camera

    def test_different_resolution_multi_tiled_camera(self):
        """Test multiple tiled cameras with different resolutions."""

        num_tiled_cameras = 2
        num_cameras_per_tiled_camera = 6

        tiled_cameras = []
        resolutions = [(4, 4), (16, 16), (64, 64), (512, 512), (23, 765), (1001, 1)]
        for i in range(num_tiled_cameras):
            for j in range(num_cameras_per_tiled_camera):
                prim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")

            # Create camera
            camera_cfg = copy.deepcopy(self.camera_cfg)
            camera_cfg.prim_path = f"/World/Origin_{i}.*/CameraSensor"
            camera_cfg.height, camera_cfg.width = resolutions[i]
            camera = TiledCamera(camera_cfg)
            tiled_cameras.append(camera)

            # Check simulation parameter is set correctly
            self.assertTrue(self.sim.has_rtx_sensors())

        # Play sim
        self.sim.reset()

        for i, camera in enumerate(tiled_cameras):
            # Check if camera is initialized
            self.assertTrue(camera.is_initialized)
            # Check if camera prim is set correctly and that it is a camera prim
            self.assertEqual(camera._sensor_prims[1].GetPath().pathString, f"/World/Origin_{i}_1/CameraSensor")
            self.assertIsInstance(camera._sensor_prims[0], UsdGeom.Camera)

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        for camera in tiled_cameras:
            # Check buffers that exists and have correct shapes
            self.assertEqual(camera.data.pos_w.shape, (num_cameras_per_tiled_camera, 3))
            self.assertEqual(camera.data.quat_w_ros.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_world.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.quat_w_opengl.shape, (num_cameras_per_tiled_camera, 4))
            self.assertEqual(camera.data.intrinsic_matrices.shape, (num_cameras_per_tiled_camera, 3, 3))
            self.assertEqual(camera.data.image_shape, (camera.cfg.height, camera.cfg.width))

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            for i, camera in enumerate(tiled_cameras):
                # update camera
                camera.update(self.dt)
                # check image data
                for data_type, im_data in camera.data.output.items():
                    if data_type == "rgb":
                        im_data = im_data.clone() / 255.0
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 3)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater((im_data[j]).mean().item(), 0.0)
                    elif data_type == "distance_to_camera":
                        im_data = im_data.clone()
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 1)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater(im_data[j].mean().item(), 0.0)

        for camera in tiled_cameras:
            del camera

    def test_frame_offset_multi_tiled_camera(self):
        """Test frame offset issue with multiple tiled cameras"""

        num_tiled_cameras = 4
        num_cameras_per_tiled_camera = 4

        tiled_cameras = []
        for i in range(num_tiled_cameras):
            for j in range(num_cameras_per_tiled_camera):
                prim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")

            # Create camera
            camera_cfg = copy.deepcopy(self.camera_cfg)
            camera_cfg.prim_path = f"/World/Origin_{i}.*/CameraSensor"
            camera = TiledCamera(camera_cfg)
            tiled_cameras.append(camera)

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
            # update cameras
            for camera in tiled_cameras:
                camera.update(self.dt)

        # collect image data
        image_befores = [camera.data.output["rgb"].clone() / 255.0 for camera in tiled_cameras]

        # update scene
        for i in range(10):
            prim = stage.GetPrimAtPath(f"/World/Objects/Obj_{i:02d}")
            color = Gf.Vec3f(0, 0, 0)
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([color])

        # update rendering
        self.sim.step()

        # update cameras
        for camera in tiled_cameras:
            camera.update(self.dt)

        # make sure the image is different
        image_afters = [camera.data.output["rgb"].clone() / 255.0 for camera in tiled_cameras]

        # check difference is above threshold
        for i in range(num_tiled_cameras):
            image_before = image_befores[i]
            image_after = image_afters[i]
            self.assertGreater(
                torch.abs(image_after - image_before).mean(), 0.05
            )  # images of same color should be below 0.001

        for camera in tiled_cameras:
            del camera

    def test_frame_different_poses_multi_tiled_camera(self):
        """Test multiple tiled cameras placed at different poses render different images."""

        num_tiled_cameras = 3
        num_cameras_per_tiled_camera = 4
        positions = [(0.0, 0.0, 4.0), (0.0, 0.0, 4.0), (0.0, 0.0, 3.0)]
        rotations = [(0.0, 0.0, 1.0, 0.0), (1.0, 0.0, 1.0, 0.0), (0.0, 0.0, 1.0, 0.0)]

        tiled_cameras = []
        for i in range(num_tiled_cameras):
            for j in range(num_cameras_per_tiled_camera):
                prim_utils.create_prim(f"/World/Origin_{i}_{j}", "Xform")

            # Create camera
            camera_cfg = copy.deepcopy(self.camera_cfg)
            camera_cfg.prim_path = f"/World/Origin_{i}.*/CameraSensor"
            camera_cfg.offset = TiledCameraCfg.OffsetCfg(pos=positions[i], rot=rotations[i], convention="ros")
            camera = TiledCamera(camera_cfg)
            tiled_cameras.append(camera)

        # Play sim
        self.sim.reset()

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()

        # Simulate physics
        for _ in range(10):
            # Initialize data arrays
            rgbs = []
            distances = []

            # perform rendering
            self.sim.step()
            for i, camera in enumerate(tiled_cameras):
                # update camera
                camera.update(self.dt)
                # check image data
                for data_type, im_data in camera.data.output.items():
                    if data_type == "rgb":
                        im_data = im_data.clone() / 255.0
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 3)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater((im_data[j]).mean().item(), 0.0)
                        rgbs.append(im_data)
                    elif data_type == "distance_to_camera":
                        im_data = im_data.clone()
                        self.assertEqual(
                            im_data.shape, (num_cameras_per_tiled_camera, camera.cfg.height, camera.cfg.width, 1)
                        )
                        for j in range(num_cameras_per_tiled_camera):
                            self.assertGreater(im_data[j].mean().item(), 0.0)
                        distances.append(im_data)

            # Check data from tiled cameras are different, assumes >1 tiled cameras
            for i in range(1, num_tiled_cameras):
                self.assertGreater(
                    torch.abs(rgbs[0] - rgbs[i]).mean(), 0.05
                )  # images of same color should be below 0.001
                self.assertGreater(
                    torch.abs(distances[0] - distances[i]).mean(), 0.01
                )  # distances of same scene should be 0

        for camera in tiled_cameras:
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
            SingleGeometryPrim(f"/World/Objects/Obj_{i:02d}", collision=True)
            SingleRigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)


if __name__ == "__main__":  #
    run_tests()
