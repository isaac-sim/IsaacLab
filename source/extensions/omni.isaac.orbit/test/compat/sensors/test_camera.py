# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import logging

from omni.isaac.kit import SimulationApp

# launch the simulator
config = {"headless": True}
simulation_app = SimulationApp(config)

# disable matplotlib debug messages
mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

"""Rest everything follows."""


import numpy as np
import os
import random
import scipy.spatial.transform as tf
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.torch import set_seed
from omni.isaac.core.utils.viewports import set_camera_view
from pxr import Gf, UsdGeom

import omni.isaac.orbit.compat.utils.kit as kit_utils
from omni.isaac.orbit.compat.sensors.camera import Camera, PinholeCameraCfg
from omni.isaac.orbit.utils.math import convert_quat
from omni.isaac.orbit.utils.timer import Timer


class TestCameraSensor(unittest.TestCase):
    """Test fixture for checking camera interface."""

    @classmethod
    def tearDownClass(cls):
        """Closes simulator after running all test fixtures."""
        simulation_app.close()

    def setUp(self) -> None:
        """Create a blank new stage for each test."""
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")
        # Set camera view
        set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
        # Fix random seed -- to generate same scene every time
        set_seed(0)
        # Spawn things into stage
        self._populate_scene()
        # Wait for spawning
        stage_utils.update_stage()

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # close all the opened viewport from before.
        rep.vp_manager.destroy_hydra_textures()
        # stop simulation
        self.sim.stop()
        self.sim.clear()

    def test_camera_resolution(self):
        """Checks that a camera provides image at the resolution specified."""
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()
        # perform rendering
        self.sim.step()
        # update camera
        camera.update(self.dt)

        # expected camera image shape
        height_expected, width_expected = camera.image_shape
        # check that the camera image shape is correct
        for im_data in camera.data.output.values():
            if not isinstance(im_data, np.ndarray):
                continue
            height, width = im_data.shape[:2]
            self.assertEqual(height, height_expected)
            self.assertEqual(width, width_expected)

    def test_default_camera(self):
        """Checks that the pre-existing stage camera is configured correctly."""
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(test_dir, "output", "camera", "kit_persp")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create replicator writer
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=180,
            width=320,
            data_types=["rgb", "distance_to_image_plane", "normals", "distance_to_camera"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        # camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize("/OmniverseKit_Persp")
        # Set camera pose
        set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

        # Simulate physics
        for i in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Save images
            rep_writer.write(camera.data.output)
            # Check image data
            # expect same frame number
            self.assertEqual(i + 1, camera.frame)
            # expected camera image shape
            height_expected, width_expected = camera.image_shape
            # check that the camera image shape is correct
            for im_data in camera.data.output.values():
                if not isinstance(im_data, np.ndarray):
                    continue
                height, width = im_data.shape[:2]
                self.assertEqual(height, height_expected)
                self.assertEqual(width, width_expected)

    def test_single_cam(self):
        """Checks that the single camera gets created properly."""
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(test_dir, "output", "camera", "single")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create replicator writer
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=180,
            width=320,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "normals",
                "distance_to_camera",
                # "instance_segmentation",
                # "semantic_segmentation",
                "bounding_box_2d_tight",
                "bounding_box_2d_loose",
                "bounding_box_2d_tight",
                "bounding_box_3d",
            ],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()
        # Set camera position directly
        # Note: Not a recommended way but was feeling lazy to do it properly.
        camera.set_world_pose_from_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

        # Simulate physics
        for i in range(4):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Save images
            rep_writer.write(camera.data.output)
            # Check image data
            # expected camera image shape
            height_expected, width_expected = camera.image_shape
            # check that the camera image shape is correct
            for im_data in camera.data.output.values():
                if not isinstance(im_data, np.ndarray):
                    continue
                height, width = im_data.shape[:2]
                self.assertEqual(height, height_expected)
                self.assertEqual(width, width_expected)

    def test_multiple_cam(self):
        """Checks that the multiple cameras created properly."""
        # Create camera instance
        # -- default viewport
        camera_def_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=180,
            width=320,
            data_types=["rgb"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera_def = Camera(cfg=camera_def_cfg, device="cpu")
        # -- camera 1
        camera1_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=180,
            width=320,
            data_types=["rgb", "distance_to_image_plane", "normals"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera_1 = Camera(cfg=camera1_cfg, device="cpu")
        # -- camera 2
        camera2_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=256,
            width=256,
            data_types=["rgb", "distance_to_image_plane", "normals", "distance_to_camera"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera_2 = Camera(cfg=camera2_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera_1.spawn("/World/CameraSensor1")
        camera_2.spawn("/World/CameraSensor2")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera_def.initialize("/OmniverseKit_Persp")
        camera_1.initialize()
        camera_2.initialize()

        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera_def.update(self.dt)
            camera_1.update(self.dt)
            camera_2.update(self.dt)
            # Check image data
            for cam in [camera_def, camera_1, camera_2]:
                # expected camera image shape
                height_expected, width_expected = cam.image_shape
                # check that the camera image shape is correct
                for im_data in cam.data.output.values():
                    if not isinstance(im_data, np.ndarray):
                        continue
                    height, width = im_data.shape[:2]
                    self.assertEqual(height, height_expected)
                    self.assertEqual(width, width_expected)

    def test_intrinsic_matrix(self):
        """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera.spawn("/World/CameraSensor")

        # Desired properties (obtained from realsense camera at 320x240 resolution)
        rs_intrinsic_matrix = [229.31640625, 0.0, 164.810546875, 0.0, 229.826171875, 122.1650390625, 0.0, 0.0, 1.0]
        rs_intrinsic_matrix = np.array(rs_intrinsic_matrix).reshape(3, 3)
        # Set matrix into simulator
        camera.set_intrinsic_matrix(rs_intrinsic_matrix)

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Check that matrix is correct
            K = camera.data.intrinsic_matrix
            # TODO: This is not correctly setting all values in the matrix since the
            #       vertical aperture and aperture offsets are not being set correctly
            #       This is a bug in the simulator.
            self.assertAlmostEqual(rs_intrinsic_matrix[0, 0], K[0, 0], 4)
            # self.assertAlmostEqual(rs_intrinsic_matrix[1, 1], K[1, 1], 4)
        # Display results
        print(f">>> Desired intrinsic matrix: \n{rs_intrinsic_matrix}")
        print(f">>> Current intrinsic matrix: \n{camera.data.intrinsic_matrix}")

    def test_set_pose_ros(self):
        """Checks that the camera's set and retrieve methods work for pose in ROS convention."""
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()

        # Simulate physics
        for _ in range(10):
            # set camera pose randomly
            camera_position = np.random.random(3) * 5.0
            camera_orientation = convert_quat(tf.Rotation.random().as_quat(), "wxyz")
            camera.set_world_pose_ros(pos=camera_position, quat=camera_orientation)
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Check that pose is correct
            # -- position
            np.testing.assert_almost_equal(camera.data.position, camera_position, 4)
            # -- orientation
            if np.sign(camera.data.orientation[0]) != np.sign(camera_orientation[0]):
                camera_orientation *= -1
            np.testing.assert_almost_equal(camera.data.orientation, camera_orientation, 4)

    def test_set_pose_from_view(self):
        """Checks that the camera's set method works for look-at pose."""
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()

        # Test look-at pose
        # -- inputs
        eye = np.array([2.5, 2.5, 2.5])
        targets = [np.array([0.0, 0.0, 0.0]), np.array([2.5, 2.5, 0.0])]
        # -- expected outputs
        camera_position = eye.copy()
        camera_orientations = [
            np.array([-0.17591989, 0.33985114, 0.82047325, -0.42470819]),
            np.array([0.0, 1.0, 0.0, 0.0]),
        ]

        # check that the camera pose is correct
        for target, camera_orientation in zip(targets, camera_orientations):
            # set camera pose
            camera.set_world_pose_from_view(eye=eye, target=target)
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Check that pose is correct
            # -- position
            np.testing.assert_almost_equal(camera.data.position, camera_position, 4)
            # # -- orientation
            if np.sign(camera.data.orientation[0]) != np.sign(camera_orientation[0]):
                camera_orientation *= -1
            np.testing.assert_almost_equal(camera.data.orientation, camera_orientation, 4)

    def test_throughput(self):
        """Checks that the single camera gets created properly with a rig."""
        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(test_dir, "output", "camera", "throughput")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Create replicator writer
        rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)
        # Create camera instance
        camera_cfg = PinholeCameraCfg(
            sensor_tick=0,
            height=480,
            width=640,
            data_types=["rgb", "distance_to_image_plane"],
            usd_params=PinholeCameraCfg.UsdCameraCfg(clipping_range=(0.1, 1.0e5)),
        )
        camera = Camera(cfg=camera_cfg, device="cpu")
        # Note: the camera is spawned by default in the stage
        camera.spawn("/World/CameraSensor")

        # Play simulator
        self.sim.reset()
        # Initialize sensor
        camera.initialize()
        # Set camera pose
        camera.set_world_pose_from_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

        # Simulate physics
        for _ in range(5):
            # perform rendering
            self.sim.step()
            # update camera
            with Timer(f"Time taken for updating camera with shape {camera.image_shape}"):
                camera.update(self.dt)
            # Save images
            with Timer(f"Time taken for writing data with shape {camera.image_shape}   "):
                rep_writer.write(camera.data.output)
            print("----------------------------------------")
            # Check image data
            # expected camera image shape
            height_expected, width_expected = camera.image_shape
            # check that the camera image shape is correct
            for im_data in camera.data.output.values():
                if not isinstance(im_data, np.ndarray):
                    continue
                height, width = im_data.shape[:2]
                self.assertEqual(height, height_expected)
                self.assertEqual(width, width_expected)

    """
    Helper functions.
    """

    @staticmethod
    def _populate_scene():
        """Add prims to the scene."""
        # Ground-plane
        kit_utils.create_ground_plane("/World/defaultGroundPlane")
        # Lights-1
        prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
        # Lights-2
        prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))
        # Random objects
        for i in range(8):
            # sample random position
            position = np.random.rand(3) - np.asarray([0.05, 0.05, -1.0])
            position *= np.asarray([1.5, 1.5, 0.5])
            # create prim
            prim_type = random.choice(["Cube", "Sphere", "Cylinder"])
            _ = prim_utils.create_prim(
                f"/World/Objects/Obj_{i:02d}",
                prim_type,
                translation=position,
                scale=(0.25, 0.25, 0.25),
                semantic_label=prim_type,
            )
            # add rigid properties
            rigid_obj = RigidPrim(f"/World/Objects/Obj_{i:02d}", mass=5.0)
            # cast to geom prim
            geom_prim = getattr(UsdGeom, prim_type)(rigid_obj.prim)
            # set random color
            color = Gf.Vec3f(random.random(), random.random(), random.random())
            geom_prim.CreateDisplayColorAttr()
            geom_prim.GetDisplayColorAttr().Set([color])


if __name__ == "__main__":
    unittest.main(verbosity=2)
