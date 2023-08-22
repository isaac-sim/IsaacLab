# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This script shows how to use the camera sensor from the Orbit framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Camera Sensor Test Script")
parser.add_argument("--gpu", action="store_false", default=False, help="Use GPU device for camera rendering output.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import shutil
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, Usd, UsdGeom

from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.sim import PinholeCameraCfg
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.kit import create_ground_plane
from omni.isaac.orbit.utils.math import convert_quat
from omni.isaac.orbit.utils.timer import Timer

# sample camera poses
POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
QUAT_OPENGL = [0.33985113, 0.17591988, 0.42470818, 0.82047324]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]


class TestCamera(unittest.TestCase):
    """Test for orbit camera sensor"""

    """
    Test Setup and Teardown
    """

    def setUp(self):
        """Create a blank new stage for each test."""
        self.camera_cfg = CameraCfg(
            height=24,
            width=32,
            prim_path="/World/Camera",
            update_period=0,
            data_types=["distance_to_image_plane"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            colorize=False,
        )

        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(
            physics_dt=self.dt, rendering_dt=self.dt, backend="torch", device="cuda" if args_cli.gpu else "cpu"
        )
        # populate scene
        self._populate_scene()

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # close all the opened viewport from before.
        rep.vp_manager.destroy_hydra_textures()
        # stop simulation
        self.sim.stop()
        self.sim.clear()

    """
    Tests
    """

    def test_camera_init(self) -> None:
        """Test camera initialization."""
        # Create camera
        camera = Camera(self.camera_cfg)
        # Play sim
        self.sim.play()
        # Check if camera is initialized
        self.assertTrue(camera._is_initialized)
        # Check if camera prim is set correctly and that it is a camera prim
        self.assertTrue(camera._sensor_prims[0].GetPath().pathString == self.camera_cfg.prim_path)
        self.assertTrue(isinstance(camera._sensor_prims[0], UsdGeom.Camera))
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        # Check buffers that exists and have correct shapes
        self.assertTrue(camera.data.pos_w.shape == (1, 3))
        self.assertTrue(camera.data.quat_w_ros.shape == (1, 4))
        self.assertTrue(camera.data.quat_w_world.shape == (1, 4))
        self.assertTrue(camera.data.quat_w_opengl.shape == (1, 4))
        self.assertTrue(camera.data.intrinsic_matrices.shape == (1, 3, 3))
        self.assertTrue(camera.data.image_shape == (self.camera_cfg.height, self.camera_cfg.width))
        self.assertTrue(camera.data.info == [{self.camera_cfg.data_types[0]: None}])
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_data in camera.data.output.to_dict().values():
                self.assertTrue(im_data.shape == (1, self.camera_cfg.height, self.camera_cfg.width))

    def test_camera_resolution(self) -> None:
        # Create camera
        camera = Camera(self.camera_cfg)
        # Play sim
        self.sim.play()
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        camera.update(self.dt)
        # access image data and compare shapes
        for im_data in camera.data.output.to_dict().values():
            self.assertTrue(im_data.shape == (1, self.camera_cfg.height, self.camera_cfg.width))

    def test_camera_init_offset(self) -> None:
        """Test camera initialization with offset."""
        # define the same offset in all conventions
        cam_cfg_offset_ros = copy.copy(self.camera_cfg)
        cam_cfg_offset_ros.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_ROS,
            convention="ros",
        )
        cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"

        cam_cfg_offset_opengl = copy.copy(self.camera_cfg)
        cam_cfg_offset_opengl.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_OPENGL,
            convention="opengl",
        )
        cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"

        cam_cfg_offset_world = copy.copy(self.camera_cfg)
        cam_cfg_offset_world.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_WORLD,
            convention="world",
        )
        cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"

        camera_ros = Camera(cam_cfg_offset_ros)
        camera_opengl = Camera(cam_cfg_offset_opengl)
        camera_world = Camera(cam_cfg_offset_world)

        # play sim
        self.sim.play()

        # retrieve camera pose
        prim_tf_ros = camera_ros._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_tf_opengl = camera_opengl._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_tf_world = camera_world._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        prim_tf_ros = np.transpose(prim_tf_ros)
        prim_tf_opengl = np.transpose(prim_tf_opengl)
        prim_tf_world = np.transpose(prim_tf_world)

        # check that all transforms are set correctly
        self.assertTrue(np.allclose(prim_tf_ros[0:3, 3], cam_cfg_offset_ros.offset.pos))
        self.assertTrue(np.allclose(prim_tf_opengl[0:3, 3], cam_cfg_offset_opengl.offset.pos))
        self.assertTrue(np.allclose(prim_tf_world[0:3, 3], cam_cfg_offset_world.offset.pos))
        self.assertTrue(
            np.allclose(
                convert_quat(tf.Rotation.from_matrix(prim_tf_ros[:3, :3]).as_quat(), "wxyz"),
                cam_cfg_offset_opengl.offset.rot,
            )
        )
        self.assertTrue(
            np.allclose(
                convert_quat(tf.Rotation.from_matrix(prim_tf_opengl[:3, :3]).as_quat(), "wxyz"),
                cam_cfg_offset_opengl.offset.rot,
            )
        )
        self.assertTrue(
            np.allclose(
                convert_quat(tf.Rotation.from_matrix(prim_tf_world[:3, :3]).as_quat(), "wxyz"),
                cam_cfg_offset_opengl.offset.rot,
            )
        )

        # check if transform correctly set in output
        self.assertTrue(np.allclose(camera_ros.data.pos_w[0], cam_cfg_offset_ros.offset.pos))
        self.assertTrue(np.allclose(camera_ros.data.quat_w_ros[0], QUAT_ROS))
        self.assertTrue(np.allclose(camera_ros.data.quat_w_opengl[0], QUAT_OPENGL))
        self.assertTrue(np.allclose(camera_ros.data.quat_w_world[0], QUAT_WORLD))

    def test_multi_camera_init(self) -> None:
        """Test camera initialization with offset."""
        # define the same offset in all conventions
        cam_cfg_1 = copy.copy(self.camera_cfg)
        cam_cfg_1.prim_path = "/World/Camera_1"

        cam_cfg_2 = copy.copy(self.camera_cfg)
        cam_cfg_2.prim_path = "/World/Camera_2"

        cam_1 = Camera(cam_cfg_1)
        cam_2 = Camera(cam_cfg_2)

        # play sim
        self.sim.play()
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            cam_1.update(self.dt)
            cam_2.update(self.dt)
            # check image data
            for cam in [cam_1, cam_2]:
                for im_data in cam.data.output.to_dict().values():
                    self.assertTrue(im_data.shape == (1, self.camera_cfg.height, self.camera_cfg.width))

    def test_camera_set_world_poses(self) -> None:
        """Test camera function to set specific world pose."""
        camera = Camera(self.camera_cfg)
        # play sim
        self.sim.play()
        # set new pose
        camera.set_world_poses([POSITION], [QUAT_WORLD], convention="world")
        self.assertTrue(np.allclose(camera.data.pos_w, [POSITION]))
        self.assertTrue(np.allclose(camera.data.quat_w_world, [QUAT_WORLD]))

    def test_camera_set_world_poses_from_view(self) -> None:
        """Test camera function to set specific world pose from view."""
        camera = Camera(self.camera_cfg)
        # play sim
        self.sim.play()
        # set new pose
        camera.set_world_poses_from_view([POSITION], [[0.0, 0.0, 0.0]])
        self.assertTrue(np.allclose(camera.data.pos_w, [POSITION]))
        self.assertTrue(np.allclose(camera.data.quat_w_ros, [QUAT_ROS]))

    def test_intrinsic_matrix(self) -> None:
        """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
        camera_cfg = copy.copy(self.camera_cfg)
        camera_cfg.height = 240
        camera_cfg.width = 320
        camera = Camera(camera_cfg)
        # play sim
        self.sim.play()
        # Desired properties (obtained from realsense camera at 320x240 resolution)
        rs_intrinsic_matrix = [229.31640625, 0.0, 164.810546875, 0.0, 229.826171875, 122.1650390625, 0.0, 0.0, 1.0]
        rs_intrinsic_matrix = np.array(rs_intrinsic_matrix).reshape(3, 3)
        # Set matrix into simulator
        camera.set_intrinsic_matrices([rs_intrinsic_matrix])
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # Check that matrix is correct
            K = camera.data.intrinsic_matrices[0].numpy()
            # TODO: This is not correctly setting all values in the matrix since the
            #       vertical aperture and aperture offsets are not being set correctly
            #       This is a bug in the simulator.
            self.assertAlmostEqual(rs_intrinsic_matrix[0, 0], K[0, 0], 4)
            # self.assertAlmostEqual(rs_intrinsic_matrix[1, 1], K[1, 1], 4)

    def test_throughput(self):
        """Checks that the single camera gets created properly with a rig."""
        # Create directory temp dir to dump the results
        file_dir = os.path.dirname(os.path.realpath(__file__))
        temp_dir = os.path.join(file_dir, "output", "camera", "throughput")
        os.makedirs(temp_dir, exist_ok=True)
        # Create replicator writer
        rep_writer = rep.BasicWriter(output_dir=temp_dir, frame_padding=3)
        # create camera
        camera_cfg = copy.copy(self.camera_cfg)
        camera_cfg.height = 480
        camera_cfg.width = 640
        camera = Camera(camera_cfg)
        # Play simulator
        self.sim.play()
        # Set camera pose
        camera.set_world_poses_from_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
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
            # Save images
            with Timer(f"Time taken for writing data with shape {camera.image_shape}   "):
                # Pack data back into replicator format to save them using its writer
                rep_output = dict()
                camera_data = convert_dict_to_backend(camera.data.output[0].to_dict(), backend="numpy")
                for key, data, info in zip(camera_data.keys(), camera_data.values(), camera.data.info[0].values()):
                    if info is not None:
                        rep_output[key] = {"data": data, "info": info}
                    else:
                        rep_output[key] = data
                # Save images
                rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
                rep_writer.write(rep_output)
            print("----------------------------------------")
            # Check image data
            for im_data in camera.data.output.values():
                self.assertTrue(im_data.shape == (1, camera_cfg.height, camera_cfg.width))

    """
    Helper functions.
    """

    @staticmethod
    def _populate_scene():
        """Add prims to the scene."""
        # Ground-plane
        create_ground_plane("/World/defaultGroundPlane")
        # Lights-1
        prim_utils.create_prim(
            "/World/Light/GreySphere",
            "SphereLight",
            translation=(4.5, 3.5, 10.0),
            attributes={"radius": 1.0, "intensity": 300.0, "color": (0.75, 0.75, 0.75)},
        )
        # Lights-2
        prim_utils.create_prim(
            "/World/Light/WhiteSphere",
            "SphereLight",
            translation=(-4.5, 3.5, 10.0),
            attributes={"radius": 1.0, "intensity": 300.0, "color": (1.0, 1.0, 1.0)},
        )
        # Random objects
        random.seed(0)
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
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
