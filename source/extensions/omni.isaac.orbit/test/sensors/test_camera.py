# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import torch
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
from omni.isaac.core.prims import GeometryPrim, RigidPrim
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf, Usd, UsdGeom

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.math import convert_quat
from omni.isaac.orbit.utils.timer import Timer

# sample camera poses
POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
QUAT_OPENGL = [0.33985113, 0.17591988, 0.42470818, 0.82047324]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]


class TestCamera(unittest.TestCase):
    """Test for USD Camera sensor."""

    def setUp(self):
        """Create a blank new stage for each test."""
        self.camera_cfg = CameraCfg(
            height=128,
            width=128,
            prim_path="/World/Camera",
            update_period=0,
            data_types=["distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
            ),
            colorize=False,
        )
        # Create a new stage
        stage_utils.create_new_stage()
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="torch", device="cpu")
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
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    """
    Tests
    """

    def test_camera_init(self):
        """Test camera initialization."""
        # Create camera
        camera = Camera(self.camera_cfg)
        # Play sim
        self.sim.reset()
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

    def test_camera_resolution(self):
        """Test camera resolution is correctly set."""
        # Create camera
        camera = Camera(self.camera_cfg)
        # Play sim
        self.sim.reset()
        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        camera.update(self.dt)
        # access image data and compare shapes
        for im_data in camera.data.output.to_dict().values():
            self.assertTrue(im_data.shape == (1, self.camera_cfg.height, self.camera_cfg.width))

    def test_camera_init_offset(self):
        """Test camera initialization with offset using different conventions."""
        # define the same offset in all conventions
        # -- ROS convention
        cam_cfg_offset_ros = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_ros.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_ROS,
            convention="ros",
        )
        cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
        camera_ros = Camera(cam_cfg_offset_ros)
        # -- OpenGL convention
        cam_cfg_offset_opengl = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_opengl.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_OPENGL,
            convention="opengl",
        )
        cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
        camera_opengl = Camera(cam_cfg_offset_opengl)
        # -- World convention
        cam_cfg_offset_world = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_world.offset = CameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_WORLD,
            convention="world",
        )
        cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
        camera_world = Camera(cam_cfg_offset_world)

        # play sim
        self.sim.reset()

        # retrieve camera pose
        prim_tf_ros = camera_ros._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_tf_opengl = camera_opengl._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_tf_world = camera_world._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())

        prim_tf_ros = np.transpose(prim_tf_ros)
        prim_tf_opengl = np.transpose(prim_tf_opengl)
        prim_tf_world = np.transpose(prim_tf_world)

        # check that all transforms are set correctly
        np.testing.assert_allclose(prim_tf_ros[0:3, 3], cam_cfg_offset_ros.offset.pos)
        np.testing.assert_allclose(prim_tf_opengl[0:3, 3], cam_cfg_offset_opengl.offset.pos)
        np.testing.assert_allclose(prim_tf_world[0:3, 3], cam_cfg_offset_world.offset.pos)
        np.testing.assert_allclose(
            convert_quat(tf.Rotation.from_matrix(prim_tf_ros[:3, :3]).as_quat(), "wxyz"),
            cam_cfg_offset_opengl.offset.rot,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            convert_quat(tf.Rotation.from_matrix(prim_tf_opengl[:3, :3]).as_quat(), "wxyz"),
            cam_cfg_offset_opengl.offset.rot,
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            convert_quat(tf.Rotation.from_matrix(prim_tf_world[:3, :3]).as_quat(), "wxyz"),
            cam_cfg_offset_opengl.offset.rot,
            rtol=1e-5,
        )

        # check if transform correctly set in output
        np.testing.assert_allclose(camera_ros.data.pos_w[0], cam_cfg_offset_ros.offset.pos, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_ros[0], QUAT_ROS, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_opengl[0], QUAT_OPENGL, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_world[0], QUAT_WORLD, rtol=1e-5)

    def test_multi_camera_init(self):
        """Test multi-camera initialization."""
        # create two cameras with different prim paths
        # -- camera 1
        cam_cfg_1 = copy.deepcopy(self.camera_cfg)
        cam_cfg_1.prim_path = "/World/Camera_1"
        cam_1 = Camera(cam_cfg_1)
        # -- camera 2
        cam_cfg_2 = copy.deepcopy(self.camera_cfg)
        cam_cfg_2.prim_path = "/World/Camera_2"
        cam_2 = Camera(cam_cfg_2)

        # play sim
        self.sim.reset()
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

    def test_camera_set_world_poses(self):
        """Test camera function to set specific world pose."""
        camera = Camera(self.camera_cfg)
        # play sim
        self.sim.reset()
        # set new pose
        camera.set_world_poses(torch.tensor([POSITION]), torch.tensor([QUAT_WORLD]), convention="world")
        np.testing.assert_allclose(camera.data.pos_w, [POSITION], rtol=1e-5)
        np.testing.assert_allclose(camera.data.quat_w_world, [QUAT_WORLD], rtol=1e-5)

    def test_camera_set_world_poses_from_view(self):
        """Test camera function to set specific world pose from view."""
        camera = Camera(self.camera_cfg)
        # play sim
        self.sim.reset()
        # set new pose
        camera.set_world_poses_from_view(torch.tensor([POSITION]), torch.tensor([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(camera.data.pos_w, [POSITION], rtol=1e-5)
        np.testing.assert_allclose(camera.data.quat_w_ros, [QUAT_ROS], rtol=1e-5)

    def test_intrinsic_matrix(self):
        """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.height = 240
        camera_cfg.width = 320
        camera = Camera(camera_cfg)
        # play sim
        self.sim.reset()
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
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.height = 480
        camera_cfg.width = 640
        camera = Camera(camera_cfg)
        # Play simulator
        self.sim.reset()
        # Set camera pose
        camera.set_world_poses_from_view(torch.tensor([[2.5, 2.5, 2.5]]), torch.tensor([[0.0, 0.0, 0.0]]))
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
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.SphereLightCfg()
        cfg.func("/World/Light/GreySphere", cfg, translation=(4.5, 3.5, 10.0))
        cfg.func("/World/Light/WhiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
        # Random objects
        random.seed(0)
        for i in range(8):
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
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
