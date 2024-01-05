# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import numpy as np
import os
import torch
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
from omni.isaac.core.simulation_context import SimulationContext
from pxr import Gf

from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from omni.isaac.orbit.sim import PinholeCameraCfg
from omni.isaac.orbit.terrains.trimesh.utils import make_plane
from omni.isaac.orbit.terrains.utils import create_prim_from_mesh
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.utils.timer import Timer

# sample camera poses
POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
QUAT_OPENGL = [0.33985113, 0.17591988, 0.42470818, 0.82047324]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]


class TestWarpCamera(unittest.TestCase):
    """Test for orbit camera sensor"""

    """
    Test Setup and Teardown
    """

    def setUp(self):
        """Create a blank new stage for each test."""
        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=480,
            width=640,
        )
        self.camera_cfg = RayCasterCameraCfg(
            prim_path="/World/Camera",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            debug_vis=False,
            pattern_cfg=camera_pattern_cfg,
            data_types=[
                "distance_to_image_plane",
            ],
        )
        # Create a new stage
        stage_utils.create_new_stage()
        # create xform because placement of camera directly under world is not supported
        prim_utils.create_prim("/World/Camera", "Xform")
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="torch", device="cpu")
        # Ground-plane
        mesh = make_plane(size=(2e1, 2e1), height=0.0, center_zero=True)
        create_prim_from_mesh("/World/defaultGroundPlane", mesh)
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
        self.sim.clear_instance()

    """
    Tests
    """

    def test_camera_init(self):
        """Test camera initialization."""
        # Create camera
        camera = RayCasterCamera(cfg=self.camera_cfg)
        # Play sim
        self.sim.reset()
        # Check if camera is initialized
        self.assertTrue(camera._is_initialized)
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
        self.assertTrue(
            camera.data.image_shape == (self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width)
        )
        self.assertTrue(camera.data.info == [{self.camera_cfg.data_types[0]: None}])
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_data in camera.data.output.to_dict().values():
                self.assertTrue(
                    im_data.shape == (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width)
                )

    def test_camera_resolution(self):
        """Test camera resolution is correctly set."""
        # Create camera
        camera = RayCasterCamera(cfg=self.camera_cfg)
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
            self.assertTrue(im_data.shape == (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width))

    def test_camera_init_offset(self):
        """Test camera initialization with offset using different conventions."""
        # define the same offset in all conventions
        # -- ROS convention
        cam_cfg_offset_ros = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_ros.offset = RayCasterCameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_ROS,
            convention="ros",
        )
        prim_utils.create_prim("/World/CameraOffsetRos", "Xform")
        cam_cfg_offset_ros.prim_path = "/World/CameraOffsetRos"
        camera_ros = RayCasterCamera(cam_cfg_offset_ros)
        # -- OpenGL convention
        cam_cfg_offset_opengl = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_opengl.offset = RayCasterCameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_OPENGL,
            convention="opengl",
        )
        prim_utils.create_prim("/World/CameraOffsetOpengl", "Xform")
        cam_cfg_offset_opengl.prim_path = "/World/CameraOffsetOpengl"
        camera_opengl = RayCasterCamera(cam_cfg_offset_opengl)
        # -- World convention
        cam_cfg_offset_world = copy.deepcopy(self.camera_cfg)
        cam_cfg_offset_world.offset = RayCasterCameraCfg.OffsetCfg(
            pos=POSITION,
            rot=QUAT_WORLD,
            convention="world",
        )
        prim_utils.create_prim("/World/CameraOffsetWorld", "Xform")
        cam_cfg_offset_world.prim_path = "/World/CameraOffsetWorld"
        camera_world = RayCasterCamera(cam_cfg_offset_world)

        # play sim
        self.sim.reset()

        # update cameras
        camera_world.update(self.dt)
        camera_opengl.update(self.dt)
        camera_ros.update(self.dt)

        # check that all transforms are set correctly
        np.testing.assert_allclose(camera_ros.data.pos_w[0].numpy(), cam_cfg_offset_ros.offset.pos)
        np.testing.assert_allclose(camera_opengl.data.pos_w[0].numpy(), cam_cfg_offset_opengl.offset.pos)
        np.testing.assert_allclose(camera_world.data.pos_w[0].numpy(), cam_cfg_offset_world.offset.pos)

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
        prim_utils.create_prim("/World/Camera_1", "Xform")
        # Create camera
        cam_1 = RayCasterCamera(cam_cfg_1)
        # -- camera 2
        cam_cfg_2 = copy.deepcopy(self.camera_cfg)
        cam_cfg_2.prim_path = "/World/Camera_2"
        prim_utils.create_prim("/World/Camera_2", "Xform")
        cam_2 = RayCasterCamera(cam_cfg_2)

        # check that the loaded meshes are equal
        self.assertTrue(cam_1.meshes == cam_2.meshes)

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
                    self.assertTrue(
                        im_data.shape == (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width)
                    )

    def test_camera_set_world_poses(self):
        """Test camera function to set specific world pose."""
        camera = RayCasterCamera(self.camera_cfg)
        # play sim
        self.sim.reset()
        # set new pose
        camera.set_world_poses(torch.tensor([POSITION]), torch.tensor([QUAT_WORLD]), convention="world")
        np.testing.assert_allclose(camera.data.pos_w, [POSITION], rtol=1e-5)
        np.testing.assert_allclose(camera.data.quat_w_world, [QUAT_WORLD], rtol=1e-5)

    def test_camera_set_world_poses_from_view(self):
        """Test camera function to set specific world pose from view."""
        camera = RayCasterCamera(self.camera_cfg)
        # play sim
        self.sim.reset()
        # set new pose
        camera.set_world_poses_from_view(torch.tensor([POSITION]), torch.tensor([[0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(camera.data.pos_w, [POSITION], rtol=1e-5)
        np.testing.assert_allclose(camera.data.quat_w_ros, [QUAT_ROS], rtol=1e-5)

    def test_intrinsic_matrix(self):
        """Checks that the camera's set and retrieve methods work for intrinsic matrix."""
        camera_cfg = copy.deepcopy(self.camera_cfg)
        camera_cfg.pattern_cfg.height = 240
        camera_cfg.pattern_cfg.width = 320
        camera = RayCasterCamera(camera_cfg)
        # play sim
        self.sim.reset()
        # Desired properties (obtained from realsense camera at 320x240 resolution)
        rs_intrinsic_matrix = [229.31640625, 0.0, 164.810546875, 0.0, 229.826171875, 122.1650390625, 0.0, 0.0, 1.0]
        rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix).reshape(3, 3).unsqueeze(0)
        # Set matrix into simulator
        camera.set_intrinsic_matrices(rs_intrinsic_matrix)
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
            self.assertAlmostEqual(rs_intrinsic_matrix[0, 0, 0].numpy(), K[0, 0], 4)
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
        camera_cfg.pattern_cfg.height = 480
        camera_cfg.pattern_cfg.width = 640
        camera = RayCasterCamera(camera_cfg)
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
                self.assertTrue(im_data.shape == (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width))

    def test_output_equal_to_usdcamera(self):
        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=240,
            width=320,
        )
        prim_utils.create_prim("/World/Camera_warp", "Xform")
        camera_cfg_warp = RayCasterCameraCfg(
            prim_path="/World/Camera",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            debug_vis=False,
            pattern_cfg=camera_pattern_cfg,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        )

        camera_warp = RayCasterCamera(camera_cfg_warp)

        # create usd camera
        camera_cfg_usd = CameraCfg(
            height=240,
            width=320,
            prim_path="/World/Camera_usd",
            update_period=0,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1.0e5)
            ),
            colorize=False,
        )
        camera_usd = Camera(camera_cfg_usd)

        # play sim
        self.sim.reset()
        self.sim.play()

        # set views
        camera_warp.set_world_poses_from_view(torch.tensor([[2.5, 2.5, 4.5]]), torch.tensor([[0.0, 0.0, 0.0]]))
        camera_usd.set_world_poses_from_view(torch.tensor([[2.5, 2.5, 4.5]]), torch.tensor([[0.0, 0.0, 0.0]]))

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # check image data
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_image_plane"].numpy(),
            camera_warp.data.output["distance_to_image_plane"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_camera"].numpy(),
            camera_warp.data.output["distance_to_camera"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["normals"].numpy()[..., :3],
            camera_warp.data.output["normals"].numpy(),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_output_equal_to_usdcamera_offset(self):
        offset_rot = [-0.1251, 0.3617, 0.8731, -0.3020]

        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=240,
            width=320,
        )
        prim_utils.create_prim("/World/Camera_warp", "Xform")
        camera_cfg_warp = RayCasterCameraCfg(
            prim_path="/World/Camera",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=offset_rot, convention="ros"),
            debug_vis=False,
            pattern_cfg=camera_pattern_cfg,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        )

        camera_warp = RayCasterCamera(camera_cfg_warp)

        # create usd camera
        camera_cfg_usd = CameraCfg(
            height=240,
            width=320,
            prim_path="/World/Camera_usd",
            update_period=0,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
            ),
            colorize=False,
            offset=CameraCfg.OffsetCfg(pos=(2.5, 2.5, 4.0), rot=offset_rot, convention="ros"),
        )
        camera_usd = Camera(camera_cfg_usd)

        # play sim
        self.sim.reset()
        self.sim.play()

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # check image data
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_image_plane"].numpy(),
            camera_warp.data.output["distance_to_image_plane"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_camera"].numpy(),
            camera_warp.data.output["distance_to_camera"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["normals"].numpy()[..., :3],
            camera_warp.data.output["normals"].numpy(),
            rtol=1e-5,
            atol=1e-4,
        )

    def test_output_equal_to_usdcamera_prim_offset(self):
        """Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
        under an XForm prim that is translated and rotated from the world origin
        ."""
        offset_rot = [-0.1251, 0.3617, 0.8731, -0.3020]

        # gf quat
        gf_quatf = Gf.Quatd()
        gf_quatf.SetReal(QUAT_OPENGL[0])
        gf_quatf.SetImaginary(tuple(QUAT_OPENGL[1:]))

        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=240,
            width=320,
        )
        prim_raycast_cam = prim_utils.create_prim("/World/Camera_warp", "Xform")
        prim_raycast_cam.GetAttribute("xformOp:translate").Set(tuple(POSITION))
        prim_raycast_cam.GetAttribute("xformOp:orient").Set(gf_quatf)

        camera_cfg_warp = RayCasterCameraCfg(
            prim_path="/World/Camera_warp",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
            debug_vis=False,
            pattern_cfg=camera_pattern_cfg,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
        )

        camera_warp = RayCasterCamera(camera_cfg_warp)

        # create usd camera
        camera_cfg_usd = CameraCfg(
            height=240,
            width=320,
            prim_path="/World/Camera_usd/camera",
            update_period=0,
            data_types=["distance_to_image_plane", "distance_to_camera", "normals"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-6, 1.0e5)
            ),
            colorize=False,
            offset=CameraCfg.OffsetCfg(pos=(0, 0, 2.0), rot=offset_rot, convention="ros"),
        )
        prim_usd = prim_utils.create_prim("/World/Camera_usd", "Xform")
        prim_usd.GetAttribute("xformOp:translate").Set(tuple(POSITION))
        prim_usd.GetAttribute("xformOp:orient").Set(gf_quatf)

        camera_usd = Camera(camera_cfg_usd)

        # play sim
        self.sim.reset()
        self.sim.play()

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # check if pos and orientation are correct
        np.testing.assert_allclose(camera_warp.data.pos_w[0].numpy(), camera_usd.data.pos_w[0].numpy(), rtol=1e-5)
        np.testing.assert_allclose(
            camera_warp.data.quat_w_ros[0].numpy(), camera_usd.data.quat_w_ros[0].numpy(), rtol=1e-5
        )

        # check image data
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_image_plane"].numpy(),
            camera_warp.data.output["distance_to_image_plane"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["distance_to_camera"].numpy(),
            camera_warp.data.output["distance_to_camera"].numpy(),
            rtol=5e-3,
        )
        np.testing.assert_allclose(
            camera_usd.data.output["normals"].numpy()[..., :3],
            camera_warp.data.output["normals"].numpy(),
            rtol=1e-5,
            atol=1e-4,
        )


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
