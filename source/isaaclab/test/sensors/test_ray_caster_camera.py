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
import os
import torch
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import omni.replicator.core as rep
from pxr import Gf

import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab.sensors.ray_caster import RayCasterCamera, RayCasterCameraCfg, patterns
from isaaclab.sim import PinholeCameraCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.timer import Timer

# sample camera poses
POSITION = [2.5, 2.5, 2.5]
QUAT_ROS = [-0.17591989, 0.33985114, 0.82047325, -0.42470819]
QUAT_OPENGL = [0.33985113, 0.17591988, 0.42470818, 0.82047324]
QUAT_WORLD = [-0.3647052, -0.27984815, -0.1159169, 0.88047623]


class TestWarpCamera(unittest.TestCase):
    """Test for isaaclab camera sensor"""

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
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt)
        self.sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)
        # Ground-plane
        mesh = make_plane(size=(100, 100), height=0.0, center_zero=True)
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
        self.sim.clear_all_callbacks()
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
        self.assertTrue(camera.is_initialized)
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
        self.assertEqual(
            camera.data.image_shape, (self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width)
        )
        self.assertEqual(camera.data.info, [{self.camera_cfg.data_types[0]: None}])
        # Simulate physics
        for _ in range(10):
            # perform rendering
            self.sim.step()
            # update camera
            camera.update(self.dt)
            # check image data
            for im_data in camera.data.output.values():
                self.assertEqual(
                    im_data.shape, (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width, 1)
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
        for im_data in camera.data.output.values():
            self.assertTrue(
                im_data.shape == (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width, 1)
            )

    def test_depth_clipping(self):
        """Test depth clipping.

        .. note::

            This test is the same for all camera models to enforce the same clipping behavior.
        """
        prim_utils.create_prim("/World/CameraZero", "Xform")
        prim_utils.create_prim("/World/CameraNone", "Xform")
        prim_utils.create_prim("/World/CameraMax", "Xform")

        # get camera cfgs
        camera_cfg_zero = RayCasterCameraCfg(
            prim_path="/World/CameraZero",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            offset=RayCasterCameraCfg.OffsetCfg(
                pos=(2.5, 2.5, 6.0), rot=(0.9914449, 0.0, 0.1305, 0.0), convention="world"
            ),
            pattern_cfg=patterns.PinholeCameraPatternCfg().from_intrinsic_matrix(
                focal_length=38.0,
                intrinsic_matrix=[380.08, 0.0, 467.79, 0.0, 380.08, 262.05, 0.0, 0.0, 1.0],
                height=540,
                width=960,
            ),
            max_distance=10.0,
            data_types=["distance_to_image_plane", "distance_to_camera"],
            depth_clipping_behavior="zero",
        )
        camera_zero = RayCasterCamera(camera_cfg_zero)

        camera_cfg_none = copy.deepcopy(camera_cfg_zero)
        camera_cfg_none.prim_path = "/World/CameraNone"
        camera_cfg_none.depth_clipping_behavior = "none"
        camera_none = RayCasterCamera(camera_cfg_none)

        camera_cfg_max = copy.deepcopy(camera_cfg_zero)
        camera_cfg_max.prim_path = "/World/CameraMax"
        camera_cfg_max.depth_clipping_behavior = "max"
        camera_max = RayCasterCamera(camera_cfg_max)

        # Play sim
        self.sim.reset()

        camera_zero.update(self.dt)
        camera_none.update(self.dt)
        camera_max.update(self.dt)

        # none clipping should contain inf values
        self.assertTrue(torch.isinf(camera_none.data.output["distance_to_camera"]).any())
        self.assertTrue(torch.isnan(camera_none.data.output["distance_to_image_plane"]).any())
        self.assertTrue(
            camera_none.data.output["distance_to_camera"][
                ~torch.isinf(camera_none.data.output["distance_to_camera"])
            ].max()
            > camera_cfg_zero.max_distance
        )
        self.assertTrue(
            camera_none.data.output["distance_to_image_plane"][
                ~torch.isnan(camera_none.data.output["distance_to_image_plane"])
            ].max()
            > camera_cfg_zero.max_distance
        )

        # zero clipping should result in zero values
        self.assertTrue(
            torch.all(
                camera_zero.data.output["distance_to_camera"][
                    torch.isinf(camera_none.data.output["distance_to_camera"])
                ]
                == 0.0
            )
        )
        self.assertTrue(
            torch.all(
                camera_zero.data.output["distance_to_image_plane"][
                    torch.isnan(camera_none.data.output["distance_to_image_plane"])
                ]
                == 0.0
            )
        )
        self.assertTrue(camera_zero.data.output["distance_to_camera"].max() <= camera_cfg_zero.max_distance)
        self.assertTrue(camera_zero.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.max_distance)

        # max clipping should result in max values
        self.assertTrue(
            torch.all(
                camera_max.data.output["distance_to_camera"][torch.isinf(camera_none.data.output["distance_to_camera"])]
                == camera_cfg_zero.max_distance
            )
        )
        self.assertTrue(
            torch.all(
                camera_max.data.output["distance_to_image_plane"][
                    torch.isnan(camera_none.data.output["distance_to_image_plane"])
                ]
                == camera_cfg_zero.max_distance
            )
        )
        self.assertTrue(camera_max.data.output["distance_to_camera"].max() <= camera_cfg_zero.max_distance)
        self.assertTrue(camera_max.data.output["distance_to_image_plane"].max() <= camera_cfg_zero.max_distance)

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
        np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos)
        np.testing.assert_allclose(camera_opengl.data.pos_w[0].cpu().numpy(), cam_cfg_offset_opengl.offset.pos)
        np.testing.assert_allclose(camera_world.data.pos_w[0].cpu().numpy(), cam_cfg_offset_world.offset.pos)

        # check if transform correctly set in output
        np.testing.assert_allclose(camera_ros.data.pos_w[0].cpu().numpy(), cam_cfg_offset_ros.offset.pos, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_ros[0].cpu().numpy(), QUAT_ROS, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_opengl[0].cpu().numpy(), QUAT_OPENGL, rtol=1e-5)
        np.testing.assert_allclose(camera_ros.data.quat_w_world[0].cpu().numpy(), QUAT_WORLD, rtol=1e-5)

    def test_camera_init_intrinsic_matrix(self):
        """Test camera initialization from intrinsic matrix."""
        # get the first camera
        camera_1 = RayCasterCamera(cfg=self.camera_cfg)
        # get intrinsic matrix
        self.sim.reset()
        intrinsic_matrix = camera_1.data.intrinsic_matrices[0].cpu().flatten().tolist()
        self.tearDown()
        # reinit the first camera
        self.setUp()
        camera_1 = RayCasterCamera(cfg=self.camera_cfg)
        # initialize from intrinsic matrix
        intrinsic_camera_cfg = RayCasterCameraCfg(
            prim_path="/World/Camera",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
            debug_vis=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                intrinsic_matrix=intrinsic_matrix,
                height=self.camera_cfg.pattern_cfg.height,
                width=self.camera_cfg.pattern_cfg.width,
                focal_length=self.camera_cfg.pattern_cfg.focal_length,
            ),
            data_types=[
                "distance_to_image_plane",
            ],
        )
        camera_2 = RayCasterCamera(cfg=intrinsic_camera_cfg)

        # play sim
        self.sim.reset()
        self.sim.play()

        # update cameras
        camera_1.update(self.dt)
        camera_2.update(self.dt)

        # check image data
        torch.testing.assert_close(
            camera_1.data.output["distance_to_image_plane"],
            camera_2.data.output["distance_to_image_plane"],
        )
        # check that both intrinsic matrices are the same
        torch.testing.assert_close(
            camera_1.data.intrinsic_matrices[0],
            camera_2.data.intrinsic_matrices[0],
        )

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
                for im_data in cam.data.output.values():
                    self.assertEqual(
                        im_data.shape, (1, self.camera_cfg.pattern_cfg.height, self.camera_cfg.pattern_cfg.width, 1)
                    )

    def test_camera_set_world_poses(self):
        """Test camera function to set specific world pose."""
        camera = RayCasterCamera(self.camera_cfg)
        # play sim
        self.sim.reset()

        # convert to torch tensors
        position = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
        orientation = torch.tensor([QUAT_WORLD], dtype=torch.float32, device=camera.device)
        # set new pose
        camera.set_world_poses(position.clone(), orientation.clone(), convention="world")

        # check if transform correctly set in output
        torch.testing.assert_close(camera.data.pos_w, position)
        torch.testing.assert_close(camera.data.quat_w_world, orientation)

    def test_camera_set_world_poses_from_view(self):
        """Test camera function to set specific world pose from view."""
        camera = RayCasterCamera(self.camera_cfg)
        # play sim
        self.sim.reset()

        # convert to torch tensors
        eyes = torch.tensor([POSITION], dtype=torch.float32, device=camera.device)
        targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
        quat_ros_gt = torch.tensor([QUAT_ROS], dtype=torch.float32, device=camera.device)
        # set new pose
        camera.set_world_poses_from_view(eyes.clone(), targets.clone())

        # check if transform correctly set in output
        torch.testing.assert_close(camera.data.pos_w, eyes)
        torch.testing.assert_close(camera.data.quat_w_ros, quat_ros_gt)

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
        rs_intrinsic_matrix = torch.tensor(rs_intrinsic_matrix, device=camera.device).reshape(3, 3).unsqueeze(0)
        # Set matrix into simulator
        camera.set_intrinsic_matrices(rs_intrinsic_matrix.clone())
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
            torch.testing.assert_close(rs_intrinsic_matrix, camera.data.intrinsic_matrices)

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
        eyes = torch.tensor([[2.5, 2.5, 2.5]], dtype=torch.float32, device=camera.device)
        targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera.device)
        camera.set_world_poses_from_view(eyes, targets)

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
                rep_output = {"annotators": {}}
                camera_data = convert_dict_to_backend({k: v[0] for k, v in camera.data.output.items()}, backend="numpy")
                for key, data, info in zip(camera_data.keys(), camera_data.values(), camera.data.info[0].values()):
                    if info is not None:
                        rep_output["annotators"][key] = {"render_product": {"data": data, **info}}
                    else:
                        rep_output["annotators"][key] = {"render_product": {"data": data}}
                # Save images
                rep_output["trigger_outputs"] = {"on_time": camera.frame[0]}
                rep_writer.write(rep_output)
            print("----------------------------------------")
            # Check image data
            for im_data in camera.data.output.values():
                self.assertEqual(im_data.shape, (1, camera_cfg.pattern_cfg.height, camera_cfg.pattern_cfg.width, 1))

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
        )
        camera_usd = Camera(camera_cfg_usd)

        # play sim
        self.sim.reset()
        self.sim.play()

        # convert to torch tensors
        eyes = torch.tensor([[2.5, 2.5, 4.5]], dtype=torch.float32, device=camera_warp.device)
        targets = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=camera_warp.device)
        # set views
        camera_warp.set_world_poses_from_view(eyes, targets)
        camera_usd.set_world_poses_from_view(eyes, targets)

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # check the intrinsic matrices
        torch.testing.assert_close(
            camera_usd.data.intrinsic_matrices,
            camera_warp.data.intrinsic_matrices,
        )

        # check the apertures
        torch.testing.assert_close(
            camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
            camera_cfg_warp.pattern_cfg.horizontal_aperture,
        )
        torch.testing.assert_close(
            camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
            camera_cfg_warp.pattern_cfg.vertical_aperture,
        )

        # check image data
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_image_plane"],
            camera_warp.data.output["distance_to_image_plane"],
        )
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_camera"],
            camera_warp.data.output["distance_to_camera"],
            atol=5e-5,
            rtol=5e-6,
        )

        # check normals
        # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
        torch.testing.assert_close(
            camera_usd.data.output["normals"][..., :3],
            camera_warp.data.output["normals"],
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
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_image_plane"],
            camera_warp.data.output["distance_to_image_plane"],
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_camera"],
            camera_warp.data.output["distance_to_camera"],
            rtol=1e-3,
            atol=1e-5,
        )

        # check normals
        # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
        torch.testing.assert_close(
            camera_usd.data.output["normals"][..., :3],
            camera_warp.data.output["normals"],
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
        torch.testing.assert_close(camera_warp.data.pos_w[0], camera_usd.data.pos_w[0])
        torch.testing.assert_close(camera_warp.data.quat_w_ros[0], camera_usd.data.quat_w_ros[0])

        # check image data
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_image_plane"],
            camera_warp.data.output["distance_to_image_plane"],
            rtol=1e-3,
            atol=1e-5,
        )
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_camera"],
            camera_warp.data.output["distance_to_camera"],
            rtol=4e-6,
            atol=2e-5,
        )

        # check normals
        # NOTE: floating point issues of ~1e-5, so using atol and rtol in this case
        torch.testing.assert_close(
            camera_usd.data.output["normals"][..., :3],
            camera_warp.data.output["normals"],
            rtol=1e-5,
            atol=1e-4,
        )

    def test_output_equal_to_usd_camera_intrinsics(self):
        """
        Test that the output of the ray caster camera and usd camera are the same when both are
        initialized with the same intrinsic matrix.
        """

        # create cameras
        offset_rot = [-0.1251, 0.3617, 0.8731, -0.3020]
        offset_pos = (2.5, 2.5, 4.0)
        intrinsics = [380.0831, 0.0, 467.7916, 0.0, 380.0831, 262.0532, 0.0, 0.0, 1.0]
        prim_utils.create_prim("/World/Camera_warp", "Xform")
        # get camera cfgs
        camera_warp_cfg = RayCasterCameraCfg(
            prim_path="/World/Camera_warp",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            offset=RayCasterCameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
            debug_vis=False,
            pattern_cfg=patterns.PinholeCameraPatternCfg.from_intrinsic_matrix(
                intrinsic_matrix=intrinsics,
                height=540,
                width=960,
                focal_length=38.0,
            ),
            max_distance=20.0,
            data_types=["distance_to_image_plane"],
        )
        camera_usd_cfg = CameraCfg(
            prim_path="/World/Camera_usd",
            offset=CameraCfg.OffsetCfg(pos=offset_pos, rot=offset_rot, convention="ros"),
            spawn=PinholeCameraCfg.from_intrinsic_matrix(
                intrinsic_matrix=intrinsics,
                height=540,
                width=960,
                clipping_range=(0.01, 20),
                focal_length=38.0,
            ),
            height=540,
            width=960,
            data_types=["distance_to_image_plane"],
        )

        # set aperture offsets to 0, as currently not supported for usd camera
        camera_warp_cfg.pattern_cfg.horizontal_aperture_offset = 0
        camera_warp_cfg.pattern_cfg.vertical_aperture_offset = 0
        camera_usd_cfg.spawn.horizontal_aperture_offset = 0
        camera_usd_cfg.spawn.vertical_aperture_offset = 0
        # init cameras
        camera_warp = RayCasterCamera(camera_warp_cfg)
        camera_usd = Camera(camera_usd_cfg)

        # play sim
        self.sim.reset()
        self.sim.play()

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # filter nan and inf from output
        cam_warp_output = camera_warp.data.output["distance_to_image_plane"].clone()
        cam_usd_output = camera_usd.data.output["distance_to_image_plane"].clone()
        cam_warp_output[torch.isnan(cam_warp_output)] = 0
        cam_warp_output[torch.isinf(cam_warp_output)] = 0
        cam_usd_output[torch.isnan(cam_usd_output)] = 0
        cam_usd_output[torch.isinf(cam_usd_output)] = 0

        # check that both have the same intrinsic matrices
        torch.testing.assert_close(camera_warp.data.intrinsic_matrices[0], camera_usd.data.intrinsic_matrices[0])

        # check the apertures
        torch.testing.assert_close(
            camera_usd._sensor_prims[0].GetHorizontalApertureAttr().Get(),
            camera_warp_cfg.pattern_cfg.horizontal_aperture,
        )
        torch.testing.assert_close(
            camera_usd._sensor_prims[0].GetVerticalApertureAttr().Get(),
            camera_warp_cfg.pattern_cfg.vertical_aperture,
        )

        # check image data
        torch.testing.assert_close(
            cam_warp_output,
            cam_usd_output,
            atol=5e-5,
            rtol=5e-6,
        )

    def test_output_equal_to_usd_camera_when_intrinsics_set(self):
        """
        Test that the output of the ray caster camera is equal to the output of the usd camera when both are placed
        under an XForm prim and an intrinsic matrix is set.
        """

        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=540,
            width=960,
        )
        camera_cfg_warp = RayCasterCameraCfg(
            prim_path="/World/Camera",
            mesh_prim_paths=["/World/defaultGroundPlane"],
            update_period=0,
            offset=RayCasterCameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            debug_vis=False,
            pattern_cfg=camera_pattern_cfg,
            data_types=["distance_to_camera"],
        )

        camera_warp = RayCasterCamera(camera_cfg_warp)

        # create usd camera
        camera_cfg_usd = CameraCfg(
            height=540,
            width=960,
            prim_path="/World/Camera_usd",
            update_period=0,
            data_types=["distance_to_camera"],
            spawn=PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(1e-4, 1.0e5)
            ),
        )
        camera_usd = Camera(camera_cfg_usd)

        # play sim
        self.sim.reset()
        self.sim.play()

        # set intrinsic matrix
        # NOTE: extend the test to cover aperture offsets once supported by the usd camera
        intrinsic_matrix = torch.tensor(
            [[380.0831, 0.0, camera_cfg_usd.width / 2, 0.0, 380.0831, camera_cfg_usd.height / 2, 0.0, 0.0, 1.0]],
            device=camera_warp.device,
        ).reshape(1, 3, 3)
        camera_warp.set_intrinsic_matrices(intrinsic_matrix, focal_length=10)
        camera_usd.set_intrinsic_matrices(intrinsic_matrix, focal_length=10)

        # set camera position
        camera_warp.set_world_poses_from_view(
            eyes=torch.tensor([[0.0, 0.0, 5.0]], device=camera_warp.device),
            targets=torch.tensor([[0.0, 0.0, 0.0]], device=camera_warp.device),
        )
        camera_usd.set_world_poses_from_view(
            eyes=torch.tensor([[0.0, 0.0, 5.0]], device=camera_usd.device),
            targets=torch.tensor([[0.0, 0.0, 0.0]], device=camera_usd.device),
        )

        # perform steps
        for _ in range(5):
            self.sim.step()

        # update camera
        camera_usd.update(self.dt)
        camera_warp.update(self.dt)

        # check image data
        torch.testing.assert_close(
            camera_usd.data.output["distance_to_camera"],
            camera_warp.data.output["distance_to_camera"],
            rtol=5e-3,
            atol=1e-4,
        )

    def test_sensor_print(self):
        """Test sensor print is working correctly."""
        # Create sensor
        sensor = RayCasterCamera(cfg=self.camera_cfg)
        # Play sim
        self.sim.reset()
        # print info
        print(sensor)


if __name__ == "__main__":
    run_tests()
