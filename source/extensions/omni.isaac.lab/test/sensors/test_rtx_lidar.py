# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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

import copy
import json
import numpy as np
import os
import scipy.spatial.transform as tf
import torch
import unittest

import omni.isaac.core.utils.stage as stage_utils
import omni.replicator.core as rep
from omni.isaac.core.utils.extensions import get_extension_path_from_name
from pxr import Usd, UsdGeom

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sensors.rtx_lidar import RTX_LIDAR_INFO_FIELDS, RtxLidar, RtxLidarCfg
from omni.isaac.lab.terrains.trimesh.utils import make_border, make_plane
from omni.isaac.lab.terrains.utils import create_prim_from_mesh
from omni.isaac.lab.utils.math import convert_quat

POSITION = (0.0, 0.0, 0.5)
QUATERNION = (0.0, 0.3461835, 0.0, 0.9381668)

# load example json
EXAMPLE_ROTARY_PATH = os.path.abspath(
    os.path.join(
        get_extension_path_from_name("omni.isaac.sensor"),
        "data/lidar_configs/NVIDIA/Simple_Example_Solid_State.json",
    )
)


class TestRtxLidar(unittest.TestCase):
    """Test for isaaclab rtx lidar"""

    """
    Test Setup and Teardown
    """

    def setUp(self):
        """Create a blank new stage for each test."""

        # Create a new stage
        stage_utils.create_new_stage()

        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=self.dt, device="cuda")
        self.sim: sim_utils.SimulationContext = sim_utils.SimulationContext(sim_cfg)

        # configure lidar
        self.lidar_cfg = RtxLidarCfg(
            prim_path="/World/Lidar",
            debug_vis=not app_launcher._headless,
            optional_data_types=[
                "azimuth",
                "elevation",
                "emitterId",
                "index",
                "materialId",
                "normal",
                "objectId",
                "velocity",
            ],
            spawn=sim_utils.LidarCfg(lidar_type=sim_utils.LidarCfg.LidarType.EXAMPLE_ROTARY),
        )

        # Ground-plane
        mesh = make_plane(size=(10, 10), height=0.0, center_zero=True)
        border = make_border(size=(10, 10), inner_size=(5, 5), height=2.0, position=(0.0, 0.0, 0.0))

        create_prim_from_mesh("/World/defaultGroundPlane", mesh)
        for i, box in enumerate(border):
            create_prim_from_mesh(f"/World/defaultBoarder{i}", box)
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

    def test_lidar_init(self):
        """Test lidar initialization and data population."""
        # Create lidar
        lidar = RtxLidar(cfg=self.lidar_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if lidar is initialized
        self.assertTrue(lidar.is_initialized)
        # Check if lidar prim is set correctly and that it is a camera prim
        self.assertEqual(lidar._sensor_prims[0].GetPath().pathString, self.lidar_cfg.prim_path)
        self.assertIsInstance(lidar._sensor_prims[0], UsdGeom.Camera)

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
            lidar.update(self.dt, force_recompute=True)
            # check info data
            for info_key, info_value in lidar.data.info[0].items():
                self.assertTrue(info_key in RTX_LIDAR_INFO_FIELDS.keys())
                self.assertTrue(isinstance(info_value, RTX_LIDAR_INFO_FIELDS[info_key]))

            # check lidar data
            for data_key, data_value in lidar.data.output.to_dict().items():
                if data_key in self.lidar_cfg.optional_data_types:
                    self.assertTrue(data_value.shape[1] > 0)

    def test_lidar_init_offset(self):
        """Test lidar offset configuration."""
        lidar_cfg_offset = copy.deepcopy(self.lidar_cfg)
        lidar_cfg_offset.offset = RtxLidarCfg.OffsetCfg(pos=POSITION, rot=QUATERNION)
        lidar_cfg_offset.prim_path = "/World/LidarOffset"
        lidar_offset = RtxLidar(lidar_cfg_offset)
        # Play sim
        self.sim.reset()

        # Retrieve lidar pose using USD API
        prim_tf = lidar_offset._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        prim_tf = np.transpose(prim_tf)

        # check that transform is set correctly
        np.testing.assert_allclose(prim_tf[0:3, 3], lidar_cfg_offset.offset.pos)
        np.testing.assert_allclose(
            convert_quat(tf.Rotation.from_matrix(prim_tf[:3, :3]).as_quat(), "wxyz"),
            lidar_cfg_offset.offset.rot,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_multi_lidar_init(self):
        """Test multiple lidar initialization and check info and data outputs."""
        lidar_cfg_1 = copy.deepcopy(self.lidar_cfg)
        lidar_cfg_1.prim_path = "/World/Lidar1"
        lidar_1 = RtxLidar(lidar_cfg_1)

        lidar_cfg_2 = copy.deepcopy(self.lidar_cfg)
        lidar_cfg_2.prim_path = "/World/Lidar2"
        lidar_2 = RtxLidar(lidar_cfg_2)

        # play sim
        self.sim.reset()

        # Simulate for a few steps
        # note: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(5):
            self.sim.step()
        # Simulate physics
        for i in range(10):
            # perform rendering
            self.sim.step()
            # update lidar
            lidar_1.update(self.dt, force_recompute=True)
            lidar_2.update(self.dt, force_recompute=True)
            # check lidar info
            for lidar_info_key in lidar_1.data.info[0].keys():
                info1 = lidar_1.data.info[0][lidar_info_key]
                info2 = lidar_2.data.info[0][lidar_info_key]
                if isinstance(info1, torch.Tensor):
                    torch.testing.assert_close(info1, info2)
                else:
                    if lidar_info_key == "renderProductPath":
                        self.assertTrue(info1 == info2.split("_")[0])
                    else:
                        self.assertTrue(info1 == info2)
            # check lidar data shape
            for lidar_data_key in lidar_1.data.output.to_dict().keys():
                data1 = lidar_1.data.output[lidar_data_key]
                data2 = lidar_2.data.output[lidar_data_key]
                self.assertTrue(data1.shape == data2.shape)

    def test_custom_lidar_config(self):
        """Test custom lidar initialization, data population, and cleanup."""
        # Create custom lidar profile dictionary
        with open(EXAMPLE_ROTARY_PATH) as json_file:
            sensor_profile = json.load(json_file)

        custom_lidar_cfg = copy.deepcopy(self.lidar_cfg)
        custom_lidar_cfg.spawn = sim_utils.LidarCfg(lidar_type="Custom", sensor_profile=sensor_profile)
        # Create custom lidar
        lidar = RtxLidar(cfg=custom_lidar_cfg)
        # Check simulation parameter is set correctly
        self.assertTrue(self.sim.has_rtx_sensors())
        # Play sim
        self.sim.reset()
        # Check if lidar is initialized
        self.assertTrue(lidar.is_initialized)
        # Check if lidar prim is set correctly and that it is a camera prim
        self.assertEqual(lidar._sensor_prims[0].GetPath().pathString, self.lidar_cfg.prim_path)
        self.assertIsInstance(lidar._sensor_prims[0], UsdGeom.Camera)

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
            lidar.update(self.dt, force_recompute=True)
            # check info data
            for info_key, info_value in lidar.data.info[0].items():
                self.assertTrue(info_key in RTX_LIDAR_INFO_FIELDS.keys())
                self.assertTrue(isinstance(info_value, RTX_LIDAR_INFO_FIELDS[info_key]))

            # check lidar data
            for data_key, data_value in lidar.data.output.to_dict().items():
                if data_key in self.lidar_cfg.optional_data_types:
                    self.assertTrue(data_value.shape[1] > 0)

        del lidar

        # check proper file cleanup
        custom_profile_name = self.lidar_cfg.spawn.sensor_profile_temp_prefix
        custom_profile_dir = self.lidar_cfg.spawn.sensor_profile_temp_dir
        files = os.listdir(custom_profile_dir)
        for file in files:
            self.assertTrue(
                custom_profile_name not in file, msg=f"{custom_profile_name} found in {custom_profile_dir}/{file}"
            )


if __name__ == "__main__":
    run_tests()
