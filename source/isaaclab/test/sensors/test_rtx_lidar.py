# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = app_launcher.app

import copy
import json
import numpy as np
import os
import scipy.spatial.transform as tf
import torch

import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.utils.extensions import get_extension_path_from_name
from pxr import Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sensors.rtx_lidar import RTX_LIDAR_INFO_FIELDS, RtxLidar, RtxLidarCfg
from isaaclab.sim import build_simulation_context
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.math import convert_quat

POSITION = (0.0, 0.0, 0.3)
QUATERNION = (0.0, 0.3461835, 0.0, 0.9381668)
# QUATERNION = (0.0, 0.0,0.0,1.0)

# load example json
EXAMPLE_ROTARY_PATH = os.path.abspath(
    os.path.join(
        get_extension_path_from_name("isaacsim.sensors.rtx"),
        "data/lidar_configs/NVIDIA/Simple_Example_Solid_State.json",
    )
)


@pytest.fixture
def sim(request):
    """Create simulation context with the specified device."""
    device = request.getfixturevalue("device")
    with build_simulation_context(device=device, dt=0.01) as sim:
        sim._app_control_on_stop_handle = None
        # Ground-plane
        mesh = make_plane(size=(10, 10), height=0.0, center_zero=True)
        border = make_border(size=(10, 10), inner_size=(5, 5), height=2.0, position=(0.0, 0.0, 0.0))

        create_prim_from_mesh("/World/defaultGroundPlane", mesh)
        for i, box in enumerate(border):
            create_prim_from_mesh(f"/World/defaultBoarder{i}", box)
        # load stage
        stage_utils.update_stage()
        yield sim


@pytest.fixture
def lidar_cfg(request):
    # configure lidar
    return RtxLidarCfg(
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_lidar_init(sim, device, lidar_cfg):
    """Test lidar initialization and data population."""
    # Create lidar
    lidar = RtxLidar(cfg=lidar_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if lidar is initialized
    assert lidar.is_initialized
    # Check if lidar prim is set correctly and that it is a camera prim
    assert lidar._sensor_prims[0].GetPath().pathString == lidar_cfg.prim_path
    assert isinstance(lidar._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        lidar.update(sim.get_physics_dt(), force_recompute=True)
        # check info data
        for info_key, info_value in lidar.data.info[0].items():
            assert info_key in RTX_LIDAR_INFO_FIELDS.keys()
            assert isinstance(info_value, RTX_LIDAR_INFO_FIELDS[info_key])

        # check lidar data
        for data_key, data_value in lidar.data.output.items():
            if data_key in lidar_cfg.optional_data_types:
                assert data_value.shape[1] > 0


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_offset_lidar_init(sim, device, lidar_cfg):
    """Test lidar offset configuration."""
    lidar_cfg_offset = copy.deepcopy(lidar_cfg)
    lidar_cfg_offset.offset = RtxLidarCfg.OffsetCfg(pos=POSITION, rot=QUATERNION)
    lidar_cfg_offset.prim_path = "/World/LidarOffset"
    lidar = RtxLidar(lidar_cfg_offset)

    # Play sim
    sim.reset()

    # Retrieve lidar pose using USD API
    prim_tf = lidar._sensor_prims[0].ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    prim_tf = np.transpose(prim_tf)

    # check that transform is set correctly
    np.testing.assert_allclose(prim_tf[0:3, 3], lidar_cfg_offset.offset.pos)
    np.testing.assert_allclose(
        convert_quat(tf.Rotation.from_matrix(prim_tf[:3, :3]).as_quat(), "wxyz"),
        lidar_cfg_offset.offset.rot,
        rtol=1e-5,
        atol=1e-5,
    )

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    lidar.update(sim.get_physics_dt())


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_multiple_lidar_init(sim, device, lidar_cfg):
    """Test multiple lidar initialization and check info and data outputs."""

    sim._app_control_on_stop_handle = None
    lidar_cfg_1 = copy.deepcopy(lidar_cfg)
    lidar_cfg_1.prim_path = "/World/Lidar1"
    lidar_1 = RtxLidar(lidar_cfg_1)

    lidar_cfg_2 = copy.deepcopy(lidar_cfg)
    lidar_cfg_2.prim_path = "/World/Lidar2"
    lidar_2 = RtxLidar(lidar_cfg_2)

    # play sim
    sim.reset()

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()
    # Simulate physics
    for i in range(10):
        # perform rendering
        sim.step()
        # update lidar
        lidar_1.update(sim.get_physics_dt())
        lidar_2.update(sim.get_physics_dt())
        # check lidar info
        for lidar_info_key in lidar_1.data.info[0].keys():
            info1 = lidar_1.data.info[0][lidar_info_key]
            info2 = lidar_2.data.info[0][lidar_info_key]
            if isinstance(info1, torch.Tensor):
                torch.testing.assert_close(info1, info2)
            else:
                if lidar_info_key == "renderProductPath":
                    assert info1 == info2.split("_")[0]
                else:
                    assert info1 == info2
        # check lidar data shape both instances should produce the same amount of data
        for lidar_data_key in lidar_1.data.output.keys():
            data1 = lidar_1.data.output[lidar_data_key]
            data2 = lidar_2.data.output[lidar_data_key]
            assert data1.shape == data2.shape


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_custom_lidar_init(sim, device, lidar_cfg):
    """Test custom lidar initialization, data population, and cleanup."""
    # Create custom lidar profile dictionary
    with open(EXAMPLE_ROTARY_PATH) as json_file:
        sensor_profile = json.load(json_file)

    custom_lidar_cfg = copy.deepcopy(lidar_cfg)
    custom_lidar_cfg.spawn = sim_utils.LidarCfg(lidar_type="Custom", sensor_profile=sensor_profile)
    # Create custom lidar
    lidar = RtxLidar(cfg=custom_lidar_cfg)
    # Check simulation parameter is set correctly
    assert sim.has_rtx_sensors()
    # Play sim
    sim.reset()
    # Check if lidar is initialized
    assert lidar.is_initialized
    # Check if lidar prim is set correctly and that it is a camera prim
    assert lidar._sensor_prims[0].GetPath().pathString == lidar_cfg.prim_path
    assert isinstance(lidar._sensor_prims[0], UsdGeom.Camera)

    # Simulate for a few steps
    # note: This is a workaround to ensure that the textures are loaded.
    #   Check "Known Issues" section in the documentation for more details.
    for _ in range(5):
        sim.step()

    # Simulate physics
    for _ in range(10):
        # perform rendering
        sim.step()
        # update camera
        lidar.update(sim.get_physics_dt(), force_recompute=True)
        # check info data
        for info_key, info_value in lidar.data.info[0].items():
            assert info_key in RTX_LIDAR_INFO_FIELDS.keys()
            assert isinstance(info_value, RTX_LIDAR_INFO_FIELDS[info_key])

        # check lidar data
        for data_key, data_value in lidar.data.output.items():
            if data_key in lidar_cfg.optional_data_types:
                assert data_value.shape[1] > 0

    del lidar

    # check proper file cleanup
    custom_profile_name = lidar_cfg.spawn.sensor_profile_temp_prefix
    custom_profile_dir = lidar_cfg.spawn.sensor_profile_temp_dir
    files = os.listdir(custom_profile_dir)
    for file in files:
        assert custom_profile_name not in file


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--maxfail=1"])
