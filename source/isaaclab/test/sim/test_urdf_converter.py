# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import numpy as np
import pytest
from packaging.version import Version

import omni.kit.app
from isaacsim.core.prims import Articulation

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils.version import get_isaac_sim_version


# Create a fixture for setup and teardown
@pytest.fixture
def sim_config():
    # Create a new stage
    sim_utils.create_new_stage()
    # pin the urdf importer extension to the older version
    manager = omni.kit.app.get_app().get_extension_manager()
    if get_isaac_sim_version() >= Version("5.1"):
        pinned_urdf_extension_name = "isaacsim.asset.importer.urdf-2.4.31"
        manager.set_extension_enabled_immediate(pinned_urdf_extension_name, True)
    else:
        pinned_urdf_extension_name = "isaacsim.asset.importer.urdf"
    # obtain the extension path
    extension_id = manager.get_enabled_extension_id(pinned_urdf_extension_name)
    extension_path = manager.get_extension_path(extension_id)
    # default configuration
    config = UrdfConverterCfg(
        asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
        fix_base=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0)
        ),
    )
    # Simulation time-step
    dt = 0.01
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    yield sim, config
    # Teardown
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_no_change(sim_config):
    """Call conversion twice. This should not generate a new USD file."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # no change to config only define the usd directory
    new_config = config
    new_config.usd_dir = urdf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created == new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_config_change(sim_config):
    """Call conversion twice but change the config in the second call. This should generate a new USD file."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)
    time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

    # change the config
    new_config = config
    new_config.fix_base = not config.fix_base
    # define the usd directory
    new_config.usd_dir = urdf_converter.usd_dir
    # convert to usd but this time in the same directory as previous step
    new_urdf_converter = UrdfConverter(new_config)
    new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

    assert time_usd_file_created != new_time_usd_file_created


@pytest.mark.isaacsim_ci
def test_create_prim_from_usd(sim_config):
    """Call conversion and create a prim from it."""
    sim, config = sim_config
    urdf_converter = UrdfConverter(config)

    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    assert sim.stage.GetPrimAtPath(prim_path).IsValid()


@pytest.mark.isaacsim_ci
def test_config_drive_type(sim_config):
    """Change the drive mechanism of the robot to be position."""
    sim, config = sim_config
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "urdf_converter")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # change the config
    config.force_usd_conversion = True
    config.joint_drive.target_type = "position"
    config.joint_drive.gains.stiffness = 42.0
    config.joint_drive.gains.damping = 4.2
    config.usd_dir = output_dir
    urdf_converter = UrdfConverter(config)
    # check the drive type of the robot
    prim_path = "/World/Robot"
    sim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

    # access the robot
    robot = Articulation(prim_path, reset_xform_properties=False)
    # play the simulator and initialize the robot
    sim.reset()
    robot.initialize()

    # check drive values for the robot (read from physx)
    drive_stiffness, drive_damping = robot.get_gains()
    np.testing.assert_array_equal(drive_stiffness.cpu().numpy(), config.joint_drive.gains.stiffness)
    np.testing.assert_array_equal(drive_damping.cpu().numpy(), config.joint_drive.gains.damping)

    # check drive values for the robot (read from usd)
    # Note: Disable the app control callback to prevent hanging during sim.stop()
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    drive_stiffness, drive_damping = robot.get_gains()
    np.testing.assert_array_equal(drive_stiffness.cpu().numpy(), config.joint_drive.gains.stiffness)
    np.testing.assert_array_equal(drive_damping.cpu().numpy(), config.joint_drive.gains.damping)
