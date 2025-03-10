# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import numpy as np
import os
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


class TestUrdfConverter(unittest.TestCase):
    """Test fixture for the UrdfConverter class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # retrieve path to urdf importer extension
        enable_extension("isaacsim.asset.importer.urdf")
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.urdf")
        # default configuration
        self.config = UrdfConverterCfg(
            asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
            fix_base=True,
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=40.0)
            ),
        )
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(
            physics_dt=self.dt, rendering_dt=self.dt, stage_units_in_meters=1.0, backend="numpy"
        )

    def tearDown(self) -> None:
        """Stops simulator after each test."""
        # stop simulation
        self.sim.stop()
        # cleanup stage and context
        self.sim.clear()
        self.sim.clear_all_callbacks()
        self.sim.clear_instance()

    def test_no_change(self):
        """Call conversion twice. This should not generate a new USD file."""

        urdf_converter = UrdfConverter(self.config)
        time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

        # no change to config only define the usd directory
        new_config = self.config
        new_config.usd_dir = urdf_converter.usd_dir
        # convert to usd but this time in the same directory as previous step
        new_urdf_converter = UrdfConverter(new_config)
        new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

        self.assertEqual(time_usd_file_created, new_time_usd_file_created)

    def test_config_change(self):
        """Call conversion twice but change the config in the second call. This should generate a new USD file."""

        urdf_converter = UrdfConverter(self.config)
        time_usd_file_created = os.stat(urdf_converter.usd_path).st_mtime_ns

        # change the config
        new_config = self.config
        new_config.fix_base = not self.config.fix_base
        # define the usd directory
        new_config.usd_dir = urdf_converter.usd_dir
        # convert to usd but this time in the same directory as previous step
        new_urdf_converter = UrdfConverter(new_config)
        new_time_usd_file_created = os.stat(new_urdf_converter.usd_path).st_mtime_ns

        self.assertNotEqual(time_usd_file_created, new_time_usd_file_created)

    def test_create_prim_from_usd(self):
        """Call conversion and create a prim from it."""

        urdf_converter = UrdfConverter(self.config)

        prim_path = "/World/Robot"
        prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

        self.assertTrue(prim_utils.is_prim_path_valid(prim_path))

    def test_config_drive_type(self):
        """Change the drive mechanism of the robot to be position."""

        # Create directory to dump results
        test_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(test_dir, "output", "urdf_converter")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # change the config
        self.config.force_usd_conversion = True
        self.config.joint_drive.target_type = "position"
        self.config.joint_drive.gains.stiffness = 42.0
        self.config.joint_drive.gains.damping = 4.2
        self.config.usd_dir = output_dir
        urdf_converter = UrdfConverter(self.config)
        # check the drive type of the robot
        prim_path = "/World/Robot"
        prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

        # access the robot
        robot = Articulation(prim_path, reset_xform_properties=False)
        # play the simulator and initialize the robot
        self.sim.reset()
        robot.initialize()

        # check drive values for the robot (read from physx)
        drive_stiffness, drive_damping = robot.get_gains()
        np.testing.assert_array_equal(drive_stiffness, self.config.joint_drive.gains.stiffness)
        np.testing.assert_array_equal(drive_damping, self.config.joint_drive.gains.damping)

        # check drive values for the robot (read from usd)
        self.sim.stop()
        drive_stiffness, drive_damping = robot.get_gains()
        np.testing.assert_array_equal(drive_stiffness, self.config.joint_drive.gains.stiffness)
        np.testing.assert_array_equal(drive_damping, self.config.joint_drive.gains.damping)


if __name__ == "__main__":
    run_tests()
