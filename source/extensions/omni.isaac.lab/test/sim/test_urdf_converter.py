# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from omni.isaac.lab.app import AppLauncher, run_tests

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import math
import numpy as np
import os
import unittest

import omni.isaac.core.utils.prims as prim_utils
import omni.usd
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name

from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.sim.converters import UrdfConverter, UrdfConverterCfg


class TestUrdfConverter(unittest.TestCase):
    """Test fixture for the UrdfConverter class."""

    def setUp(self):
        # retrieve path to urdf importer extension
        enable_extension("omni.importer.urdf")
        extension_path = get_extension_path_from_name("omni.importer.urdf")
        # default configuration
        self.config = UrdfConverterCfg(
            asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf",
            fix_base=True,
        )

    def test_no_change(self):
        """Call conversion twice. This should not generate a new USD file."""
        with build_simulation_context():
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
        with build_simulation_context():
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
        with build_simulation_context():
            urdf_converter = UrdfConverter(self.config)

            prim_path = "/World/Robot"
            prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            self.assertTrue(stage.GetPrimAtPath(prim_path).IsValid())

    def test_config_drive_type(self):
        """Change the drive mechanism of the robot to be position."""
        with build_simulation_context() as sim:
            # Create directory to dump results
            test_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(test_dir, "output", "urdf_converter")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            # change the config
            self.config.force_usd_conversion = True
            self.config.default_drive_type = "position"
            self.config.default_drive_stiffness = 400.0
            self.config.default_drive_damping = 40.0
            self.config.override_joint_dynamics = True
            self.config.usd_dir = output_dir
            urdf_converter = UrdfConverter(self.config)
            # check the drive type of the robot
            prim_path = "/World/Robot"
            prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

            # access the robot
            robot = ArticulationView(prim_path, reset_xform_properties=False)
            # play the simulator and initialize the robot
            sim.reset()
            robot.initialize()

            # check drive values for the robot (read from physx)
            drive_stiffness, drive_damping = robot.get_gains()
            # convert to numpy
            drive_stiffness = drive_stiffness.cpu().numpy()
            drive_damping = drive_damping.cpu().numpy()

            # -- for the arm (revolute joints)
            # user provides the values in radians but simulator sets them as in degrees
            expected_drive_stiffness = math.degrees(self.config.default_drive_stiffness)
            expected_drive_damping = math.degrees(self.config.default_drive_damping)
            np.testing.assert_array_equal(drive_stiffness[:, :7], expected_drive_stiffness)
            np.testing.assert_array_equal(drive_damping[:, :7], expected_drive_damping)
            # -- for the hand (prismatic joints)
            # note: from isaac sim 2023.1, the test asset has mimic joints for the hand
            #  so the mimic joint doesn't have drive values
            expected_drive_stiffness = self.config.default_drive_stiffness
            expected_drive_damping = self.config.default_drive_damping
            np.testing.assert_array_equal(drive_stiffness[:, 7], expected_drive_stiffness)
            np.testing.assert_array_equal(drive_damping[:, 7], expected_drive_damping)

            # stop the simulator
            sim.stop()
            # check drive values for the robot (read from usd since the simulator is stopped)
            drive_stiffness, drive_damping = robot.get_gains()
            # convert to numpy
            drive_stiffness = drive_stiffness.cpu().numpy()
            drive_damping = drive_damping.cpu().numpy()

            # -- for the arm (revolute joints)
            # user provides the values in radians but simulator sets them as in degrees
            expected_drive_stiffness = math.degrees(self.config.default_drive_stiffness)
            expected_drive_damping = math.degrees(self.config.default_drive_damping)
            np.testing.assert_array_equal(drive_stiffness[:, :7], expected_drive_stiffness)
            np.testing.assert_array_equal(drive_damping[:, :7], expected_drive_damping)
            # -- for the hand (prismatic joints)
            # note: from isaac sim 2023.1, the test asset has mimic joints for the hand
            #  so the mimic joint doesn't have drive values
            expected_drive_stiffness = self.config.default_drive_stiffness
            expected_drive_damping = self.config.default_drive_damping
            np.testing.assert_array_equal(drive_stiffness[:, 7], expected_drive_stiffness)
            np.testing.assert_array_equal(drive_damping[:, 7], expected_drive_damping)


if __name__ == "__main__":
    run_tests()
