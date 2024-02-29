# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
config = {"headless": True}
simulation_app = AppLauncher(config).app

"""Rest everything follows."""

import math
import numpy as np
import os
import traceback
import unittest

import carb
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.extensions import get_extension_path_from_name

from omni.isaac.orbit.sim.converters import UrdfConverter, UrdfConverterCfg


class TestUrdfConverter(unittest.TestCase):
    """Test fixture for the UrdfConverter class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # retrieve path to urdf importer extension
        extension_path = get_extension_path_from_name("omni.importer.urdf")
        # default configuration
        self.config = UrdfConverterCfg(
            asset_path=f"{extension_path}/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf", fix_base=True
        )
        # Simulation time-step
        self.dt = 0.01
        # Load kit helper
        self.sim = SimulationContext(physics_dt=self.dt, rendering_dt=self.dt, backend="numpy")

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
        self.config.default_drive_type = "position"
        self.config.default_drive_stiffness = 400.0
        self.config.default_drive_damping = 40.0
        self.config.usd_dir = output_dir
        urdf_converter = UrdfConverter(self.config)
        # check the drive type of the robot
        prim_path = "/World/Robot"
        prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

        # access the robot
        robot = ArticulationView(prim_path, reset_xform_properties=False)
        # play the simulator and initialize the robot
        self.sim.reset()
        robot.initialize()

        # check drive values for the robot (read from physx)
        drive_stiffness, drive_damping = robot.get_gains()
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

        # check drive values for the robot (read from usd)
        self.sim.stop()
        drive_stiffness, drive_damping = robot.get_gains()
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
    try:
        unittest.main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
