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


import os
import unittest

import omni.kit.app
import omni.usd

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.sim.converters import MjcfConverter, MjcfConverterCfg


class TestMjcfConverter(unittest.TestCase):
    """Test fixture for the MjcfConverter class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # enable mjcf importer extension
        extension_manager = omni.kit.app.get_app().get_extension_manager()
        extension_manager.set_extension_enabled_immediate("omni.importer.mjcf", True)
        # retrieve path to mjcf importer extension
        extension_id = extension_manager.get_enabled_extension_id("omni.importer.mjcf")
        extension_path = extension_manager.get_extension_path(extension_id)

        # default configuration
        self.config = MjcfConverterCfg(
            asset_path=f"{extension_path}/data/mjcf/nv_ant.xml",
            import_sites=True,
            fix_base=False,
            make_instanceable=True,
        )

    def test_no_change(self):
        """Call conversion twice. This should not generate a new USD file."""
        with build_simulation_context():
            mjcf_converter = MjcfConverter(self.config)
            time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

            # no change to config only define the usd directory
            new_config = self.config
            new_config.usd_dir = mjcf_converter.usd_dir
            # convert to usd but this time in the same directory as previous step
            new_mjcf_converter = MjcfConverter(new_config)
            new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

            self.assertEqual(time_usd_file_created, new_time_usd_file_created)

    def test_config_change(self):
        """Call conversion twice but change the config in the second call. This should generate a new USD file."""
        with build_simulation_context():
            mjcf_converter = MjcfConverter(self.config)
            time_usd_file_created = os.stat(mjcf_converter.usd_path).st_mtime_ns

            # change the config
            new_config = self.config
            new_config.fix_base = not self.config.fix_base
            # define the usd directory
            new_config.usd_dir = mjcf_converter.usd_dir
            # convert to usd but this time in the same directory as previous step
            new_mjcf_converter = MjcfConverter(new_config)
            new_time_usd_file_created = os.stat(new_mjcf_converter.usd_path).st_mtime_ns

            self.assertNotEqual(time_usd_file_created, new_time_usd_file_created)

    def test_create_prim_from_usd(self):
        """Call conversion and create a prim from it."""
        with build_simulation_context():
            mjcf_converter = MjcfConverter(self.config)

            prim_path = "/World/Robot"
            sim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)

            # get current stage
            stage = omni.usd.get_context().get_stage()
            self.assertTrue(stage.GetPrimAtPath(prim_path).IsValid())


if __name__ == "__main__":
    run_tests()
