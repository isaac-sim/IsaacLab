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

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.extensions import enable_extension, get_extension_path_from_name

from omni.isaac.lab.sim import build_simulation_context
from omni.isaac.lab.sim.converters import MjcfConverter, MjcfConverterCfg


class TestMjcfConverter(unittest.TestCase):
    """Test fixture for the MjcfConverter class."""

    def setUp(self):
        # retrieve path to mjcf importer extension
        enable_extension("omni.importer.mjcf")
        extension_path = get_extension_path_from_name("omni.importer.mjcf")
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
            prim_utils.create_prim(prim_path, usd_path=mjcf_converter.usd_path)

            self.assertTrue(prim_utils.is_prim_path_valid(prim_path))


if __name__ == "__main__":
    run_tests()
