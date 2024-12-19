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


import os
import unittest

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.extensions import enable_extension, get_extension_path_from_name

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg


class TestMjcfConverter(unittest.TestCase):
    """Test fixture for the MjcfConverter class."""

    def setUp(self):
        """Create a blank new stage for each test."""
        # Create a new stage
        stage_utils.create_new_stage()
        # retrieve path to mjcf importer extension
        enable_extension("isaacsim.asset.importer.mjcf")
        extension_path = get_extension_path_from_name("isaacsim.asset.importer.mjcf")
        # default configuration
        self.config = MjcfConverterCfg(
            asset_path=f"{extension_path}/data/mjcf/nv_ant.xml",
            import_sites=True,
            fix_base=False,
            make_instanceable=True,
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

        urdf_converter = MjcfConverter(self.config)

        prim_path = "/World/Robot"
        prim_utils.create_prim(prim_path, usd_path=urdf_converter.usd_path)

        self.assertTrue(prim_utils.is_prim_path_valid(prim_path))


if __name__ == "__main__":
    run_tests()
