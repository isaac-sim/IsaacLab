# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from omni.isaac.orbit.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import unittest

import omni.isaac.orbit_assets as orbit_assets  # noqa: F401

from omni.isaac.orbit.assets import AssetBase, AssetBaseCfg
from omni.isaac.orbit.sensors import SensorBase, SensorBaseCfg
from omni.isaac.orbit.sim import build_simulation_context


class TestValidEntitiesConfigs(unittest.TestCase):
    """Test cases for all registered entities configurations."""

    @classmethod
    def setUpClass(cls):
        # load all registered entities configurations from the module
        cls.registered_entities: dict[str, AssetBaseCfg | SensorBaseCfg] = {}
        # inspect all classes from the module
        for obj_name in dir(orbit_assets):
            obj = getattr(orbit_assets, obj_name)
            # store all registered entities configurations
            if isinstance(obj, (AssetBaseCfg, SensorBaseCfg)):
                cls.registered_entities[obj_name] = obj
        # print all existing entities names
        print(">>> All registered entities:", list(cls.registered_entities.keys()))

    """
    Test fixtures.
    """

    def test_asset_configs(self):
        """Check all registered asset configurations."""
        # iterate over all registered assets
        for asset_name, entity_cfg in self.registered_entities.items():
            for device in ("cuda:0", "cpu"):
                with self.subTest(asset_name=asset_name, device=device):
                    with build_simulation_context(device=device, auto_add_lighting=True) as sim:
                        # print the asset name
                        print(">>> Testing entities:", asset_name)
                        # name the prim path
                        entity_cfg.prim_path = "/World/asset"
                        # create the asset / sensors
                        entity: AssetBase | SensorBase = entity_cfg.class_type(entity_cfg)  # type: ignore

                        # play the sim
                        sim.reset()

                        # check asset is initialized successfully
                        self.assertTrue(entity._is_initialized)


if __name__ == "__main__":
    run_tests()
