# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher, run_tests

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import unittest

import isaaclab_assets as lab_assets  # noqa: F401

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.sim import build_simulation_context


class TestValidEntitiesConfigs(unittest.TestCase):
    """Test cases for all registered entities configurations."""

    @classmethod
    def setUpClass(cls):
        # load all registered entities configurations from the module
        cls.registered_entities: dict[str, AssetBaseCfg] = {}
        # inspect all classes from the module
        for obj_name in dir(lab_assets):
            obj = getattr(lab_assets, obj_name)
            # store all registered entities configurations
            if isinstance(obj, AssetBaseCfg):
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
                        sim._app_control_on_stop_handle = None
                        # print the asset name
                        print(f">>> Testing entity {asset_name} on device {device}")
                        # name the prim path
                        entity_cfg.prim_path = "/World/asset"
                        # create the asset / sensors
                        entity: AssetBase = entity_cfg.class_type(entity_cfg)  # type: ignore

                        # play the sim
                        sim.reset()

                        # check asset is initialized successfully
                        self.assertTrue(entity.is_initialized)


if __name__ == "__main__":
    run_tests()
