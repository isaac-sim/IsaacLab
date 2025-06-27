# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

# Define a fixture to replace setUpClass
import pytest

import isaaclab_assets as lab_assets  # noqa: F401

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.sim import build_simulation_context


@pytest.fixture(scope="module")
def registered_entities():
    # load all registered entities configurations from the module
    registered_entities: dict[str, AssetBaseCfg] = {}
    # inspect all classes from the module
    for obj_name in dir(lab_assets):
        obj = getattr(lab_assets, obj_name)
        # store all registered entities configurations
        if isinstance(obj, AssetBaseCfg):
            registered_entities[obj_name] = obj
    # print all existing entities names
    print(">>> All registered entities:", list(registered_entities.keys()))
    return registered_entities


# Add parameterization for the device
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_asset_configs(registered_entities, device):
    """Check all registered asset configurations."""
    # iterate over all registered assets
    for asset_name, entity_cfg in registered_entities.items():
        # Use pytest's subtests
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
            assert entity.is_initialized
