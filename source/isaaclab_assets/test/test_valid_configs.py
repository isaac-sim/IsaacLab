# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of variables warning
# pyright: reportPrivateUsage=none

from isaaclab.test.utils import launch_test_simulation
from isaaclab.utils import instantiate

launch_test_simulation()

# Define a fixture to replace setUpClass
import pytest

from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.sim import build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_assets as lab_assets  # noqa: F401
import isaaclab_assets.robots.franka as franka_assets


def test_franka_legacy_configs_warn_and_keep_their_contract() -> None:
    """Deprecated Franka names retain their previous asset and actuator contracts."""
    with pytest.warns(DeprecationWarning, match="FRANKA_PANDA_CFG.*removed in Isaac Lab 4.0"):
        legacy_cfg = franka_assets.FRANKA_PANDA_CFG
    assert legacy_cfg.spawn.usd_path.endswith("/Legacy/panda_instanceable.usd")
    assert set(legacy_cfg.actuators) == {"panda_shoulder", "panda_forearm", "panda_hand"}

    with pytest.warns(DeprecationWarning, match="FRANKA_PANDA_HIGH_PD_CFG.*removed in Isaac Lab 4.0"):
        high_pd_cfg = franka_assets.FRANKA_PANDA_HIGH_PD_CFG
    assert high_pd_cfg.spawn.usd_path == legacy_cfg.spawn.usd_path
    assert high_pd_cfg.actuators["panda_shoulder"].stiffness == 400.0
    assert high_pd_cfg.actuators["panda_forearm"].stiffness == 400.0

    with pytest.warns(DeprecationWarning, match="franka_panda_nestedInstance.usda"):
        menagerie_cfg = franka_assets.FRANKA_PANDA_MENAGERIE_CFG
    assert menagerie_cfg.spawn.usd_path.endswith("/franka_panda_nestedInstance.usda")
    assert set(menagerie_cfg.actuators) == {"panda_arm", "panda_hand"}


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


# config validity (USD paths, joint/body names, actuator patterns) does not depend on the device
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
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
            entity: AssetBase = instantiate(entity_cfg)  # type: ignore

            # play the sim
            sim.reset()

            # check asset is initialized successfully
            assert entity.is_initialized
