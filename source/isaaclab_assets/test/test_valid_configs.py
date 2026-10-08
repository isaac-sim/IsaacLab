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
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators import BamActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBase, AssetBaseCfg
from isaaclab.cloner import CloneCfg, clone_plan_from_env_0, replicate
from isaaclab.sim import SimulationCfg, build_simulation_context
from isaaclab.test.utils import DeviceScope, test_devices

import isaaclab_assets as lab_assets  # noqa: F401


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
        # BAM actuators run only on Newton MJWarp, which builds articulations from a clone plan
        uses_bam = (
            isinstance(entity_cfg, ArticulationCfg)
            and isinstance(entity_cfg.actuators, dict)
            and any(isinstance(actuator, BamActuatorCfg) for actuator in entity_cfg.actuators.values())
        )
        sim_cfg = None
        if uses_bam:
            sim_cfg = SimulationCfg(
                device=device, use_newton_actuators=True, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg())
            )
        with build_simulation_context(device=device, auto_add_lighting=True, sim_cfg=sim_cfg) as sim:
            sim._app_control_on_stop_handle = None
            # print the asset name
            print(f">>> Testing entity {asset_name} on device {device}")
            if uses_bam:
                sim_utils.create_prim("/World/Env_0", "Xform")
                entity_cfg.prim_path = "/World/Env_[^/]*/asset"
                entity: AssetBase = instantiate(entity_cfg)  # type: ignore
                clone_plan_from_env_0(CloneCfg(clone_template="/World/Env_{}"), [entity_cfg], 1, 1.0)
                replicate(sim.get_clone_plan())
            else:
                # name the prim path
                entity_cfg.prim_path = "/World/asset"
                # create the asset / sensors
                entity: AssetBase = instantiate(entity_cfg)  # type: ignore

            # play the sim
            sim.reset()

            # check asset is initialized successfully
            assert entity.is_initialized
