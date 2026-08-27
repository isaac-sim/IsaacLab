# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify PhysX lifecycle ownership when its package is imported before Kit starts."""

import sys

from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device

# Launch Kit only after importing the PhysX config to reproduce normal entry-point config resolution.
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest
from isaaclab_physx.physics import PhysxManager

import carb

from isaaclab.sim.utils import enable_extension

enable_extension("isaacsim.core.simulation_manager")
import isaacsim.core.simulation_manager as simulation_manager_module

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def setup_teardown():
    """Create a fresh stage and simulation context for each test."""
    SimulationContext.clear_instance()
    sim_utils.create_new_stage()
    yield
    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_initialize_claims_simulation_manager_lifecycle():
    """PhysxManager disables conflicting callbacks without patching supported Isaac Sim versions."""
    original_manager = simulation_manager_module.SimulationManager
    assert original_manager is not PhysxManager
    implementation_module = sys.modules["isaacsim.core.simulation_manager.impl.simulation_manager"]
    supports_startup_setting = hasattr(implementation_module, "_SETTING_ENABLE_DEFAULT_CALLBACKS")

    SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))

    setting = "/exts/isaacsim.core.simulation_manager/enable_default_callbacks"
    assert not carb.settings.get_settings().get_as_bool(setting)
    assert not any(original_manager.get_default_callback_status().values())
    if supports_startup_setting:
        assert simulation_manager_module.SimulationManager is original_manager
    else:
        assert simulation_manager_module.SimulationManager is PhysxManager
