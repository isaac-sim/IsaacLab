# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Verify PhysX lifecycle ownership when its package is imported before Kit starts."""

from isaaclab_physx.physics import PhysxCfg

from isaaclab.app import AppLauncher
from isaaclab.test.utils import resolve_test_sim_device

# Launch Kit only after importing the PhysX config to reproduce normal entry-point config resolution.
simulation_app = AppLauncher(headless=True, device=resolve_test_sim_device()).app

import pytest
from isaaclab_physx.physics import PhysxManager

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
    """PhysxManager initialization disables Isaac Sim's original lifecycle callbacks."""
    original_manager = simulation_manager_module.SimulationManager
    assert original_manager is not PhysxManager

    SimulationContext(cfg=SimulationCfg(physics=PhysxCfg()))

    assert simulation_manager_module.SimulationManager is PhysxManager
    assert not any(original_manager.get_default_callback_status().values())
