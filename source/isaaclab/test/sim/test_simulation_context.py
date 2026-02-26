# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np

import pytest
from isaaclab_newton.physics import NewtonManager

import isaaclab.sim.utils.prims as prim_utils
import isaaclab.sim.utils.stage as stage_utils
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear any existing simulation context
    SimulationContext.clear_instance()

    # Yield for the test
    yield

    # Teardown: Clear the simulation context after each test
    # Only clear stage if SimulationContext exists
    if SimulationContext.instance() is not None:
        stage_utils.clear_stage()
    SimulationContext.clear_instance()


def test_singleton():
    """Tests that the IsaacLab SimulationContext singleton is working."""
    sim1 = SimulationContext()
    sim2 = SimulationContext()
    assert sim1 is sim2

    # try to delete the singleton
    SimulationContext.clear_instance()
    assert SimulationContext.instance() is None
    # create new instance
    sim4 = SimulationContext()
    assert sim1 is not sim4
    assert SimulationContext.instance() is sim4
    # Note: Don't clear instance here - let fixture handle it


@pytest.mark.skip(reason="TODO: Physics prim creation needs to be fixed in SimulationContext")
def test_initialization():
    """Test the simulation config."""
    cfg = SimulationCfg(physics_prim_path="/Physics/PhysX", render_interval=5, gravity=(0.0, -0.5, -0.5))
    sim = SimulationContext(cfg)

    # check valid settings
    assert sim.get_physics_dt() == cfg.dt
    assert not sim._renderer_interface.has_rtx_sensors()
    # check valid paths
    assert prim_utils.is_prim_path_valid("/Physics/PhysX")
    assert prim_utils.is_prim_path_valid("/Physics/PhysX/defaultMaterial")
    # check valid gravity
    gravity = np.array(NewtonManager._gravity_vector)
    np.testing.assert_almost_equal(gravity, cfg.gravity)


def test_sim_version():
    """Test obtaining the version from isaacsim.core.version."""
    from isaacsim.core.version import get_version

    _ = SimulationContext()
    version = get_version()
    assert len(version) > 0
    assert int(version[2]) >= 4  # Major version is at index 2


def test_carb_setting():
    """Test setting carb settings."""
    sim = SimulationContext()
    # known carb setting
    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False
    # unknown carb setting
    sim.set_setting("/myExt/test_value", 42)
    assert sim.get_setting("/myExt/test_value") == 42


def test_headless_mode():
    """Test that render mode is headless since we are running in headless mode."""
    sim = SimulationContext()
    # check that we're in headless mode using settings
    has_gui = bool(sim.get_setting("/isaaclab/has_gui"))
    offscreen_render = bool(sim.get_setting("/isaaclab/render/offscreen"))
    # In headless mode, we should have no GUI and no offscreen rendering
    assert not has_gui, "Expected headless mode (no GUI)"
    assert not offscreen_render, "Expected headless mode (no offscreen rendering)"


# def test_boundedness():
#     """Test that the boundedness of the simulation context remains constant.
#
#     Note: This test fails right now because Isaac Sim does not handle boundedness correctly. On creation,
#     it is registering itself to various callbacks and hence the boundedness is more than 1. This may not be
#     critical for the simulation context since we usually call various clear functions before deleting the
#     simulation context.
#     """
#     sim = SimulationContext()
#     # manually set the boundedness to 1? -- this is not possible because of Isaac Sim.
#     sim.clear_all_callbacks()
#     sim._stage_open_callback = None
#     sim._physics_timer_callback = None
#     sim._event_timer_callback = None
#
#     # check that boundedness of simulation context is correct
#     sim_ref_count = ctypes.c_long.from_address(id(sim)).value
#     # reset the simulation
#     sim.reset()
#     assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count
#     # step the simulation
#     for _ in range(10):
#         sim.step()
#         assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count
#     # clear the simulation
#     sim.clear_instance()
#     assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count - 1


@pytest.mark.skip(reason="TODO: fix gravity")
def test_zero_gravity():
    """Test that gravity can be properly disabled."""
    cfg = SimulationCfg(gravity=(0.0, 0.0, 0.0))

    _ = SimulationContext(cfg)

    gravity = np.array(NewtonManager._gravity_vector)
    np.testing.assert_almost_equal(gravity, cfg.gravity)
