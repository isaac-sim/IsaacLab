# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np

import isaacsim.core.utils.prims as prim_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext as IsaacSimulationContext

from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear any existing simulation context
    SimulationContext.clear_instance()

    # Yield for the test
    yield

    # Teardown: Clear the simulation context after each test
    SimulationContext.clear_instance()


@pytest.mark.isaacsim_ci
def test_singleton():
    """Tests that the singleton is working."""
    sim1 = SimulationContext()
    sim2 = SimulationContext()
    sim3 = IsaacSimulationContext()
    assert sim1 is sim2
    assert sim1 is sim3

    # try to delete the singleton
    sim2.clear_instance()
    assert sim1.instance() is None
    # create new instance
    sim4 = SimulationContext()
    assert sim1 is not sim4
    assert sim3 is not sim4
    assert sim1.instance() is sim4.instance()
    assert sim3.instance() is sim4.instance()
    # clear instance
    sim3.clear_instance()


@pytest.mark.isaacsim_ci
def test_initialization():
    """Test the simulation config."""
    cfg = SimulationCfg(physics_prim_path="/Physics/PhysX", render_interval=5, gravity=(0.0, -0.5, -0.5))
    sim = SimulationContext(cfg)
    # TODO: Figure out why keyword argument doesn't work.
    # note: added a fix in Isaac Sim 2023.1 for this.
    # sim = SimulationContext(cfg=cfg)

    # check valid settings
    assert sim.get_physics_dt() == cfg.dt
    assert sim.get_rendering_dt() == cfg.dt * cfg.render_interval
    assert not sim.has_rtx_sensors()
    # check valid paths
    assert prim_utils.is_prim_path_valid("/Physics/PhysX")
    assert prim_utils.is_prim_path_valid("/Physics/PhysX/defaultMaterial")
    # check valid gravity
    gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_sim_version():
    """Test obtaining the version."""
    sim = SimulationContext()
    version = sim.get_version()
    assert len(version) > 0
    assert version[0] >= 4


@pytest.mark.isaacsim_ci
def test_carb_setting():
    """Test setting carb settings."""
    sim = SimulationContext()
    # known carb setting
    sim.set_setting("/physics/physxDispatcher", False)
    assert sim.get_setting("/physics/physxDispatcher") is False
    # unknown carb setting
    sim.set_setting("/myExt/using_omniverse_version", sim.get_version())
    assert tuple(sim.get_setting("/myExt/using_omniverse_version")) == tuple(sim.get_version())


@pytest.mark.isaacsim_ci
def test_headless_mode():
    """Test that render mode is headless since we are running in headless mode."""
    sim = SimulationContext()
    # check default render mode
    assert sim.render_mode == sim.RenderMode.NO_GUI_OR_RENDERING


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


@pytest.mark.isaacsim_ci
def test_zero_gravity():
    """Test that gravity can be properly disabled."""
    cfg = SimulationCfg(gravity=(0.0, 0.0, 0.0))

    sim = SimulationContext(cfg)

    gravity_dir, gravity_mag = sim.get_physics_context().get_gravity()
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_cpu_readback_default_cuda():
    """Test default behavior with CUDA device (enable_cpu_readback=False)."""
    import carb

    # Create simulation context with default settings on CUDA
    cfg = SimulationCfg(device="cuda:0")  # enable_cpu_readback defaults to False
    sim = SimulationContext(cfg)

    # Check the carb setting - default (False) should not override omni_isaac_sim's behavior
    # omni_isaac_sim sets suppressReadback=True for CUDA by default
    carb_settings = carb.settings.get_settings()
    suppress_readback = carb_settings.get_as_bool("/physics/suppressReadback")

    # With default settings (enable_cpu_readback=False), we don't override, so omni_isaac_sim's
    # default behavior applies (suppressReadback=True for CUDA)
    assert suppress_readback is True, "Default CUDA behavior should have suppressReadback=True"


@pytest.mark.isaacsim_ci
def test_cpu_readback_enabled():
    """Test enabling CPU readback (enable_cpu_readback=True)."""
    import carb

    # Create simulation context with CPU readback enabled
    cfg = SimulationCfg(device="cuda:0", enable_cpu_readback=True)
    sim = SimulationContext(cfg)

    # Check the carb setting - should be suppressReadback=False
    carb_settings = carb.settings.get_settings()
    suppress_readback = carb_settings.get_as_bool("/physics/suppressReadback")

    assert suppress_readback is False, "enable_cpu_readback=True should set suppressReadback=False"


@pytest.mark.isaacsim_ci
def test_cpu_readback_disabled():
    """Test with CPU readback disabled (enable_cpu_readback=False, explicit)."""
    import carb

    # Create simulation context with CPU readback explicitly disabled
    cfg = SimulationCfg(device="cuda:0", enable_cpu_readback=False)
    sim = SimulationContext(cfg)

    # Check the carb setting - should use omni_isaac_sim's default (suppressReadback=True)
    carb_settings = carb.settings.get_settings()
    suppress_readback = carb_settings.get_as_bool("/physics/suppressReadback")

    # enable_cpu_readback=False means we don't override, so default applies
    assert suppress_readback is True, "enable_cpu_readback=False should use default suppressReadback=True"


@pytest.mark.isaacsim_ci
def test_cpu_readback_override():
    """Test that enable_cpu_readback properly overrides omni_isaac_sim's default behavior."""
    import carb
    import isaacsim.core.utils.stage as stage_utils

    # First create with default settings
    cfg_default = SimulationCfg(device="cuda:0")
    sim_default = SimulationContext(cfg_default)

    carb_settings = carb.settings.get_settings()
    default_value = carb_settings.get_as_bool("/physics/suppressReadback")

    # Clean up
    sim_default.clear_all_callbacks()
    sim_default.clear_instance()

    # Create stage again
    stage_utils.create_new_stage()

    # Now create with explicit enable_cpu_readback=True (opposite of default)
    cfg_override = SimulationCfg(device="cuda:0", enable_cpu_readback=True)
    sim_override = SimulationContext(cfg_override)

    override_value = carb_settings.get_as_bool("/physics/suppressReadback")

    # The override should be different from default (if default was True, override should be False)
    # enable_cpu_readback=True -> suppressReadback=False
    assert override_value is False, "enable_cpu_readback=True should result in suppressReadback=False"

    # If default was True (GPU optimized), then override should be False (CPU readback enabled)
    if default_value is True:
        assert override_value is False, "Override successfully changed suppressReadback from True to False"

    # Clean up
    sim_override.clear_all_callbacks()
    sim_override.clear_instance()


@pytest.mark.isaacsim_ci
def test_cpu_readback_ignored_on_cpu_device():
    """Test that enable_cpu_readback is ignored when simulation device is CPU."""
    import carb

    # Create simulation context with CPU device and enable_cpu_readback=True
    # This should trigger a warning but not apply any settings
    cfg = SimulationCfg(device="cpu", enable_cpu_readback=True)
    sim = SimulationContext(cfg)

    # The flag should be ignored for CPU devices
    # We can't really check the carb setting as CPU device doesn't use suppressReadback
    # but we verify that the simulation still initializes successfully
    assert sim.device == "cpu", "Simulation device should be CPU"

    # Clean up
    sim.clear_all_callbacks()
    sim.clear_instance()

