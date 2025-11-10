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
def test_disable_sleeping_setting():
    """Test that the disable_sleeping PhysX setting is applied correctly."""
    from pxr import PhysxSchema

    # Test with disable_sleeping set to True (default)
    cfg = SimulationCfg()
    assert cfg.physx.disable_sleeping is True

    sim = SimulationContext(cfg)

    # Get the physics scene prim and check the attribute
    stage = sim.stage
    physics_scene_prim = stage.GetPrimAtPath(cfg.physics_prim_path)
    assert physics_scene_prim.IsValid()

    # Check that the attribute exists and is set to True
    disable_sleeping_attr = physics_scene_prim.GetAttribute("physxSceneAPI:disableSleeping")
    assert disable_sleeping_attr.IsValid()
    assert disable_sleeping_attr.Get() is True


@pytest.mark.isaacsim_ci
def test_disable_sleeping_false_with_cpu():
    """Test that disable_sleeping can be set to False when using CPU simulation."""
    from pxr import PhysxSchema

    # Test with disable_sleeping set to False and CPU device
    cfg = SimulationCfg(device="cpu")
    cfg.physx.disable_sleeping = False

    sim = SimulationContext(cfg)

    # Get the physics scene prim and check the attribute
    stage = sim.stage
    physics_scene_prim = stage.GetPrimAtPath(cfg.physics_prim_path)
    assert physics_scene_prim.IsValid()

    # Check that the attribute exists and is set to False
    disable_sleeping_attr = physics_scene_prim.GetAttribute("physxSceneAPI:disableSleeping")
    assert disable_sleeping_attr.IsValid()
    assert disable_sleeping_attr.Get() is False


@pytest.mark.isaacsim_ci
def test_disable_sleeping_false_with_gpu_raises_error():
    """Test that disable_sleeping=False with GPU simulation raises an error."""
    # Test with disable_sleeping set to False and GPU device
    cfg = SimulationCfg(device="cuda:0")
    cfg.physx.disable_sleeping = False

    # This should raise a RuntimeError because GPU pipeline + disable_sleeping=False is not supported
    with pytest.raises(RuntimeError, match="disable_sleeping.*GPU pipeline"):
        SimulationContext(cfg)
