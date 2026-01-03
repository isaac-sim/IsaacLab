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
import torch

import pytest

import isaaclab.sim as sim_utils
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


"""
Basic Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_init(device):
    """Test the simulation context initialization."""
    cfg = SimulationCfg(physics_prim_path="/Physics/PhysX", render_interval=5, gravity=(0.0, -0.5, -0.5), device=device)
    # sim = SimulationContext(cfg)
    # TODO: Figure out why keyword argument doesn't work.
    # note: added a fix in Isaac Sim 2023.1 for this.
    sim = SimulationContext(cfg=cfg)

    # verify app interface is valid
    assert sim.app is not None
    # verify stage is valid
    assert sim.stage is not None
    # verify device property
    assert sim.device == device
    # verify no RTX sensors are available
    assert not sim.has_rtx_sensors()

    # obtain physics scene api
    physx_scene_api = sim._physx_scene_api  # type: ignore
    physics_scene = sim._physics_scene  # type: ignore

    # check valid settings
    physics_hz = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
    physics_dt = 1.0 / physics_hz
    assert physics_dt == cfg.dt

    # check valid paths
    assert sim.stage.GetPrimAtPath("/Physics/PhysX").IsValid()
    assert sim.stage.GetPrimAtPath("/Physics/PhysX/defaultMaterial").IsValid()
    # check valid gravity
    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity)


@pytest.mark.isaacsim_ci
def test_instance_before_creation():
    """Test accessing instance before creating returns None."""
    # clear any existing instance
    SimulationContext.clear_instance()

    # accessing instance before creation should return None
    assert SimulationContext.instance() is None


@pytest.mark.isaacsim_ci
def test_singleton():
    """Tests that the singleton is working."""
    sim1 = SimulationContext()
    sim2 = SimulationContext()
    assert sim1 is sim2

    # try to delete the singleton
    sim2.clear_instance()
    assert sim1.instance() is None
    # create new instance
    sim3 = SimulationContext()
    assert sim1 is not sim3
    assert sim1.instance() is sim3.instance()
    # clear instance
    sim3.clear_instance()


"""
Property Tests.
"""


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
    assert not sim.has_gui()


"""
Timeline Operations Tests.
"""


@pytest.mark.isaacsim_ci
def test_timeline_play_stop():
    """Test timeline play and stop operations."""
    sim = SimulationContext()

    # initially simulation should be stopped
    assert sim.is_stopped()
    assert not sim.is_playing()

    # start the simulation
    sim.play()
    assert sim.is_playing()
    assert not sim.is_stopped()

    # disable callback to prevent app from continuing
    sim._disable_app_control_on_stop_handle = True  # type: ignore
    # stop the simulation
    sim.stop()
    assert sim.is_stopped()
    assert not sim.is_playing()


@pytest.mark.isaacsim_ci
def test_timeline_pause():
    """Test timeline pause operation."""
    sim = SimulationContext()

    # start the simulation
    sim.play()
    assert sim.is_playing()

    # pause the simulation
    sim.pause()
    assert not sim.is_playing()
    assert not sim.is_stopped()  # paused is different from stopped


"""
Reset and Step Tests
"""


@pytest.mark.isaacsim_ci
def test_reset():
    """Test simulation reset."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple cube to test with
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # reset the simulation
    sim.reset()

    # check that simulation is playing after reset
    assert sim.is_playing()

    # check that physics sim view is created
    assert sim.physics_sim_view is not None


@pytest.mark.isaacsim_ci
def test_reset_soft():
    """Test soft reset (without stopping simulation)."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create a simple cube
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    # perform initial reset
    sim.reset()
    assert sim.is_playing()

    # perform soft reset
    sim.reset(soft=True)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_forward():
    """Test forward propagation for fabric updates."""
    cfg = SimulationCfg(dt=0.01, use_fabric=True)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # call forward
    sim.forward()

    # should not raise any errors
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("render", [True, False])
def test_step(render):
    """Test stepping simulation with and without rendering."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # step with rendering
    for _ in range(10):
        sim.step(render=render)

    # simulation should still be playing
    assert sim.is_playing()


@pytest.mark.isaacsim_ci
def test_render():
    """Test rendering simulation."""
    cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(cfg)

    # create simple scene
    cube_cfg = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg.func("/World/Cube", cube_cfg)

    sim.reset()

    # render
    for _ in range(10):
        sim.render()

    # simulation should still be playing
    assert sim.is_playing()


"""
Stage Operations Tests
"""


@pytest.mark.isaacsim_ci
def test_get_initial_stage():
    """Test getting the initial stage."""
    sim = SimulationContext()

    # get initial stage
    stage = sim.get_initial_stage()

    # verify stage is valid
    assert stage is not None
    assert stage == sim.stage


@pytest.mark.isaacsim_ci
def test_clear_stage():
    """Test clearing the stage."""
    sim = SimulationContext()

    # create some objects
    cube_cfg1 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg1.func("/World/Cube1", cube_cfg1)
    cube_cfg2 = sim_utils.CuboidCfg(size=(0.1, 0.1, 0.1))
    cube_cfg2.func("/World/Cube2", cube_cfg2)

    # verify objects exist
    assert sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube2").IsValid()

    # clear the stage
    sim.clear()

    # verify objects are removed but World and Physics remain
    assert not sim.stage.GetPrimAtPath("/World/Cube1").IsValid()
    assert not sim.stage.GetPrimAtPath("/World/Cube2").IsValid()
    assert sim.stage.GetPrimAtPath("/World").IsValid()
    assert sim.stage.GetPrimAtPath(sim.cfg.physics_prim_path).IsValid()


"""
Physics Configuration Tests
"""


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("solver_type", [0, 1])  # 0=PGS, 1=TGS
def test_solver_type(solver_type):
    """Test different solver types."""
    from isaaclab.sim.simulation_cfg import PhysxCfg

    cfg = SimulationCfg(physx=PhysxCfg(solver_type=solver_type))
    sim = SimulationContext(cfg)

    # obtain physics scene api
    physx_scene_api = sim._physx_scene_api  # type: ignore
    # check solver type is set
    solver_type_str = "PGS" if solver_type == 0 else "TGS"
    assert physx_scene_api.GetSolverTypeAttr().Get() == solver_type_str


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("use_fabric", [True, False])
def test_fabric_setting(use_fabric):
    """Test that fabric setting is properly set."""
    cfg = SimulationCfg(use_fabric=use_fabric)
    sim = SimulationContext(cfg)

    # check fabric is enabled
    assert sim.is_fabric_enabled() == use_fabric


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("dt", [0.01, 0.02, 0.005])
def test_physics_dt(dt):
    """Test that physics time step is properly configured."""
    cfg = SimulationCfg(dt=dt)
    sim = SimulationContext(cfg)

    # obtain physics scene api
    physx_scene_api = sim._physx_scene_api  # type: ignore
    # check physics dt
    physics_hz = physx_scene_api.GetTimeStepsPerSecondAttr().Get()
    physics_dt = 1.0 / physics_hz
    assert abs(physics_dt - dt) < 1e-6


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize("gravity", [(0.0, 0.0, 0.0), (0.0, 0.0, -9.81), (0.5, 0.5, 0.5)])
def test_custom_gravity(gravity):
    """Test that gravity can be properly set."""
    cfg = SimulationCfg(gravity=gravity)
    sim = SimulationContext(cfg)

    # obtain physics scene api
    physics_scene = sim._physics_scene  # type: ignore

    gravity_dir, gravity_mag = (
        physics_scene.GetGravityDirectionAttr().Get(),
        physics_scene.GetGravityMagnitudeAttr().Get(),
    )
    gravity = np.array(gravity_dir) * gravity_mag
    np.testing.assert_almost_equal(gravity, cfg.gravity, decimal=6)


"""
Edge Cases and Error Handling
"""


def test_boundedness():
    """Test that the boundedness of the simulation context remains constant.

    Note: This test fails right now because Isaac Sim does not handle boundedness correctly. On creation,
    it is registering itself to various callbacks and hence the boundedness is more than 1. This may not be
    critical for the simulation context since we usually call various clear functions before deleting the
    simulation context.
    """
    import ctypes

    sim = SimulationContext()
    # manually set the boundedness to 1? -- this is not possible because of Isaac Sim.

    # check that boundedness of simulation context is correct
    sim_ref_count = ctypes.c_long.from_address(id(sim)).value
    # reset the simulation
    sim.reset()
    assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count
    # step the simulation
    for _ in range(10):
        sim.step()
        assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count
    # clear the simulation
    sim.clear_instance()
    assert ctypes.c_long.from_address(id(sim)).value == sim_ref_count - 1
