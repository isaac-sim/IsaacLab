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
from collections.abc import Generator

import omni.physx
import omni.usd
import pytest
import usdrt
from isaacsim.core.api.simulation_context import SimulationContext as IsaacSimulationContext
from isaacsim.core.cloner import GridCloner
from isaacsim.core.version import get_version

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Setup and teardown for each test."""
    # Setup: Clear any existing simulation context
    SimulationContext.clear_instance()

    # Yield for the test
    yield

    # Teardown: Clear the simulation context after each test
    SimulationContext.clear_instance()


@pytest.fixture
def sim_with_stage_in_memory() -> Generator[SimulationContext, None, None]:
    """Create a simulation context with stage in memory."""
    # create stage in memory
    cfg = SimulationCfg(create_stage_in_memory=True)
    sim = SimulationContext(cfg=cfg)
    # update stage
    sim_utils.update_stage()
    # yield simulation context
    yield sim
    # stop simulation
    omni.physx.get_physx_simulation_interface().detach_stage()
    sim.stop()
    # clear simulation context
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


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
    assert sim.stage.GetPrimAtPath("/Physics/PhysX").IsValid()
    assert sim.stage.GetPrimAtPath("/Physics/PhysX/defaultMaterial").IsValid()
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


"""
Integration tests for simulation context with stage in memory.
"""


def test_stage_in_memory_with_shapes(sim_with_stage_in_memory):
    """Test spawning of shapes with stage in memory."""

    # skip test if stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    num_clones = 10

    # grab stage in memory and set as current stage via the with statement
    stage_in_memory = sim_with_stage_in_memory.get_initial_stage()
    with sim_utils.use_stage(stage_in_memory):
        # create cloned cone stage
        for i in range(num_clones):
            sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        prim_path_regex = "/World/env_.*/Cone"
        cfg.func(prim_path_regex, cfg)

        # verify stage is in memory
        assert sim_utils.is_current_stage_in_memory()

        # verify prims exist in stage in memory
        prims = sim_utils.find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_clones

        # verify prims do not exist in context stage
        context_stage = omni.usd.get_context().get_stage()
        with sim_utils.use_stage(context_stage):
            prims = sim_utils.find_matching_prim_paths(prim_path_regex)
            assert len(prims) != num_clones

        # attach stage to context
        sim_utils.attach_stage_to_usd_context()

    # verify stage is no longer in memory
    assert not sim_utils.is_current_stage_in_memory()

    # verify prims now exist in context stage
    prims = sim_utils.find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_clones


def test_stage_in_memory_with_usd_references(sim_with_stage_in_memory):
    """Test spawning of USDs with stage in memory."""

    # skip test if stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    num_clones = 10
    usd_paths = [
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
    ]

    # grab stage in memory and set as current stage via the with statement
    stage_in_memory = sim_with_stage_in_memory.get_initial_stage()
    with sim_utils.use_stage(stage_in_memory):
        # create cloned robot stage
        for i in range(num_clones):
            sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        cfg = sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            activate_contact_sensors=True,
        )
        prim_path_regex = "/World/env_.*/Robot"
        cfg.func(prim_path_regex, cfg)

        # verify stage is in memory
        assert sim_utils.is_current_stage_in_memory()

        # verify prims exist in stage in memory
        prims = sim_utils.find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_clones

        # verify prims do not exist in context stage
        context_stage = omni.usd.get_context().get_stage()
        with sim_utils.use_stage(context_stage):
            prims = sim_utils.find_matching_prim_paths(prim_path_regex)
            assert len(prims) != num_clones

        # attach stage to context
        sim_utils.attach_stage_to_usd_context()

    # verify stage is no longer in memory
    assert not sim_utils.is_current_stage_in_memory()

    # verify prims now exist in context stage
    prims = sim_utils.find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_clones


def test_stage_in_memory_with_clone_in_fabric(sim_with_stage_in_memory):
    """Test cloning in fabric with stage in memory."""

    # skip test if stage in memory is not supported
    isaac_sim_version = float(".".join(get_version()[2]))
    if isaac_sim_version < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    num_clones = 100

    # grab stage in memory and set as current stage via the with statement
    stage_in_memory = sim_with_stage_in_memory.get_initial_stage()
    with sim_utils.use_stage(stage_in_memory):
        # set up paths
        base_env_path = "/World/envs"
        source_prim_path = f"{base_env_path}/env_0"

        # create cloner
        cloner = GridCloner(spacing=3, stage=stage_in_memory)
        cloner.define_base_env(base_env_path)

        # create source prim
        sim_utils.create_prim(f"{source_prim_path}/Robot", "Xform", usd_path=usd_path)

        # generate target paths
        target_paths = cloner.generate_paths("/World/envs/env", num_clones)

        # clone robots at target paths
        cloner.clone(
            source_prim_path=source_prim_path,
            base_env_path=base_env_path,
            prim_paths=target_paths,
            replicate_physics=True,
            clone_in_fabric=True,
        )
        prim_path_regex = "/World/envs/env_.*"

        # verify prims do not exist in context stage
        context_stage = omni.usd.get_context().get_stage()
        with sim_utils.use_stage(context_stage):
            prims = sim_utils.find_matching_prim_paths(prim_path_regex)
            assert len(prims) != num_clones

        # attach stage to context
        sim_utils.attach_stage_to_usd_context()

    # verify stage is no longer in memory
    assert not sim_utils.is_current_stage_in_memory()

    # verify prims now exist in fabric stage using usdrt apis
    stage_id = sim_utils.get_current_stage_id()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    for i in range(num_clones):
        prim = usdrt_stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        assert prim.IsValid()
