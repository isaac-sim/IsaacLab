# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Integration tests for simulation context with stage in memory."""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# FIXME (mmittal): Stage in memory requires cameras to be enabled.
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""


import pytest

import omni.physx
import usdrt
from isaacsim.core.cloner import GridCloner

import isaaclab.sim as sim_utils
from isaaclab.sim.simulation_context import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.version import get_isaac_sim_version


@pytest.fixture
def sim():
    """Create a simulation context."""
    cfg = SimulationCfg(create_stage_in_memory=True)
    sim = SimulationContext(cfg=cfg)
    sim_utils.update_stage()
    yield sim
    omni.physx.get_physx_simulation_interface().detach_stage()
    sim.stop()
    sim.clear_instance()


"""
Tests
"""


def test_stage_in_memory_with_shapes(sim):
    """Test spawning of shapes with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    num_clones = 10

    # verify stage is attached to USD context (happens automatically now with create_stage_in_memory)
    assert not sim_utils.is_current_stage_in_memory()

    # grab stage and set as current stage via the with statement
    stage_in_memory = sim.stage
    with sim_utils.use_stage(stage_in_memory):
        # create cloned cone stage
        for i in range(num_clones):
            sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

        cfg = sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                    ),
                ),
                sim_utils.MeshCuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                        project_uvw=True,
                        texture_scale=(0.25, 0.25),
                    ),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                        project_uvw=True,
                        texture_scale=(0.25, 0.25),
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        friction_combine_mode="multiply",
                        restitution_combine_mode="multiply",
                        static_friction=1.0,
                        dynamic_friction=1.0,
                    ),
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

        # verify prims exist in stage
        prims = sim_utils.find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_clones

    # verify prims exist in context stage
    prims = sim_utils.find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_clones


def test_stage_in_memory_with_usds(sim):
    """Test spawning of USDs with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    num_clones = 10
    usd_paths = [
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
    ]

    # verify stage is attached to USD context (happens automatically now with create_stage_in_memory)
    assert not sim_utils.is_current_stage_in_memory()

    # grab stage and set as current stage via the with statement
    stage_in_memory = sim.stage
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

        # verify prims exist in stage
        prims = sim_utils.find_matching_prim_paths(prim_path_regex)
        assert len(prims) == num_clones

    # verify prims exist in context stage
    prims = sim_utils.find_matching_prim_paths(prim_path_regex)
    assert len(prims) == num_clones


def test_stage_in_memory_with_clone_in_fabric(sim):
    """Test cloning in fabric with stage in memory."""

    # skip test if stage in memory is not supported
    if get_isaac_sim_version().major < 5:
        pytest.skip("Stage in memory is not supported in this version of Isaac Sim")

    # define parameters
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    num_clones = 100

    # verify stage is attached to USD context (happens automatically now with create_stage_in_memory)
    assert not sim_utils.is_current_stage_in_memory()

    # grab stage and set as current stage via the with statement
    stage_in_memory = sim.stage
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

    # verify prims exist in fabric stage using usdrt apis
    stage_id = sim_utils.get_current_stage_id()
    usdrt_stage = usdrt.Usd.Stage.Attach(stage_id)
    for i in range(num_clones):
        prim = usdrt_stage.GetPrimAtPath(f"/World/envs/env_{i}/Robot")
        assert prim.IsValid()
