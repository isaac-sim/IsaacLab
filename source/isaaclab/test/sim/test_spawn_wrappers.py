# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from isaacsim.core.api.simulation_context import SimulationContext

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@pytest.fixture
def sim():
    """Create a simulation context."""
    stage_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, backend="numpy")
    stage_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


def test_spawn_multiple_shapes_with_global_settings(sim):
    """Test spawning of shapes randomly with global rigid body settings."""
    num_clones = 10
    for i in range(num_clones):
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                mass_props=sim_utils.MassPropertiesCfg(mass=100.0),  # this one should get overridden
            ),
            sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
            ),
            sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            ),
        ],
        random_choice=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    prim = cfg.func("/World/env_.*/Cone", cfg)

    assert prim.IsValid()
    assert prim_utils.get_prim_path(prim) == "/World/env_0/Cone"
    prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
    assert len(prim_paths) == num_clones

    for prim_path in prim_paths:
        prim = prim_utils.get_prim_at_path(prim_path)
        assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass


def test_spawn_multiple_shapes_with_individual_settings(sim):
    """Test spawning of shapes randomly with individual rigid object settings."""
    num_clones = 10
    for i in range(num_clones):
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

    mass_variations = [2.0, 3.0, 4.0]
    cfg = sim_utils.MultiAssetSpawnerCfg(
        assets_cfg=[
            sim_utils.ConeCfg(
                radius=0.3,
                height=0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[0]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            sim_utils.CuboidCfg(
                size=(0.3, 0.3, 0.3),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[1]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            sim_utils.SphereCfg(
                radius=0.3,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=mass_variations[2]),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
        ],
        random_choice=True,
    )
    prim = cfg.func("/World/env_.*/Cone", cfg)

    assert prim.IsValid()
    assert prim_utils.get_prim_path(prim) == "/World/env_0/Cone"
    prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
    assert len(prim_paths) == num_clones

    for prim_path in prim_paths:
        prim = prim_utils.get_prim_at_path(prim_path)
        assert prim.GetAttribute("physics:mass").Get() in mass_variations


"""
Tests - Multiple USDs.
"""


def test_spawn_multiple_files_with_global_settings(sim):
    """Test spawning of files randomly with global articulation settings."""
    num_clones = 10
    for i in range(num_clones):
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))

    cfg = sim_utils.MultiUsdFileCfg(
        usd_path=[
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        ],
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
    prim = cfg.func("/World/env_.*/Robot", cfg)

    assert prim.IsValid()
    assert prim_utils.get_prim_path(prim) == "/World/env_0/Robot"
    prim_paths = prim_utils.find_matching_prim_paths("/World/env_*/Robot")
    assert len(prim_paths) == num_clones
