# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""


import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


@pytest.fixture
def sim():
    """Create a simulation context."""
    sim_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    sim_utils.update_stage()
    yield sim
    sim.stop()
    sim.clear_instance()


def test_spawn_multiple_shapes_with_regex_prefix(sim):
    """Ensure assets are spawned and cloned when using regex prefix paths."""
    num_envs = 3
    num_assets = 3
    for env_idx in range(num_envs):
        env_path = f"/World/env_{env_idx}"
        sim_utils.create_prim(env_path, "Xform", translation=(0, 0, 0))
        sim_utils.create_prim(f"{env_path}/Cone", "Xform")

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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )

    prim = cfg.func("/World/env_.*/Cone/asset_.*", cfg)
    assert str(prim.GetPath()) == "/World/env_0/Cone/asset_0"

    prim_paths = sim_utils.find_matching_prim_paths("/World/env_.*/Cone/asset_.*")
    assert len(prim_paths) == num_assets * num_envs

    for env_idx in range(num_envs):
        for asset_idx in range(num_assets):
            path = f"/World/env_{env_idx}/Cone/asset_{asset_idx}"
            assert path in prim_paths
            assert sim.stage.GetPrimAtPath(path).GetAttribute("physics:mass").Get() == cfg.mass_props.mass


def test_spawn_multiple_shapes_with_global_settings(sim):
    """Test spawning of shapes randomly with global rigid body settings."""
    sim_utils.create_prim("/World/template", "Xform", translation=(0, 0, 0))

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
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    )
    prim = cfg.func("/World/template/Cone/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Cone/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/template/Cone/asset_.*")
    assert len(prim_paths) == 3

    for prim_path in prim_paths:
        prim = sim.stage.GetPrimAtPath(prim_path)
        assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass


def test_spawn_multiple_shapes_with_individual_settings(sim):
    """Test spawning of shapes randomly with individual rigid object settings."""
    sim_utils.create_prim("/World/template", "Xform", translation=(0, 0, 0))

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
    )
    prim = cfg.func("/World/template/Cone/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Cone/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/template/Cone/asset_.*")
    assert len(prim_paths) == 3

    for prim_path in prim_paths:
        prim = sim.stage.GetPrimAtPath(prim_path)
        assert prim.GetAttribute("physics:mass").Get() in mass_variations


"""
Tests - Multiple USDs.
"""


def test_spawn_multiple_files_with_global_settings(sim):
    """Test spawning of files randomly with global articulation settings."""
    sim_utils.create_prim("/World/template", "Xform", translation=(0, 0, 0))

    cfg = sim_utils.MultiUsdFileCfg(
        usd_path=[
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
            f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
        ],
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
    prim = cfg.func("/World/template/Robot/asset_.*", cfg)

    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/template/Robot/asset_0"
    prim_paths = sim_utils.find_matching_prim_paths("/World/template/Robot/asset_.*")
    assert len(prim_paths) == 2
