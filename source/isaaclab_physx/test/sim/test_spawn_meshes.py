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
from isaaclab_physx.sim.schemas.schemas_cfg import DeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import DeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


@pytest.fixture
def sim():
    """Create a simulation context for testing."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    # Wait for spawning
    sim_utils.update_stage()
    yield sim
    # Cleanup
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


"""
Physics properties.
"""


def test_spawn_cone_with_deformable_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body API."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=DeformableBodyPropertiesCfg(deformable_body_enabled=True),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()

    # Check properties
    # Unlike rigid body, deformable body properties are on the mesh prim
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("omniphysics:deformableBodyEnabled").Get() == cfg.deformable_props.deformable_body_enabled


def test_spawn_cone_with_deformable_and_mass_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=DeformableBodyPropertiesCfg(deformable_body_enabled=True, mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("omniphysics:mass").Get() == cfg.deformable_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_deformable_and_density_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API,
    specifying density instead of mass.
    """
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=DeformableBodyPropertiesCfg(deformable_body_enabled=True),
        physics_material=DeformableBodyMaterialCfg(density=10.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/material")
    assert prim.GetAttribute("omniphysics:density").Get() == cfg.physics_material.density
    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_deformable_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with all deformable properties."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=DeformableBodyPropertiesCfg(mass=1.0),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=DeformableBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    # -- deformable body
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("omniphysics:deformableBodyEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()
