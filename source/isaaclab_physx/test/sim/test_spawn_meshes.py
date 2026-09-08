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
from isaaclab_physx.sim.schemas.schemas_cfg import PhysxDeformableBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxDeformableBodyMaterialCfg

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.isaacsim_ci


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


def test_spawn_cone_with_deformable_and_mass_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=PhysxDeformableBodyPropertiesCfg(deformable_body_enabled=True, mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("omniphysics:deformableBodyEnabled").Get() == cfg.deformable_props.deformable_body_enabled
    assert prim.GetAttribute("omniphysics:mass").Get() == cfg.deformable_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_deformable_and_collision_fragment_mapping(sim):
    """A deformable mesh accepts ``collision_props`` as a target-pattern mapping.

    Regression: the mapping was wrapped in a list, failed the fragment type check and raised, and
    the writer call neither dispatched mapping entries nor anchored their patterns.
    """
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=PhysxDeformableBodyPropertiesCfg(deformable_body_enabled=True, mass=1.0),
        collision_props={"/sim_mesh": [PhysxCollisionCfg(contact_offset=0.02, rest_offset=0.001)]},
    )
    cfg.func("/World/ConeMap", cfg)

    sim_mesh = sim.stage.GetPrimAtPath("/World/ConeMap/sim_mesh")
    assert sim_mesh.IsValid()
    assert abs(sim_mesh.GetAttribute("physxCollision:contactOffset").Get() - 0.02) < 1e-6
    assert abs(sim_mesh.GetAttribute("physxCollision:restOffset").Get() - 0.001) < 1e-6


def test_spawn_cone_with_deformable_and_density_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API,
    specifying density instead of mass.
    """
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=PhysxDeformableBodyPropertiesCfg(deformable_body_enabled=True),
        physics_material=PhysxDeformableBodyMaterialCfg(density=10.0),
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
