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


@pytest.fixture
def sim():
    """Create a simulation context for testing."""
    # Create a new stage
    stage_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(physics_dt=dt, rendering_dt=dt, device="cuda:0")
    # Wait for spawning
    stage_utils.update_stage()
    yield sim
    # Cleanup
    sim.stop()
    sim.clear()
    sim.clear_all_callbacks()
    sim.clear_instance()


"""
Basic spawning.
"""


def test_spawn_cone(sim):
    """Test spawning of UsdGeomMesh as a cone prim."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_capsule(sim):
    """Test spawning of UsdGeomMesh as a capsule prim."""
    # Spawn capsule
    cfg = sim_utils.MeshCapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Capsule")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Capsule/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cylinder(sim):
    """Test spawning of UsdGeomMesh as a cylinder prim."""
    # Spawn cylinder
    cfg = sim_utils.MeshCylinderCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cylinder", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cylinder")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cylinder/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cuboid(sim):
    """Test spawning of UsdGeomMesh as a cuboid prim."""
    # Spawn cuboid
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0))
    prim = cfg.func("/World/Cube", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cube")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_sphere(sim):
    """Test spawning of UsdGeomMesh as a sphere prim."""
    # Spawn sphere
    cfg = sim_utils.MeshSphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Sphere")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Sphere/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


"""
Physics properties.
"""


def test_spawn_cone_with_deformable_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body API."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
    )
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")

    # Check properties
    # Unlike rigid body, deformable body properties are on the mesh prim
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physxDeformable:deformableEnabled").Get() == cfg.deformable_props.deformable_enabled


def test_spawn_cone_with_deformable_and_mass_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_deformable_and_density_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with deformable body and mass API.

    Note:
        In this case, we specify the density instead of the mass. In that case, physics need to know
        the collision shape to compute the mass. Thus, we have to set the collider properties. In
        order to not have a collision shape, we disable the collision.
    """
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(deformable_enabled=True),
        mass_props=sim_utils.MassPropertiesCfg(density=10.0),
    )
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:density").Get() == cfg.mass_props.density
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
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.DeformableBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    assert prim_utils.is_prim_path_valid("/World/Cone/geometry/material")
    # Check properties
    # -- deformable body
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physxDeformable:deformableEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_rigid_props(sim):
    """Test spawning of UsdGeomMesh prim for a cone with all rigid properties."""
    # Spawn cone
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    assert prim_utils.is_prim_path_valid("/World/Cone/geometry/material")
    # Check properties
    # -- rigid body
    prim = prim_utils.get_prim_at_path("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() == cfg.rigid_props.rigid_body_enabled
    assert (
        prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get()
        == cfg.rigid_props.solver_position_iteration_count
    )
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(cfg.rigid_props.sleep_threshold)
    # -- mass
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass
    # -- collision shape
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()
