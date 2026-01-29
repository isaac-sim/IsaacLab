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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_capsule(sim):
    """Test spawning of UsdGeomMesh as a capsule prim."""
    # Spawn capsule
    cfg = sim_utils.MeshCapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Capsule").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Capsule/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cylinder(sim):
    """Test spawning of UsdGeomMesh as a cylinder prim."""
    # Spawn cylinder
    cfg = sim_utils.MeshCylinderCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cylinder", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cylinder").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cylinder/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_cuboid(sim):
    """Test spawning of UsdGeomMesh as a cuboid prim."""
    # Spawn cuboid
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0))
    prim = cfg.func("/World/Cube", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_spawn_sphere(sim):
    """Test spawning of UsdGeomMesh as a sphere prim."""
    # Spawn sphere
    cfg = sim_utils.MeshSphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Sphere").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Sphere/geometry/mesh")
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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()

    # Check properties
    # Unlike rigid body, deformable body properties are on the mesh prim
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    # -- deformable body
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
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
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()
    # Check properties
    # -- rigid body
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() == cfg.rigid_props.rigid_body_enabled
    assert (
        prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get()
        == cfg.rigid_props.solver_position_iteration_count
    )
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(cfg.rigid_props.sleep_threshold)
    # -- mass
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass
    # -- collision shape
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()
