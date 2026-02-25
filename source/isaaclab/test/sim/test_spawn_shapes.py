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
    """Create a simulation context."""
    sim_utils.create_new_stage()
    dt = 0.1
    sim = SimulationContext(SimulationCfg(dt=dt))
    sim_utils.update_stage()
    yield sim
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


"""
Basic spawning.
"""


def test_spawn_cone(sim):
    """Test spawning of UsdGeom.Cone prim."""
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cone"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_capsule(sim):
    """Test spawning of UsdGeom.Capsule prim."""
    cfg = sim_utils.CapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Capsule").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Capsule/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Capsule"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_cylinder(sim):
    """Test spawning of UsdGeom.Cylinder prim."""
    cfg = sim_utils.CylinderCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cylinder", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cylinder").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cylinder/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cylinder"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_cuboid(sim):
    """Test spawning of UsdGeom.Cube prim."""
    cfg = sim_utils.CuboidCfg(size=(1.0, 2.0, 3.0))
    prim = cfg.func("/World/Cube", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cube"
    assert prim.GetAttribute("size").Get() == min(cfg.size)


def test_spawn_sphere(sim):
    """Test spawning of UsdGeom.Sphere prim."""
    cfg = sim_utils.SphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Sphere").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Sphere/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Sphere"
    assert prim.GetAttribute("radius").Get() == cfg.radius


"""
Physics properties.
"""


def test_spawn_cone_with_rigid_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body API.

    Note:
        Playing the simulation in this case will give a warning that no mass is specified!
        Need to also setup mass and colliders.
    """
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() == cfg.rigid_props.rigid_body_enabled
    assert (
        prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get()
        == cfg.rigid_props.solver_position_iteration_count
    )
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(cfg.rigid_props.sleep_threshold)


def test_spawn_cone_with_rigid_and_mass_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body and mass API."""
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:mass").Get() == cfg.mass_props.mass

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_rigid_and_density_props(sim):
    """Test spawning of UsdGeom.Cone prim with rigid body and mass API.

    Note:
        In this case, we specify the density instead of the mass. In that case, physics need to know
        the collision shape to compute the mass. Thus, we have to set the collider properties. In
        order to not have a collision shape, we disable the collision.
    """
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True, solver_position_iteration_count=8, sleep_threshold=0.1
        ),
        mass_props=sim_utils.MassPropertiesCfg(density=10.0),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    )
    prim = cfg.func("/World/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cone").IsValid()
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:density").Get() == cfg.mass_props.density

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


def test_spawn_cone_with_all_props(sim):
    """Test spawning of UsdGeom.Cone prim with all properties."""
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
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
    # -- rigid body properties
    prim = sim.stage.GetPrimAtPath("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    # -- collision properties
    prim = sim.stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True

    # check sim playing
    sim.play()
    for _ in range(10):
        sim.step()


"""
Cloning.
"""


def test_spawn_cone_clones_invalid_paths(sim):
    """Test spawning of cone clones on invalid cloning paths."""
    num_clones = 10
    for i in range(num_clones):
        sim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on invalid cloning path -- should raise an error
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    with pytest.raises(RuntimeError):
        cfg.func("/World/env/env_.*/Cone", cfg)


def test_spawn_cone_clones(sim):
    """Test spawning of cone clones."""
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0, 0, 0))
    # Spawn cone on valid cloning path
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    prim = cfg.func("/World/env_.*/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/env_0/Cone"


def test_spawn_cone_clone_with_all_props_global_material(sim):
    """Test spawning of cone clones with global material reference."""
    sim_utils.create_prim("/World/env_0", "Xform", translation=(0, 0, 0))
    # Spawn cone on valid cloning path
    cfg = sim_utils.ConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.materials.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.materials.RigidBodyMaterialCfg(),
        visual_material_path="/Looks/visualMaterial",
        physics_material_path="/Looks/physicsMaterial",
    )
    prim = cfg.func("/World/env_.*/Cone", cfg)

    # Check validity
    assert prim.IsValid()
    assert str(prim.GetPath()) == "/World/env_0/Cone"
    # find matching material prims
    prims = sim_utils.find_matching_prim_paths("/Looks/visualMaterial.*")
    assert len(prims) == 1
