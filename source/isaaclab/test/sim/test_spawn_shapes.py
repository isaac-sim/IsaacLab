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


"""
Basic spawning.
"""


def test_spawn_cone(sim):
    """Test spawning of UsdGeom.Cone prim."""
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Cone")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cone"
    assert prim.GetAttribute("radius").Get() == cfg.radius
    assert prim.GetAttribute("height").Get() == cfg.height
    assert prim.GetAttribute("axis").Get() == cfg.axis


def test_spawn_capsule(sim):
    """Test spawning of UsdGeom.Capsule prim."""
    cfg = sim_utils.CapsuleCfg(radius=1.0, height=2.0, axis="Y")
    prim = cfg.func("/World/Capsule", cfg)
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Capsule")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    prim = prim_utils.get_prim_at_path("/World/Capsule/geometry/mesh")
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
    assert prim_utils.is_prim_path_valid("/World/Cylinder")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cylinder/geometry/mesh")
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
    assert prim_utils.is_prim_path_valid("/World/Cube")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Cube"
    assert prim.GetAttribute("size").Get() == min(cfg.size)


def test_spawn_sphere(sim):
    """Test spawning of UsdGeom.Sphere prim."""
    cfg = sim_utils.SphereCfg(radius=1.0)
    prim = cfg.func("/World/Sphere", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.is_prim_path_valid("/World/Sphere")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Sphere/geometry/mesh")
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
    assert prim_utils.is_prim_path_valid("/World/Cone")
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone")
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
    assert prim_utils.is_prim_path_valid("/World/Cone")
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone")
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
    assert prim_utils.is_prim_path_valid("/World/Cone")
    # Check properties
    prim = prim_utils.get_prim_at_path("/World/Cone")
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
    assert prim_utils.is_prim_path_valid("/World/Cone")
    assert prim_utils.is_prim_path_valid("/World/Cone/geometry/material")
    # Check properties
    # -- rigid body properties
    prim = prim_utils.get_prim_at_path("/World/Cone")
    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    # -- collision properties
    prim = prim_utils.get_prim_at_path("/World/Cone/geometry/mesh")
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
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on invalid cloning path -- should raise an error
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    with pytest.raises(RuntimeError):
        cfg.func("/World/env/env_.*/Cone", cfg)


def test_spawn_cone_clones(sim):
    """Test spawning of cone clones."""
    num_clones = 10
    for i in range(num_clones):
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
    # Spawn cone on valid cloning path
    cfg = sim_utils.ConeCfg(radius=1.0, height=2.0, copy_from_source=True)
    prim = cfg.func("/World/env_.*/Cone", cfg)
    # Check validity
    assert prim.IsValid()
    assert prim_utils.get_prim_path(prim) == "/World/env_0/Cone"
    # find matching prims
    prims = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
    assert len(prims) == num_clones


def test_spawn_cone_clone_with_all_props_global_material(sim):
    """Test spawning of cone clones with global material reference."""
    num_clones = 10
    for i in range(num_clones):
        prim_utils.create_prim(f"/World/env_{i}", "Xform", translation=(i, i, 0))
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
    assert prim_utils.get_prim_path(prim) == "/World/env_0/Cone"
    # find matching prims
    prims = prim_utils.find_matching_prim_paths("/World/env_*/Cone")
    assert len(prims) == num_clones
    # find matching material prims
    prims = prim_utils.find_matching_prim_paths("/Looks/visualMaterial.*")
    assert len(prims) == 1
