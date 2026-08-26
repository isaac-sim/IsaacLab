# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""


import numpy as np
import pytest

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.meshes import meshes as mesh_spawner

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


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
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0), edge_refinement=1.0)
    prim = cfg.func("/World/Cube", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Cube").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"
    assert len(prim.GetAttribute("points").Get()) == 8
    assert len(prim.GetAttribute("faceVertexCounts").Get()) == 12


def test_mesh_edge_refinement_default():
    """Test the default mesh edge refinement."""
    assert sim_utils.MeshCfg().edge_refinement == 4.0
    assert not hasattr(sim_utils.MeshRectangleCfg(size=(1.0, 1.0)), "resolution")


@pytest.mark.parametrize(
    "cfg_type,kwargs,edge_refinement",
    [
        (sim_utils.MeshSphereCfg, {"radius": 1.0}, 25.0),
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 2.0, 3.0)}, 3.0),
        (sim_utils.MeshCylinderCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshCapsuleCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshConeCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshRectangleCfg, {"size": (1.0, 1.0)}, 3.0),
    ],
)
def test_spawn_mesh_with_edge_refinement(sim, cfg_type, kwargs, edge_refinement):
    """Test surface edge refinement for mesh primitives."""
    cfg = cfg_type(**kwargs, edge_refinement=edge_refinement)
    cfg.func("/World/Refined", cfg)
    prim = sim.stage.GetPrimAtPath("/World/Refined/geometry/mesh")
    points = np.asarray(prim.GetAttribute("points").Get())
    faces = np.asarray(prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
    edges = points[faces[:, [0, 1, 1, 2, 2, 0]]].reshape(-1, 2, 3)
    max_edge = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max()
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    assert max_edge <= diagonal / edge_refinement


@pytest.mark.parametrize(
    "cfg_type,geometry_kwargs,refinement_kwargs,physics_material,expected_factor",
    [
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 1.0, 1.0)}, {}, None, 0.25),
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 1.0, 1.0)}, {"edge_refinement": 2.0}, None, 0.5),
        (sim_utils.MeshRectangleCfg, {"size": (1.0, 1.0)}, {}, sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(), None),
    ],
)
def test_edge_refinement_sets_tetrahedralization_resolution(
    sim, monkeypatch, cfg_type, geometry_kwargs, refinement_kwargs, physics_material, expected_factor
):
    """Test edge refinement is forwarded to volume tetrahedralization."""
    captured_kwargs = {}

    def capture_deformable_properties(*args, **kwargs):
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(mesh_spawner.schemas, "define_deformable_body_properties", capture_deformable_properties)
    cfg = cfg_type(
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        physics_material=physics_material,
        **geometry_kwargs,
        **refinement_kwargs,
    )
    cfg.func("/World/Deformable", cfg)

    if expected_factor is None:
        assert "tetrahedralization_edge_length_fac" not in captured_kwargs
    else:
        assert captured_kwargs["tetrahedralization_edge_length_fac"] == pytest.approx(expected_factor)


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


@pytest.mark.parametrize("edge_refinement", [1.0, 2.4])
@pytest.mark.parametrize("size", [(1.0, 1.0), (1.5, 0.8)])
def test_spawn_rectangle(sim, edge_refinement, size):
    """Test spawning of UsdGeomMesh as a rectangle prim."""
    # Spawn rectangle
    cfg = sim_utils.MeshRectangleCfg(size=size, edge_refinement=edge_refinement)
    prim = cfg.func("/World/Rectangle", cfg)

    # Check validity
    assert prim.IsValid()
    assert sim.stage.GetPrimAtPath("/World/Rectangle").IsValid()
    assert prim.GetPrimTypeInfo().GetTypeName() == "Xform"
    # Check properties
    prim = sim.stage.GetPrimAtPath("/World/Rectangle/geometry/mesh")
    assert prim.GetPrimTypeInfo().GetTypeName() == "Mesh"


def test_invalid_edge_refinement(sim):
    """Test spawning with invalid edge refinement."""
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0), edge_refinement=0.5)
    with pytest.raises(ValueError, match="Mesh edge refinement must be at least 1.0"):
        cfg.func("/World/Invalid", cfg)


"""
Physics properties.
"""


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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
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
