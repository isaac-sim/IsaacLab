# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.meshes import meshes as mesh_spawner

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


@pytest.fixture
def capture_deformable_props(monkeypatch):
    """Record the deformable writer call: volume tetrahedralization is an optional dependency."""
    captured = {}
    monkeypatch.setattr(mesh_spawner.schemas, "define_deformable_body_properties", lambda *a, **k: captured.update(k))
    return captured


@pytest.mark.parametrize(
    ("cfg", "num_points", "num_faces"),
    [
        (sim_utils.MeshConeCfg(radius=1.0, height=2.0, axis="Y"), None, None),
        (sim_utils.MeshCapsuleCfg(radius=1.0, height=2.0, axis="Y"), None, None),
        (sim_utils.MeshCylinderCfg(radius=1.0, height=2.0, axis="Y"), None, None),
        (sim_utils.MeshSphereCfg(radius=1.0), None, None),
        (sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0)), 8, 12),
        (sim_utils.MeshRectangleCfg(size=(1.5, 0.8)), 4, 2),
    ],
    ids=["cone", "capsule", "cylinder", "sphere", "cuboid", "rectangle"],
)
def test_spawn_mesh(stage, cfg, num_points, num_faces):
    prim = cfg.func("/World/Shape", cfg)
    assert prim.GetPath() == "/World/Shape"
    assert prim.GetTypeName() == "Xform"
    mesh = stage.GetPrimAtPath("/World/Shape/geometry/mesh")
    assert mesh.GetTypeName() == "Mesh"
    if num_points is not None:
        assert len(mesh.GetAttribute("points").Get()) == num_points
        assert len(mesh.GetAttribute("faceVertexCounts").Get()) == num_faces
    with pytest.raises(ValueError, match="already exists"):
        cfg.func("/World/Shape", cfg)


def test_spawn_mesh_with_rigid_props(stage):
    cfg = sim_utils.MeshConeCfg(
        radius=1.0,
        height=2.0,
        mass_props=sim_utils.MassCfg(mass=5.0),
        rigid_props=[
            sim_utils.UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            PhysxRigidBodyCfg(solver_position_iteration_count=8, sleep_threshold=0.1),
        ],
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.75, 0.5)),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )
    prim = cfg.func("/World/Cone", cfg)

    assert prim.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert prim.GetAttribute("physxRigidBody:solverPositionIterationCount").Get() == 8
    assert prim.GetAttribute("physxRigidBody:sleepThreshold").Get() == pytest.approx(0.1)
    assert prim.GetAttribute("physics:mass").Get() == 5.0
    mesh = stage.GetPrimAtPath("/World/Cone/geometry/mesh")
    assert mesh.GetAttribute("physics:collisionEnabled").Get() is True
    assert mesh.GetAttribute("physics:approximation").Get() == "convexHull"
    assert stage.GetPrimAtPath("/World/Cone/geometry/material").IsValid()


def test_mesh_edge_refinement_default():
    assert sim_utils.MeshCfg().edge_refinement == 4.0
    assert not hasattr(sim_utils.MeshRectangleCfg(size=(1.0, 1.0)), "resolution")


@pytest.mark.parametrize(
    ("cfg_type", "kwargs", "edge_refinement"),
    [
        (sim_utils.MeshSphereCfg, {"radius": 1.0}, 25.0),
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 2.0, 3.0)}, 3.0),
        (sim_utils.MeshCylinderCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshCapsuleCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshConeCfg, {"radius": 1.0, "height": 2.0}, 3.0),
        (sim_utils.MeshRectangleCfg, {"size": (1.0, 1.0)}, 3.0),
    ],
)
def test_spawn_mesh_with_edge_refinement(stage, capture_deformable_props, cfg_type, kwargs, edge_refinement):
    """Deformable surface meshes are subdivided until no edge exceeds the bounding-box diagonal / refinement."""
    cfg = cfg_type(**kwargs, edge_refinement=edge_refinement, deformable_props=sim_utils.DeformableBodyPropertiesCfg())
    cfg.func("/World/Refined", cfg)
    prim = stage.GetPrimAtPath("/World/Refined/geometry/mesh")
    points = np.asarray(prim.GetAttribute("points").Get())
    faces = np.asarray(prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
    edges = points[faces[:, [0, 1, 1, 2, 2, 0]]].reshape(-1, 2, 3)
    max_edge = np.linalg.norm(edges[:, 0] - edges[:, 1], axis=1).max()
    diagonal = np.linalg.norm(points.max(axis=0) - points.min(axis=0))

    assert max_edge <= diagonal / edge_refinement


@pytest.mark.parametrize(
    ("cfg_type", "geometry_kwargs", "refinement_kwargs", "physics_material", "expected_factor"),
    [
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 1.0, 1.0)}, {}, None, 0.25),
        (sim_utils.MeshCuboidCfg, {"size": (1.0, 1.0, 1.0)}, {"edge_refinement": 2.0}, None, 0.5),
        (sim_utils.MeshRectangleCfg, {"size": (1.0, 1.0)}, {}, sim_utils.PhysxSurfaceDeformableBodyMaterialCfg(), None),
    ],
    ids=["volume_default", "volume_refined", "surface"],
)
def test_edge_refinement_sets_tetrahedralization_resolution(
    stage, capture_deformable_props, cfg_type, geometry_kwargs, refinement_kwargs, physics_material, expected_factor
):
    """Edge refinement is forwarded to volume tetrahedralization only."""
    cfg = cfg_type(
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        physics_material=physics_material,
        **geometry_kwargs,
        **refinement_kwargs,
    )
    cfg.func("/World/Deformable", cfg)

    if expected_factor is None:
        assert "tetrahedralization_edge_length_fac" not in capture_deformable_props
    else:
        assert capture_deformable_props["tetrahedralization_edge_length_fac"] == pytest.approx(expected_factor)


def test_spawn_mesh_rejects_invalid_configs(stage, capture_deformable_props):
    cfg = sim_utils.MeshCuboidCfg(size=(1.0, 2.0, 3.0), edge_refinement=0.5)
    with pytest.raises(ValueError, match="Mesh edge refinement must be at least 1.0"):
        cfg.func("/World/Invalid", cfg)

    cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
    )
    with pytest.raises(ValueError, match="both deformable and rigid"):
        cfg.func("/World/Both", cfg)

    cfg = sim_utils.MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        deformable_props=sim_utils.DeformableBodyPropertiesCfg(),
        physics_material=sim_utils.RigidBodyMaterialCfg(),
    )
    with pytest.raises(ValueError, match="deformable physics material"):
        cfg.func("/World/WrongMaterial", cfg)
