# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Declared deformable prototype classification and clone-plan expansion."""

from __future__ import annotations

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.cloner import ClonePlan
from isaaclab.scene_data.deformable_discovery import (
    _matrix4d_to_numpy,
    _transform_points,
    deformable_entry,
    deformable_prototypes,
    expand_deformable_entries,
)


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def test_transform_points_matches_usd_matrix4d_transform():
    """Numpy baking must use USD row-vector convention (``p @ M``)."""
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), 30.0))
    matrix.SetTranslateOnly(Gf.Vec3d(0.5, -0.1, 0.25))
    points = np.array(
        [
            [0.15, -0.025, 0.025],
            [-0.15, 0.025, -0.025],
            [0.0, 0.0, 0.1],
        ],
        dtype=np.float32,
    )

    baked = _transform_points(_matrix4d_to_numpy(matrix), points)
    expected = np.array(
        [list(matrix.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))) for p in points],
        dtype=np.float32,
    )
    assert np.allclose(baked, expected, atol=1e-6)


def test_deformable_entry_volume_tet_mesh():
    """Classify tetrahedra and prefer the named visual over unrelated child meshes."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
    for name in ("decoration", "visual", "props/unrelated"):
        mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_0/SoftBody/{name}")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entry = deformable_entry(root)
    assert entry.deformable_type == "volume"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 4
    assert entry.root_path.endswith("/SoftBody")
    assert entry.sim_mesh_path.endswith("/simulation")
    assert entry.vis_mesh_path.endswith("/visual")


def test_deformable_entry_surface_mesh():
    stage = Usd.Stage.CreateInMemory()
    sim_mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth")
    _add_api_schemas(
        sim_mesh.GetPrim(),
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsSurfaceDeformableSimAPI",
        ],
    )
    sim_points = [
        Gf.Vec3f(0.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 0.0, 0.0),
        Gf.Vec3f(1.0, 1.0, 0.0),
        Gf.Vec3f(0.0, 1.0, 0.0),
    ]
    sim_mesh.CreatePointsAttr(sim_points)
    sim_mesh.CreateFaceVertexCountsAttr([3, 3])
    sim_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    vis_mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth/visual")
    vis_points = sim_points + [Gf.Vec3f(0.5, 0.5, 0.1)]
    vis_mesh.CreatePointsAttr(vis_points)
    vis_mesh.CreateFaceVertexCountsAttr([3, 3])
    vis_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])

    entry = deformable_entry(sim_mesh.GetPrim())
    assert entry.deformable_type == "surface"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 5


def test_deformable_entry_skips_missing_mesh(caplog):
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/EmptyDeformable").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])

    with caplog.at_level("WARNING", logger="isaaclab.scene_data.deformable_discovery"):
        entry = deformable_entry(root)

    assert entry is None
    assert any("Skipping deformable prim" in record.message for record in caplog.records)


def test_deformable_entry_surface_without_sim_api_uses_first_mesh():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth")
    _add_api_schemas(mesh.GetPrim(), ["OmniPhysicsDeformableBodyAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entry = deformable_entry(mesh.GetPrim())
    assert entry.deformable_type == "surface"
    assert entry.vertex_count == 3
    assert entry.vis_vertex_count == 3


def test_backend_geometry_nearest_owner_partial_rows_and_shared_roots():
    """Import declared prototypes once; bind exact paths without retaining geometry on the plan."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Lab/Cell3")
    parent.AddTranslateOp().Set((10.0, 0.0, 0.0))
    parent.AddRotateZOp().Set(30.0)
    for path, schema in (
        ("/Lab/Cell3/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Nested/Cloth", "PhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Dormant/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Lab/Cell7/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Shared/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Undeclared/Cloth", "OmniPhysicsDeformableBodyAPI"),
    ):
        mesh = UsdGeom.Mesh.Define(stage, path)
        _add_api_schemas(mesh.GetPrim(), [schema])
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    UsdGeom.Points.Define(stage, "/Shared/Particles").CreatePointsAttr([(1.0, 2.0, 3.0)])
    mpm_points = UsdGeom.Points.Define(stage, "/Lab/Cell3/SimulationPoints")
    mpm_points.CreatePointsAttr([(1.0, 2.0, 3.0), (0.0, 0.0, 0.0)])
    _add_api_schemas(mpm_points.GetPrim(), ["PhysicsDeformableBodyAPI"])
    plan = ClonePlan(
        sources=("/Lab/Cell3", "/Lab/Cell3/Nested", "/Missing", "/Lab/Cell3/Dormant"),
        destinations=("/Lab/Cell{}", "/Lab/Cell{}/Nested", "/Other/{}", "/Lab/Cell{}/Dormant"),
        clone_mask=np.asarray([[1, 1, 0], [1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=np.bool_),
        env_ids=np.asarray([3, 7, 11]),
        positions=np.asarray([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [35.0, 0.0, 0.0]]),
        global_paths=("/Shared", "/Shared/Cloth", "/Lab"),
    )

    prototypes = deformable_prototypes(stage, plan)
    assert len(prototypes) == 3
    assert {entry.root_path for entry in prototypes} == {"/Lab/Cell3/Cloth", "/Lab/Cell3/Nested/Cloth", "/Shared/Cloth"}
    assert {entry.root_path for entry in deformable_prototypes(stage, plan, (1,))} == {
        "/Lab/Cell3/Nested/Cloth",
        "/Shared/Cloth",
    }
    prototype = next(entry for entry in prototypes if entry.root_path == "/Lab/Cell3/Cloth")
    stage.RemovePrim("/Lab")

    expanded = {entry.root_path: entry for entry in expand_deformable_entries(plan, prototypes)}
    assert set(expanded) == {
        "/Lab/Cell3/Cloth",
        "/Lab/Cell7/Cloth",
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    assert {entry.root_path for entry in expand_deformable_entries(plan, prototypes, (1,))} == {
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    clone = expanded["/Lab/Cell7/Cloth"]
    assert clone.sim_mesh_path == clone.vis_mesh_path == clone.root_path
    assert clone.vertices is prototype.vertices and clone.indices is prototype.indices
    np.testing.assert_allclose(clone.init_pos, (20.0, 0.0, 0.0))
    np.testing.assert_allclose(clone.init_rot, (0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)))

    # Distinct source roots can target the same subtree; its nearest destination owner wins.
    override_plan = ClonePlan(
        sources=("/Lab/Cell3", "/Shared"),
        destinations=("/Lab/Cell{}", "/Lab/Cell{}/Nested"),
        clone_mask=np.ones((2, 1), dtype=np.bool_),
        env_ids=np.asarray([3]),
    )
    shared = next(entry for entry in prototypes if entry.root_path == "/Shared/Cloth")
    for ordered in (prototypes, prototypes[::-1]):
        expanded = {entry.root_path: entry for entry in expand_deformable_entries(override_plan, ordered)}
        assert expanded["/Lab/Cell3/Nested/Cloth"].vertices is shared.vertices
