# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery tests for PhysX/OVPhysX deformable classification."""

from __future__ import annotations

import numpy as np

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.scene_data.deformable_discovery import (
    _matrix4d_to_numpy,
    _transform_points,
    discover_deformables_on_stage,
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


def test_discover_volume_tet_mesh_deformable():
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
    visual = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.deformable_type == "volume"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 4
    assert entry.root_path.endswith("/SoftBody")
    assert entry.sim_mesh_path.endswith("/simulation")
    assert entry.vis_mesh_path.endswith("/visual")


def test_discover_surface_mesh_deformable():
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

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.deformable_type == "surface"
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 5


def test_discover_skips_deformable_without_mesh(caplog):
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/EmptyDeformable").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])

    with caplog.at_level("WARNING", logger="isaaclab.scene_data.deformable_discovery"):
        entries = discover_deformables_on_stage(stage)

    assert entries == []
    assert any("Skipping deformable prim" in record.message for record in caplog.records)


def test_discover_surface_without_sim_api_uses_first_mesh():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Cloth")
    _add_api_schemas(mesh.GetPrim(), ["OmniPhysicsDeformableBodyAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    assert entries[0].deformable_type == "surface"
    assert entries[0].vertex_count == 3
    assert entries[0].vis_vertex_count == 3


def test_discover_volume_prefers_named_visual_over_unrelated_child_mesh():
    """When several child meshes exist under the BodyAPI root, select the visual mesh."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody").GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])

    # Lexicographically first child mesh (must not win over the named visual).
    deco = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/decoration")
    deco.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    deco.CreateFaceVertexCountsAttr([3])
    deco.CreateFaceVertexIndicesAttr([0, 1, 2])

    visual = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])

    # Nested under an unrelated child branch — scoring must prefer the named visual.
    nested = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/props/unrelated")
    nested.CreatePointsAttr(points)
    nested.CreateFaceVertexCountsAttr([3])
    nested.CreateFaceVertexIndicesAttr([0, 1, 2])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    assert entries[0].vis_mesh_path.endswith("/visual")
    assert entries[0].vis_vertex_count == 4
