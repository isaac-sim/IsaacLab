# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery tests for PhysX/OVPhysX deformable classification."""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.scene_data.deformable_discovery import discover_deformables_on_stage


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def test_discover_volume_tet_mesh_deformable():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody")
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(
        tet.GetPrim(),
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsVolumeDeformableSimAPI",
        ],
    )
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
    assert entry.sim_mesh_path.endswith("/simulation")
    # Visual Mesh is a sibling of the API-bearing TetMesh under the SoftBody parent.
    assert entry.vis_mesh_path.endswith("/visual")
    assert entry.vis_vertex_count == 4


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


def test_discover_volume_prefers_named_visual_over_unrelated_sibling_mesh():
    """When several sibling meshes exist, select the visual mesh—not the first Mesh."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody")
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(
        tet.GetPrim(),
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsVolumeDeformableSimAPI",
        ],
    )
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])

    # Lexicographically first sibling mesh (must not win over the named visual).
    deco = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/decoration")
    deco.CreatePointsAttr([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)])
    deco.CreateFaceVertexCountsAttr([3])
    deco.CreateFaceVertexIndicesAttr([0, 1, 2])

    visual = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/visual")
    visual.CreatePointsAttr(points)
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 1, 2])

    # Nested under an unrelated sibling branch — must not be considered.
    nested = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/props/unrelated")
    nested.CreatePointsAttr(points)
    nested.CreateFaceVertexCountsAttr([3])
    nested.CreateFaceVertexIndicesAttr([0, 1, 2])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    assert entries[0].vis_mesh_path.endswith("/visual")
    assert entries[0].vis_vertex_count == 4


def test_discover_volume_marks_sim_vis_count_mismatch():
    """Volume deformables with fewer visual verts are flagged by mismatched counts."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0/SoftBody")
    tet = UsdGeom.TetMesh.Define(stage, "/World/envs/env_0/SoftBody/simulation")
    _add_api_schemas(
        tet.GetPrim(),
        [
            "OmniPhysicsDeformableBodyAPI",
            "OmniPhysicsVolumeDeformableSimAPI",
        ],
    )
    points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]
    tet.CreatePointsAttr(points)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])
    visual = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/SoftBody/visual")
    visual.CreatePointsAttr([Gf.Vec3f(0.25, 0.25, 0.25)])
    visual.CreateFaceVertexCountsAttr([3])
    visual.CreateFaceVertexIndicesAttr([0, 0, 0])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.vertex_count == 4
    assert entry.vis_vertex_count == 1
    assert len(entry.vis_vertices) == 1
    assert entry.vis_indices
