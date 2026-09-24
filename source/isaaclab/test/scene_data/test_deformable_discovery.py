# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery tests for PhysX/OVPhysX deformable classification."""

from __future__ import annotations

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.geometry import compile_geometry
from isaaclab.scene_data.deformable_discovery import (
    _matrix4d_to_numpy,
    _transform_points,
    deformable_entries,
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


def test_discover_declared_roots_once_without_undeclared_siblings():
    stage = Usd.Stage.CreateInMemory()
    for path in ("/Scene/Declared/Cloth", "/Scene/Undeclared/Cloth"):
        mesh = UsdGeom.Mesh.Define(stage, path)
        _add_api_schemas(mesh.GetPrim(), ["OmniPhysicsDeformableBodyAPI"])
        mesh.CreatePointsAttr([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        mesh.CreateFaceVertexCountsAttr([3])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2])

    entries = discover_deformables_on_stage(
        stage, root_paths=("/Scene/Declared/Cloth", "/Scene/Declared", "/Scene/Declared/Cloth")
    )

    assert [entry.root_path for entry in entries] == ["/Scene/Declared/Cloth"]


def test_plan_geometry_nearest_owner_partial_rows_and_shared_roots():
    """Compile declared prototypes once; consumers expand exact paths without copying geometry."""
    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Xform.Define(stage, "/Lab/Cell3")
    parent.AddTranslateOp().Set((10.0, 0.0, 0.0))
    parent.AddRotateZOp().Set(30.0)
    for path, schema in (
        ("/Lab/Cell3/Cloth", "OmniPhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Nested/Cloth", "PhysicsDeformableBodyAPI"),
        ("/Lab/Cell3/Dormant/Cloth", "OmniPhysicsDeformableBodyAPI"),
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
        global_paths=("/Shared", "/Shared/Cloth"),
    )

    compile_geometry(plan, stage)
    prototype = plan.deformables[0][0]
    assert [entry.root_path for entry in plan.deformables[0]] == ["/Lab/Cell3/Cloth"]
    assert [entry.root_path for entry in plan.deformables[1]] == ["/Lab/Cell3/Nested/Cloth"]
    assert plan.deformables[2] == plan.deformables[3] == ()
    stage.RemovePrim("/Lab")
    compile_geometry(plan, stage)
    assert plan.deformables[0][0] is prototype

    expanded = {entry.root_path: entry for entry in deformable_entries(plan)}
    assert set(expanded) == {
        "/Lab/Cell3/Cloth",
        "/Lab/Cell7/Cloth",
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    assert {entry.root_path for entry in deformable_entries(plan, (1,))} == {
        "/Lab/Cell3/Nested/Cloth",
        "/Lab/Cell11/Nested/Cloth",
        "/Shared/Cloth",
    }
    clone = expanded["/Lab/Cell7/Cloth"]
    assert clone.sim_mesh_path == clone.vis_mesh_path == clone.root_path
    assert clone.vertices is prototype.vertices and clone.indices is prototype.indices
    np.testing.assert_allclose(clone.init_pos, (20.0, 0.0, 0.0))
    np.testing.assert_allclose(clone.init_rot, (0.0, 0.0, np.sin(np.pi / 12), np.cos(np.pi / 12)))


@pytest.mark.parametrize(
    "counts, curve_type, wrap, expected",
    [
        ([4], "linear", "nonperiodic", 3),
        ([4], "linear", "periodic", None),
        ([4], "cubic", "nonperiodic", None),
        ([2, 2], "linear", "nonperiodic", None),
        ([1], "linear", "nonperiodic", None),
    ],
)
def test_plan_cables_preserve_supported_topology(counts, curve_type, wrap, expected):
    stage = Usd.Stage.CreateInMemory()
    cable = UsdGeom.BasisCurves.Define(stage, "/Prototype/Cable")
    _add_api_schemas(cable.GetPrim(), ["PhysicsCurvesDeformableSimAPI"])
    cable.CreateCurveVertexCountsAttr(counts)
    cable.CreateTypeAttr(curve_type)
    cable.CreateWrapAttr(wrap)
    plan = ClonePlan(("/Prototype",), ("/Scene/Cell{}",), np.ones((1, 1), dtype=np.bool_))

    compile_geometry(plan, stage)

    assert plan.cables[0] == (() if expected is None else (("/Prototype/Cable", expected),))


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
