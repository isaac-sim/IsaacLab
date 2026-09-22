# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD discovery tests for PhysX/OVPhysX deformable classification."""

from __future__ import annotations

import numpy as np
import pytest

from pxr import Gf, Sdf, Usd, UsdGeom

from isaaclab.scene_data.deformable_discovery import (
    _matrix4d_to_numpy,
    _transform_points,
    discover_deformables_on_stage,
)

pytestmark = pytest.mark.unit

_TET_POINTS = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0), Gf.Vec3f(0.0, 0.0, 1.0)]


def _add_api_schemas(prim: Usd.Prim, schemas: list[str]) -> None:
    api_schemas = Sdf.TokenListOp()
    api_schemas.explicitItems = schemas
    prim.SetMetadata("apiSchemas", api_schemas)


def _define_triangle_mesh(stage: Usd.Stage, path: str, points, schemas: list[str] = ()) -> UsdGeom.Mesh:
    """Define a single-triangle-fan ``Mesh`` over ``points`` with the given applied API schemas."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * (len(points) - 2))
    mesh.CreateFaceVertexIndicesAttr([index for i in range(len(points) - 2) for index in (0, i + 1, i + 2)])
    if schemas:
        _add_api_schemas(mesh.GetPrim(), list(schemas))
    return mesh


def _define_volume_root(stage: Usd.Stage, root_path: str) -> None:
    """Define a deformable root with one tet simulation mesh below it."""
    root = UsdGeom.Xform.Define(stage, root_path).GetPrim()
    _add_api_schemas(root, ["OmniPhysicsDeformableBodyAPI"])
    tet = UsdGeom.TetMesh.Define(stage, f"{root_path}/simulation")
    _add_api_schemas(tet.GetPrim(), ["OmniPhysicsVolumeDeformableSimAPI"])
    tet.CreatePointsAttr(_TET_POINTS)
    tet.CreateTetVertexIndicesAttr([Gf.Vec4i(0, 1, 2, 3)])


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
    """A tet sim mesh yields a volume entry whose vertices are baked into the deformable parent frame."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World/envs/env_0").AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    _define_volume_root(stage, "/World/envs/env_0/SoftBody")
    UsdGeom.Xform(stage.GetPrimAtPath("/World/envs/env_0/SoftBody")).AddTranslateOp().Set(Gf.Vec3d(0.0, 2.0, 0.0))
    _define_triangle_mesh(stage, "/World/envs/env_0/SoftBody/visual", _TET_POINTS)

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.deformable_type == "volume"
    assert (entry.vertex_count, entry.vis_vertex_count) == (4, 4)
    assert entry.root_path == "/World/envs/env_0/SoftBody"
    assert entry.sim_mesh_path == "/World/envs/env_0/SoftBody/simulation"
    assert entry.vis_mesh_path == "/World/envs/env_0/SoftBody/visual"
    assert entry.indices.tolist() == [0, 1, 2, 3]
    assert entry.vis_indices.tolist() == [0, 1, 2, 0, 2, 3]
    # the env translation is the parent frame and drops out; the root's own offset is baked in
    expected = np.asarray(_TET_POINTS, dtype=np.float32) + np.array([0.0, 2.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(entry.vertices, expected, atol=1e-6)
    np.testing.assert_allclose(entry.vis_vertices, expected, atol=1e-6)


def test_discover_surface_mesh_deformable():
    stage = Usd.Stage.CreateInMemory()
    sim_points = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(1.0, 1.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    _define_triangle_mesh(
        stage,
        "/World/envs/env_0/Cloth",
        sim_points,
        schemas=["OmniPhysicsDeformableBodyAPI", "OmniPhysicsSurfaceDeformableSimAPI"],
    )
    _define_triangle_mesh(stage, "/World/envs/env_0/Cloth/visual", sim_points + [Gf.Vec3f(0.5, 0.5, 0.1)])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    entry = entries[0]
    assert entry.deformable_type == "surface"
    assert (entry.vertex_count, entry.vis_vertex_count) == (4, 5)
    assert entry.indices.tolist() == [0, 1, 2, 0, 2, 3]


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
    _define_triangle_mesh(stage, "/World/envs/env_0/Cloth", _TET_POINTS[:3], schemas=["OmniPhysicsDeformableBodyAPI"])

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    assert entries[0].deformable_type == "surface"
    assert (entries[0].vertex_count, entries[0].vis_vertex_count) == (3, 3)
    assert entries[0].vis_mesh_path == entries[0].sim_mesh_path


def test_discover_declared_roots_once_without_undeclared_siblings():
    stage = Usd.Stage.CreateInMemory()
    for path in ("/Scene/Declared/Cloth", "/Scene/Undeclared/Cloth"):
        _define_triangle_mesh(stage, path, _TET_POINTS[:3], schemas=["OmniPhysicsDeformableBodyAPI"])

    entries = discover_deformables_on_stage(
        stage, root_paths=("/Scene/Declared/Cloth", "/Scene/Declared", "/Scene/Declared/Cloth")
    )

    assert [entry.root_path for entry in entries] == ["/Scene/Declared/Cloth"]


def test_discover_volume_prefers_named_visual_over_unrelated_child_mesh():
    """When several child meshes exist under the BodyAPI root, select the visual mesh."""
    stage = Usd.Stage.CreateInMemory()
    _define_volume_root(stage, "/World/envs/env_0/SoftBody")
    # lexicographically first child mesh (must not win over the named visual)
    _define_triangle_mesh(stage, "/World/envs/env_0/SoftBody/decoration", _TET_POINTS[:3])
    _define_triangle_mesh(stage, "/World/envs/env_0/SoftBody/visual", _TET_POINTS)
    # nested under an unrelated child branch; scoring must prefer the named sibling
    _define_triangle_mesh(stage, "/World/envs/env_0/SoftBody/props/unrelated", _TET_POINTS)

    entries = discover_deformables_on_stage(stage)
    assert len(entries) == 1
    assert entries[0].vis_mesh_path.endswith("/visual")
    assert entries[0].vis_vertex_count == 4
