# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests: the Newton cloner honors USD-authored ``physics:approximation``."""

import newton
import pytest
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from newton import GeoType, ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics

_SOURCE = "/World/Asset"


def _make_stage(approximation: str | None) -> Usd.Stage:
    """Author a rigid concave L-prism; optionally set ``physics:approximation``."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, _SOURCE)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())

    mesh = UsdGeom.Mesh.Define(stage, f"{_SOURCE}/geom")
    # Concave hexagonal cross-section (an "L"), extruded along z.
    base = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    points = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    tris = []
    # Bottom and top caps: a fan from vertex 0 (which sees the whole L polygon).
    for a, b in ((1, 2), (2, 3), (3, 4), (4, 5)):
        tris.append((0, b, a))
        tris.append((6, 6 + a, 6 + b))
    # Side walls.
    for a in range(6):
        b = (a + 1) % 6
        tris.append((a, b, b + 6))
        tris.append((a, b + 6, a + 6))
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([3] * len(tris))
    mesh.CreateFaceVertexIndicesAttr([i for t in tris for i in t])

    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    if approximation is not None:
        mesh_collision.CreateApproximationAttr().Set(approximation)
    return stage


def _collision_shapes(builder: newton.ModelBuilder) -> list[GeoType]:
    """Geo types of the colliding shapes in the builder."""
    return [
        GeoType(shape_type)
        for i, shape_type in enumerate(builder.shape_type)
        if builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
    ]


def _build(stage: Usd.Stage, **kwargs) -> newton.ModelBuilder:
    builders = build_source_builders(
        stage,
        [_SOURCE],
        create_builder=lambda: newton.ModelBuilder(up_axis=newton.Axis.Z),
        schema_resolvers=[],
        **kwargs,
    )
    return builders[_SOURCE]


class TestClonerCollisionApproximation:
    """build_source_builders must honor authored approximations and hull the rest."""

    def test_authored_convex_decomposition_produces_multiple_hulls(self):
        """A concave mesh authored with convexDecomposition decomposes into 2+ hulls."""
        pytest.importorskip("coacd", reason="convexDecomposition requires the coacd dependency")
        shapes = _collision_shapes(_build(_make_stage("convexDecomposition")))
        assert len(shapes) >= 2, f"expected a multi-hull decomposition, got {shapes}"
        assert all(s == GeoType.CONVEX_MESH for s in shapes)

    def test_authored_bounding_sphere_produces_sphere(self):
        """boundingSphere becomes a SPHERE collision shape, not a convex hull."""
        shapes = _collision_shapes(_build(_make_stage("boundingSphere")))
        assert shapes == [GeoType.SPHERE]

    def test_unauthored_mesh_still_simplifies_to_single_hull(self):
        """Meshes with no authored approximation keep the default convex-hull treatment."""
        shapes = _collision_shapes(_build(_make_stage(None)))
        assert len(shapes) == 1
        assert shapes[0] in (GeoType.MESH, GeoType.CONVEX_MESH)

    def test_simplify_meshes_false_keeps_raw_mesh(self):
        """simplify_meshes=False leaves an unauthored mesh untouched."""
        shapes = _collision_shapes(_build(_make_stage(None), simplify_meshes=False))
        assert shapes == [GeoType.MESH]
