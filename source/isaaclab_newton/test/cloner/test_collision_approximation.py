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


def _add_l_prism(stage: Usd.Stage, path: str, approximation: str | None, offset: float = 0.0) -> None:
    """Author a concave L-prism collision mesh; optionally set ``physics:approximation``."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    base = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    points = [(x + offset, y, 0.0) for x, y in base] + [(x + offset, y, 1.0) for x, y in base]
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


def _make_stage(approximation: str | None) -> Usd.Stage:
    """A rigid body with one L-prism collision mesh."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    xform = UsdGeom.Xform.Define(stage, _SOURCE)
    UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
    _add_l_prism(stage, f"{_SOURCE}/geom", approximation)
    return stage


def _collision_shapes(builder: newton.ModelBuilder) -> dict[str, GeoType]:
    """Label -> geo type of the colliding shapes in the builder."""
    return {
        builder.shape_label[i]: GeoType(shape_type)
        for i, shape_type in enumerate(builder.shape_type)
        if builder.shape_flags[i] & ShapeFlags.COLLIDE_SHAPES
    }


def _make_two_source_stage(approximation_a: str | None, approximation_b: str | None) -> tuple[Usd.Stage, list[str]]:
    """Two rigid-body sources, each with one L-prism collision mesh."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    sources = ["/World/AssetA", "/World/AssetB"]
    for source, approximation in zip(sources, (approximation_a, approximation_b)):
        xform = UsdGeom.Xform.Define(stage, source)
        UsdPhysics.RigidBodyAPI.Apply(xform.GetPrim())
        _add_l_prism(stage, f"{source}/geom", approximation)
    return stage, sources


def _build_sources(stage: Usd.Stage, sources: list[str], **kwargs) -> dict[str, newton.ModelBuilder]:
    return build_source_builders(
        stage,
        sources,
        create_builder=lambda: newton.ModelBuilder(up_axis=newton.Axis.Z),
        schema_resolvers=[],
        **kwargs,
    )


def _build(stage: Usd.Stage, **kwargs) -> newton.ModelBuilder:
    return _build_sources(stage, [_SOURCE], **kwargs)[_SOURCE]


class TestClonerCollisionApproximation:
    """build_source_builders must honor authored approximations and hull the rest."""

    def test_authored_convex_decomposition_produces_multiple_hulls(self):
        """A concave mesh authored with convexDecomposition decomposes into 2+ hulls."""
        shapes = _collision_shapes(_build(_make_stage("convexDecomposition")))
        assert len(shapes) >= 2, f"expected a multi-hull decomposition, got {shapes}"
        assert all(geo_type == GeoType.CONVEX_MESH for geo_type in shapes.values())

    @pytest.mark.parametrize(
        ("approximation", "expected"),
        [
            ("boundingSphere", GeoType.SPHERE),
            ("boundingCube", GeoType.BOX),
            ("meshSimplification", GeoType.MESH),
            ("none", GeoType.MESH),
        ],
    )
    def test_authored_approximation_produces_expected_shape(self, approximation, expected):
        """Each authored mode maps to its shape type instead of a convex hull."""
        shapes = _collision_shapes(_build(_make_stage(approximation)))
        assert list(shapes.values()) == [expected]

    def test_unauthored_mesh_still_simplifies_to_single_hull(self):
        """Meshes with no authored approximation keep the default convex-hull treatment."""
        shapes = _collision_shapes(_build(_make_stage(None)))
        assert len(shapes) == 1
        assert next(iter(shapes.values())) in (GeoType.MESH, GeoType.CONVEX_MESH)

    def test_simplify_meshes_false_keeps_raw_mesh(self):
        """simplify_meshes=False leaves an unauthored mesh untouched."""
        shapes = _collision_shapes(_build(_make_stage(None), simplify_meshes=False))
        assert list(shapes.values()) == [GeoType.MESH]

    def test_simplify_pass_only_touches_unauthored_meshes(self):
        """In a mixed stage, only the unauthored mesh gets the default hull treatment."""
        stage = _make_stage("none")
        _add_l_prism(stage, f"{_SOURCE}/geom_plain", None, offset=4.0)

        shapes = _collision_shapes(_build(stage))

        # The authored "none" mesh keeps its raw trimesh; the unauthored sibling is hulled.
        assert shapes[f"{_SOURCE}/geom"] == GeoType.MESH
        assert shapes[f"{_SOURCE}/geom_plain"] == GeoType.CONVEX_MESH

    def test_heterogeneous_sources_fall_back_to_uniform_hulls(self):
        """Differing shape sequences across sources revert to hull-everything with a warning.

        SolverMuJoCo requires homogeneous worlds, so honoring authored modes that make
        clone sources structurally different would break solver initialization.
        """
        stage, sources = _make_two_source_stage("boundingSphere", None)

        with pytest.warns(UserWarning, match="homogeneous-worlds"):
            builders = _build_sources(stage, sources)

        for source in sources:
            assert list(_collision_shapes(builders[source]).values()) == [GeoType.CONVEX_MESH]

    def test_heterogeneous_sources_with_equal_sequences_stay_honored(self):
        """Identically authored sources keep their authored modes (no fallback)."""
        stage, sources = _make_two_source_stage("boundingSphere", "boundingSphere")

        builders = _build_sources(stage, sources)

        for source in sources:
            assert list(_collision_shapes(builders[source]).values()) == [GeoType.SPHERE]
