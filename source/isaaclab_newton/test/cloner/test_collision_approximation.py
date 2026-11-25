# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests: the Newton cloner honors USD-authored ``physics:approximation``."""

import newton
import numpy as np
import pytest
import trimesh
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from isaaclab_newton.physics.newton_builder import usd_schema_resolvers
from newton import GeoType, ShapeFlags

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.test.utils import DeviceScope, test_devices

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


def _apply_sdf_collision(stage: Usd.Stage, path: str) -> None:
    """Author ``NewtonSDFCollisionAPI`` on a collision mesh, as the Lab schema fragment does."""
    prim = stage.GetPrimAtPath(path)
    prim.AddAppliedSchema("NewtonSDFCollisionAPI")
    prim.CreateAttribute("newton:sdfMaxResolution", Sdf.ValueTypeNames.Int).Set(64)


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


def _make_mixed_visual_stage() -> Usd.Stage:
    """Create dynamic and static colliders with and without separate visual shapes."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, _SOURCE)

    primitive_body = UsdGeom.Xform.Define(stage, f"{_SOURCE}/Primitive")
    UsdPhysics.RigidBodyAPI.Apply(primitive_body.GetPrim())
    primitive = UsdGeom.Cube.Define(stage, f"{_SOURCE}/Primitive/geometry")
    UsdPhysics.CollisionAPI.Apply(primitive.GetPrim())

    authored_body = UsdGeom.Xform.Define(stage, f"{_SOURCE}/Authored")
    UsdPhysics.RigidBodyAPI.Apply(authored_body.GetPrim())
    UsdGeom.Sphere.Define(stage, f"{_SOURCE}/Authored/visual")
    collider = UsdGeom.Cube.Define(stage, f"{_SOURCE}/Authored/collider")
    collider.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    UsdGeom.Xform.Define(stage, f"{_SOURCE}/StaticPrimitive")
    static_primitive = UsdGeom.Cube.Define(stage, f"{_SOURCE}/StaticPrimitive/geometry")
    UsdPhysics.CollisionAPI.Apply(static_primitive.GetPrim())

    UsdGeom.Xform.Define(stage, f"{_SOURCE}/StaticAuthored")
    UsdGeom.Sphere.Define(stage, f"{_SOURCE}/StaticAuthored/visual")
    static_collider = UsdGeom.Cube.Define(stage, f"{_SOURCE}/StaticAuthored/collider")
    UsdPhysics.CollisionAPI.Apply(static_collider.GetPrim())

    mesh_body = UsdGeom.Xform.Define(stage, f"{_SOURCE}/MeshOnly")
    UsdPhysics.RigidBodyAPI.Apply(mesh_body.GetPrim())
    _add_l_prism(stage, f"{_SOURCE}/MeshOnly/geom", None, offset=8.0)
    return stage


def _build_sources(stage: Usd.Stage, sources: list[str], **kwargs) -> dict[str, newton.ModelBuilder]:
    return build_source_builders(
        stage,
        sources,
        create_builder=lambda: newton.ModelBuilder(up_axis=newton.Axis.Z),
        schema_resolvers=usd_schema_resolvers(None),
        **kwargs,
    )


def _build(stage: Usd.Stage, **kwargs) -> newton.ModelBuilder:
    return _build_sources(stage, [_SOURCE], **kwargs)[_SOURCE]


class TestClonerCollisionApproximation:
    """build_source_builders must leave every collider at its USD-authored approximation."""

    @pytest.mark.parametrize("dynamic", [True, False])
    def test_authored_convex_decomposition_produces_multiple_hulls(self, dynamic):
        """Decompose collisions while preserving the authored visual mesh and material binding."""
        stage = _make_stage("convexDecomposition")
        if not dynamic:
            stage.GetPrimAtPath(_SOURCE).RemoveAPI(UsdPhysics.RigidBodyAPI)
        material = UsdShade.Material.Define(stage, f"{_SOURCE}/material")
        UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(_SOURCE)).Bind(material)
        builder = _build(stage)
        shapes = _collision_shapes(builder)
        assert len(shapes) >= 2, f"expected a multi-hull decomposition, got {shapes}"
        assert all(geo_type == GeoType.CONVEX_MESH for geo_type in shapes.values())
        named_flags = zip(builder.shape_label, builder.shape_flags, strict=True)
        assert [path for path, flags in named_flags if flags & ShapeFlags.VISIBLE] == [f"{_SOURCE}/geom_visual"]
        visual = builder.shape_label.index(f"{_SOURCE}/geom_visual")
        assert builder.finalize(device="cpu").isaaclab.visual_material_path[visual] == str(material.GetPath())

    def test_skip_mesh_approximation_bypasses_authored_convex_decomposition(self):
        """A render-only import can bypass decomposition and retain the original mesh."""
        shapes = _collision_shapes(_build(_make_stage("convexDecomposition"), skip_mesh_approximation=True))

        assert list(shapes.values()) == [GeoType.MESH]

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

    def test_only_the_authored_mesh_is_remeshed_in_a_mixed_stage(self):
        """A sibling that authors nothing keeps its trimesh while the authored one is remeshed."""
        stage = _make_stage("boundingCube")
        _add_l_prism(stage, f"{_SOURCE}/geom_plain", None, offset=4.0)

        shapes = _collision_shapes(_build(stage))

        assert shapes[f"{_SOURCE}/geom"] == GeoType.BOX
        assert shapes[f"{_SOURCE}/geom_plain"] == GeoType.MESH

    def test_differing_sources_each_keep_their_own_authored_mode(self):
        """Sources that resolve to different shape types are left alone, not rewritten.

        Distinct assets in a scene are *expected* to differ; only per-world alternatives for
        the same slot must agree, and that is settled by authoring them consistently rather
        than by the cloner discarding what the assets declared.
        """
        stage, sources = _make_two_source_stage("boundingSphere", None)

        builders = _build_sources(stage, sources)

        assert list(_collision_shapes(builders[sources[0]]).values()) == [GeoType.SPHERE]
        assert list(_collision_shapes(builders[sources[1]]).values()) == [GeoType.MESH]

    @pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
    def test_sdf_reduction_is_opt_in_and_preserves_visuals_and_cavities(self, device):
        """Source import reduces only SDF collisions, retaining authored visuals, mass and the central hole."""
        stage = _make_stage("none")
        ring = trimesh.creation.annulus(r_min=0.04, r_max=0.05, height=0.03, sections=64)
        path = f"{_SOURCE}/ring"
        mesh = UsdGeom.Mesh.Define(stage, path)
        # Duplicate vertices as in authored meshes with UV/normal seams.
        mesh.CreatePointsAttr(ring.vertices[ring.faces].reshape(-1, 3))
        mesh.CreateFaceVertexCountsAttr(np.full(len(ring.faces), 3, dtype=np.int32))
        mesh.CreateFaceVertexIndicesAttr(np.arange(ring.faces.size, dtype=np.int32))
        UsdGeom.Xformable(mesh).AddTranslateOp().Set((0.03, -0.02, 0.01))
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        # SDF must still ignore a non-mesh approximation, while allowing explicit triangle reduction.
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("boundingCube")
        _apply_sdf_collision(stage, path)
        original = _build(stage)
        reduced = _build(stage, sdf_mesh_reduction=0.75)
        index = reduced.shape_label.index(path)
        baseline = original.shape_source[original.shape_label.index(path)]
        cooked = reduced.shape_source[index]
        assert reduced.shape_type[index] == GeoType.MESH
        assert reduced.shape_sdf_max_resolution[index] == 64
        assert len(cooked.indices) <= len(baseline.indices) * 0.3
        assert len(baseline.indices) == ring.faces.size
        assert np.linalg.norm(cooked.vertices[cooked.indices.reshape(-1, 3)].mean(axis=1)[:, :2], axis=1).min() > 0.035
        plain = f"{_SOURCE}/geom"
        np.testing.assert_array_equal(
            reduced.shape_source[reduced.shape_label.index(plain)].indices,
            original.shape_source[original.shape_label.index(plain)].indices,
        )
        for before in range(original.shape_count):
            if not original.shape_flags[before] & ShapeFlags.VISIBLE:
                continue
            candidates = [
                i
                for i, label in enumerate(reduced.shape_label)
                if label.startswith(original.shape_label[before]) and reduced.shape_flags[i] & ShapeFlags.VISIBLE
            ]
            assert len(candidates) == 1
            after = candidates[0]
            np.testing.assert_array_equal(reduced.shape_source[after].vertices, original.shape_source[before].vertices)
            np.testing.assert_array_equal(reduced.shape_source[after].indices, original.shape_source[before].indices)
            assert reduced.shape_transform[after] == original.shape_transform[before]
        np.testing.assert_allclose(reduced.body_mass, original.body_mass)
        np.testing.assert_allclose(reduced.body_inertia, original.body_inertia)
        bypass = _build(stage, sdf_mesh_reduction=0.75, skip_mesh_approximation=True)
        assert len(bypass.shape_source[bypass.shape_label.index(path)].indices) == ring.faces.size
        # Two slots using the same asset must share the cooked SDF allocation, without merging their bodies.
        duplicate = "/World/Copy"
        Sdf.CopySpec(stage.GetRootLayer(), _SOURCE, stage.GetRootLayer(), duplicate)
        sources = _build_sources(stage, [_SOURCE, duplicate], sdf_mesh_reduction=0.75)
        combined = newton.ModelBuilder()
        for builder in sources.values():
            combined.add_builder(builder)
        model = combined.finalize(device=device)
        assert model.body_count == 2
        assert len(model._texture_sdf_data) == 1, "Repeated slots must not allocate duplicate SDF textures"

    def test_primitive_collider_remains_visible_in_mixed_visual_model(self):
        """Colliders remain visible only when their body or static parent has no visual shape.

        A lone default-purpose collision mesh also stays visible: assets that author a single mesh as
        both collider and render geometry have no visual-only shape to fall back on.
        """
        builder = _build(_make_mixed_visual_stage())
        flags_by_label = dict(zip(builder.shape_label, builder.shape_flags, strict=True))

        assert flags_by_label[f"{_SOURCE}/Primitive/geometry"] & ShapeFlags.VISIBLE
        assert not flags_by_label[f"{_SOURCE}/Authored/collider"] & ShapeFlags.VISIBLE
        assert flags_by_label[f"{_SOURCE}/StaticPrimitive/geometry"] & ShapeFlags.VISIBLE
        assert not flags_by_label[f"{_SOURCE}/StaticAuthored/collider"] & ShapeFlags.VISIBLE
        assert flags_by_label[f"{_SOURCE}/MeshOnly/geom"] & ShapeFlags.VISIBLE
