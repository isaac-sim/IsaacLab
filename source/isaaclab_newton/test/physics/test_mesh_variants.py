# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reset-time Newton mesh variants."""

import ast
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import warp as wp
from isaaclab_newton.physics import mesh_variants
from isaaclab_newton.physics.mesh_variants import (
    MeshVariants,
    _mesh_variant_body_properties,
    _prepare_mesh_variant,
    _render_shapes,
)
from newton import GeoType, Mesh, ModelBuilder, ShapeFlags


def _variant_builder(
    source: str,
    half_extent: float,
    body_suffix: str = "/body",
    fixed: bool = False,
    kinematic: bool = False,
    convex: bool = False,
) -> ModelBuilder:
    vertices = np.asarray(
        [
            (-half_extent, -half_extent, -half_extent),
            (half_extent, -half_extent, -half_extent),
            (0.0, half_extent, -half_extent),
            (0.0, 0.0, half_extent),
        ],
        dtype=np.float32,
    )
    faces = np.asarray((0, 2, 1, 0, 1, 3, 1, 2, 3, 2, 0, 3), dtype=np.int32)
    builder = ModelBuilder()
    body = builder.add_link(
        mass=1.0, inertia=wp.mat33(np.eye(3)), label=source + body_suffix, lock_inertia=True, is_kinematic=kinematic
    )
    if convex:
        builder.add_shape_convex_hull(body, mesh=Mesh.create_sphere(half_extent, num_latitudes=16, num_longitudes=32))
    else:
        builder.add_shape_mesh(body, mesh=Mesh(vertices, faces))
    joint = builder.add_joint_fixed(parent=-1, child=body) if fixed else builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    return builder


def test_mesh_variant_resources_do_not_add_rigid_bodies() -> None:
    """A multi-asset catalog retains unused meshes without adding parked bodies."""
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/IsaacLabPrototypes/group_0/variant_1", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(small)
    body_count = builder.body_count
    definition = _prepare_mesh_variant(
        builder,
        (small, large),
        np.asarray(((0,), (1,)), dtype=np.int32),
        _mesh_variant_body_properties((small, large), 0.0),
        np.asarray((0, 0), dtype=np.int32),
        None,
    )
    assert builder.body_count == body_count
    assert definition.source_shape_indices.shape == (2, 1)
    model = builder.finalize(device="cpu")
    variants = MeshVariants(model, definition)
    np.testing.assert_array_equal(variants.variant_ids.numpy(), (0, 0))
    np.testing.assert_array_equal(model.shape_flags.numpy()[definition.source_shape_indices.flatten()], 0)


def test_mesh_variant_catalogs_preserve_reused_sources() -> None:
    """Catalogs may reuse source geometry while keeping independent targets and selection IDs."""
    sources = (_variant_builder("/World/small", 0.1), _variant_builder("/World/large", 0.2))
    builder = ModelBuilder()
    for world_sources in (sources, tuple(reversed(sources))):
        builder.begin_world()
        for source in world_sources:
            builder.add_builder(source)
        builder.end_world()
    duplicate_sources = (*sources, sources[0])
    first = _prepare_mesh_variant(
        builder,
        duplicate_sources,
        np.asarray(((0,), (2,)), dtype=np.int32),
        _mesh_variant_body_properties(duplicate_sources, 0.0),
        np.asarray((2, 1), dtype=np.int32),
        None,
    )
    second = _prepare_mesh_variant(
        builder,
        sources,
        np.asarray(((1,), (3,)), dtype=np.int32),
        _mesh_variant_body_properties(sources, 0.0),
        np.asarray((1, 0), dtype=np.int32),
        None,
    )
    assert builder.body_count == 4
    assert builder.shape_count == 9
    model = builder.finalize(device="cpu")
    selected = MeshVariants(model, first)
    reused = MeshVariants(model, second)
    np.testing.assert_array_equal(selected.variant_ids.numpy(), (2, 1))
    np.testing.assert_array_equal(reused.variant_ids.numpy(), (1, 0))
    untouched = model.shape_source_ptr.numpy()[[1, 3]].copy()
    selected.select(wp.array([0], dtype=wp.int32, device="cpu"), wp.array([1], dtype=wp.int32, device="cpu"))
    np.testing.assert_array_equal(selected.variant_ids.numpy(), (2, 0))
    np.testing.assert_array_equal(reused.variant_ids.numpy(), (1, 0))
    np.testing.assert_array_equal(model.shape_source_ptr.numpy()[[1, 3]], untouched)


@pytest.mark.parametrize(
    ("body_type", "convex"), [("free", False), ("fixed", False), ("kinematic", False), ("free", True)]
)
def test_mesh_selection_updates_geometry_and_inertia_without_solver(body_type: str, convex: bool) -> None:
    """Switch selected worlds, including a previously unused mesh and off-diagonal inertia."""
    fixed, kinematic = body_type == "fixed", body_type == "kinematic"
    paths = ("/World/envs/env_0/object", "/World/envs/env_1/object", "/World/IsaacLabPrototypes/unused")
    sources = (
        _variant_builder(paths[0], 0.1, fixed=fixed, kinematic=kinematic, convex=convex),
        _variant_builder(paths[1], 0.2, "/other_body", fixed=fixed, kinematic=kinematic, convex=convex),
        _variant_builder(paths[2], 0.3, fixed=fixed, kinematic=kinematic, convex=convex),
    )
    for variant, source in enumerate(sources):
        source.shape_margin[0] = 0.001 * (variant + 1)
        source.shape_gap[0] = 0.002 * (variant + 1)
        source.shape_is_solid[0] = variant != 2
    sources[2].body_mass[0] = 4.0
    sources[2].body_com[0] = wp.vec3(0.01, 0.02, 0.03)
    sources[2].body_inertia[0] = wp.mat33(2.0, 0.5, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0)
    builder = ModelBuilder()
    builder.add_world(sources[0])
    builder.add_world(sources[1])
    definition = _prepare_mesh_variant(
        builder,
        sources,
        np.asarray(((0,), (1,)), dtype=np.int32),
        _mesh_variant_body_properties(sources, 0.25),
        np.asarray((0, 1), dtype=np.int32),
        tuple(_render_shapes(source, ShapeFlags.VISIBLE) for source in sources),
    )
    resource_shapes = definition.source_shape_indices
    model = builder.finalize(device="cpu")
    selected = MeshVariants(model, definition)
    render_set = selected.render_set
    assert render_set is not None
    ids = selected.variant_ids
    original_body_flags = model.body_flags.numpy().copy()
    np.testing.assert_array_equal(ids.numpy(), (0, 1))
    np.testing.assert_allclose(model.body_inertia.numpy(), np.tile(np.eye(3), (2, 1, 1)))
    model.shape_material_mu.fill_(0.87)
    untouched_inertia = model.body_inertia.numpy()[1].copy()
    untouched_inv_mass = model.body_inv_mass.numpy()[1]
    resources_before = model.shape_source_ptr.numpy()[np.asarray(resource_shapes).flatten()].copy()
    geometry_before = {
        field: getattr(model, field).numpy().copy()
        for field in ("shape_margin", "shape_gap", "shape_is_solid", "_shape_support_data", "bvh_shape_bounds")
    }
    if convex:
        assert not np.array_equal(geometry_before["_shape_support_data"][0], geometry_before["_shape_support_data"][-1])
    expected_inertia = np.asarray(((2.25, 0.5, 0.0), (0.5, 3.25, 0.0), (0.0, 0.0, 4.25)))
    for variant in (2, 1, 2):
        selected.select(wp.array([variant], dtype=wp.int32, device="cpu"), wp.array([0], dtype=wp.int32, device="cpu"))
        assert ids is selected.variant_ids
        np.testing.assert_array_equal(ids.numpy(), (variant, 1))
        assert model.shape_source_ptr.numpy()[0] == model.shape_source_ptr.numpy()[resource_shapes[variant][0]]
        np.testing.assert_array_equal(model.body_inertia.numpy()[1], untouched_inertia)
        for field, original in geometry_before.items():
            values = getattr(model, field).numpy()
            np.testing.assert_array_equal(values[0], original[resource_shapes[variant][0]], err_msg=field)
            np.testing.assert_array_equal(values[1:], original[1:], err_msg=field)
    np.testing.assert_allclose(model.body_mass.numpy(), (4.0, 1.0))
    np.testing.assert_allclose(model.body_com.numpy()[0], (0.01, 0.02, 0.03))
    np.testing.assert_allclose(model.body_inertia.numpy()[0], expected_inertia)
    np.testing.assert_array_equal(model.body_flags.numpy(), original_body_flags)
    np.testing.assert_allclose(model.body_inv_mass.numpy(), (0.0 if fixed or kinematic else 0.25, untouched_inv_mass))
    np.testing.assert_allclose(
        model.body_inv_inertia.numpy()[0],
        np.zeros((3, 3)) if fixed or kinematic else np.linalg.inv(expected_inertia),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(model.shape_material_mu.numpy(), 0.87)
    np.testing.assert_array_equal(
        model.shape_source_ptr.numpy()[np.asarray(resource_shapes).flatten()], resources_before
    )
    np.testing.assert_array_equal(render_set.body_indices.numpy(), (0, 1))
    assert render_set.model_shape_indices == (0, 1)
    assert tuple(len(shapes) for shapes in render_set.visual_variants) == (1, 1, 1)
    assert render_set.collision_variants[2][0].source is sources[2].shape_source[0]
    assert render_set.collision_variants[2][0].scale == tuple(sources[2].shape_scale[0])
    np.testing.assert_allclose(np.asarray(sources[0].body_inertia[0]).reshape(3, 3), np.eye(3))


def test_mesh_variants_do_not_depend_on_fork_solver_banks() -> None:
    """Keep declarations in ClonePlan and selections/render metadata in the runtime bank."""
    physics_dir = Path(mesh_variants.__file__).parent
    assert not (physics_dir / "mesh_variant_batch.py").exists()
    forbidden = {
        "MeshVariantSet",
        "MJWarpMeshVariantBatch",
        "set_mesh_variant_index",
        "mesh_variant_names",
        "registrations",
        "_register_mesh_variant_cfg",
        "_set_mesh_variant_index",
        "_mesh_variant_ids",
        "mesh_variant_ids",
        "mesh_variant_render_sets",
        "_mesh_variant_cfgs",
        "_mesh_variant_resource_shapes",
        "_mesh_variant_visual_sources",
        "visual_source_builders",
        "build_mesh_variants",
        "_prepare_mesh_variant_resources",
        "mesh_variant_resources",
        "mesh_variant_visual_shapes",
    }
    paths = [physics_dir / name for name in ("mesh_variants.py", "newton_manager.py", "mjwarp_manager.py")]
    paths.append(physics_dir.parent / "cloner" / "replicate.py")
    source_dir = physics_dir.parents[2]
    flag_paths = [*paths, physics_dir.parent / "assets/rigid_object/rigid_object.py"]
    flag_paths.extend(
        source_dir / "isaaclab/isaaclab" / path
        for path in (
            "assets/rigid_object/rigid_object_cfg.py",
            "assets/rigid_object/base_rigid_object.py",
            "cloner/replicate_session.py",
        )
    )
    for path in flag_paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Attribute):
                assert node.attr != "mesh_variants_enabled", f"{path.name} gates the spawner catalog behind an opt-in"
            elif isinstance(node, (ast.Name, ast.Constant)):
                assert (node.id if isinstance(node, ast.Name) else node.value) != "mesh_variants_enabled"
    for path in paths:
        filename = path.name
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if path.name == "mesh_variants.py":
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("isaaclab.cloner"), "Bank must consume resolved targets"
                elif isinstance(node, ast.Name):
                    assert node.id not in {"ClonePlan", "cloner_path"}, "Bank must consume resolved targets"
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"{filename}:{node.lineno} depends on {node.attr}"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden, f"{filename}:{node.lineno} depends on {node.id}"
            elif isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                assert node.name not in forbidden, f"{filename}:{node.lineno} defines {node.name}"
            elif isinstance(node, ast.arg):
                assert node.arg not in forbidden, f"{filename}:{node.lineno} accepts {node.arg}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.endswith("mesh_variant_batch"), f"{filename} imports the removed batch owner"
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            if cls.name not in {"MeshVariants", "NewtonBackend"}:
                continue
            for node in ast.walk(cls):
                if isinstance(node, ast.Name):
                    assert node.id not in {"ClonePlan", "cloner_path"}, f"{cls.name} reinterprets the clone plan"
                elif isinstance(node, ast.Attribute):
                    assert node.attr not in {"get_clone_plan", "_cl_protos"}, f"{cls.name} rediscovers source ownership"


@pytest.mark.parametrize(
    "mismatch",
    [
        "missing_resources",
        "model_root_joint",
        "collider_count",
        "filtering",
        "primitive",
        "source_root_joint",
        "kinematic_root_joint",
    ],
)
def test_mesh_variants_reject_incompatible_resources(mismatch: str) -> None:
    """Unsupported catalogs keep static spawning; malformed prepared model mappings still fail."""
    kinematic = mismatch == "kinematic_root_joint"
    small = _variant_builder("/World/envs/env_0/object", 0.1, kinematic=kinematic)
    large = _variant_builder("/World/envs/env_1/object", 0.2, fixed=kinematic, kinematic=kinematic)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(
        _variant_builder("/World/envs/env_1/object", 0.2, fixed=True) if mismatch == "model_root_joint" else large
    )
    if mismatch == "source_root_joint":
        large = _variant_builder("/World/envs/env_1/object", 0.2, fixed=True)
    if mismatch == "collider_count":
        large.add_shape_mesh(0, mesh=large.shape_source[0])
    if mismatch == "filtering":
        large.shape_collision_group[0] = 2
    if mismatch == "primitive":
        large.shape_type[0], large.shape_source[0] = GeoType.BOX, None
    counts = builder.body_count, builder.shape_count
    properties = _mesh_variant_body_properties((small, large), 0.0)
    if mismatch not in ("missing_resources", "model_root_joint"):
        assert properties is None
        assert (builder.body_count, builder.shape_count) == counts
        return
    definition = _prepare_mesh_variant(
        builder,
        (small, large),
        np.asarray(((0,), (1,)), dtype=np.int32),
        properties,
        np.asarray((0, 1), dtype=np.int32),
        None,
    )
    model = builder.finalize(device="cpu")
    if mismatch == "missing_resources":
        definition = replace(definition, source_shape_indices=np.empty((0, 1), dtype=np.int32))
    expected = "collision-shape topology" if mismatch == "missing_resources" else "root-joint"
    with pytest.raises(ValueError, match=expected):
        MeshVariants(model, definition)
