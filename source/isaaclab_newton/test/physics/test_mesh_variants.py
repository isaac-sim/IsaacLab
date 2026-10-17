# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for reset-time Newton mesh variants."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp
from isaaclab_newton.physics import mesh_variants
from isaaclab_newton.physics.mesh_variants import (
    _disable_mesh_variant_resources,
    _prepare_mesh_variant_resources,
    build_mesh_variants,
)
from newton import Mesh, ModelBuilder

from isaaclab.cloner import make_clone_plan
from isaaclab.cloner import path as cloner_path


def _variant_builder(
    source: str, half_extent: float, body_suffix: str = "/body", fixed: bool = False, kinematic: bool = False
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
    builder.add_shape_mesh(body, mesh=Mesh(vertices, faces))
    joint = builder.add_joint_fixed(parent=-1, child=body) if fixed else builder.add_joint_free(child=body)
    builder.add_articulation([joint])
    return builder


def _clone_plan(cfg, active_mask: torch.Tensor, sources=None):
    cfg.prim_path = "/World/envs/env_.*/object"
    sources = sources or ("/World/envs/env_0/object", "/World/envs/env_1/object")
    assets = [SimpleNamespace(prim_path=cfg.prim_path, spawn=SimpleNamespace(spawn_path=source)) for source in sources]
    return make_clone_plan(
        assets, [tuple(torch.where(column)[0].tolist()) for column in active_mask.T], active_mask.shape[1]
    )


def test_mesh_variant_resources_do_not_add_rigid_bodies() -> None:
    """Keep unselected source meshes alive without adding parked physical objects."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large_path = "/World/IsaacLabPrototypes/group_0/variant_1"
    large = _variant_builder(large_path, 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(small)
    body_count = builder.body_count
    plan = _clone_plan(cfg, torch.tensor(((True, True), (False, False))), ("/World/envs/env_0/object", large_path))
    resources = _prepare_mesh_variant_resources(
        {"object": cfg},
        builder,
        dict(zip(cloner_path.get_asset_prototype_paths(plan), (small, large), strict=True)),
        plan,
    )
    assert builder.body_count == body_count
    assert tuple(len(shapes) for shapes in resources["object"]) == (1, 1)
    model = builder.finalize(device="cpu")
    _disable_mesh_variant_resources(model, resources)
    np.testing.assert_array_equal(model.shape_flags.numpy()[np.asarray(resources["object"]).flatten()], 0)


@pytest.mark.parametrize("body_type", ["free", "fixed", "kinematic"])
def test_mesh_selection_updates_geometry_and_inertia_without_solver(body_type: str) -> None:
    """Switch selected worlds, including a previously unused mesh and off-diagonal inertia."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.25)
    fixed, kinematic = body_type == "fixed", body_type == "kinematic"
    paths = ("/World/envs/env_0/object", "/World/envs/env_1/object", "/World/IsaacLabPrototypes/unused")
    sources = [
        _variant_builder(paths[0], 0.1, fixed=fixed, kinematic=kinematic),
        _variant_builder(paths[1], 0.2, "/other_body", fixed=fixed, kinematic=kinematic),
        _variant_builder(paths[2], 0.3, fixed=fixed, kinematic=kinematic),
    ]
    sources[2].body_mass[0] = 4.0
    sources[2].body_com[0] = wp.vec3(0.01, 0.02, 0.03)
    sources[2].body_inertia[0] = wp.mat33(2.0, 0.5, 0.0, 0.5, 3.0, 0.0, 0.0, 0.0, 4.0)
    builder = ModelBuilder()
    builder.add_world(sources[0])
    builder.add_world(sources[1])
    plan = _clone_plan(cfg, torch.tensor(((True, False), (False, True), (False, False))), paths)
    source_builders = dict(zip(cloner_path.get_asset_prototype_paths(plan), sources, strict=True))
    resources = _prepare_mesh_variant_resources({"object": cfg}, builder, source_builders, plan)
    model = builder.finalize(device="cpu")
    _disable_mesh_variant_resources(model, resources)
    selections, (render_set,) = build_mesh_variants(
        {"object": cfg}, model, source_builders, plan, source_builders, resources
    )
    selected = selections["object"]
    ids = selected.variant_ids
    original_body_flags = model.body_flags.numpy().copy()
    np.testing.assert_array_equal(ids.numpy(), (0, 1))
    np.testing.assert_allclose(model.body_inertia.numpy(), np.tile(np.eye(3) * 1.25, (2, 1, 1)))
    model.shape_material_mu.fill_(0.87)
    untouched_inertia = model.body_inertia.numpy()[1].copy()
    resources_before = model.shape_source_ptr.numpy()[np.asarray(resources["object"]).flatten()].copy()
    expected_inertia = np.asarray(((2.25, 0.5, 0.0), (0.5, 3.25, 0.0), (0.0, 0.0, 4.25)))
    for variant in (2, 1, 2):
        selected.select(wp.array([variant], dtype=wp.int32, device="cpu"), wp.array([0], dtype=wp.int32, device="cpu"))
        assert ids is selected.variant_ids
        np.testing.assert_array_equal(ids.numpy(), (variant, 1))
        assert model.shape_source_ptr.numpy()[0] == model.shape_source_ptr.numpy()[resources["object"][variant][0]]
        np.testing.assert_array_equal(model.body_inertia.numpy()[1], untouched_inertia)
    np.testing.assert_allclose(model.body_mass.numpy(), (4.0, 1.0))
    np.testing.assert_allclose(model.body_com.numpy()[0], (0.01, 0.02, 0.03))
    np.testing.assert_allclose(model.body_inertia.numpy()[0], expected_inertia)
    np.testing.assert_array_equal(model.body_flags.numpy(), original_body_flags)
    np.testing.assert_allclose(model.body_inv_mass.numpy(), (0.0, 0.0) if fixed or kinematic else (0.25, 1.0))
    np.testing.assert_allclose(
        model.body_inv_inertia.numpy()[0],
        np.zeros((3, 3)) if fixed or kinematic else np.linalg.inv(expected_inertia),
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(model.shape_material_mu.numpy(), 0.87)
    np.testing.assert_array_equal(
        model.shape_source_ptr.numpy()[np.asarray(resources["object"]).flatten()], resources_before
    )
    np.testing.assert_array_equal(render_set.body_indices.numpy(), (0, 1))
    assert render_set.source_paths == paths
    assert render_set.model_shape_indices == (0, 1)
    assert tuple(len(shapes) for shapes in render_set.visual_variants) == (1, 1, 1)
    assert render_set.collision_variants[2][0].source is sources[2].shape_source[0]
    assert render_set.collision_variants[2][0].scale == tuple(sources[2].shape_scale[0])
    np.testing.assert_allclose(np.asarray(sources[0].body_inertia[0]).reshape(3, 3), np.eye(3))


def test_mesh_variants_do_not_depend_on_fork_solver_banks() -> None:
    """Keep variant ownership in Isaac Lab rather than reintroducing a fork-only Newton solver API."""
    physics_dir = Path(mesh_variants.__file__).parent
    assert not (physics_dir / "mesh_variant_batch.py").exists()
    forbidden = {"MeshVariantSet", "MJWarpMeshVariantBatch", "set_mesh_variant_index", "mesh_variant_names"}
    for filename in ("mesh_variants.py", "newton_manager.py", "mjwarp_manager.py"):
        tree = ast.parse((physics_dir / filename).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in forbidden, f"{filename}:{node.lineno} depends on {node.attr}"
                if node.attr == "mesh_variant_ids" and isinstance(node.value, ast.Name):
                    assert node.value.id != "solver", f"{filename}:{node.lineno} uses solver-owned variant IDs"
            elif isinstance(node, ast.Name):
                assert node.id not in forbidden, f"{filename}:{node.lineno} depends on {node.id}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.endswith("mesh_variant_batch"), f"{filename} imports the removed batch owner"


@pytest.mark.parametrize("mismatch", ["missing_resources", "collider_count", "filtering"])
def test_mesh_variants_reject_incompatible_resources(mismatch: str) -> None:
    """Reject inputs whose geometry cannot be switched without changing collision topology."""
    cfg = SimpleNamespace(mesh_variant_inertia_diagonal_offset=0.0)
    small = _variant_builder("/World/envs/env_0/object", 0.1)
    large = _variant_builder("/World/envs/env_1/object", 0.2)
    builder = ModelBuilder()
    builder.add_world(small)
    builder.add_world(large)
    plan = _clone_plan(cfg, torch.eye(2, dtype=torch.bool))
    sources = dict(zip(cloner_path.get_asset_prototype_paths(plan), (small, large), strict=True))
    if mismatch == "collider_count":
        large.add_shape_mesh(0, mesh=large.shape_source[0])
        with pytest.raises(ValueError, match="collision-shape topology"):
            _prepare_mesh_variant_resources({"object": cfg}, builder, sources, plan)
        return
    resources = _prepare_mesh_variant_resources({"object": cfg}, builder, sources, plan)
    model = builder.finalize(device="cpu")
    _disable_mesh_variant_resources(model, resources)
    if mismatch == "filtering":
        large.shape_collision_group[0] = 2
    with pytest.raises(ValueError, match="resources" if mismatch == "missing_resources" else "collision filtering"):
        build_mesh_variants(
            {"object": cfg},
            model,
            sources,
            plan,
            resource_shape_indices=None if mismatch == "missing_resources" else resources,
        )
