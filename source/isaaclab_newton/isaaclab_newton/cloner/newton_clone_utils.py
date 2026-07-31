# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch
import warp as wp
from newton import GeoType, ModelBuilder, ShapeFlags, solvers

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab import cloner
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

# USD ``physics:approximation`` token (lower case) -> Newton remeshing method.
# Mirrors Newton's own importer mapping; ``none`` keeps the raw trimesh.
_APPROXIMATION_TO_REMESHING_METHOD = {
    "convexdecomposition": "coacd",
    "convexhull": "convex_hull",
    "boundingsphere": "bounding_sphere",
    "boundingcube": "bounding_box",
    "meshsimplification": "quadratic",
}


def _authored_collision_approximations(stage: Usd.Stage) -> dict[str, str]:
    """Prim path -> authored ``physics:approximation`` token (lower case).

    SDF collision prims are excluded: the attribute has no meaning on a shape with
    ``NewtonSDFCollisionAPI`` applied (matching Newton's importer semantics).
    """
    authored: dict[str, str] = {}
    for prim in stage.Traverse():
        attr = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr()
        if attr and attr.HasAuthoredValue() and "NewtonSDFCollisionAPI" not in prim.GetAppliedSchemas():
            authored[prim.GetPath().pathString] = str(attr.Get()).lower()
    return authored


def _apply_authored_approximations(builder: ModelBuilder, path_shape_map: dict, authored: dict[str, str]) -> set[int]:
    """Remesh authored collision shapes (visual shapes preserved); return their indices."""
    authored_shape_indices: set[int] = set()
    for path, mode in authored.items():
        index = path_shape_map.get(path)
        if index is None:
            continue
        authored_shape_indices.add(index)
        method = _APPROXIMATION_TO_REMESHING_METHOD.get(mode)
        if method is not None:
            builder.approximate_meshes(method, shape_indices=[index], keep_visual_shapes=True)
    return authored_shape_indices


def _unauthored_collision_mesh_shapes(builder: ModelBuilder, authored_shape_indices: set[int]) -> list[int]:
    """Colliding mesh shapes not covered by an authored ``physics:approximation``."""
    return [
        index
        for index, shape_type in enumerate(builder.shape_type)
        if shape_type == GeoType.MESH
        and (builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES)
        and index not in authored_shape_indices
    ]


def _has_visible_non_collision_geometry(stage: Usd.Stage, prim_path: str) -> bool:
    """Return whether a prim hierarchy contains visible geometry without collision."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        return False
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() != UsdGeom.Tokens.invisible and imageable.ComputePurpose() in (
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.proxy,
        ):
            return True
    return False


def _restore_visible_colliders_without_visual_shapes(
    builder: ModelBuilder, stage: Usd.Stage, path_shape_map: dict[str, int] | None
) -> None:
    """Show viewport-visible colliders on bodies without separate visual shapes.

    Newton normally hides every collider when any visual-only shape exists in the
    imported model. Isaac Lab procedural shapes use one default-purpose USD geometry
    for both collision and visualization, so an unrelated visual asset must not hide
    them. Imported collision meshes, guide-purpose collision geometry, and
    colliders on bodies with separate visual shapes remain hidden.
    """
    if not path_shape_map:
        return
    bodies_with_visual_shapes = {
        builder.shape_body[index]
        for index, flags in enumerate(builder.shape_flags)
        if builder.shape_body[index] >= 0 and flags & ShapeFlags.VISIBLE and not flags & ShapeFlags.COLLIDE_SHAPES
    }
    static_parent_paths = {
        path.rpartition("/")[0] for path, index in path_shape_map.items() if builder.shape_body[index] < 0
    }
    static_parents_with_visual_shapes = {
        path for path in static_parent_paths if _has_visible_non_collision_geometry(stage, path)
    }
    for path, index in path_shape_map.items():
        flags = builder.shape_flags[index]
        body_index = builder.shape_body[index]
        if (
            not flags & ShapeFlags.COLLIDE_SHAPES
            or builder.shape_type[index] == GeoType.MESH
            or body_index in bodies_with_visual_shapes
            or (body_index < 0 and path.rpartition("/")[0] in static_parents_with_visual_shapes)
        ):
            continue
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        if (
            imageable
            and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
            and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy)
        ):
            builder.shape_flags[index] = flags | ShapeFlags.VISIBLE


def build_source_builders(
    stage: Usd.Stage,
    sources: Sequence[str],
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    *,
    ignore_paths: Sequence[str] | None = None,
    simplify_meshes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path.

    USD-authored ``physics:approximation`` modes are honored (applied after import so
    visual shapes are preserved for visualization/rendering). Exception: when the
    honored modes leave multiple sources with differing shape-type sequences (e.g.
    heterogeneous asset variants), every mesh falls back to the uniform convex-hull
    treatment, because :class:`SolverMuJoCo` requires homogeneous worlds.
    """
    authored = _authored_collision_approximations(stage)
    builders = {
        source: _build_source_builder(
            stage, source, create_builder, schema_resolvers, ignore_paths, simplify_meshes, authored
        )
        for source in sources
    }

    if authored and len(builders) > 1:
        shape_sequences = {tuple(int(t) for t in b.shape_type) for b in builders.values()}
        if len(shape_sequences) > 1:
            warnings.warn(
                "Clone sources have differing collision shape sequences after honoring authored"
                " physics:approximation modes, which SolverMuJoCo's homogeneous-worlds requirement"
                " does not support. Falling back to uniform convex-hull approximation for all"
                " collision meshes.",
                stacklevel=2,
            )
            builders = {
                source: _build_source_builder(
                    stage, source, create_builder, schema_resolvers, ignore_paths, simplify_meshes, {}
                )
                for source in sources
            }
    return builders


def _build_source_builder(
    stage: Usd.Stage,
    source: str,
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    ignore_paths: Sequence[str] | None,
    simplify_meshes: bool,
    authored: dict[str, str],
) -> ModelBuilder:
    """Build one source builder; an empty ``authored`` map restores hull-everything."""
    builder = create_builder()
    solvers.SolverMuJoCo.register_custom_attributes(builder)
    solvers.SolverKamino.register_custom_attributes(builder)
    import_result = builder.add_usd(
        stage,
        root_path=source,
        load_visual_shapes=True,
        hide_collision_shapes=True,
        skip_mesh_approximation=True,
        schema_resolvers=schema_resolvers,
        ignore_paths=ignore_paths,
    )
    _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
    if authored:
        authored_shape_indices = _apply_authored_approximations(builder, import_result["path_shape_map"], authored)
        if simplify_meshes:
            builder.approximate_meshes(
                "convex_hull",
                shape_indices=_unauthored_collision_mesh_shapes(builder, authored_shape_indices),
                keep_visual_shapes=True,
            )
    elif simplify_meshes:
        builder.approximate_meshes("convex_hull", keep_visual_shapes=True)
    replace_newton_builder_shape_colors(builder, stage)
    return builder


def replicate_builder_mapping(
    builder: ModelBuilder,
    sources: Sequence[str],
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    source_builders: dict[str, ModelBuilder],
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, list[float], list[float]], None]] = (),
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    world_xforms = [wp.transform(positions[col], quaternions[col]) for col in range(num_worlds)]

    can_batch = (
        len(sources) == 1
        and mapping.size(0) == 1
        and num_worlds > 0
        and bool(mapping.all().item())
        and not per_world_builder_hooks
    )
    if can_batch:
        source_builder = source_builders[sources[0]]

        # Inject env-root sites into the source so replicate() copies them. Prefixed
        # by world_xforms[0] so R_w = world_xform_w * inv(world_xform_0) lands each
        # copy at world_xform_w * xform.
        site_local_indices: dict[str, list[int]] = {}
        for label, xform in env_root_sites.items():
            idx = source_builder.add_site(body=-1, xform=wp.transform_multiply(world_xforms[0], xform), label=label)
            site_local_indices.setdefault(label, []).append(idx)
        for label, indices in source_site_indices.get(id(source_builder), {}).items():
            site_local_indices.setdefault(label, []).extend(indices)

        # Site index after replicate: base_shape + world * stride + source_local_index.
        base_shape = builder.shape_count
        stride = source_builder.shape_count
        source_xform_inv = wp.transform_inverse(world_xforms[0])
        xforms = [wp.transform_multiply(world_xform, source_xform_inv) for world_xform in world_xforms]
        builder.replicate(source_builder, num_worlds, xforms=xforms)

        for label, local_indices in site_local_indices.items():
            local_site_map[label] = [
                [base_shape + world * stride + local for local in local_indices] for world in range(num_worlds)
            ]

        for hook in post_replicate_hooks:
            hook(builder)
        return local_site_map, world_xforms

    source_world_indices = mapping.to(dtype=torch.int64).argmax(dim=1)

    for col in range(num_worlds):
        builder.begin_world()
        world_xform = world_xforms[col]

        for label, xform in env_root_sites.items():
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(world_xform, xform), label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            source_builder = source_builders[sources[int(row)]]
            offset = builder.shape_count
            source_col = int(source_world_indices[int(row)])
            source_xform = world_xforms[source_col]
            builder.add_builder(
                source_builder, xform=wp.transform_multiply(world_xform, wp.transform_inverse(source_xform))
            )

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, positions[col].tolist(), quaternions[col].tolist())
        builder.end_world()

    for hook in post_replicate_hooks:
        hook(builder)
    return local_site_map, world_xforms


_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> list[tuple[str, int]]:
    """Rewrite source-root labels to per-env destination roots and return Fabric body bindings."""
    fabric_body_bindings: list[tuple[str, int]] = []
    bound_body_indices: set[int] = set()

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/")
        world_cols = torch.nonzero(mapping[source_index], as_tuple=True)[0].tolist()
        world_roots = {int(env_ids[col]): destinations[source_index].format(int(env_ids[col])) for col in world_cols}

        def _rename_pair(values, worlds, *, collect_body_bindings: bool = False):
            for index, (value, world) in enumerate(zip(values, worlds, strict=True)):
                if world is None:
                    continue
                world_root = world_roots.get(int(world))
                if isinstance(value, str) and world_root is not None:
                    renamed_value = cloner.path.rebase(value, source_root, world_root)
                    if renamed_value != value:
                        values[index] = renamed_value
                        if collect_body_bindings:
                            fabric_body_bindings.append((renamed_value, index))
                            bound_body_indices.add(index)

        for labels, worlds, collect_body_bindings in (
            (builder.body_label, builder.body_world, True),
            (builder.joint_label, builder.joint_world, False),
            (builder.shape_label, builder.shape_world, False),
            (builder.articulation_label, builder.articulation_world, False),
            (builder.constraint_mimic_label, builder.constraint_mimic_world, False),
        ):
            _rename_pair(labels, worlds, collect_body_bindings=collect_body_bindings)

        custom_attrs = builder.custom_attributes.values()
        worlds_by_freq = {attr.frequency: attr.values for attr in custom_attrs if attr.references == "world"}
        for attr in custom_attrs:
            if attr.dtype is str and attr.values and (worlds := worlds_by_freq.get(attr.frequency)):
                _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
