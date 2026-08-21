# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import warp as wp
from newton import GeoType, ModelBuilder, ShapeFlags, solvers

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import path as clone_path
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors

from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths


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
    builder: ModelBuilder,
    stage: Usd.Stage,
    path_shape_map: dict[str, int] | None,
    load_visual_shapes: bool = True,
) -> None:
    """Show viewport-visible colliders on bodies without separate visual shapes.

    Newton normally hides every collider when any visual-only shape exists in the
    imported model. Isaac Lab procedural shapes use one default-purpose USD geometry
    for both collision and visualization, so an unrelated visual asset must not hide
    them. Imported collision meshes, guide-purpose collision geometry, and
    colliders on bodies with separate visual shapes remain hidden.

    With ``load_visual_shapes=False`` the pass is skipped: Newton never hides a collider
    when the model holds no visual-only shapes, so every flag it would set is already set,
    and nothing draws them in a run that opted out of visual geometry. The skipped USD
    visibility/purpose resolution is per collider shape, so it is worth avoiding.
    """
    if not path_shape_map or not load_visual_shapes:
        return
    bodies_with_visual_shapes = {
        builder.shape_body[index]
        for index, flags in enumerate(builder.shape_flags)
        if builder.shape_body[index] >= 0 and flags & ShapeFlags.VISIBLE and not flags & ShapeFlags.COLLIDE_SHAPES
    }
    # Resolved on first use: a static parent whose colliders are all filtered out below is
    # never traversed at all.
    static_parents_with_visual_shapes: dict[str, bool] = {}
    for path, index in path_shape_map.items():
        flags = builder.shape_flags[index]
        body_index = builder.shape_body[index]
        if (
            not flags & ShapeFlags.COLLIDE_SHAPES
            or builder.shape_type[index] == GeoType.MESH
            or body_index in bodies_with_visual_shapes
        ):
            continue
        if body_index < 0:
            parent_path = path.rpartition("/")[0]
            if parent_path not in static_parents_with_visual_shapes:
                static_parents_with_visual_shapes[parent_path] = _has_visible_non_collision_geometry(stage, parent_path)
            if static_parents_with_visual_shapes[parent_path]:
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
    load_visual_shapes: bool = True,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path.

    The cloner approximates nothing. Collision geometry is whatever the asset authored:
    Newton's importer applies each shape's ``physics:approximation`` while importing, and
    USD defaults that token to ``none``, meaning "use the mesh as-is". Change it where it
    is authored -- the mesh-collision schema fragments on the spawner -- not here.

    Args:
        stage: USD stage containing the source prims.
        sources: Source prim paths to build a builder for.
        create_builder: Factory returning a fresh :class:`ModelBuilder`.
        schema_resolvers: Schema resolvers forwarded to Newton's USD importer.
        ignore_paths: Prim paths skipped during import.
        load_visual_shapes: Whether to import visual-only geometry. Importing it costs
            USD parse time and memory that only pays off when the shapes are rendered
            or ray cast.
    """
    return {
        source: _build_source_builder(stage, source, create_builder, schema_resolvers, ignore_paths, load_visual_shapes)
        for source in sources
    }


def _build_source_builder(
    stage: Usd.Stage,
    source: str,
    create_builder: Callable[[], ModelBuilder],
    schema_resolvers: Sequence[Any],
    ignore_paths: Sequence[str] | None,
    load_visual_shapes: bool = True,
) -> ModelBuilder:
    """Build one source builder."""
    builder = create_builder()
    solvers.SolverMuJoCo.register_custom_attributes(builder)
    solvers.SolverKamino.register_custom_attributes(builder)
    import_result = builder.add_usd(
        stage,
        root_path=source,
        load_visual_shapes=load_visual_shapes,
        hide_collision_shapes=True,
        skip_mesh_approximation=False,
        schema_resolvers=schema_resolvers,
        ignore_paths=ignore_paths,
    )
    _restore_visible_colliders_without_visual_shapes(
        builder, stage, import_result["path_shape_map"], load_visual_shapes
    )
    replace_newton_builder_shape_colors(builder, stage)
    if load_visual_shapes:
        import_builder_visual_material_paths(builder, stage)
    return builder


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of xyzw quaternion arrays, broadcast over the leading axes."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)
    out[..., 0] = aw * bx + ax * bw + ay * bz - az * by
    out[..., 1] = aw * by - ax * bz + ay * bw + az * bx
    out[..., 2] = aw * bz + ax * by - ay * bx + az * bw
    out[..., 3] = aw * bw - ax * bx - ay * by - az * bz
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by xyzw quaternions ``q``, broadcast over the leading axes."""
    axis, angle_w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(axis, v)
    return (v + angle_w * t + np.cross(axis, t)).astype(np.float32)


def _compose_world_xforms(world_p: np.ndarray, world_q: np.ndarray, local: Sequence[float]) -> np.ndarray:
    """``world_xform_w * local`` for every world, as one ``[num_worlds, 7]`` xyzw array."""
    local = np.asarray(local, dtype=np.float32)
    out = np.empty((world_p.shape[0], 7), dtype=np.float32)
    out[:, :3] = world_p + _quat_rotate(world_q, np.broadcast_to(local[:3], world_p.shape))
    out[:, 3:] = _quat_multiply(world_q, local[3:])
    return out


def _invert_xform(xform: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of a single xyzw transform, assuming a unit quaternion."""
    xform = np.asarray(xform, dtype=np.float32)
    quat_inv = np.array([-xform[0 + 3], -xform[1 + 3], -xform[2 + 3], xform[6]], dtype=np.float32)
    return np.concatenate([-_quat_rotate(quat_inv, xform[:3]), quat_inv])


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
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
    quaternions_np = quaternions.detach().cpu().numpy().astype(np.float32, copy=False)
    xforms_np = np.concatenate((positions_np, quaternions_np), axis=1)
    xform_rows = xforms_np.tolist()
    world_xforms = [wp.transform(*row) for row in xform_rows]

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
        source_xform_inv = _invert_xform(xforms_np[0])
        xforms = _compose_world_xforms(positions_np, quaternions_np, source_xform_inv)
        builder.replicate(source_builder, num_worlds, xforms=xforms)

        for label, local_indices in site_local_indices.items():
            local_site_map[label] = [
                [base_shape + world * stride + local for local in local_indices] for world in range(num_worlds)
            ]

        return local_site_map, world_xforms

    source_world_indices = mapping.to(dtype=torch.int64).argmax(dim=1).tolist()

    # Per-world placements for every env-root site, composed up front so the per-world loop
    # below only indexes rows.
    root_site_xforms = {
        label: _compose_world_xforms(positions_np, quaternions_np, xform) for label, xform in env_root_sites.items()
    }
    # Same for the source placements, but only for the occupied ``(row, col)`` pairs of the
    # mapping: composing a dense ``num_rows x num_worlds`` table would blow up on heterogeneous
    # plans where each row is present in a handful of worlds.
    # One scan of the transposed mapping yields the occupied pairs in world-major order, which
    # is the order both indices want; scanning per world instead costs a torch call and a
    # device sync per world.
    rows_per_world: list[list[int]] = [[] for _ in range(num_worlds)]
    worlds_per_row: dict[int, list[int]] = {}
    for col, row in mapping.t().nonzero(as_tuple=False).tolist():
        rows_per_world[col].append(row)
        worlds_per_row.setdefault(row, []).append(col)
    source_xforms: dict[tuple[int, int], np.ndarray] = {}
    for row, cols in worlds_per_row.items():
        source_col = source_world_indices[row]
        row_xforms = _compose_world_xforms(
            positions_np[cols],
            quaternions_np[cols],
            _invert_xform(xforms_np[source_col]),
        )
        source_xforms.update(((row, col), row_xforms[index]) for index, col in enumerate(cols))

    for col in range(num_worlds):
        builder.begin_world()

        for label, world_site_xforms in root_site_xforms.items():
            site_idx = builder.add_site(body=-1, xform=world_site_xforms[col], label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in rows_per_world[col]:
            source_builder = source_builders[sources[row]]
            offset = builder.shape_count
            builder.add_builder(source_builder, xform=source_xforms[row, col])

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, xform_rows[col][:3], xform_rows[col][3:])
        builder.end_world()

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
    env_ids_list = env_ids.tolist()

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/") or "/"
        world_cols = torch.nonzero(mapping[source_index], as_tuple=True)[0].tolist()
        # Pre-normalize the destination roots
        destination = destinations[source_index]
        world_roots = {
            env_id: (destination.format(env_id).rstrip("/") or "/")
            for env_id in (env_ids_list[col] for col in world_cols)
        }

        def _rename_pair(values, worlds, src_root=source_root, roots=world_roots, *, collect_body_bindings=False):
            rows = (
                ((index, value, worlds[index]) for index, value in values.items())
                if isinstance(values, dict)
                else ((index, value, world) for index, (value, world) in enumerate(zip(values, worlds, strict=True)))
            )
            for index, value, world in rows:
                suffix = clone_path.relative_to(value, src_root) if isinstance(value, str) else None
                if world is None or suffix is None:
                    continue
                world_root = roots.get(int(world))
                if world_root is None:
                    continue
                renamed_value = world_root + suffix
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
            if attr.dtype is not str or not attr.values:
                continue
            if attr.namespace == "isaaclab" and attr.name == "visual_material_path":
                _rename_pair(attr.values, builder.shape_world)
            elif worlds := worlds_by_freq.get(attr.frequency):
                _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
