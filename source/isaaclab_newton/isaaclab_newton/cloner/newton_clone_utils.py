# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
import warp as wp
from newton import GeoType, ModelBuilder, ShapeFlags, solvers

from pxr import Usd, UsdPhysics

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
        skip_mesh_approximation=True,
        schema_resolvers=schema_resolvers,
        ignore_paths=ignore_paths,
    )
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


def _invert_xform(xform: Sequence[float]) -> np.ndarray:
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
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Replicate source builders into per-env Newton worlds."""
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    # One bulk transfer, then plain Python rows: feeding torch slices to wp.transform walks
    # each component through the tensor's __getitem__ once per world.
    positions_np = positions.detach().cpu().numpy().astype(np.float32, copy=False)
    quaternions_np = quaternions.detach().cpu().numpy().astype(np.float32, copy=False)
    positions_rows = positions_np.tolist()
    quaternions_rows = quaternions_np.tolist()
    world_xforms = [wp.transform(positions_rows[col], quaternions_rows[col]) for col in range(num_worlds)]

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
        source_xform_inv = _invert_xform(positions_rows[0] + quaternions_rows[0])
        xforms = _compose_world_xforms(positions_np, quaternions_np, source_xform_inv)
        builder.replicate(source_builder, num_worlds, xforms=xforms)

        for label, local_indices in site_local_indices.items():
            local_site_map[label] = [
                [base_shape + world * stride + local for local in local_indices] for world in range(num_worlds)
            ]

        for hook in post_replicate_hooks:
            hook(builder)
        return local_site_map, world_xforms

    source_world_indices = mapping.to(dtype=torch.int64).argmax(dim=1).tolist()

    # Per-world placements for every source and every env-root site, composed up front so the
    # per-world loop below only indexes rows.
    root_site_xforms = {
        label: _compose_world_xforms(positions_np, quaternions_np, xform) for label, xform in env_root_sites.items()
    }
    source_xforms = [
        _compose_world_xforms(
            positions_np,
            quaternions_np,
            _invert_xform(positions_rows[source_col] + quaternions_rows[source_col]),
        )
        for source_col in source_world_indices
    ]
    rows_per_world = [torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist() for col in range(num_worlds)]

    for col in range(num_worlds):
        builder.begin_world()

        for label, world_site_xforms in root_site_xforms.items():
            site_idx = builder.add_site(body=-1, xform=world_site_xforms[col], label=label)
            local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col].append(site_idx)

        for row in rows_per_world[col]:
            source_builder = source_builders[sources[row]]
            offset = builder.shape_count
            builder.add_builder(source_builder, xform=source_xforms[row][col])

            for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
                local_indices = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])[col]
                local_indices.extend(offset + shape_idx for shape_idx in source_shape_indices)

        for hook in per_world_builder_hooks:
            hook(builder, col, positions_rows[col], quaternions_rows[col])
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
    env_ids_list = env_ids.tolist()

    for source_index, source in enumerate(sources):
        source_root = source.rstrip("/") or "/"
        source_root_len = len(source_root)
        world_cols = torch.nonzero(mapping[source_index], as_tuple=True)[0].tolist()
        # Pre-normalize the destination roots
        destination = destinations[source_index]
        world_roots = {
            env_id: (destination.format(env_id).rstrip("/") or "/")
            for env_id in (env_ids_list[col] for col in world_cols)
        }

        def _rename_pair(values, worlds, *, collect_body_bindings: bool = False):
            for index, (value, world) in enumerate(zip(values, worlds, strict=True)):
                if world is None or not isinstance(value, str) or not value.startswith(source_root):
                    continue
                suffix = value[source_root_len:]
                if suffix and not suffix.startswith("/"):
                    continue
                world_root = world_roots.get(int(world))
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
            if attr.dtype is str and attr.values and (worlds := worlds_by_freq.get(attr.frequency)):
                _rename_pair(attr.values, worlds)

    fabric_body_bindings.extend(
        (label, index) for index, label in enumerate(builder.body_label) if index not in bound_body_indices
    )
    return fabric_body_bindings
