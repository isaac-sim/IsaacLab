# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import warp as wp
from newton import GeoType, JointType, ModelBuilder, ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan
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


def _static_collider_owner_path(stage: Usd.Stage, collider_path: str) -> str:
    """Return the nearest rigid-body ancestor or the collider's immediate parent."""
    collider_prim = stage.GetPrimAtPath(collider_path)
    prim = collider_prim.GetParent() if collider_prim else None
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return str(prim.GetPath())
        prim = prim.GetParent()
    return collider_path.rpartition("/")[0]


def _restore_visible_colliders_without_visual_shapes(
    builder: ModelBuilder,
    stage: Usd.Stage,
    path_shape_map: dict[str, int] | None,
    load_visual_shapes: bool = True,
) -> None:
    """Show viewport-visible colliders on bodies without separate visual shapes.

    Newton groups static colliders under the world body, where unrelated visual
    geometry can hide them. Isaac Lab procedural shapes use one default-purpose USD
    geometry for both collision and visualization. Imported collision meshes,
    guide-purpose geometry, and colliders with separate visuals remain hidden.

    With ``load_visual_shapes=False`` the pass is skipped: Newton never hides a collider
    when the model holds no visual-only shapes, so every flag it would set is already set,
    and nothing draws them in a run that opted out of visual geometry. The skipped USD
    visibility/purpose resolution is per collider shape, so it is worth avoiding.
    """
    if not path_shape_map or not load_visual_shapes:
        return
    # Newton may synthesize a visible ``*_visual`` mesh for a proxy-purpose collider.
    # It is not an authored USD prim and must remain hidden alongside its source collider.
    for index, path in enumerate(builder.shape_label):
        if not path.endswith("_visual") or stage.GetPrimAtPath(path):
            continue
        collider_prim = stage.GetPrimAtPath(path.removesuffix("_visual"))
        if collider_prim and collider_prim.HasAPI(UsdPhysics.CollisionAPI):
            builder.shape_flags[index] &= ~ShapeFlags.VISIBLE
    bodies_with_visual_shapes = {
        builder.shape_body[index]
        for index, flags in enumerate(builder.shape_flags)
        if builder.shape_body[index] >= 0 and flags & ShapeFlags.VISIBLE and not flags & ShapeFlags.COLLIDE_SHAPES
    }
    # Resolved on first use: a static parent whose colliders are all filtered out below is
    # never traversed at all.
    static_owners_with_visual_shapes: dict[str, bool] = {}
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
            owner_path = _static_collider_owner_path(stage, path)
            if owner_path not in static_owners_with_visual_shapes:
                static_owners_with_visual_shapes[owner_path] = _has_visible_non_collision_geometry(stage, owner_path)
            if static_owners_with_visual_shapes[owner_path]:
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
    skip_mesh_approximation: bool = False,
    import_results_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, ModelBuilder]:
    """Build one Newton builder for each clone source prim path.

    By default, Newton's importer applies each shape's authored
    ``physics:approximation``. Render-only callers can bypass collision mesh
    approximation with ``skip_mesh_approximation``.

    Args:
        stage: USD stage containing the source prims.
        sources: Source prim paths to build a builder for.
        create_builder: Factory returning a fresh :class:`ModelBuilder`.
        schema_resolvers: Schema resolvers forwarded to Newton's USD importer.
        ignore_paths: Prim paths skipped during import.
        load_visual_shapes: Whether to import visual-only geometry. Importing it costs
            USD parse time and memory that only pays off when the shapes are rendered
            or ray cast.
        skip_mesh_approximation: Whether to skip collision mesh approximation during import.
        import_results_out: Optional caller-owned output mapping populated in place with each
            source's USD import result.
    """
    builders = {}
    for source in dict.fromkeys(sources):
        builder = create_builder()
        import_result = builder.add_usd(
            stage,
            root_path=source,
            load_visual_shapes=load_visual_shapes,
            hide_collision_shapes=True,
            skip_mesh_approximation=skip_mesh_approximation,
            schema_resolvers=schema_resolvers,
            ignore_paths=[
                *(ignore_paths or ()),
                *(path for path in sources if path != source and clone_path.under(path, source)),
            ],
            return_deformable_results=True,
        )
        _restore_visible_colliders_without_visual_shapes(
            builder, stage, import_result["path_shape_map"], load_visual_shapes
        )
        replace_newton_builder_shape_colors(builder, stage)
        if load_visual_shapes:
            import_builder_visual_material_paths(builder, stage)
        _name_root_joints_after_their_body(builder)
        builders[source] = builder
        if import_results_out is not None:
            import_results_out[source] = import_result
    return builders


def _name_root_joints_after_their_body(builder: ModelBuilder) -> None:
    """Name importer-generated free root joints after their child bodies, in place."""
    for index, label in enumerate(builder.joint_label):
        if not isinstance(label, str) or not label.startswith("joint_") or not label[6:].isdigit():
            continue
        if builder.joint_type[index] != JointType.FREE or builder.joint_parent[index] != -1:
            continue
        body_label = builder.body_label[builder.joint_child[index]]
        if isinstance(body_label, str) and body_label.startswith("/"):
            builder.joint_label[index] = f"{body_label}_free_joint"


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


def _label_groups(builder: ModelBuilder) -> dict[str, list]:
    """Return every entity-label container owned by a Newton builder."""
    groups = {
        name: value for name, value in vars(builder).items() if name.endswith("_label") and isinstance(value, list)
    }
    for frequency in builder.custom_frequencies.values():
        if frequency.label_attribute is not None:
            groups[frequency.label_attribute] = builder.custom_attributes[frequency.label_attribute].values
    groups["mujoco:equality_constraint_label"] = builder.custom_attributes["mujoco:equality_constraint_label"].values
    return groups


@contextmanager
def _rebase_builder_paths(
    builder: ModelBuilder, source: str, destination: str, prefix: str, reference_paths: Sequence[tuple[str, str]]
) -> Iterator[None]:
    """Borrow one prototype with instance-relative labels, restoring its names after the copy."""
    builder._resolve_custom_frequency_articulation_owners()
    labels = _label_groups(builder)
    world_frequencies = {attr.frequency for attr in builder.custom_attributes.values() if attr.references == "world"}
    paths = [
        attr
        for attr in builder.custom_attributes.values()
        if attr.dtype is str
        and (
            attr.frequency in world_frequencies
            or (attr.namespace == "isaaclab" and attr.name == "visual_material_path")
        )
        and not any(attr.values is values for values in labels.values())
    ]
    original_labels = {name: list(values) for name, values in labels.items()}
    original_paths = [attr.values.copy() for attr in paths]
    source, destination = source.rstrip("/") or "/", destination.rstrip("/") or "/"
    reference_destinations = dict(reference_paths)
    reference_destinations[source] = destination
    reference_sources = sorted(reference_destinations, key=len, reverse=True)
    try:
        for values in labels.values():
            for index, label in enumerate(values):
                if not isinstance(label, str) or not label.startswith("/"):
                    continue
                suffix = clone_path.relative_to(label, source)
                if suffix is None:
                    suffix = label[len(source) :] if label.startswith(source + "_") else None
                if suffix is None:
                    raise ValueError(f"Newton label {label!r} is outside clone source {source!r}.")
                values[index] = (destination[len(prefix) :] + suffix).lstrip("/") if prefix else destination + suffix
                if not values[index]:
                    raise ValueError(f"Newton label {label!r} cannot be prefixed by destination {destination!r}.")
        for attr in paths:
            for index in attr.values if isinstance(attr.values, dict) else range(len(attr.values)):
                value = attr.values[index]
                if isinstance(value, str):
                    for reference_source in reference_sources:
                        if clone_path.under(value, reference_source):
                            attr.values[index] = clone_path.rebase(
                                value, reference_source, reference_destinations[reference_source]
                            )
                            break
        yield
    finally:
        for name, values in original_labels.items():
            labels[name][:] = values
        for attr, values in zip(paths, original_paths, strict=True):
            attr.values = values


def replicate_builder_mapping(
    builder: ModelBuilder,
    plan: ClonePlan,
    instances: Sequence[tuple[int, str | None, str, np.ndarray]],
    positions: np.ndarray,
    quaternions: np.ndarray,
    source_builders: dict[str, ModelBuilder],
    *,
    env_template: str,
    env_ids: np.ndarray,
    reference_instances: Sequence[tuple[int, str | None, str, np.ndarray]] = (),
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = (),
    source_builder_added: Callable[[str, int, ModelBuilder, Sequence[float]], None] | None = None,
) -> tuple[dict[str, list[list[int]]], list[wp.transform], list[tuple[str, int]]]:
    """Compose each declared world prototype once, then replicate its selected worlds.

    Additional ``reference_instances`` supply path bindings, not geometry to import.
    """
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = len(plan.world_prototype_layout)
    xforms_np = np.concatenate((positions, quaternions), axis=1).astype(np.float32, copy=False)
    world_xforms = [wp.transform(*xform) for xform in xforms_np]
    local_site_map = {
        label: [indices.copy() for _ in range(num_worlds)]
        for label, indices in source_site_indices.get(id(builder), {}).items()
    }
    source_inverse = {}
    for _, source, _, world_ids in instances:
        if len(world_ids) and source not in source_inverse:
            source_inverse[source] = (
                np.asarray(wp.transform(), dtype=np.float32)
                if world_ids[0] == -1 or clone_path.match(source, env_template) is None
                else _invert_xform(xforms_np[world_ids[0]])
            )
    world_builders = {}
    prototype_ids, first_world_ids = np.unique(plan.world_prototype_layout, return_index=True)
    for world_prototype_id, first_world in zip((-1, *prototype_ids), (-1, *first_world_ids), strict=True):
        start, end = plan.world_prototype_starts[world_prototype_id + 1 : world_prototype_id + 3]
        members = plan.world_prototypes[start:end]
        # Native names belong to the importer. The plan supplies composition and cardinality.
        by_asset = {}
        for asset_prototype_id, source, destination, targets in instances:
            if first_world in targets:
                by_asset.setdefault(asset_prototype_id, []).append((source, destination))
        by_asset = {index: iter(targets) for index, targets in by_asset.items()}
        components = [next(by_asset[int(index)]) for index in members if int(index) in by_asset]
        reference_paths = components + [
            (source, destination) for _, source, destination, targets in reference_instances if first_world in targets
        ]
        prototype = ModelBuilder(up_axis=builder.up_axis)
        sites, particle_offsets = {}, []
        for source, destination in components:
            asset = source_builders[source]
            particle_offsets.append(prototype.particle_count)
            for label, indices in source_site_indices.get(id(asset), {}).items():
                sites.setdefault(label, []).extend(prototype.shape_count + index for index in indices)
            with _rebase_builder_paths(
                asset, source, destination, "" if world_prototype_id == -1 else env_template, reference_paths
            ):
                prototype.add_builder(asset, xform=source_inverse[source])
        if world_prototype_id == -1:
            base_shape, base_particle = builder.shape_count, builder.particle_count
            builder.add_builder(prototype)
            for label, indices in sites.items():
                local_site_map[label] = np.tile(base_shape + np.asarray(indices), (num_worlds, 1)).tolist()
            if source_builder_added is not None:
                for (source, _), offset in zip(components, particle_offsets, strict=True):
                    source_builder_added(
                        source, base_particle + offset, source_builders[source], source_inverse[source]
                    )
            continue
        for label, xform in env_root_sites.items():
            sites.setdefault(label, []).append(prototype.add_site(body=-1, xform=xform, label=label))
        world_builders[world_prototype_id] = prototype, sites, components, particle_offsets

    # Preserve destination order, batching each contiguous run of an identical world prototype.
    boundaries = np.r_[0, np.flatnonzero(np.diff(plan.world_prototype_layout)) + 1, num_worlds]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start == stop:
            continue
        prototype, sites, components, particle_offsets = world_builders[int(plan.world_prototype_layout[start])]
        base_shape, base_particle = builder.shape_count, builder.particle_count
        shape_offsets = base_shape + np.arange(stop - start) * prototype.shape_count
        particle_bases = base_particle + np.arange(stop - start) * prototype.particle_count
        if per_world_builder_hooks:
            for world in range(start, stop):
                shape_offsets[world - start], particle_bases[world - start] = (
                    builder.shape_count,
                    builder.particle_count,
                )
                builder.begin_world()
                builder.add_builder(prototype, xform=xforms_np[world], label_prefix=env_template.format(env_ids[world]))
                labels = _label_groups(builder)
                label_starts = {name: len(values) for name, values in labels.items()}
                for hook in per_world_builder_hooks:
                    hook(builder, world, positions[world].copy(), quaternions[world].copy())
                for name, values in labels.items():
                    for index in range(label_starts[name], len(values)):
                        for source, destination in components:
                            values[index] = clone_path.rebase(values[index], source, destination.format(env_ids[world]))
                builder.end_world()
        else:
            builder.replicate(
                prototype,
                int(stop - start),
                xforms=xforms_np[start:stop],
                label_prefixes=[env_template.format(env_ids[world]) for world in range(start, stop)],
            )
        for label, indices in sites.items():
            per_world = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])
            per_world[start:stop] = (shape_offsets[:, None] + np.asarray(indices)).tolist()
        if source_builder_added is not None:
            for (source, _), offset in zip(components, particle_offsets, strict=True):
                transforms = _compose_world_xforms(
                    positions[start:stop], quaternions[start:stop], source_inverse[source]
                )
                for index, xform in enumerate(transforms):
                    source_builder_added(source, int(particle_bases[index]) + offset, source_builders[source], xform)
    worlds_by_frequency = {
        attr.frequency: attr.values for attr in builder.custom_attributes.values() if attr.references == "world"
    }
    for attr in builder.custom_attributes.values():
        worlds = (
            builder.shape_world
            if attr.namespace == "isaaclab" and attr.name == "visual_material_path"
            else worlds_by_frequency.get(attr.frequency)
        )
        if attr.dtype is str and worlds is not None:
            for index in attr.values if isinstance(attr.values, dict) else range(len(attr.values)):
                value = attr.values[index]
                if isinstance(value, str) and "{}" in value and worlds[index] is not None:
                    attr.values[index] = value.format(env_ids[worlds[index]])
    return local_site_map, world_xforms, [(label, index) for index, label in enumerate(builder.body_label)]
