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

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.cloner import ClonePlan
from isaaclab.cloner import path as clone_path
from isaaclab.scene_data.deformable_discovery import DeformableStageEntry, deformable_entry
from isaaclab.sim.utils.newton_model_utils import replace_newton_builder_shape_colors
from isaaclab.utils.string import to_camel_case

from isaaclab_newton.renderers.visual_material import import_builder_visual_material_paths
from isaaclab_newton.sim.spawners.materials import (
    NewtonDeformableBodyMaterialCfg,
    NewtonSurfaceDeformableBodyMaterialCfg,
)


def add_deformable_from_usd(builder: ModelBuilder, stage: Usd.Stage, *, root_path: str) -> DeformableStageEntry:
    """Import one declared deformable's geometry, Newton material, and native element ranges.

    Args:
        builder: Source builder that receives the complete deformable prototype.
        stage: Stage containing the authored geometry and bound physics material.
        root_path: Deformable-body prim path.

    Returns:
        Prototype geometry for SDP's visual-mesh binding.

    Note:
        Replace this importer with native ``add_usd`` when Isaac Lab's authored schemas and
        Newton material attributes have parity (newton-physics/newton#3036 and #3038).
        The native USD importer landed in #3192; that alone does not establish material parity.
        Remove private group recording when the pinned Newton includes #3326's native recording.
    """
    prim = stage.GetPrimAtPath(root_path)
    geometry = deformable_entry(prim)
    if geometry is None:
        raise ValueError(f"No simulation mesh found under deformable {root_path!r}.")
    material = next(
        (
            stage.GetPrimAtPath(path)
            for path in UsdShade.MaterialBindingAPI(prim).GetDirectBindingRel("physics").GetTargets()
            if stage.GetPrimAtPath(path).GetAttribute("newton:density").IsValid()
        ),
        None,
    )
    if material is None:
        raise ValueError(f"Deformable {root_path!r} requires a bound Newton physics material.")
    if geometry.deformable_type == "volume":
        add_mesh = builder.add_soft_mesh
        defaults = NewtonDeformableBodyMaterialCfg()
        names = ("density", "particle_radius", "k_mu", "k_lambda", "k_damp")
    else:
        add_mesh = builder.add_cloth_mesh
        defaults = NewtonSurfaceDeformableBodyMaterialCfg()
        names = ("density", "particle_radius", "tri_ke", "tri_ka", "tri_kd", "edge_ke", "edge_kd")
    material_kwargs = {}
    for name in names:
        attr = material.GetAttribute(f"newton:{to_camel_case(name, to='cC')}")
        material_kwargs[name] = attr.Get() if attr.IsValid() else getattr(defaults, name)

    particle_start, tri_start = builder.particle_count, len(builder.tri_indices)
    edge_start, tet_start = len(builder.edge_indices), len(builder.tet_indices)
    add_mesh(
        vertices=geometry.vertices,
        indices=geometry.indices,
        pos=wp.vec3(*geometry.init_pos),
        rot=wp.quat(*geometry.init_rot),
        scale=1.0,
        vel=wp.vec3(),
        label=root_path,
        **material_kwargs,
    )
    particle_range = (particle_start, builder.particle_count)
    if geometry.deformable_type == "volume":
        builder._record_soft_group(root_path, particle_range, (tet_start, len(builder.tet_indices)))
    else:
        builder._record_cloth_group(
            root_path, particle_range, (tri_start, len(builder.tri_indices)), (edge_start, len(builder.edge_indices))
        )
    return geometry


def _has_visible_non_collision_geometry(stage: Usd.Stage, prim_path: str) -> bool:
    """Return whether a prim hierarchy contains visible geometry without collision."""
    root_prim = stage.GetPrimAtPath(prim_path)
    if not root_prim:
        return False
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Gprim) or prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        imageable = UsdGeom.Imageable(prim)
        is_visible = imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
        if is_visible and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy):
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
    bodies_with_visual_shapes = set()
    for body, flags in zip(builder.shape_body, builder.shape_flags, strict=True):
        is_visible = flags & ShapeFlags.VISIBLE
        is_collider = flags & ShapeFlags.COLLIDE_SHAPES
        if body >= 0 and is_visible and not is_collider:
            bodies_with_visual_shapes.add(body)
    # Resolved on first use: a static parent whose colliders are all filtered out below is
    # never traversed at all.
    static_owners_with_visual_shapes: dict[str, bool] = {}
    for path, index in path_shape_map.items():
        flags, body_index = builder.shape_flags[index], builder.shape_body[index]
        is_collider = bool(flags & ShapeFlags.COLLIDE_SHAPES)
        is_mesh = builder.shape_type[index] == GeoType.MESH
        has_visuals = body_index in bodies_with_visual_shapes
        if not is_collider or is_mesh or has_visuals:
            continue
        if body_index < 0:
            owner_path = _static_collider_owner_path(stage, path)
            if owner_path not in static_owners_with_visual_shapes:
                static_owners_with_visual_shapes[owner_path] = _has_visible_non_collision_geometry(stage, owner_path)
            if static_owners_with_visual_shapes[owner_path]:
                continue
        imageable = UsdGeom.Imageable(stage.GetPrimAtPath(path))
        is_visible = imageable and imageable.ComputeVisibility() != UsdGeom.Tokens.invisible
        if is_visible and imageable.ComputePurpose() in (UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy):
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
    sources = tuple(dict.fromkeys(sources))
    for source in sources:
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
                *(path for path in sources if path != source and clone_path.relative_to(path, source) is not None),
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


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` by xyzw quaternions ``q``, broadcast over the leading axes."""
    axis, angle_w = q[..., :3], q[..., 3:4]
    t = 2.0 * np.cross(axis, v)
    return (v + angle_w * t + np.cross(axis, t)).astype(np.float32)


def _rotate_builder_particles(
    builder: ModelBuilder, source: ModelBuilder, particle_start: int, tet_start: int, xform: np.ndarray
) -> None:
    """Apply particle and rest-frame rotations omitted by Newton's builder composition."""
    # Remove when the pinned Newton fixes newton-physics/newton#4115, including tet rest frames.
    if source.particle_count == 0 or np.array_equal(xform[3:], (0.0, 0.0, 0.0, 1.0)):
        return
    particle_slice = slice(particle_start, particle_start + source.particle_count)
    builder.particle_q[particle_slice] = (
        _quat_rotate(xform[3:], np.asarray(source.particle_q, dtype=np.float32)) + xform[:3]
    ).tolist()
    builder.particle_qd[particle_slice] = _quat_rotate(
        xform[3:], np.asarray(source.particle_qd, dtype=np.float32)
    ).tolist()
    if source.tet_count:
        # Dm^-1 rotates on the right: (R Dm)^-1 = Dm^-1 R^T.
        poses = np.asarray(source.tet_poses, dtype=np.float32).reshape(-1, 3, 3)
        builder.tet_poses[tet_start : tet_start + source.tet_count] = _quat_rotate(xform[3:], poses).tolist()


def _invert_xform(xform: Sequence[float] | np.ndarray) -> np.ndarray:
    """Inverse of a single xyzw transform, assuming a unit quaternion."""
    xform = np.asarray(xform, dtype=np.float32)
    quat_inv = np.array([-xform[3], -xform[4], -xform[5], xform[6]], dtype=np.float32)
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
    paths = []
    for attr in builder.custom_attributes.values():
        if attr.dtype is not str:
            continue
        is_world_path = attr.frequency in world_frequencies
        is_material_path = attr.namespace == "isaaclab" and attr.name == "visual_material_path"
        if (is_world_path or is_material_path) and not any(attr.values is values for values in labels.values()):
            paths.append(attr)
    original_labels = {name: list(values) for name, values in labels.items()}
    original_paths = [attr.values.copy() for attr in paths]
    source, destination = source.rstrip("/") or "/", destination.rstrip("/") or "/"
    destinations = dict(reference_paths)
    destinations[source] = destination
    sources = sorted(destinations, key=len, reverse=True)
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
                    for root in sources:
                        if (suffix := clone_path.relative_to(value, root)) is not None:
                            attr.values[index] = destinations[root].rstrip("/") + suffix or "/"
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
    positions: np.ndarray,
    quaternions: np.ndarray,
    source_builders: dict[str, ModelBuilder],
    *,
    env_ids: np.ndarray,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    per_world_builder_hooks: Sequence[Callable[[ModelBuilder, int, np.ndarray, np.ndarray], None]] = (),
    source_builder_added: Callable[[str, str, int], None] | None = None,
) -> tuple[dict[str, list[list[int]]], list[wp.transform]]:
    """Compose routed source builders once per world prototype, then batch their selected worlds."""
    topology, env_template = plan.topology, plan.env_template
    layout = topology.world_prototype_layout
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}
    num_worlds = len(layout)
    xforms_np = np.concatenate((positions, quaternions), axis=1).astype(np.float32, copy=False)
    world_xforms = [wp.transform(*xform) for xform in xforms_np]
    initial_sites = source_site_indices.get(id(builder), {})
    local_site_map = {label: [indices.copy() for _ in range(num_worlds)] for label, indices in initial_sites.items()}
    source_inverse = {}
    sources = clone_path.get_asset_prototype_paths(plan)
    templates, starts = clone_path.get_world_prototype_asset_templates(plan)
    world_builders = {}
    prototype_ids, first_world_ids = np.unique(layout, return_index=True)
    for prototype_id, first_world in zip((-1, *prototype_ids), (-1, *first_world_ids), strict=True):
        start, end = starts[prototype_id + 1 : prototype_id + 3]
        reference_paths = [(sources[topology.world_prototypes[index]], templates[index]) for index in range(start, end)]
        components = [(source, template) for source, template in reference_paths if source in source_builders]
        prototype = builder if prototype_id == -1 else ModelBuilder(up_axis=builder.up_axis)
        sites, particle_offsets = {}, []
        for source, destination in components:
            # Remove the source world's placement before composing assets into other world prototypes.
            if source not in source_inverse:
                source_inverse[source] = (
                    np.asarray(wp.transform(), dtype=np.float32)
                    if first_world == -1 or clone_path.match(source, env_template) is None
                    else _invert_xform(xforms_np[first_world])
                )
            asset = source_builders[source]
            particle_offsets.append(prototype.particle_count)
            tet_start = prototype.tet_count
            for label, indices in source_site_indices.get(id(asset), {}).items():
                sites.setdefault(label, []).extend(prototype.shape_count + index for index in indices)
            prefix = "" if prototype_id == -1 else env_template
            with _rebase_builder_paths(asset, source, destination, prefix, reference_paths):
                prototype.add_builder(asset, xform=source_inverse[source])
            _rotate_builder_particles(prototype, asset, particle_offsets[-1], tet_start, source_inverse[source])
        if prototype_id == -1:
            for label, indices in sites.items():
                local_site_map[label] = np.tile(indices, (num_worlds, 1)).tolist()
            if source_builder_added is not None:
                for (source, destination), offset in zip(components, particle_offsets, strict=True):
                    source_builder_added(source, destination, offset)
            continue
        for label, xform in env_root_sites.items():
            sites.setdefault(label, []).append(prototype.add_site(body=-1, xform=xform, label=label))
        world_builders[prototype_id] = prototype, sites, components, particle_offsets

    # Preserve destination order, batching each contiguous run of an identical world prototype.
    boundaries = np.r_[0, np.flatnonzero(np.diff(layout)) + 1, num_worlds]
    for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
        if start == stop:
            continue
        prototype, sites, components, particle_offsets = world_builders[layout[start]]
        base_shape, base_particle, base_tet = builder.shape_count, builder.particle_count, builder.tet_count
        shape_offsets = base_shape + np.arange(stop - start) * prototype.shape_count
        particle_bases = base_particle + np.arange(stop - start) * prototype.particle_count
        if per_world_builder_hooks:
            for world in range(start, stop):
                shape_offsets[world - start] = builder.shape_count
                particle_bases[world - start] = builder.particle_count
                tet_start = builder.tet_count
                builder.begin_world()
                builder.add_builder(prototype, xform=xforms_np[world], label_prefix=env_template.format(env_ids[world]))
                offset = particle_bases[world - start]
                _rotate_builder_particles(builder, prototype, offset, tet_start, xforms_np[world])
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
            prefixes = [env_template.format(env_ids[world]) for world in range(start, stop)]
            builder.replicate(prototype, int(stop - start), xforms=xforms_np[start:stop], label_prefixes=prefixes)
            if prototype.particle_count:
                for world in np.flatnonzero(np.any(xforms_np[start:stop, 3:] != (0.0, 0.0, 0.0, 1.0), axis=1)):
                    tet_start = base_tet + world * prototype.tet_count
                    _rotate_builder_particles(
                        builder, prototype, particle_bases[world], tet_start, xforms_np[start + world]
                    )
        for label, indices in sites.items():
            per_world = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])
            per_world[start:stop] = (shape_offsets[:, None] + np.asarray(indices)).tolist()
        if source_builder_added is not None:
            for (source, destination), offset in zip(components, particle_offsets, strict=True):
                for index, world in enumerate(range(start, stop)):
                    target = destination.format(env_ids[world])
                    source_builder_added(source, target, int(particle_bases[index]) + offset)
    # Resolve remaining world-slot placeholders using each native element's owning world.
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
    return local_site_map, world_xforms
