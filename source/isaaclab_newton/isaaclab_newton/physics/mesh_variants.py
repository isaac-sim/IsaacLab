# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-selectable rigid-body meshes owned by the Newton backend."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp
from newton import BodyFlags, Gaussian, GeoType, Heightfield, JointType, Mesh, Model, ModelBuilder, ShapeFlags
from newton._src.geometry.inertia import verify_and_correct_inertia

from isaaclab.cloner import ClonePlan
from isaaclab.cloner import path as cloner_path


@dataclass(frozen=True, slots=True)
class MeshVariantRenderShape:
    """One shape stored once in a mesh variant render bank."""

    geometry_type: int
    source: Mesh | Heightfield | Gaussian | None
    transform: tuple[float, ...]
    scale: tuple[float, float, float]
    thickness: float
    is_solid: bool
    color: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MeshVariantRenderSet:
    """Render prototypes and live bodies belonging to one mesh selection."""

    name: str
    source_paths: tuple[str, ...]
    body_indices: wp.array(dtype=wp.int32)
    model_shape_indices: tuple[int, ...]
    visual_variants: tuple[tuple[MeshVariantRenderShape, ...], ...]
    collision_variants: tuple[tuple[MeshVariantRenderShape, ...], ...]


@wp.struct
class _VariantBody:
    mass: wp.float32
    inv_mass: wp.float32
    com: wp.vec3f
    inertia: wp.mat33f
    inv_inertia: wp.mat33f


@wp.kernel(enable_backward=False)
def _select_mesh_variants(
    selections: wp.array(dtype=wp.int32),
    worlds: wp.array(dtype=wp.int32),
    targets: wp.array2d(dtype=wp.int32),
    sources: wp.array2d(dtype=wp.int32),
    variants: wp.array(dtype=_VariantBody),
    selected: wp.array(dtype=wp.int32),
    shape_bodies: wp.array(dtype=wp.int32),
    scales: wp.array(dtype=wp.vec3f),
    transforms: wp.array(dtype=wp.transformf),
    radii: wp.array(dtype=wp.float32),
    mesh_ptrs: wp.array(dtype=wp.uint64),
    mesh_properties: wp.array(dtype=wp.int32),
    sdf_indices: wp.array(dtype=wp.int32),
    edge_ranges: wp.array(dtype=wp.vec2i),
    aabb_lowers: wp.array(dtype=wp.vec3f),
    aabb_uppers: wp.array(dtype=wp.vec3f),
    voxel_resolutions: wp.array(dtype=wp.vec3i),
    masses: wp.array(dtype=wp.float32),
    inv_masses: wp.array(dtype=wp.float32),
    coms: wp.array(dtype=wp.vec3f),
    inertias: wp.array(dtype=wp.mat33f),
    inv_inertias: wp.array(dtype=wp.mat33f),
):
    selection, slot = wp.tid()
    world = worlds[selection]
    variant = selections[selection]
    if world < 0 or world >= targets.shape[0] or variant < 0 or variant >= sources.shape[0]:
        return
    target = targets[world, slot]
    source = sources[variant, slot]
    scales[target] = scales[source]
    transforms[target] = transforms[source]
    radii[target] = radii[source]
    mesh_ptrs[target] = mesh_ptrs[source]
    if mesh_properties:
        mesh_properties[target] = mesh_properties[source]
    if sdf_indices:
        sdf_indices[target] = sdf_indices[source]
    if edge_ranges:
        edge_ranges[target] = edge_ranges[source]
    if aabb_lowers:
        aabb_lowers[target] = aabb_lowers[source]
        aabb_uppers[target] = aabb_uppers[source]
    if voxel_resolutions:
        voxel_resolutions[target] = voxel_resolutions[source]
    if slot == 0:
        body = shape_bodies[target]
        properties = variants[variant]
        masses[body] = properties.mass
        inv_masses[body] = properties.inv_mass
        coms[body] = properties.com
        inertias[body] = properties.inertia
        inv_inertias[body] = properties.inv_inertia
        selected[world] = variant


class MeshVariants:
    """Select Newton mesh resources without changing solver topology.

    One fixed-root or free rigid body is retained in each world. Variants must have the same
    ordered collider types and collision filtering. Selection changes geometry and inertia;
    per-environment material randomization is retained. The caller publishes model changes to
    the solver and resets contact history before stepping with Newton's external contacts.

    Args:
        model: Finalized Newton model retaining inactive mesh resources.
        shape_indices: Target colliders per world, shape ``(world_count, collider_count)``.
        source_shape_indices: Inactive colliders per variant, shape ``(variant_count, collider_count)``.
        variant_builders: Imported single-body sources in selection order.
        initial_variant_ids: Initial selection per world, shape ``(world_count,)``.
        inertia_diagonal_offset: Diagonal inertia added to every candidate [kg·m²].
    """

    def __init__(
        self,
        model: Model,
        shape_indices: np.ndarray,
        source_shape_indices: np.ndarray,
        variant_builders: list[ModelBuilder],
        initial_variant_ids: np.ndarray,
        inertia_diagonal_offset: float = 0.0,
    ) -> None:
        if not math.isfinite(inertia_diagonal_offset) or inertia_diagonal_offset < 0.0:
            raise ValueError("inertia_diagonal_offset must be finite and non-negative.")
        targets = np.asarray(shape_indices, dtype=np.int32)
        sources = np.asarray(source_shape_indices, dtype=np.int32)
        if targets.ndim != 2 or targets.shape[0] != model.world_count or not targets.shape[1]:
            raise ValueError("Mesh variants require non-empty collision-shape rows for every world.")
        if sources.shape != (len(variant_builders), targets.shape[1]):
            raise ValueError("Mesh variants must preserve collision-shape topology.")
        if (
            np.any(targets < 0)
            or np.any(sources < 0)
            or np.any(targets >= model.shape_count)
            or np.any(sources >= model.shape_count)
        ):
            raise ValueError("Mesh variants reference an invalid shape.")
        if np.intersect1d(targets, sources).size or np.unique(targets).size != targets.size:
            raise ValueError("Mesh variant resources must be separate from uniquely owned target shapes.")
        types, flags = model.shape_type.numpy(), model.shape_flags.numpy()
        if np.any(flags[sources]) or not np.all(flags[targets] & int(ShapeFlags.COLLIDE_SHAPES)):
            raise ValueError("Mesh variants require inactive source resources and colliding targets.")
        if not np.all(np.isin(types[sources], (GeoType.MESH, GeoType.CONVEX_MESH))):
            raise ValueError("Mesh variants require mesh-only collision geometry.")
        if not np.all(types[sources] == types[targets[0]]) or not np.all(types[targets] == types[targets[0]]):
            raise ValueError("Mesh variants must preserve ordered collision-shape types.")
        bodies = model.shape_body.numpy()[targets]
        if np.any(bodies < 0) or not np.all(bodies == bodies[:, :1]):
            raise ValueError("Mesh variants require one rigid body per world.")
        if not np.array_equal(model.body_world.numpy()[bodies[:, 0]], np.arange(model.world_count)):
            raise ValueError("Mesh variants cannot cross world boundaries.")
        parents, children, joint_types = model.joint_parent.numpy(), model.joint_child.numpy(), model.joint_type.numpy()
        body_ids = bodies[:, 0]
        if np.any(np.isin(parents, body_ids)):
            raise ValueError("Mesh variants require root leaf bodies.")
        if np.any(np.bincount(children, minlength=model.body_count)[body_ids] > 1):
            raise ValueError("Mesh variants require a fixed-root or free rigid body.")
        joints_by_body = np.full(model.body_count, -1, dtype=np.int32)
        joints_by_body[children] = np.arange(len(children))
        joints = joints_by_body[body_ids]
        present = joints >= 0
        if np.any(parents[joints[present]] != -1) or not np.all(
            np.isin(joint_types[joints[present]], (JointType.FREE, JointType.FIXED))
        ):
            raise ValueError("Mesh variants require a fixed-root or free rigid body.")
        dynamic = np.zeros(len(body_ids), dtype=bool)
        dynamic[present] = joint_types[joints[present]] == JointType.FREE
        body_flags = model.body_flags.numpy()[body_ids]
        if np.any(body_flags != body_flags[0]):
            raise ValueError("Mesh variants must preserve kinematic body flags across worlds.")
        dynamic &= (body_flags & int(BodyFlags.KINEMATIC)) == 0
        if np.any(dynamic != dynamic[0]):
            raise ValueError("Mesh variants must preserve the root-joint type across worlds.")
        properties = np.zeros(len(variant_builders), dtype=_VariantBody.numpy_dtype())
        collider_contract = None
        for variant, builder in enumerate(variant_builders):
            if builder.body_count != 1:
                raise ValueError("Mesh variants require one rigid body per source.")
            if builder.body_flags[0] != body_flags[0]:
                raise ValueError("Mesh variants must preserve kinematic body flags across sources.")
            slots = [
                shape
                for shape in builder.body_shapes[0]
                if int(builder.shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
            ]
            if len(slots) != sources.shape[1]:
                raise ValueError("Mesh variants must preserve collision-shape topology.")
            condim = builder.custom_attributes.get("mujoco:condim")
            contract = tuple(
                (
                    builder.shape_collision_group[shape],
                    int(builder.shape_flags[shape]) & ~int(ShapeFlags.VISIBLE),
                    3 if condim is None else (condim.values or {}).get(shape, condim.default),
                )
                for shape in slots
            )
            if collider_contract is not None and contract != collider_contract:
                raise ValueError("Mesh variants must preserve collision filtering and contact dimensionality.")
            collider_contract = contract
            for slot, shape in enumerate(slots):
                source = sources[variant, slot]
                if not isinstance(builder.shape_source[shape], Mesh) or hash(model.shape_source[source]) != hash(
                    builder.shape_source[shape]
                ):
                    raise ValueError("Mesh variant resources do not match their imported mesh sources.")
            mass, inertia, _ = verify_and_correct_inertia(
                builder.body_mass[0],
                builder.body_inertia[0],
                builder.balance_inertia,
                builder.bound_mass,
                builder.bound_inertia,
                builder.body_label[0],
            )
            inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3).copy()
            inertia.flat[::4] += inertia_diagonal_offset
            if dynamic[0] and (mass <= 0.0 or np.any(np.linalg.eigvalsh(inertia) <= 0.0)):
                raise ValueError("Free mesh variants require positive mass and inertia.")
            properties[variant]["mass"] = mass
            properties[variant]["inv_mass"] = 1.0 / mass if dynamic[0] else 0.0
            properties[variant]["com"] = builder.body_com[0]
            properties[variant]["inertia"] = inertia
            properties[variant]["inv_inertia"] = np.linalg.inv(inertia) if dynamic[0] else np.zeros((3, 3))
        if (
            initial_variant_ids.shape != (model.world_count,)
            or np.any(initial_variant_ids < 0)
            or np.any(initial_variant_ids >= len(variant_builders))
        ):
            raise ValueError("Mesh variants have invalid initial selections.")
        self._model = model
        self._shape_indices = wp.array(targets, dtype=wp.int32, device=model.device)
        self._source_shape_indices = wp.array(sources, dtype=wp.int32, device=model.device)
        self._body_properties = wp.array(properties, dtype=_VariantBody, device=model.device)
        self.variant_ids = wp.array(initial_variant_ids, dtype=wp.int32, device=model.device)
        self.select(self.variant_ids, wp.array(np.arange(model.world_count), dtype=wp.int32, device=model.device))

    def select(self, variant_ids: wp.array[wp.int32], world_ids: wp.array[wp.int32]) -> None:
        """Apply selections without allocating or publishing solver changes.

        Args:
            variant_ids: Variant index per selected world, shape ``(len(world_ids),)``.
            world_ids: Valid, unique world indices.
        """
        if variant_ids.shape != world_ids.shape:
            raise ValueError("Mesh variants require one selection per world index.")
        if not world_ids.shape[0]:
            return
        model = self._model
        wp.launch(
            _select_mesh_variants,
            dim=(world_ids.shape[0], self._shape_indices.shape[1]),
            inputs=[
                variant_ids,
                world_ids,
                self._shape_indices,
                self._source_shape_indices,
                self._body_properties,
                self.variant_ids,
                model.shape_body,
                model.shape_scale,
                model.shape_transform,
                model.shape_collision_radius,
                model.shape_source_ptr,
                model._shape_mesh_properties,
                model._shape_sdf_index,
                model.shape_edge_range,
                model.shape_collision_aabb_lower,
                model.shape_collision_aabb_upper,
                model._shape_voxel_resolution,
                model.body_mass,
                model.body_inv_mass,
                model.body_com,
                model.body_inertia,
                model.body_inv_inertia,
            ],
            device=model.device,
        )


@wp.kernel(enable_backward=False)
def _clear_shape_flags(shape_indices: wp.array(dtype=wp.int32), shape_flags: wp.array(dtype=wp.int32)):
    shape_flags[shape_indices[wp.tid()]] = 0


def _render_shapes(builder: ModelBuilder, flag: ShapeFlags) -> tuple[MeshVariantRenderShape, ...]:
    shapes = []
    for shape in builder.body_shapes[0]:
        flags = int(builder.shape_flags[shape])
        if not flags & int(flag) or flags & int(ShapeFlags.SITE):
            continue
        shapes.append(
            MeshVariantRenderShape(
                geometry_type=int(builder.shape_type[shape]),
                source=builder.shape_source[shape],
                transform=tuple(builder.shape_transform[shape]),
                scale=tuple(builder.shape_scale[shape]),
                thickness=float(builder.shape_margin[shape]),
                is_solid=bool(builder.shape_is_solid[shape]),
                color=tuple(builder.shape_color[shape]),
            )
        )
    return tuple(shapes)


def _prepare_mesh_variant_resources(
    registrations: dict[str, Any],
    builder: ModelBuilder,
    source_builders: dict[str, ModelBuilder],
    clone_plan: ClonePlan,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    """Add collision resources without adding parked rigid bodies."""
    if not registrations:
        return {}
    if clone_plan is None:
        raise ValueError("Runtime mesh variants require a heterogeneous ClonePlan.")

    resources = {}
    sources = cloner_path.get_asset_prototype_paths(clone_plan)
    for name, cfg in registrations.items():
        rows = cloner_path.get_asset_prototypes(clone_plan, cfg.prim_path)
        if len(rows) < 2:
            raise ValueError(f"Mesh variants for {name!r} are missing from the ClonePlan.")

        variants = []
        for row in rows:
            source_path = sources[row]
            if source_path is None or source_path not in source_builders:
                raise ValueError(f"Mesh variant prototype {row} for {name!r} was not authored and imported.")
            source = source_builders[source_path]
            if source.body_count != 1:
                raise ValueError(f"Mesh variant source {source_path!r} must contain one rigid body.")

            shapes = []
            for source_shape in source.body_shapes[0]:
                flags = int(source.shape_flags[source_shape])
                if not flags & int(ShapeFlags.COLLIDE_SHAPES):
                    continue
                shape_cfg = ModelBuilder.ShapeConfig(
                    density=0.0,
                    ke=source.shape_material_ke[source_shape],
                    kd=source.shape_material_kd[source_shape],
                    kf=source.shape_material_kf[source_shape],
                    ka=source.shape_material_ka[source_shape],
                    mu=source.shape_material_mu[source_shape],
                    restitution=source.shape_material_restitution[source_shape],
                    mu_torsional=source.shape_material_mu_torsional[source_shape],
                    mu_rolling=source.shape_material_mu_rolling[source_shape],
                    margin=source.shape_margin[source_shape],
                    gap=source.shape_gap[source_shape],
                    is_solid=source.shape_is_solid[source_shape],
                    collision_group=source.shape_collision_group[source_shape],
                    collision_filter_parent=False,
                    has_shape_collision=True,
                    has_particle_collision=bool(flags & int(ShapeFlags.COLLIDE_PARTICLES)),
                    is_visible=False,
                    is_hydroelastic=bool(flags & int(ShapeFlags.HYDROELASTIC)),
                    kh=source.shape_material_kh[source_shape],
                )
                shape = builder.add_shape(
                    body=-1,
                    type=source.shape_type[source_shape],
                    xform=source.shape_transform[source_shape],
                    cfg=shape_cfg,
                    scale=source.shape_scale[source_shape],
                    src=source.shape_source[source_shape],
                    is_static=True,
                    label=f"mesh_variant/{name}/{len(variants)}/{len(shapes)}",
                )
                for field in (
                    "shape_sdf_narrow_band_range",
                    "shape_sdf_target_voxel_size",
                    "shape_sdf_max_resolution",
                    "shape_force_sdf",
                    "shape_sdf_texture_format",
                    "shape_sdf_padding",
                ):
                    getattr(builder, field)[shape] = getattr(source, field)[source_shape]
                shapes.append(shape)
            variants.append(tuple(shapes))

        shape_counts = {len(shapes) for shapes in variants}
        if shape_counts == {0} or len(shape_counts) != 1:
            raise ValueError(f"Mesh variants for {name!r} require fixed, non-empty collision-shape topology.")
        resources[name] = tuple(variants)
    return resources


def _disable_mesh_variant_resources(model: Model, resources: dict[str, tuple[tuple[int, ...], ...]]) -> None:
    shape_indices = [shape for variants in resources.values() for shapes in variants for shape in shapes]
    if not shape_indices:
        return
    indices = wp.array(shape_indices, dtype=wp.int32, device=model.device)
    wp.launch(_clear_shape_flags, dim=len(shape_indices), inputs=[indices, model.shape_flags], device=model.device)


def build_mesh_variants(
    registrations: dict[str, Any],
    model: Model,
    source_builders: dict[str, ModelBuilder],
    clone_plan: ClonePlan,
    visual_source_builders: dict[str, ModelBuilder] | None = None,
    resource_shape_indices: dict[str, tuple[tuple[int, ...], ...]] | None = None,
) -> tuple[dict[str, MeshVariants], tuple[MeshVariantRenderSet, ...]]:
    """Map native prototype memberships to per-world Newton shape selections."""
    if not registrations:
        return {}, ()
    if clone_plan is None:
        raise ValueError("Runtime mesh variants require a heterogeneous ClonePlan.")

    topology = clone_plan.topology
    sources = cloner_path.get_asset_prototype_paths(clone_plan)
    templates, starts = cloner_path.get_world_prototype_asset_templates(clone_plan)
    if len(topology.world_prototype_layout) != model.world_count:
        raise ValueError("ClonePlan and Newton model world counts differ.")

    body_by_label = {label: index for index, label in enumerate(model.body_label)}
    shape_flags = model.shape_flags.numpy()
    variants = {}
    claimed_bodies = set()
    render_sets = []
    for name, cfg in registrations.items():
        rows = cloner_path.get_asset_prototypes(clone_plan, cfg.prim_path)
        if len(rows) < 2:
            raise ValueError(f"Mesh variants for {name!r} are missing from the ClonePlan.")

        variant_by_asset = {int(asset): variant for variant, asset in enumerate(rows)}
        initial_variant_ids, roots = [], []
        for world, prototype in enumerate(topology.world_prototype_layout):
            start, end = starts[prototype + 1 : prototype + 3]
            members = [
                index for index in range(start, end) if int(topology.world_prototypes[index]) in variant_by_asset
            ]
            if len(members) != 1:
                raise ValueError(f"Mesh variants for {name!r} require one active source per world.")
            member = members[0]
            initial_variant_ids.append(variant_by_asset[int(topology.world_prototypes[member])])
            roots.append(templates[member].format(world).rstrip("/") or "/")
        initial_variant_ids = np.asarray(initial_variant_ids, dtype=np.int32)
        source_shapes = None if resource_shape_indices is None else resource_shape_indices.get(name)
        if source_shapes is None or len(source_shapes) != len(rows):
            raise ValueError(f"Mesh variant resources for {name!r} do not match the ClonePlan.")

        builders = []
        body_suffixes = []
        for row in rows:
            source = sources[row]
            if source is None or source not in source_builders:
                raise ValueError(f"Mesh variant prototype {row} for {name!r} was not authored and imported.")
            builder = source_builders[source]
            if builder.body_count != 1:
                raise ValueError(f"Mesh variant source {source!r} must contain one rigid body.")
            body_label = builder.body_label[0]
            if body_label != source and not body_label.startswith(source + "/"):
                raise ValueError(f"Mesh variant body {body_label!r} is outside source {source!r}.")
            builders.append(builder)
            body_suffixes.append(body_label[len(source) :])

        body_indices = []
        shape_rows = []
        for world, variant in enumerate(initial_variant_ids):
            root = roots[world]
            body_label = root + body_suffixes[variant]
            try:
                body = body_by_label[body_label]
            except KeyError as error:
                raise ValueError(f"Mesh variant body {body_label!r} is missing from the Newton model.") from error
            body_indices.append(body)
            shapes = tuple(
                shape for shape in model.body_shapes[body] if int(shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
            )
            shape_rows.append(shapes)

        if claimed_bodies.intersection(body_indices):
            raise ValueError(f"Mesh variants for {name!r} share bodies with another selection.")
        claimed_bodies.update(body_indices)
        variants[name] = MeshVariants(
            model=model,
            shape_indices=np.asarray(shape_rows, dtype=np.int32),
            source_shape_indices=np.asarray(source_shapes, dtype=np.int32),
            variant_builders=builders,
            initial_variant_ids=initial_variant_ids,
            inertia_diagonal_offset=cfg.mesh_variant_inertia_diagonal_offset,
        )
        if visual_source_builders:
            render_sets.append(
                MeshVariantRenderSet(
                    name=name,
                    source_paths=tuple(sources[row] for row in rows),
                    body_indices=wp.array(body_indices, dtype=wp.int32, device=model.device),
                    model_shape_indices=tuple(shape for body in body_indices for shape in model.body_shapes[body]),
                    visual_variants=tuple(
                        _render_shapes(visual_source_builders[sources[row]], ShapeFlags.VISIBLE) for row in rows
                    ),
                    collision_variants=tuple(
                        _render_shapes(builder, ShapeFlags.COLLIDE_SHAPES) for builder in builders
                    ),
                )
            )

    return variants, tuple(render_sets)
