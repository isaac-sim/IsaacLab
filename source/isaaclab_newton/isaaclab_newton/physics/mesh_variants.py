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
from newton import BodyFlags, GeoType, JointType, Mesh, Model, ModelBuilder, ShapeFlags
from newton._src.geometry.inertia import verify_and_correct_inertia

from isaaclab.utils.warp import ProxyArray
from isaaclab.utils.warp.index_kernel import IndexKernelDispatcher


@dataclass(frozen=True, slots=True)
class _MeshVariantDefinition:
    """Native clone rows retained by the builder for each subsequent model allocation."""

    shape_indices: np.ndarray
    source_shape_indices: np.ndarray
    body_properties: np.ndarray
    root_joint_type: int
    body_flags: int
    initial_variant_ids: np.ndarray
    render_model: Model | None
    source_meshes: tuple[tuple[Mesh, ...], ...]


@wp.struct
class _VariantBody:
    mass: wp.float32
    inv_mass: wp.float32
    com: wp.vec3f
    inertia: wp.mat33f
    inv_inertia: wp.mat33f


def _mesh_variant_body_properties(builders: tuple[ModelBuilder, ...], inertia_offset: float) -> np.ndarray | None:
    """Prepare compatible source inertia, or leave an unsupported catalog as ordinary multi-asset spawning."""
    if not math.isfinite(inertia_offset) or inertia_offset < 0.0:
        raise ValueError("mesh_variant_inertia_diagonal_offset must be finite and non-negative.")
    properties = np.zeros(len(builders), dtype=_VariantBody.numpy_dtype())
    source_contract = None
    for variant, builder in enumerate(builders):
        if builder.body_count != 1 or builder.joint_count > 1:
            return None
        joint_type = builder.joint_type[0] if builder.joint_count else -1
        if builder.joint_count and (
            joint_type not in (JointType.FREE, JointType.FIXED)
            or builder.joint_parent[0] != -1
            or builder.joint_child[0] != 0
        ):
            return None
        slots = [
            shape
            for shape in builder.body_shapes[0]
            if int(builder.shape_flags[shape]) & int(ShapeFlags.COLLIDE_SHAPES)
        ]
        if not slots or any(
            builder.shape_type[shape] not in (GeoType.MESH, GeoType.CONVEX_MESH)
            or not isinstance(builder.shape_source[shape], Mesh)
            for shape in slots
        ):
            return None
        condim = builder.custom_attributes.get("mujoco:condim")
        colliders = tuple(
            (
                builder.shape_type[shape],
                builder.shape_collision_group[shape],
                int(builder.shape_flags[shape]) & ~int(ShapeFlags.VISIBLE),
                3 if condim is None else (condim.values or {}).get(shape, condim.default),
            )
            for shape in slots
        )
        contract = joint_type, builder.body_flags[0], colliders
        if source_contract is not None and contract != source_contract:
            return None
        source_contract = contract
        dynamic = joint_type == JointType.FREE and not builder.body_flags[0] & int(BodyFlags.KINEMATIC)
        mass, inertia, _ = verify_and_correct_inertia(
            builder.body_mass[0],
            builder.body_inertia[0],
            builder.balance_inertia,
            builder.bound_mass,
            builder.bound_inertia,
            builder.body_label[0],
        )
        inertia = np.asarray(inertia, dtype=np.float64).reshape(3, 3).copy()
        inertia.flat[::4] += inertia_offset
        if not np.isfinite(mass) or not np.isfinite(inertia).all():
            return None
        if dynamic and (mass <= 0.0 or np.any(np.linalg.eigvalsh(inertia) <= 0.0)):
            return None
        properties[variant]["mass"] = mass
        properties[variant]["inv_mass"] = 1.0 / mass if dynamic else 0.0
        properties[variant]["com"] = builder.body_com[0]
        properties[variant]["inertia"] = inertia
        properties[variant]["inv_inertia"] = np.linalg.inv(inertia) if dynamic else np.zeros((3, 3))
    return properties


@wp.kernel(enable_backward=False)
def _select_mesh_variants(
    selections: wp.array(dtype=Any),
    worlds: wp.array(dtype=Any),
    targets: wp.array2d(dtype=wp.int32),
    sources: wp.array2d(dtype=wp.int32),
    variants: wp.array(dtype=_VariantBody),
    selected: wp.array(dtype=wp.int32),
    shape_bodies: wp.array(dtype=wp.int32),
    scales: wp.array(dtype=wp.vec3f),
    transforms: wp.array(dtype=wp.transformf),
    radii: wp.array(dtype=wp.float32),
    margins: wp.array(dtype=wp.float32),
    gaps: wp.array(dtype=wp.float32),
    solids: wp.array(dtype=wp.bool),
    mesh_ptrs: wp.array(dtype=wp.uint64),
    mesh_properties: wp.array(dtype=wp.int32),
    support_data: wp.array(dtype=wp.vec4i),
    sdf_indices: wp.array(dtype=wp.int32),
    edge_ranges: wp.array(dtype=wp.vec2i),
    aabb_lowers: wp.array(dtype=wp.vec3f),
    aabb_uppers: wp.array(dtype=wp.vec3f),
    bvh_bounds: wp.array2d(dtype=wp.vec3f),
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
    margins[target] = margins[source]
    gaps[target] = gaps[source]
    solids[target] = solids[source]
    mesh_ptrs[target] = mesh_ptrs[source]
    if mesh_properties:
        mesh_properties[target] = mesh_properties[source]
    if support_data:
        support_data[target] = support_data[source]
    if sdf_indices:
        sdf_indices[target] = sdf_indices[source]
    if edge_ranges:
        edge_ranges[target] = edge_ranges[source]
    if aabb_lowers:
        aabb_lowers[target] = aabb_lowers[source]
        aabb_uppers[target] = aabb_uppers[source]
    if bvh_bounds:
        bvh_bounds[target, 0] = bvh_bounds[source, 0]
        bvh_bounds[target, 1] = bvh_bounds[source, 1]
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
        selected[world] = wp.int32(variant)


_SELECT_MESH_VARIANTS = IndexKernelDispatcher(_select_mesh_variants, ("selections", "worlds"))


class MeshVariants:
    """Own validated mesh resources and selected IDs for one finalized Newton model.

    One fixed-root or free rigid body is retained in each world. Variants must have the same
    root-joint type, ordered collider types, and collision filtering. Selection changes
    geometry, collision margins and gaps, and inertia;
    per-environment material randomization is retained. The caller publishes model changes to
    the solver and resets contact history before stepping with Newton's external contacts.

    Args:
        model: Finalized Newton model retaining inactive mesh resources.
        definition: Source resources and target rows recorded during native cloning.
    """

    def __init__(self, model: Model, definition: _MeshVariantDefinition) -> None:
        targets, sources = definition.shape_indices, definition.source_shape_indices
        properties, initial_variant_ids = definition.body_properties, definition.initial_variant_ids
        if targets.ndim != 2 or targets.shape[0] != model.world_count or not targets.shape[1]:
            raise ValueError("Mesh variants require non-empty collision-shape rows for every world.")
        if sources.shape != (len(properties), targets.shape[1]):
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
        self.source_shape_indices = wp.array(sources, dtype=wp.int32, device=model.device)
        wp.indexedarray(model.shape_flags, self.source_shape_indices.flatten()).zero_()
        types, flags = model.shape_type.numpy(), model.shape_flags.numpy()
        if not np.all(flags[targets] & int(ShapeFlags.COLLIDE_SHAPES)):
            raise ValueError("Mesh variants require colliding target shapes.")
        if not np.all(np.isin(types[sources], (GeoType.MESH, GeoType.CONVEX_MESH))):
            raise ValueError("Mesh variants require mesh-only collision geometry.")
        if not np.all(types[sources] == types[targets[0]]) or not np.all(types[targets] == types[targets[0]]):
            raise ValueError("Mesh variants must preserve ordered collision-shape types.")
        for row, meshes in zip(sources, definition.source_meshes, strict=True):
            if any(hash(model.shape_source[index]) != hash(mesh) for index, mesh in zip(row, meshes, strict=True)):
                raise ValueError("Mesh variant resources do not match their imported mesh sources.")
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
        root_joint_types = np.full(len(body_ids), -1, dtype=np.int32)
        root_joint_types[present] = joint_types[joints[present]]
        if np.any(root_joint_types != definition.root_joint_type):
            raise ValueError("Mesh variants must preserve the root-joint type across sources and worlds.")
        body_flags = model.body_flags.numpy()[body_ids]
        if np.any(body_flags != definition.body_flags):
            raise ValueError("Mesh variants must preserve kinematic body flags across sources and worlds.")
        if (
            initial_variant_ids.shape != (model.world_count,)
            or np.any(initial_variant_ids < 0)
            or np.any(initial_variant_ids >= len(properties))
        ):
            raise ValueError("Mesh variants have invalid initial selections.")
        self.shape_indices = wp.array(targets, dtype=wp.int32, device=model.device)
        self.body_properties = wp.array(properties, dtype=_VariantBody, device=model.device)
        self.variant_ids = ProxyArray(wp.array(initial_variant_ids, dtype=wp.int32, device=model.device))
        self.render_model = definition.render_model
        self.body_indices = tuple(body_ids.tolist())


def _prepare_mesh_variant(
    builder: ModelBuilder,
    sources: tuple[ModelBuilder, ...],
    targets: np.ndarray,
    properties: np.ndarray,
    initial_variant_ids: np.ndarray,
    render_model: Model | None,
) -> _MeshVariantDefinition:
    """Append one catalog's inactive collision resources after its resolved native target rows."""
    resources = []
    for source in sources:
        shapes = []
        for source_shape in source.body_shapes[0]:
            flags = int(source.shape_flags[source_shape])
            if not flags & int(ShapeFlags.COLLIDE_SHAPES):
                continue
            shape_cfg = ModelBuilder.ShapeConfig(
                density=0.0,
                margin=source.shape_margin[source_shape],
                gap=source.shape_gap[source_shape],
                is_solid=source.shape_is_solid[source_shape],
                collision_group=source.shape_collision_group[source_shape],
                collision_filter_parent=False,
                has_shape_collision=True,
                has_particle_collision=bool(flags & int(ShapeFlags.COLLIDE_PARTICLES)),
                is_visible=False,
                is_hydroelastic=bool(flags & int(ShapeFlags.HYDROELASTIC)),
            )
            shape = builder.add_shape(
                body=-1,
                type=source.shape_type[source_shape],
                xform=source.shape_transform[source_shape],
                cfg=shape_cfg,
                scale=source.shape_scale[source_shape],
                src=source.shape_source[source_shape],
                is_static=True,
                label=f"mesh_variant/{builder.shape_count}",
            )
            builder.shape_sdf_narrow_band_range[shape] = source.shape_sdf_narrow_band_range[source_shape]
            builder.shape_sdf_target_voxel_size[shape] = source.shape_sdf_target_voxel_size[source_shape]
            builder.shape_sdf_max_resolution[shape] = source.shape_sdf_max_resolution[source_shape]
            builder.shape_force_sdf[shape] = source.shape_force_sdf[source_shape]
            builder.shape_sdf_texture_format[shape] = source.shape_sdf_texture_format[source_shape]
            builder.shape_sdf_padding[shape] = source.shape_sdf_padding[source_shape]
            shapes.append(shape)
        resources.append(shapes)
    return _MeshVariantDefinition(
        shape_indices=targets,
        source_shape_indices=np.asarray(resources, dtype=np.int32),
        body_properties=properties,
        root_joint_type=sources[0].joint_type[0] if sources[0].joint_count else -1,
        body_flags=sources[0].body_flags[0],
        initial_variant_ids=initial_variant_ids,
        render_model=render_model,
        source_meshes=tuple(
            tuple(
                source.shape_source[shape]
                for shape in source.body_shapes[0]
                if source.shape_flags[shape] & ShapeFlags.COLLIDE_SHAPES
            )
            for source in sources
        ),
    )
