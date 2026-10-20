# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared Newton mesh-variant rendering for desktop and web visualizers."""

from __future__ import annotations

from collections.abc import Callable, Collection
from dataclasses import dataclass

import numpy as np
import warp as wp
from isaaclab_newton.physics.mesh_variants import MeshVariants
from newton import GeoType, Model
from newton.viewer import ViewerBase


def populate_shapes_without_mesh_variants(
    viewer: ViewerBase, shape_indices: Collection[int], populate: Callable[[], None]
) -> None:
    """Populate viewer batches without stale model shapes, preserving the physics model's flags."""
    if not shape_indices:
        populate()
        return
    model = viewer.model
    if viewer._shape_sdf_index_host is not None:
        # Lazy collision overlays must not recreate the initial variant's SDF mesh.
        viewer._shape_sdf_index_host = viewer._shape_sdf_index_host.copy()
        viewer._shape_sdf_index_host[list(shape_indices)] = -1
    model_flags = model.shape_flags
    render_flags = model_flags.numpy().copy()
    render_flags[list(shape_indices)] = 0
    model.shape_flags = wp.array(render_flags, dtype=wp.int32, device=model_flags.device)
    try:
        populate()
    finally:
        model.shape_flags = model_flags


class MeshVariantRenderer:
    """Bind viewer geometry and render reset-selectable shapes outside the physics model.

    Collision overlays show each selected variant's triangle mesh, not a cached SDF isosurface.
    """

    def __init__(
        self,
        viewer: ViewerBase,
        model: Model,
        banks: Collection[MeshVariants],
        visible_worlds: list[int] | None,
    ) -> None:
        banks = tuple(bank for bank in banks if bank.visual_variants is not None)
        viewer._mesh_variant_model_shapes = tuple(shape for bank in banks for shape in bank.model_shape_indices)
        viewer.set_model(model)
        worlds = np.arange(model.world_count, dtype=np.int32) if visible_worlds is None else visible_worlds
        worlds = wp.array(worlds, dtype=wp.int32, device=model.device)
        instance_count = len(worlds)
        groups = []

        for set_index, bank in enumerate(banks):
            shapes = tuple(
                (variant, collision, shape)
                for collision, variants in (
                    (False, bank.visual_variants),
                    (True, bank.collision_variants),
                )
                for variant, variant_shapes in enumerate(variants)
                for shape in variant_shapes
            )
            if not shapes or instance_count == 0:
                continue

            local_xforms = wp.array(
                [shape.transform for _, _, shape in shapes], dtype=wp.transformf, device=model.device
            )
            local_scales = []
            shape_variants = wp.array([variant for variant, _, _ in shapes], dtype=wp.int32, device=model.device)
            world_xforms = wp.empty(len(shapes) * instance_count, dtype=wp.transformf, device=model.device)
            world_scales = wp.empty(len(shapes) * instance_count, dtype=wp.vec3, device=model.device)
            batches = []

            for shape_index, (_, collision, shape) in enumerate(shapes):
                is_mesh = shape.geometry_type in (int(GeoType.MESH), int(GeoType.CONVEX_MESH))
                mirror = is_mesh and np.prod(shape.scale) < 0.0
                geometry_scale = (1.0, 1.0, 1.0) if is_mesh else shape.scale
                mesh = viewer._populate_geometry(
                    shape.geometry_type,
                    geometry_scale,
                    shape.thickness,
                    shape.is_solid,
                    geo_src=shape.source,
                    mirror=mirror,
                )
                local_scales.append(shape.scale if is_mesh else (1.0, 1.0, 1.0))

                if collision:
                    color = viewer._collision_color_map(shape_index)
                    roughness, metallic, textured = 0.3, 0.0, False
                else:
                    color = shape.color
                    roughness = shape.source.roughness if is_mesh else None
                    metallic = shape.source.metallic if is_mesh else None
                    textured = is_mesh and shape.source.uvs is not None and shape.source.texture is not None
                start = shape_index * instance_count
                stop = start + instance_count
                batches.append(
                    _MeshVariantBatch(
                        name=f"/mesh_variants/set_{set_index}/shape_{shape_index}",
                        mesh=mesh,
                        collision=collision,
                        xforms=world_xforms[start:stop],
                        scales=world_scales[start:stop],
                        colors=wp.full(instance_count, wp.vec3(*color), dtype=wp.vec3, device=model.device),
                        materials=wp.full(
                            instance_count,
                            wp.vec4(
                                0.5 if roughness is None else float(roughness),
                                0.0 if metallic is None else float(metallic),
                                0.0,
                                float(textured),
                            ),
                            dtype=wp.vec4,
                            device=model.device,
                        ),
                    )
                )

            groups.append(
                _MeshVariantGroup(
                    kernel_inputs=(
                        bank.body_indices,
                        bank.variant_ids,
                        worlds,
                        local_xforms,
                        wp.array(local_scales, dtype=wp.vec3, device=model.device),
                        shape_variants,
                        instance_count,
                    ),
                    kernel_outputs=(world_xforms, world_scales),
                    batches=tuple(batches),
                )
            )

        self._groups = tuple(groups)
        self._last_visibility: tuple[bool, bool, bool] | None = None

    def render(self, viewer: ViewerBase, body_q: wp.array(dtype=wp.transformf)) -> None:
        """Update and draw the selected variant for each visible world."""
        layer_visible = not viewer._layer_force_hidden()
        visibility = (layer_visible, viewer.show_visual, viewer.show_collision)
        appearance_changed = visibility != self._last_visibility
        for group in self._groups:
            if layer_visible and (viewer.show_visual or viewer.show_collision):
                wp.launch(
                    _update_mesh_variant_instances,
                    dim=len(group.kernel_outputs[0]),
                    inputs=[
                        body_q,
                        *group.kernel_inputs,
                        viewer._visible_worlds_mask,
                        viewer.world_offsets,
                        viewer.layer.xform,
                    ],
                    outputs=group.kernel_outputs,
                    device=viewer.device,
                    record_tape=False,
                )
            for batch in group.batches:
                visible = layer_visible and (viewer.show_collision if batch.collision else viewer.show_visual)
                viewer.log_instances(
                    batch.name,
                    batch.mesh,
                    batch.xforms,
                    batch.scales,
                    batch.colors if appearance_changed else None,
                    batch.materials if appearance_changed else None,
                    hidden=not visible,
                )
        self._last_visibility = visibility


@wp.kernel(enable_backward=False)
def _update_mesh_variant_instances(
    body_q: wp.array(dtype=wp.transformf),
    body_indices: wp.array(dtype=wp.int32),
    selected_variants: wp.array(dtype=wp.int32),
    visible_worlds: wp.array(dtype=wp.int32),
    local_xforms: wp.array(dtype=wp.transformf),
    local_scales: wp.array(dtype=wp.vec3f),
    shape_variants: wp.array(dtype=wp.int32),
    instance_count: int,
    visible_worlds_mask: wp.array(dtype=wp.int32),
    world_offsets: wp.array(dtype=wp.vec3f),
    layer_xform: wp.transformf,
    world_xforms: wp.array(dtype=wp.transformf),
    world_scales: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    shape = tid // instance_count
    instance = tid - shape * instance_count
    world = visible_worlds[instance]

    xform = wp.transform_multiply(body_q[body_indices[world]], local_xforms[shape])
    xform = wp.transform(xform.p + world_offsets[world], xform.q)
    world_xforms[tid] = wp.transform_multiply(layer_xform, xform)

    scale = wp.vec3(0.0)
    visible = True
    if visible_worlds_mask:
        visible = visible_worlds_mask[world] != 0
    if visible and selected_variants[world] == shape_variants[shape]:
        scale = local_scales[shape]
    world_scales[tid] = scale


@dataclass(slots=True)
class _MeshVariantBatch:
    name: str
    mesh: str
    collision: bool
    xforms: wp.array(dtype=wp.transformf)
    scales: wp.array(dtype=wp.vec3f)
    colors: wp.array(dtype=wp.vec3f)
    materials: wp.array(dtype=wp.vec4f)


@dataclass(slots=True)
class _MeshVariantGroup:
    kernel_inputs: tuple
    kernel_outputs: tuple[wp.array(dtype=wp.transformf), wp.array(dtype=wp.vec3f)]
    batches: tuple[_MeshVariantBatch, ...]
