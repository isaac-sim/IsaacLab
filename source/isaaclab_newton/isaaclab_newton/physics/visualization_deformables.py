# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow-model deformable discovery and registry helpers for PhysX/OVPhysX visualization."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import warp as wp
from newton import ModelBuilder

from pxr import Usd

from isaaclab.cloner import ClonePlan
from isaaclab.cloner.path import rebase, under
from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    sort_deformable_entries_for_geometry_sync,
)
from isaaclab.scene_data.deformable_vis_remap import VolumeVisRemap, build_volume_vis_barycentric_remap
from isaaclab.sim.utils.transforms import resolve_prim_pose

logger = logging.getLogger(__name__)


@dataclass
class ShadowDeformableEntity:
    """One shadow-model deformable entity used for SceneData geometry mapping."""

    root_path: str
    sim_particle_offset: int
    sim_particle_count: int
    vis_particle_offset: int
    vis_particle_count: int
    volume_vis_remap: VolumeVisRemap | None = None


@dataclass
class ShadowDeformableRegistryGroup:
    """Registry metadata for one replicated deformable asset."""

    prim_path: str
    sim_mesh_prim_path: str
    vis_mesh_prim_path: str
    deformable_type: str
    particles_per_body: int
    register_usd_vis_point_bindings: bool
    """Whether USD visual-mesh ``points`` can be bound 1:1 to shadow ``particle_q`` slices.

    Used by consumers such as OVRTX that push remapped (or matching) particle
    positions into the authored visual mesh. Newton Warp does not need this;
    it renders the shadow model directly.
    """
    particle_offsets: list[int] = field(default_factory=list)
    entities: list[ShadowDeformableEntity] = field(default_factory=list)


def _needs_volume_vis_remap(entry: DeformableStageEntry) -> bool:
    """Return ``True`` if this volume body needs barycentric sim-to-visual remapping.

    Required when a visual triangle mesh exists on a different prim from the sim mesh.
    """
    return (
        entry.deformable_type == "volume"
        and entry.vis_mesh_path != entry.sim_mesh_path
        and entry.vis_vertices.size > 0
        and entry.vis_indices.size > 0
    )


def _vertices_to_wp_vec3(vertices: np.ndarray | list) -> list[wp.vec3]:
    """Convert baked ``(N, 3)`` vertex arrays to ``wp.vec3`` lists for Newton builders."""
    if isinstance(vertices, np.ndarray):
        return [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vertices]
    return vertices


def _mesh_indices(indices: np.ndarray | list) -> list:
    """Return mesh topology indices as a Python list for Newton builders."""
    if isinstance(indices, np.ndarray):
        return indices.tolist()
    return list(indices)


def _build_volume_vis_remap(entry: DeformableStageEntry, device: str) -> VolumeVisRemap | None:
    """Build device-resident barycentric sim-to-visual remap tables for one volume body.

    Args:
        entry: Discovered deformable with sim tet topology and visual mesh verts.
        device: Warp device for the uploaded remap arrays (typically the sim device).

    Returns:
        A :class:`~isaaclab.scene_data.deformable_vis_remap.VolumeVisRemap` on success,
        or ``None`` when no tet can be assigned (logs a warning).
    """
    remap = build_volume_vis_barycentric_remap(
        entry.vertices,
        entry.indices,
        entry.vis_vertices,
        device=device,
    )
    if remap is None:
        logger.warning(
            "Volume deformable '%s' could not build a sim-to-visual barycentric remap; "
            "renderers may fall back to rest-pose visual geometry.",
            entry.root_path,
        )
    return remap


def _expand_clone_plan_deformable_entries(
    entries: Sequence[DeformableStageEntry],
    clone_plan: ClonePlan,
    rows: tuple[int, ...],
) -> list[DeformableStageEntry]:
    """Expand prototype deformables into every destination selected by a clone plan.

    Kit-less replication may retain only a source deformable on the USD stage. The
    shadow builder nevertheless needs one deformable particle block and one visual
    mesh binding for every cloned environment.
    """
    expanded: dict[str, DeformableStageEntry] = {}
    for entry in entries:
        for source_idx, (source, destination) in enumerate(
            zip(clone_plan.sources, clone_plan.destinations, strict=True)
        ):
            if not under(entry.root_path, source):
                continue
            if source_idx not in rows:
                break
            columns = np.flatnonzero(clone_plan.clone_mask[source_idx])
            for col in columns:
                target = destination.format(int(clone_plan.env_ids[col]))
                offset = (
                    0 if clone_plan.positions is None else clone_plan.positions[col] - clone_plan.positions[columns[0]]
                )
                cloned_entry = replace(
                    entry,
                    root_path=rebase(entry.root_path, source, target),
                    sim_mesh_path=rebase(entry.sim_mesh_path, source, target),
                    vis_mesh_path=rebase(entry.vis_mesh_path, source, target),
                    init_pos=tuple(np.asarray(entry.init_pos) + offset),
                )
                expanded.setdefault(cloned_entry.root_path, cloned_entry)
            break
        else:
            expanded.setdefault(entry.root_path, entry)

    return list(expanded.values())


def add_shadow_deformables_to_builder(
    builder: ModelBuilder,
    stage: Usd.Stage,
    entries: Sequence[DeformableStageEntry],
    clone_plan: ClonePlan,
    rows: tuple[int, ...],
    *,
    device: str = "cpu",
) -> tuple[list[ShadowDeformableEntity], list[ShadowDeformableRegistryGroup]]:
    """Add PhysX/OVPhysX deformable meshes to a shadow Newton builder.

    Volume deformables with a sim-to-visual barycentric remap are added as visual
    triangle meshes (:meth:`~newton.ModelBuilder.add_cloth_mesh`) so Newton Warp and
    OVRTX render the paired visual mesh rather than the tet simulation topology.

    Args:
        builder: Shadow :class:`~newton.ModelBuilder` under construction.
        stage: Current USD stage.
        entries: Deformable geometry imported from declared clone sources and shared roots.
        clone_plan: Replication layout used to expand prototypes into destination environments.
        rows: Plan rows routed to this Newton representation.
        device: Warp device for barycentric remap tables uploaded during shadow build.

    Returns:
        Flat entity list for geometry mapping and grouped registry metadata for
        USD visual-mesh point bindings (e.g. OVRTX).
    """
    if not entries:
        return [], []
    wildcard_groups: dict[tuple[str, str, str], list[DeformableStageEntry]] = {}
    for entry in entries:
        # Discovery bakes vertices into the source root's parent frame. Retain that
        # pose and translate clones through the plan, without fetching destination prims.
        pos, quat = resolve_prim_pose(stage.GetPrimAtPath(entry.root_path).GetParent())
        entry = replace(entry, init_pos=tuple(pos), init_rot=tuple(quat))
        key = (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path)
        for source, destination in zip(clone_plan.sources, clone_plan.destinations, strict=True):
            if under(entry.root_path, source):
                key = tuple(rebase(path, source, destination.format("[^/]+")) for path in key)
                break
        wildcard_groups.setdefault(key, []).extend(_expand_clone_plan_deformable_entries([entry], clone_plan, rows))
    entries = [entry for group in wildcard_groups.values() for entry in group]

    flat_entities: list[ShadowDeformableEntity] = []
    registry_groups: list[ShadowDeformableRegistryGroup] = []
    sim_particle_cursor = 0

    for (wildcard_root, wildcard_sim, wildcard_vis), group_entries in sorted(wildcard_groups.items()):
        if not group_entries:
            continue
        template = group_entries[0]
        uses_remap = _needs_volume_vis_remap(template)
        render_count = template.vis_vertex_count if uses_remap else template.vertex_count
        group = ShadowDeformableRegistryGroup(
            prim_path=wildcard_root,
            sim_mesh_prim_path=wildcard_sim,
            vis_mesh_prim_path=wildcard_vis,
            deformable_type=template.deformable_type,
            particles_per_body=render_count,
            register_usd_vis_point_bindings=uses_remap or template.sim_mesh_path == template.vis_mesh_path,
        )

        group_volume_vis_remap = _build_volume_vis_remap(template, device) if uses_remap else None

        for entry in sorted(group_entries, key=lambda item: item.root_path):
            body_pos = wp.vec3(*entry.init_pos)
            body_rot = wp.quat(*entry.init_rot)

            before_render = int(getattr(builder, "particle_count", 0))
            volume_vis_remap = None

            if entry.deformable_type == "surface":
                builder.add_cloth_mesh(
                    pos=body_pos,
                    rot=body_rot,
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=_vertices_to_wp_vec3(entry.vertices),
                    indices=_mesh_indices(entry.indices),
                    density=1.0,
                    tri_ke=1e4,
                    tri_ka=1e4,
                    tri_kd=1.5e-6,
                    edge_ke=5.0,
                    edge_kd=1e-2,
                    particle_radius=0.008,
                )
            else:
                # Build remap before allocating render slots so a failed embed can fall
                # back to sim tet topology instead of leaving mismatched vis slots.
                # Replicated clones share one template table (rest topology is identical).
                volume_vis_remap = group_volume_vis_remap if _needs_volume_vis_remap(entry) else None

                if volume_vis_remap is not None:
                    builder.add_cloth_mesh(
                        pos=body_pos,
                        rot=body_rot,
                        scale=1.0,
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=_vertices_to_wp_vec3(entry.vis_vertices),
                        indices=_mesh_indices(entry.vis_indices),
                        density=1.0,
                        tri_ke=1e4,
                        tri_ka=1e4,
                        tri_kd=1.5e-6,
                        edge_ke=5.0,
                        edge_kd=1e-2,
                        particle_radius=0.008,
                    )
                else:
                    builder.add_soft_mesh(
                        pos=body_pos,
                        rot=body_rot,
                        scale=1.0,
                        vel=wp.vec3(0.0, 0.0, 0.0),
                        vertices=_vertices_to_wp_vec3(entry.vertices),
                        indices=_mesh_indices(entry.indices),
                        density=1000.0,
                        k_mu=1e5,
                        k_lambda=1e5,
                        k_damp=0.0,
                    )

            added_render = int(getattr(builder, "particle_count", 0)) - before_render
            render_count = added_render if added_render > 0 else render_count
            sim_count = entry.vertex_count
            # Use the builder particle cursor so pre-existing particles (e.g. from
            # USD import) keep shadow sync / OVRTX offsets aligned with particle_q.
            entity = ShadowDeformableEntity(
                root_path=entry.root_path,
                sim_particle_offset=sim_particle_cursor,
                sim_particle_count=sim_count,
                vis_particle_offset=before_render,
                vis_particle_count=render_count,
                volume_vis_remap=volume_vis_remap,
            )
            sim_particle_cursor += sim_count
            flat_entities.append(entity)
            group.particle_offsets.append(entity.vis_particle_offset)
            group.entities.append(entity)

        if group.entities:
            group.particles_per_body = group.entities[0].vis_particle_count
            if uses_remap:
                # Only bind USD visual points when every body successfully remapped.
                group.register_usd_vis_point_bindings = all(
                    entity.volume_vis_remap is not None for entity in group.entities
                )
            if not group.register_usd_vis_point_bindings:
                logger.warning(
                    "Skipping USD visual-mesh point bindings for %s deformable '%s'",
                    template.deformable_type,
                    wildcard_root,
                )
            registry_groups.append(group)

    ordered_roots = [entry.root_path for entry in sort_deformable_entries_for_geometry_sync(entries)]
    entity_by_root = {entity.root_path: entity for entity in flat_entities}
    flat_entities = [entity_by_root[root_path] for root_path in ordered_roots]

    return flat_entities, registry_groups


def populate_shadow_deformable_registry(
    manager_cls,
    registry_groups: Sequence[ShadowDeformableRegistryGroup],
) -> None:
    """Populate ``manager_cls._deformable_registry`` for USD visual-mesh point bindings.

    Under PhysX/OVPhysX sim, OVRTX (and any similar consumer) uses this registry to
    bind authored visual-mesh ``points`` to shadow ``particle_q`` render slots.

    Args:
        manager_cls: Physics manager class that owns ``_deformable_registry``.
        registry_groups: Groups produced by :func:`add_shadow_deformables_to_builder`.
    """
    try:
        from isaaclab_contrib.deformable.deformable_object import DeformableRegistryEntry
    except ImportError:
        logger.debug("isaaclab_contrib deformable registry unavailable; skipping shadow registry population.")
        return

    for group in registry_groups:
        if not group.register_usd_vis_point_bindings:
            continue

        manager_cls._deformable_registry.append(
            DeformableRegistryEntry(
                prim_path=group.prim_path,
                sim_mesh_prim_path=group.sim_mesh_prim_path,
                vis_mesh_prim_path=group.vis_mesh_prim_path,
                vertices=[],
                indices=[],
                deformable_type=group.deformable_type,
                init_pos=(0.0, 0.0, 0.0),
                init_rot=(0.0, 0.0, 0.0, 1.0),
                particle_offsets=list(group.particle_offsets),
                particles_per_body=group.particles_per_body,
            )
        )
