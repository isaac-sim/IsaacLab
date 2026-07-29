# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shadow-model deformable discovery and registry helpers for PhysX/OVPhysX visualization."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import warp as wp
from newton import ModelBuilder

from pxr import Usd

from isaaclab.scene_data.deformable_discovery import (
    DeformableStageEntry,
    discover_deformables_on_stage,
    path_to_env_regex,
    path_to_env_wildcard,
    sort_deformable_entries_for_geometry_sync,
)
from isaaclab.sim.utils.transforms import resolve_prim_pose

logger = logging.getLogger(__name__)


@dataclass
class ShadowDeformableEntity:
    """One shadow-model deformable entity used for SceneData geometry mapping."""

    root_path: str
    particle_offset: int
    particle_count: int


@dataclass
class ShadowDeformableRegistryGroup:
    """Registry metadata for one replicated deformable asset."""

    prim_path: str
    sim_mesh_prim_path: str
    vis_mesh_prim_path: str
    deformable_type: str
    particles_per_body: int
    register_for_ovrtx: bool
    particle_offsets: list[int] = field(default_factory=list)
    entities: list[ShadowDeformableEntity] = field(default_factory=list)


def add_shadow_deformables_to_builder(
    builder: ModelBuilder,
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
) -> tuple[list[ShadowDeformableEntity], list[ShadowDeformableRegistryGroup]]:
    """Add PhysX/OVPhysX deformable meshes to a shadow Newton builder.

    Args:
        builder: Shadow :class:`~newton.ModelBuilder` under construction.
        stage: Current USD stage.
        env_paths: Sorted ``(env_id, env_prim_path)`` pairs.

    Returns:
        Flat entity list for geometry mapping and grouped registry metadata for OVRTX.
    """
    entries = discover_deformables_on_stage(stage)
    if not entries:
        return [], []

    env_path_by_id = dict(env_paths)
    wildcard_groups: dict[tuple[str, str, str], list[DeformableStageEntry]] = {}
    for entry in entries:
        wildcard_root = path_to_env_wildcard(entry.root_path)
        key = (wildcard_root, path_to_env_wildcard(entry.sim_mesh_path), path_to_env_wildcard(entry.vis_mesh_path))
        wildcard_groups.setdefault(key, []).append(entry)

    flat_entities: list[ShadowDeformableEntity] = []
    registry_groups: list[ShadowDeformableRegistryGroup] = []

    for (_wildcard_root, _wildcard_sim_key, _wildcard_vis_key), group_entries in sorted(wildcard_groups.items()):
        template = group_entries[0]
        wildcard_root = path_to_env_regex(template.root_path)
        wildcard_sim = path_to_env_regex(template.sim_mesh_path)
        wildcard_vis = path_to_env_regex(template.vis_mesh_path)
        asset_suffix = (
            wildcard_root.split("/World/envs/env_.*/", 1)[-1]
            if "/World/envs/" in wildcard_root
            else wildcard_root.rsplit("/", 1)[-1]
        )
        group = ShadowDeformableRegistryGroup(
            prim_path=f"/World/envs/env_.*/{asset_suffix}",
            sim_mesh_prim_path=wildcard_sim,
            vis_mesh_prim_path=wildcard_vis,
            deformable_type=template.deformable_type,
            particles_per_body=template.vertex_count,
            register_for_ovrtx=template.vertex_count == template.vis_vertex_count,
        )

        if not group.register_for_ovrtx and template.deformable_type == "surface":
            logger.debug(
                "Skipping OVRTX registry for shadow deformable '%s' because sim (%d) and visual (%d) "
                "vertex counts differ.",
                wildcard_root,
                template.vertex_count,
                template.vis_vertex_count,
            )

        for entry in sorted(group_entries, key=lambda item: item.root_path):
            root_prim = stage.GetPrimAtPath(entry.root_path)
            if root_prim.IsValid():
                pos, quat = resolve_prim_pose(root_prim)
            else:
                env_prefix = (
                    entry.root_path.split("/World/envs/")[1].split("/", 1)[0]
                    if "/World/envs/" in entry.root_path
                    else None
                )
                env_id = (
                    int(env_prefix.replace("env_", ""))
                    if env_prefix is not None and env_prefix.startswith("env_")
                    else 0
                )
                env_path = env_path_by_id.get(env_id)
                if env_path is None:
                    pos = (0.0, 0.0, 0.0)
                    quat = (0.0, 0.0, 0.0, 1.0)
                else:
                    pos, quat = resolve_prim_pose(stage.GetPrimAtPath(env_path))

            before_count = int(getattr(builder, "particle_count", 0))
            body_pos = wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))
            body_rot = wp.quat(float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))

            # Visualization-only material parameters; they do not drive PhysX/OVPhysX simulation.
            if entry.deformable_type == "volume":
                builder.add_soft_mesh(
                    pos=body_pos,
                    rot=body_rot,
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=entry.vertices,
                    indices=entry.indices,
                    density=1000.0,
                    k_mu=1e5,
                    k_lambda=1e5,
                    k_damp=0.0,
                )
            else:
                builder.add_cloth_mesh(
                    pos=body_pos,
                    rot=body_rot,
                    scale=1.0,
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    vertices=entry.vertices,
                    indices=entry.indices,
                    density=1.0,
                    tri_ke=1e4,
                    tri_ka=1e4,
                    tri_kd=1.5e-6,
                    edge_ke=5.0,
                    edge_kd=1e-2,
                    particle_radius=0.008,
                )

            offset = before_count
            added = int(getattr(builder, "particle_count", 0)) - before_count
            count = added if added > 0 else entry.vertex_count
            if added > 0 and added != entry.vertex_count:
                logger.warning(
                    "Shadow deformable '%s' allocated %d Newton particles but USD "
                    "reports %d vertices; using allocated count for SceneData mapping.",
                    entry.root_path,
                    added,
                    entry.vertex_count,
                )
            flat_entities.append(
                ShadowDeformableEntity(root_path=entry.root_path, particle_offset=offset, particle_count=count)
            )
            group.particle_offsets.append(offset)
            group.entities.append(flat_entities[-1])

        if group.entities:
            group.particles_per_body = group.entities[0].particle_count
            registry_groups.append(group)

    ordered_roots = [entry.root_path for entry in sort_deformable_entries_for_geometry_sync(entries)]
    entity_by_root = {entity.root_path: entity for entity in flat_entities}
    flat_entities = [entity_by_root[root_path] for root_path in ordered_roots]

    return flat_entities, registry_groups


def populate_shadow_deformable_registry(
    manager_cls,
    registry_groups: Sequence[ShadowDeformableRegistryGroup],
) -> None:
    """Populate ``manager_cls._deformable_registry`` for OVRTX under PhysX/OVPhysX sim."""
    try:
        from isaaclab_contrib.deformable.deformable_object import DeformableRegistryEntry
    except ImportError:
        logger.debug("isaaclab_contrib deformable registry unavailable; skipping shadow registry population.")
        return

    for group in registry_groups:
        if not group.register_for_ovrtx:
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
