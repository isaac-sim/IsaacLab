# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.scene_data.deformable_discovery import DeformableStageEntry, discover_deformables_on_stage
from isaaclab.sim.utils.transforms import resolve_prim_pose

from isaaclab_newton.cloner.newton_clone_utils import (
    _restore_visible_colliders_without_visual_shapes,
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)
from isaaclab_newton.physics.visualization_deformables import add_shadow_deformables_to_builder


def _deformable_ignore_paths(
    stage: Usd.Stage,
    sources: Sequence[str] | None = None,
    entries: Sequence[DeformableStageEntry] | None = None,
) -> list[str]:
    """Collect deformable prim paths for ``add_usd`` ignore lists.

    Shadow deformables are added explicitly by :func:`add_shadow_deformables_to_builder`.
    Ignoring them during USD import prevents double-allocating particles in the
    shadow ``particle_q`` buffer (which breaks geometry mapping offsets).

    Args:
        stage: USD stage to scan.
        sources: Optional clone-source roots to restrict the ignore list. When ``None``,
            every discovered deformable path is ignored (standalone imports).
        entries: Optional pre-discovered deformable entries to avoid a second stage walk.
    """
    source_prefixes: tuple[str, ...] | None = None
    if sources is not None:
        source_prefixes = tuple(f"{source.rstrip('/')}/" for source in sources)

    if entries is None:
        entries = discover_deformables_on_stage(stage)

    ignore_paths: list[str] = []

    for entry in entries:
        for path in (entry.root_path, entry.sim_mesh_path, entry.vis_mesh_path):
            if source_prefixes is None:
                ignore_paths.append(path)
                continue

            under_source = any(
                path == source.rstrip("/") or path.startswith(prefix)
                for source, prefix in zip(sources, source_prefixes, strict=True)
            )
            if under_source:
                ignore_paths.append(path)

    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(ignore_paths))


def build_visualization_builder_from_stage_envs(
    stage: Usd.Stage,
    env_paths: Sequence[tuple[int, str]],
    clone_plan: Any | None,
    *,
    up_axis: str = "Z",
    device: str = "cpu",
) -> tuple[ModelBuilder, tuple[list, list]]:
    """Build a Newton shadow visualization builder from a USD stage.

    Cloned scenes use the clone plan to preserve per-environment world layout and
    labels. Standalone scenes without a clone plan are imported as one world, then
    deformables are added as shadow particles so OVRTX registry bindings stay populated.

    Args:
        stage: USD stage to import into the shadow builder.
        env_paths: Sorted ``(env_id, env_prim_path)`` pairs.
        clone_plan: Optional clone plan; ``None`` imports the stage as a standalone world.
        up_axis: Up axis for the :class:`~newton.ModelBuilder`.
        device: Warp device for volume sim-to-visual remap tables.

    Returns:
        A tuple of the populated :class:`~newton.ModelBuilder` and shadow-deformable
        metadata ``(shadow_entities, registry_groups)``.
    """
    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    builder = ModelBuilder(up_axis=up_axis)
    # Discover once and reuse via ``entries=`` below (ignore paths + shadow add).
    deformable_entries = discover_deformables_on_stage(stage)

    if clone_plan is None:
        # Ignore deformables during USD import; add them as shadow particles below so
        # SceneData mapping and OVRTX registry receive the same particle offsets.
        deformable_ignore_paths = _deformable_ignore_paths(stage, entries=deformable_entries)
        import_result = builder.add_usd(
            stage,
            schema_resolvers=schema_resolvers,
            ignore_paths=deformable_ignore_paths or None,
        )
        _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
        shadow_entities, registry_groups = add_shadow_deformables_to_builder(
            builder, stage, env_paths, device=device, entries=deformable_entries
        )
        return builder, (shadow_entities, registry_groups)

    if not env_paths:
        raise ValueError("clone plan requires at least one environment path")

    env_path_by_id = dict(env_paths)

    sources = tuple(clone_plan.sources)
    destinations = tuple(clone_plan.destinations)
    env_ids = clone_plan.env_ids.detach().cpu()
    mapping = clone_plan.clone_mask.detach().cpu()

    poses = [resolve_prim_pose(stage.GetPrimAtPath(env_path_by_id[int(env_id)])) for env_id in env_ids.tolist()]
    positions = torch.tensor([pos for pos, _ in poses], dtype=torch.float32)
    quaternions = torch.tensor([quat for _, quat in poses], dtype=torch.float32)
    # Ignore every deformable on the stage for the world import — not only those under
    # clone sources. Otherwise a non-env deformable (e.g. ``/World/Assets/Cloth``) is
    # imported here and added again by ``add_shadow_deformables_to_builder``.
    deformable_ignore_paths = _deformable_ignore_paths(stage, entries=deformable_entries)
    import_result = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources, *deformable_ignore_paths],
        schema_resolvers=schema_resolvers,
    )
    _restore_visible_colliders_without_visual_shapes(builder, stage, import_result["path_shape_map"])
    source_deformable_ignore_paths = _deformable_ignore_paths(stage, sources, entries=deformable_entries)
    source_builders = build_source_builders(
        stage,
        sources,
        lambda: ModelBuilder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=source_deformable_ignore_paths or None,
        simplify_meshes=False,
    )
    replicate_builder_mapping(builder, sources, mapping, positions, quaternions, source_builders)
    rename_builder_labels(builder, sources, destinations, env_ids, mapping)
    shadow_entities, registry_groups = add_shadow_deformables_to_builder(
        builder, stage, env_paths, device=device, entries=deformable_entries
    )
    return builder, (shadow_entities, registry_groups)
