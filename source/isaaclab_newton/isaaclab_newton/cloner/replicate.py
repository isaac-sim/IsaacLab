# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import torch
from newton import ModelBuilder
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.cloner.replicate_session import REPLICATION_QUEUE
from isaaclab.physics import PhysicsManager

from isaaclab_newton.cloner.newton_clone_utils import (
    build_source_builders,
    rename_builder_labels,
    replicate_builder_mapping,
)
from isaaclab_newton.physics import NewtonManager

if TYPE_CHECKING:
    _MappingBatch: TypeAlias = tuple[
        tuple[str, ...], tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None
    ]
else:
    _MappingBatch = tuple


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object, dict, list, dict[str, ModelBuilder]]:
    """Build a Newton model builder from clone mapping inputs.

    Also returns the per-source builders (``{source_path: ModelBuilder}``) so the
    committing path can retain them for single-model consumers such as the
    batched Newton IK action.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]
    manager_cls = PhysicsManager._sim.physics_manager

    builder = manager_cls.create_builder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources],
        schema_resolvers=schema_resolvers,
    )

    # Deformable prim paths are handled by per_world_builder_hooks, not add_usd.
    # Resolve the regex prim_path patterns to concrete env_0 paths so add_usd
    # can skip them via ignore_paths.
    deformable_patterns = tuple(
        re.compile(entry.prim_path.replace(".*", "[^/]*")) for entry in NewtonManager._deformable_registry
    )
    deformable_ignore_paths = []
    if deformable_patterns:
        for source in sources:
            for child in Usd.PrimRange(stage.GetPrimAtPath(source)):
                child_path = str(child.GetPath())
                if any(pattern.fullmatch(child_path) for pattern in deformable_patterns):
                    deformable_ignore_paths.append(child_path)

    source_builders = build_source_builders(
        stage,
        sources,
        lambda: manager_cls.create_builder(up_axis=up_axis),
        schema_resolvers,
        ignore_paths=deformable_ignore_paths or None,
        simplify_meshes=simplify_meshes,
    )

    # Inject registered sites into source builders (and global sites into main builder).
    global_sites, source_sites, root_sites = NewtonManager._cl_inject_sites(builder, source_builders)

    replicate_args = (builder, sources, mapping, positions, quaternions, source_builders)
    local_site_map, world_xforms = replicate_builder_mapping(
        *replicate_args,
        source_site_indices=source_sites,
        env_root_sites=root_sites,
        per_world_builder_hooks=NewtonManager._per_world_builder_hooks,
        post_replicate_hooks=NewtonManager._post_replicate_hooks,
    )

    site_index_map = {label: (idx, None) for label, idx in global_sites.items()}
    site_index_map.update((label, (None, per_world)) for label, per_world in local_site_map.items())
    return builder, stage_info, site_index_map, world_xforms, source_builders


class NewtonReplicateContext:
    """Queue and run Newton replication work for one stage."""

    def __init__(
        self,
        stage: Usd.Stage,
        *,
        device: str = "cpu",
        up_axis: str = "Z",
        simplify_meshes: bool | None = None,
        commit_to_manager: bool = True,
    ):
        """Initialize the context.

        Args:
            stage: USD stage containing source assets.
            device: Device used by the finalized Newton model builder.
            up_axis: Up axis for the Newton model builder.
            simplify_meshes: Whether to run convex-hull mesh approximation. If
                ``None``, read from the active :class:`NewtonCfg`.
            commit_to_manager: Whether :meth:`replicate` should publish the builder to
                :class:`NewtonManager`.
        """
        self.stage = stage
        self.device = device
        self.up_axis = up_axis
        if simplify_meshes is None:
            from isaaclab_newton.physics import NewtonCfg

            cfg = PhysicsManager._cfg
            simplify_meshes = cfg.simplify_meshes if isinstance(cfg, NewtonCfg) else True
        self.simplify_meshes = simplify_meshes
        self.commit_to_manager = commit_to_manager
        self._queue: list[_MappingBatch] = []

    def queue_mapping(
        self,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: torch.Tensor,
        mapping: torch.Tensor,
        *,
        positions: torch.Tensor | None = None,
        quaternions: torch.Tensor | None = None,
    ) -> None:
        """Queue replication rows from the current flat clone mapping.

        Args:
            sources: Source prim paths used for cloning.
            destinations: Destination prim path templates.
            env_ids: Environment ids for destination worlds.
            mapping: Boolean source-to-environment mapping matrix.
            positions: Optional per-environment world positions [m].
            quaternions: Optional per-environment orientations in xyzw order.
        """
        self._queue.append((tuple(sources), tuple(destinations), env_ids, mapping, positions, quaternions))

    @staticmethod
    def _merge_optional_tensor(
        name: str, current: torch.Tensor | None, incoming: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Merge optional tensors, requiring equal values when both are present."""
        if current is None:
            return incoming
        if incoming is None:
            return current
        if current.device != incoming.device or current.shape != incoming.shape or not torch.equal(current, incoming):
            raise ValueError(f"Queued Newton mappings must use the same {name} tensor.")
        return current

    def _merged_mapping(self) -> _MappingBatch:
        """Merge queued mapping batches into the legacy flat mapping shape."""
        if not self._queue:
            raise RuntimeError("Cannot replicate without queued Newton mappings.")

        sources: list[str] = []
        destinations: list[str] = []
        mappings: list[torch.Tensor] = []
        env_ids = self._queue[0][2]
        positions = self._queue[0][4]
        quaternions = self._queue[0][5]

        for (
            queued_sources,
            queued_destinations,
            queued_env_ids,
            mapping,
            queued_positions,
            queued_quaternions,
        ) in self._queue:
            if (
                env_ids.device != queued_env_ids.device
                or env_ids.shape != queued_env_ids.shape
                or not torch.equal(env_ids, queued_env_ids)
            ):
                raise ValueError("Queued Newton mappings must use the same env_ids tensor.")
            sources.extend(queued_sources)
            destinations.extend(queued_destinations)
            mappings.append(mapping)
            positions = self._merge_optional_tensor("positions", positions, queued_positions)
            quaternions = self._merge_optional_tensor("quaternions", quaternions, queued_quaternions)

        return tuple(sources), tuple(destinations), env_ids, torch.cat(mappings, dim=0), positions, quaternions

    def replicate(self) -> tuple[ModelBuilder, object, dict]:
        """Build the Newton model builder from queued mappings and optionally publish it."""
        sources, destinations, env_ids, mapping, positions, quaternions = self._merged_mapping()
        builder, stage_info, site_index_map, world_xforms, source_builders = _build_newton_builder_from_mapping(
            stage=self.stage,
            sources=sources,
            destinations=destinations,
            env_ids=env_ids,
            mapping=mapping,
            positions=positions,
            quaternions=quaternions,
            up_axis=self.up_axis,
            simplify_meshes=self.simplify_meshes,
        )
        fabric_body_bindings = rename_builder_labels(builder, sources, destinations, env_ids, mapping)
        if self.commit_to_manager:
            NewtonManager._cl_site_index_map = site_index_map
            NewtonManager._cl_fabric_body_bindings = fabric_body_bindings
            NewtonManager._world_xforms = world_xforms
            NewtonManager._cl_protos = source_builders
            NewtonManager.set_builder(builder)
            NewtonManager._num_envs = mapping.size(1)
        self._queue.clear()
        return builder, stage_info, site_index_map


def queue_newton_physics_replication(cfg: Any) -> None:
    """Register ``cfg`` for Newton replication when :func:`~isaaclab.cloner.replicate` next runs.

    Appends ``(cfg, NewtonReplicateContext)`` to
    :data:`~isaaclab.cloner.REPLICATION_QUEUE`. The actual row resolution and dispatch
    happen inside :func:`~isaaclab.cloner.replicate`, so this helper is safe to call from
    any asset constructor — no active session is required.
    """
    REPLICATION_QUEUE.append((cfg, NewtonReplicateContext))


def newton_physics_replicate(
    stage: Usd.Stage,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    device: str = "cpu",
    up_axis: str = "Z",
    simplify_meshes: bool = True,
):
    """Replicate prims into a Newton ``ModelBuilder`` using a per-source mapping.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        destinations: Destination prim path templates.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        device: Device used by the finalized Newton model builder.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder and stage metadata.
    """
    ctx = NewtonReplicateContext(
        stage, device=device, up_axis=up_axis, simplify_meshes=simplify_meshes, commit_to_manager=True
    )
    ctx.queue_mapping(sources, destinations, env_ids, mapping, positions=positions, quaternions=quaternions)
    builder, stage_info, _site_index_map = ctx.replicate()
    return builder, stage_info
