# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import warp as wp
from newton import ModelBuilder, solvers
from newton._src.usd.schemas import SchemaResolverNewton, SchemaResolverPhysx

from pxr import Usd

from isaaclab.cloner.replicate_session import REPLICATION_QUEUE

from isaaclab_newton.physics import NewtonManager


def _build_newton_builder_from_mapping(
    stage: Usd.Stage,
    sources: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor | None = None,
    quaternions: torch.Tensor | None = None,
    up_axis: str = "Z",
    simplify_meshes: bool = True,
) -> tuple[ModelBuilder, object, dict, list]:
    """Build a Newton model builder from clone mapping inputs.

    Args:
        stage: USD stage containing source assets.
        sources: Source prim paths used for cloning.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix.
        positions: Optional per-environment world positions.
        quaternions: Optional per-environment orientations in xyzw order.
        up_axis: Up axis for the Newton model builder.
        simplify_meshes: Whether to run convex-hull mesh approximation.

    Returns:
        Tuple of the populated Newton model builder, stage metadata returned
        by ``add_usd``, a site index map for
        :attr:`NewtonManager._cl_site_index_map`, and the absolute per-world
        transforms for :attr:`NewtonManager._world_xforms`.
    """
    if positions is None:
        positions = torch.zeros((mapping.size(1), 3), device=mapping.device, dtype=torch.float32)
    if quaternions is None:
        quaternions = torch.zeros((mapping.size(1), 4), device=mapping.device, dtype=torch.float32)
        quaternions[:, 3] = 1.0

    schema_resolvers = [SchemaResolverNewton(), SchemaResolverPhysx()]

    builder = NewtonManager.create_builder(up_axis=up_axis)
    stage_info = builder.add_usd(
        stage,
        ignore_paths=["/World/envs", *sources],
        schema_resolvers=schema_resolvers,
    )

    # The prototype is built from env_0 in absolute world coordinates.
    # add_builder xforms are deltas from env_0 so positions don't get double-counted.
    env0_pos = positions[0]

    # Deformable prim paths are handled by per_world_builder_hooks, not add_usd.
    # Resolve the regex prim_path patterns to concrete env_0 paths so add_usd
    # can skip them via ignore_paths.
    import re

    _deformable_ignore_paths: list[str] = []
    if hasattr(NewtonManager, "_deformable_registry"):
        for entry in NewtonManager._deformable_registry:
            pat = re.compile(entry.prim_path.replace(".*", "[^/]*") + "$")
            for src_path in sources:
                # Check if any prim under this source matches the deformable pattern
                prim = stage.GetPrimAtPath(src_path)
                if prim.IsValid():
                    for child in Usd.PrimRange(prim):
                        child_path = str(child.GetPath())
                        if pat.match(child_path):
                            _deformable_ignore_paths.append(child_path)

    protos: dict[str, ModelBuilder] = {}
    for src_path in sources:
        p = NewtonManager.create_builder(up_axis=up_axis)
        solvers.SolverMuJoCo.register_custom_attributes(p)
        p.add_usd(
            stage,
            root_path=src_path,
            load_visual_shapes=True,
            skip_mesh_approximation=True,
            schema_resolvers=schema_resolvers,
            ignore_paths=_deformable_ignore_paths if _deformable_ignore_paths else None,
        )
        if simplify_meshes:
            p.approximate_meshes("convex_hull", keep_visual_shapes=True)
        protos[src_path] = p

    # Inject registered sites into prototypes (and global sites into main builder)
    global_sites, proto_sites, world_sites = NewtonManager._cl_inject_sites(builder, protos)

    # Global sites: (int, None)
    global_site_map: dict[str, tuple[int, None]] = {label: (idx, None) for label, idx in global_sites.items()}

    # Local sites: per-world sublists, populated in the loop below
    num_worlds = mapping.size(1)
    local_site_map: dict[str, list[list[int]]] = {}
    # Absolute per-world transforms (env-root local-to-world). Consumed by
    # FrameView to place non-physics frames (e.g. cameras) relative to each
    # cloned env, mirroring the legacy ``_replicate_from_stage`` path.
    world_xforms: list[wp.transform] = []

    # create a separate world for each environment (heterogeneous spawning)
    # Newton assigns sequential world IDs (0, 1, 2, ...), so we need to track the mapping
    for col, _ in enumerate(env_ids.tolist()):
        # begin a new world context (Newton assigns world ID = col)
        builder.begin_world()
        # ``add_builder`` xforms are deltas from env_0 (the proto is baked in env_0's
        # absolute coords), while bodyless world sites and ``world_xforms`` live in the
        # global frame and therefore use the env's absolute transform.
        delta_pos = (positions[col] - env0_pos).tolist()
        env_xform = wp.transform(delta_pos, quaternions[col].tolist())
        world_xform = wp.transform(positions[col].tolist(), quaternions[col].tolist())
        world_xforms.append(world_xform)
        # Per-world bodyless sites are placed in each world's (global) frame.
        for label, xform in world_sites.items():
            if label not in local_site_map:
                local_site_map[label] = [[] for _ in range(num_worlds)]
            site_idx = builder.add_site(body=-1, xform=wp.transform_multiply(world_xform, xform), label=label)
            local_site_map[label][col].append(site_idx)
        for row in torch.nonzero(mapping[:, col], as_tuple=True)[0].tolist():
            proto = protos[sources[row]]
            offset = builder.shape_count
            builder.add_builder(
                proto,
                xform=env_xform,
            )
            # Compute final shape indices for sites in this proto
            for label, proto_shape_indices in proto_sites.get(id(proto), {}).items():
                if label not in local_site_map:
                    local_site_map[label] = [[] for _ in range(num_worlds)]
                for proto_shape_idx in proto_shape_indices:
                    local_site_map[label][col].append(offset + proto_shape_idx)

        # Run per-world builder hooks (e.g. deformable body registration).
        if hasattr(NewtonManager, "_per_world_builder_hooks"):
            for hook in NewtonManager._per_world_builder_hooks:
                hook(builder, col, positions[col].tolist(), quaternions[col].tolist())

        # end the world context
        builder.end_world()

    # Run post-replicate hooks (e.g. builder.color() for deformable coloring).
    if hasattr(NewtonManager, "_post_replicate_hooks"):
        for hook in NewtonManager._post_replicate_hooks:
            hook(builder)

    site_index_map = {
        **global_site_map,
        **{label: (None, per_world) for label, per_world in local_site_map.items()},
    }

    return builder, stage_info, site_index_map, world_xforms


# Built-in label arrays that ``_rename_builder_labels`` rewrites in Pass 1.
# Each type ``t`` has a paired ``<t>_label`` (or ``<t>_key``) string column
# and a ``<t>_world`` int column on Newton's ``ModelBuilder``. Exposed as a
# module-level constant so tests can import it instead of duplicating.
_BUILTIN_LABEL_TYPES: tuple[str, ...] = (
    "body",
    "joint",
    "shape",
    "articulation",
    "constraint_mimic",
    "equality_constraint",
)


def _rename_builder_labels(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
) -> None:
    """Rename builder labels/keys from source roots to destination roots.

    Walks both built-in label arrays (see :data:`_BUILTIN_LABEL_TYPES`) and any
    string-typed custom-attribute column whose frequency declares a sibling
    world column (``references="world"``).
    The boundary-safe match (exact source root, or source root followed by ``/``)
    makes the rewrite a no-op for strings that are not paths under the source.
    Non-path custom string columns are passed through untouched and any future
    solver-registered string column is handled automatically without changes here.

    Args:
        builder: Newton model builder to update in-place.
        sources: Source prim root paths.
        destinations: Destination prim path templates.
        env_ids: Environment ids corresponding to mapping columns.
        mapping: Boolean source-to-environment mapping matrix.
    """
    # per-source, per-world renaming (strict prefix swap), compact style preserved
    for i, src_path in enumerate(sources):
        # Canonicalize the source root (drop any trailing ``/``) so the
        # boundary-safe match logic in ``_rename_pair`` is unambiguous.
        src_root = src_path.rstrip("/")
        world_cols = torch.nonzero(mapping[i], as_tuple=True)[0].tolist()
        # Map Newton world IDs (sequential) to destination paths using env_ids
        world_roots = {int(env_ids[c]): destinations[i].format(int(env_ids[c])) for c in world_cols}

        def _rename_pair(values, worlds):
            if len(values) != len(worlds):
                raise ValueError(f"label/world column length mismatch: {len(values)} vs {len(worlds)}")
            for k in range(len(values)):
                v = values[k]
                if not isinstance(v, str):
                    continue
                world_id = int(worlds[k])
                if world_id not in world_roots:
                    continue
                # Gate on an explicit prefix test before slicing. ``str.removeprefix``
                # is tempting but conflates "match with empty suffix" and "no match"
                # (both return a string starting with "/"), so a label already
                # rewritten in an earlier source-iteration would be re-prepended to
                # the next iteration's dst root.
                if not v.startswith(src_root):
                    continue
                suffix = v[len(src_root) :]
                # ``suffix == ""``     -> exact source-root match (rewrite to dst root).
                # ``suffix[0] == "/"`` -> child path under source.
                # otherwise           -> boundary-bleed sibling like "/Sources/protoAB/x"
                #                        when src_root is "/Sources/protoA" -> skip.
                if suffix and not suffix.startswith("/"):
                    continue
                values[k] = world_roots[world_id] + suffix

        # Pass 1: built-in label arrays. Each has a paired ``*_world`` int column.
        # Use ``is None`` (not ``or``) so an empty-but-defined ``*_label`` column
        # is recognized — falling through to ``*_key`` would over-match a
        # builder that legitimately exposes both attributes.
        for t in _BUILTIN_LABEL_TYPES:
            labels = getattr(builder, f"{t}_label", None)
            if labels is None:
                labels = getattr(builder, f"{t}_key", None)
            worlds_arr = getattr(builder, f"{t}_world", None)
            if labels is None or worlds_arr is None:
                continue
            _rename_pair(labels, worlds_arr)

        # Pass 2: string-typed custom-attribute columns (e.g. ``mujoco:tendon_label``)
        # paired with a world companion declared via ``references="world"``. Index
        # world companions by frequency for O(1) lookup, then walk the str columns.
        custom = builder.custom_attributes
        world_by_freq: dict[str, ModelBuilder.CustomAttribute] = {}
        for attr in custom.values():
            if getattr(attr, "references", None) == "world":
                world_by_freq[attr.frequency] = attr
        for attr in custom.values():
            if attr.dtype is not str:
                continue
            world_attr = world_by_freq.get(attr.frequency)
            if world_attr is None:
                continue
            values = attr.values
            worlds = world_attr.values
            if not values or not worlds:
                continue
            _rename_pair(values, worlds)


class NewtonReplicateContext:
    """Queue and run Newton replication work for one stage."""

    def __init__(
        self,
        stage: Usd.Stage,
        *,
        device: str = "cpu",
        up_axis: str = "Z",
        simplify_meshes: bool = True,
        commit_to_manager: bool = True,
    ):
        """Initialize the context.

        Args:
            stage: USD stage containing source assets.
            device: Device used by the finalized Newton model builder.
            up_axis: Up axis for the Newton model builder.
            simplify_meshes: Whether to run convex-hull mesh approximation.
            commit_to_manager: Whether :meth:`replicate` should publish the builder to
                :class:`NewtonManager`.
        """
        self.stage = stage
        self.device = device
        self.up_axis = up_axis
        self.simplify_meshes = simplify_meshes
        self.commit_to_manager = commit_to_manager
        self._queue: list[
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                torch.Tensor,
                torch.Tensor,
                torch.Tensor | None,
                torch.Tensor | None,
            ]
        ] = []

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

    def _merged_mapping(
        self,
    ) -> tuple[tuple[str, ...], tuple[str, ...], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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
        builder, stage_info, site_index_map, world_xforms = _build_newton_builder_from_mapping(
            stage=self.stage,
            sources=sources,
            env_ids=env_ids,
            mapping=mapping,
            positions=positions,
            quaternions=quaternions,
            up_axis=self.up_axis,
            simplify_meshes=self.simplify_meshes,
        )
        _rename_builder_labels(builder, sources, destinations, env_ids, mapping)
        if self.commit_to_manager:
            NewtonManager._cl_site_index_map = site_index_map
            NewtonManager._world_xforms = world_xforms
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
    ctx.queue_mapping(
        sources,
        destinations,
        env_ids,
        mapping,
        positions=positions,
        quaternions=quaternions,
    )
    builder, stage_info, _site_index_map = ctx.replicate()
    return builder, stage_info
