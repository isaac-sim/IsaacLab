# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile backend-neutral collision groups into Newton shape filter pairs."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
from newton import ModelBuilder, ShapeFlags

from isaaclab.cloner import path as clone_path
from isaaclab.physics import CollisionFilterCfg
from isaaclab.physics._collision_filter import CompiledCollisionFilter

_CONVEX_PART_SUFFIX = re.compile(r"^(?P<label>.+)_convex_[1-9][0-9]*$")

ColliderShapeMap = dict[str, tuple[int, ...]]
"""Authored collider path to every Newton shape generated for that collider."""


def collider_shape_map(
    builder: ModelBuilder,
    path_shape_map: Mapping[str, int],
    shape_range: range | None = None,
) -> ColliderShapeMap:
    """Map authored collider paths to all colliding shapes created by one USD import.

    Newton's USD importer maps an authored path only to the first generated shape.
    Convex decomposition pieces use ``<primary-label>_convex_N`` labels, so they
    are folded back into their authored collider. Absolute labels produced by
    schema resolvers are retained as independent collider paths.
    """
    candidates = set(shape_range if shape_range is not None else range(builder.shape_count))
    result: dict[str, list[int]] = defaultdict(list)
    claimed: set[int] = set()
    primary_paths: dict[str, str] = {}
    for path, index in path_shape_map.items():
        if index not in candidates or not _shape_collides(builder, index):
            continue
        result[path].append(index)
        claimed.add(index)
        label = builder.shape_label[index]
        if isinstance(label, str):
            primary_paths[label] = path

    for index in sorted(candidates - claimed):
        if not _shape_collides(builder, index):
            continue
        label = builder.shape_label[index]
        if not isinstance(label, str) or not label.startswith("/"):
            continue
        match = _CONVEX_PART_SUFFIX.fullmatch(label)
        path = primary_paths.get(match.group("label")) if match is not None else None
        if path is not None and builder.shape_body[index] == builder.shape_body[result[path][0]]:
            result[path].append(index)
        else:
            result[label].append(index)
    return {path: tuple(indices) for path, indices in result.items()}


def merge_collider_shape_maps(*shape_maps: Mapping[str, Sequence[int]]) -> ColliderShapeMap:
    """Merge collider maps while preserving each path's unique shape order."""
    merged: ColliderShapeMap = {}
    for shape_map in shape_maps:
        _merge_collider_shape_map_into(merged, shape_map)
    return merged


def _merge_collider_shape_map_into(target: ColliderShapeMap, incoming: Mapping[str, Sequence[int]]) -> None:
    for path, indices in incoming.items():
        existing = target.get(path, ())
        seen = set(existing)
        target[path] = (*existing, *(index for index in indices if index not in seen))


def _memberships_by_shape(
    policy: CompiledCollisionFilter,
    collider_shapes: Mapping[str, Sequence[int]],
    universe: set[int],
) -> dict[int, set[str]]:
    memberships_by_shape: dict[int, set[str]] = defaultdict(set)
    for path, shape_indices in collider_shapes.items():
        memberships = policy.memberships(path)
        for shape_index in shape_indices:
            if shape_index in universe:
                memberships_by_shape[shape_index].update(memberships)
    return memberships_by_shape


def _collision_filter_pairs(
    policy: CompiledCollisionFilter,
    collider_shapes: Mapping[str, Sequence[int]],
    universe: Iterable[int],
    shape_worlds: Sequence[int] | None = None,
    shape_owners: Mapping[int, int] | None = None,
    preapplied_owners: set[int] | None = None,
) -> frozenset[tuple[int, int]]:
    """Expand path memberships into exact deny pairs within each collision world."""
    universe = set(universe)
    memberships_by_shape = _memberships_by_shape(policy, collider_shapes, universe)
    preapplied_owners = preapplied_owners or set()
    profiles: dict[tuple[int | None, tuple[str, ...], int | None], set[int]] = defaultdict(set)
    for shape_index in sorted(universe):
        memberships = memberships_by_shape[shape_index]
        group_profile = tuple(name for name in policy.groups if name in memberships)
        collision_world = None
        if shape_worlds is not None and int(shape_worlds[shape_index]) >= 0:
            collision_world = int(shape_worlds[shape_index])
        owner = None if shape_owners is None else shape_owners.get(shape_index)
        owner = owner if owner in preapplied_owners else None
        profiles[collision_world, group_profile, owner].add(shape_index)

    pairs: set[tuple[int, int]] = set()
    profile_items = tuple(profiles.items())
    indices_by_world: dict[int | None, list[int]] = defaultdict(list)
    for index, ((world, _, _), _) in enumerate(profile_items):
        indices_by_world[world].append(index)
    global_indices = indices_by_world.get(None, [])
    for index, ((world, first_groups, first_owner), first_shapes) in enumerate(profile_items):
        candidates = range(index, len(profile_items)) if world is None else (*global_indices, *indices_by_world[world])
        for second_index in candidates:
            if second_index < index:
                continue
            (_, second_groups, second_owner), second_shapes = profile_items[second_index]
            if first_owner is not None and first_owner == second_owner:
                continue
            if not policy.filters(first_groups, second_groups):
                continue
            pairs.update(
                (min(first, second), max(first, second))
                for first in first_shapes
                for second in second_shapes
                if first != second
            )
    return frozenset(pairs)


def _membership_signature(
    policy: CompiledCollisionFilter,
    collider_shapes: Mapping[str, Sequence[int]],
    universe: Iterable[int],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    selected = set(universe)
    memberships = _memberships_by_shape(policy, collider_shapes, selected)
    return tuple(
        (shape, tuple(name for name in policy.groups if name in memberships[shape])) for shape in sorted(selected)
    )


class NewtonCollisionFilter:
    """Apply declarative collision policy before Newton model finalization.

    Replication-invariant source-local pairs are added to prototype builders so
    Newton retains its compact replicated filter blocks. Cross-source, global,
    and environment-specific pairs are added after builder assembly.
    """

    def __init__(
        self,
        cfg: CollisionFilterCfg,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: np.ndarray,
        mapping: np.ndarray,
        env_template: str,
    ):
        self._policy = CompiledCollisionFilter(cfg, env_template)
        self._sources = tuple(sources)
        self._destinations = tuple(destinations)
        self._env_ids = env_ids
        self._mapping = mapping
        self._global_shapes: ColliderShapeMap = {}
        self._global_shape_indices: frozenset[int] = frozenset()
        self._source_builders: dict[str, ModelBuilder] = {}
        self._source_shapes: dict[str, ColliderShapeMap] = {}
        self._source_shape_counts: dict[str, int] = {}
        self._preapplied_rows: set[int] = set()

    def prepare(
        self,
        builder: ModelBuilder,
        global_shapes: ColliderShapeMap,
        source_builders: Mapping[str, ModelBuilder],
        source_shapes: Mapping[str, ColliderShapeMap],
    ) -> None:
        """Apply reusable global and homogeneous source-local pairs before replication."""
        self._global_shapes = global_shapes
        self._global_shape_indices = frozenset(_colliding_shape_indices(builder))
        self._source_builders = dict(source_builders)
        self._source_shapes = dict(source_shapes)
        self._source_shape_counts = {
            source: len(_colliding_shape_indices(source_builder)) for source, source_builder in source_builders.items()
        }
        _add_pairs(builder, _collision_filter_pairs(self._policy, global_shapes, self._global_shape_indices))

        rows_by_source: dict[str, list[int]] = defaultdict(list)
        for row, source in enumerate(self._sources):
            rows_by_source[source].append(row)
        for source, rows in rows_by_source.items():
            source_builder = source_builders[source]
            universe = _colliding_shape_indices(source_builder)
            templates = []
            cache = {}
            active_rows = set()
            for row in rows:
                for column in np.flatnonzero(self._mapping[row]):
                    active_rows.add(row)
                    destination = self._destinations[row].format(int(self._env_ids[column]))
                    concrete_shapes = _rebase_collider_paths(source_shapes[source], source, destination)
                    signature = _membership_signature(self._policy, concrete_shapes, universe)
                    template = cache.get(signature)
                    if template is None:
                        template = _collision_filter_pairs(self._policy, concrete_shapes, universe)
                        cache[signature] = template
                    templates.append(template)
            if templates and all(template == templates[0] for template in templates[1:]):
                _add_pairs(source_builder, templates[0])
                self._preapplied_rows.update(active_rows)

    def apply_to_replicated_builder(
        self,
        builder: ModelBuilder,
        source_shape_offsets: Mapping[tuple[int, int], int],
    ) -> None:
        """Apply residual policy after all source and per-world builders are assembled."""
        if self._can_skip_final_expansion(builder):
            return

        final_shapes = dict(self._global_shapes)
        shape_owners: dict[int, int] = {}
        for row, source in enumerate(self._sources):
            source_builder = self._source_builders[source]
            for column in np.flatnonzero(self._mapping[row]):
                column = int(column)
                offset = source_shape_offsets[row, column]
                for local_index in _colliding_shape_indices(source_builder):
                    shape_owners[offset + local_index] = row
                destination = self._destinations[row].format(int(self._env_ids[column]))
                concrete_shapes = {
                    _rebase_path(path, source, destination): tuple(offset + index for index in indices)
                    for path, indices in self._source_shapes[source].items()
                }
                _merge_collider_shape_map_into(final_shapes, concrete_shapes)

        mapped_shapes = {index for indices in final_shapes.values() for index in indices}
        supplement = {
            path: tuple(index for index in indices if index not in mapped_shapes)
            for path, indices in collider_shape_map(builder, {}).items()
        }
        _merge_collider_shape_map_into(final_shapes, {path: indices for path, indices in supplement.items() if indices})
        pairs = _collision_filter_pairs(
            self._policy,
            final_shapes,
            _colliding_shape_indices(builder),
            builder.shape_world,
            shape_owners,
            self._preapplied_rows,
        )
        _add_pairs(
            builder,
            (
                pair
                for pair in pairs
                if not (pair[0] in self._global_shape_indices and pair[1] in self._global_shape_indices)
            ),
        )

    def _can_skip_final_expansion(self, builder: ModelBuilder) -> bool:
        if self._global_shape_indices:
            return False
        active_rows = {row for row in range(len(self._sources)) if np.any(self._mapping[row])}
        if len(active_rows) != 1 or not active_rows.issubset(self._preapplied_rows):
            return False
        expected = sum(
            self._source_shape_counts[self._sources[row]] * int(np.count_nonzero(self._mapping[row]))
            for row in active_rows
        )
        return expected == len(_colliding_shape_indices(builder))


def _shape_collides(builder: ModelBuilder, index: int) -> bool:
    return bool(builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES)


def _colliding_shape_indices(builder: ModelBuilder) -> tuple[int, ...]:
    return tuple(index for index in range(builder.shape_count) if _shape_collides(builder, index))


def _rebase_collider_paths(collider_shapes: ColliderShapeMap, source: str, destination: str) -> ColliderShapeMap:
    return {_rebase_path(path, source, destination): indices for path, indices in collider_shapes.items()}


def _rebase_path(path: str, source: str, destination: str) -> str:
    suffix = clone_path.relative_to(path, source.rstrip("/") or "/")
    if suffix is None:
        raise ValueError(f"Newton collider path {path!r} is outside clone source {source!r}.")
    return (destination.rstrip("/") or "/") + suffix


def _add_pairs(builder: ModelBuilder, pairs: Iterable[tuple[int, int]]) -> None:
    for first, second in sorted(pairs):
        builder.add_shape_collision_filter_pair(first, second)
