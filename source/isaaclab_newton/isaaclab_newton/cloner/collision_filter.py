# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile backend-neutral collision groups into Newton shape filter pairs."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from newton import ModelBuilder, ShapeFlags

from isaaclab.cloner import path as clone_path
from isaaclab.physics import CollisionFilterCfg
from isaaclab.physics._collision_filter import CompiledCollisionFilter

if TYPE_CHECKING:
    from pxr import Usd

_CONVEX_PART_SUFFIX = re.compile(r"^(?P<label>.+)_convex_[1-9][0-9]*$")

ColliderShapeMap = dict[str, tuple[int, ...]]
"""Authored collider path to every Newton shape generated for that collider."""

CollisionEndpointShapeMap = dict[str, tuple[int, ...]]
"""Authored collider or rigid-body endpoint to its directly owned Newton shapes."""

FilteredPathPairs = frozenset[tuple[str, str]]
"""Canonical USD paths connected by ``physics:filteredPairs``."""


def authored_filtered_path_pairs(stage: Usd.Stage) -> FilteredPathPairs:
    """Snapshot every composed ``physics:filteredPairs`` relationship on *stage*."""
    from pxr import Usd  # noqa: PLC0415

    pairs = set()
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        relationship = prim.GetRelationship("physics:filteredPairs")
        if not relationship:
            continue
        source = str(prim.GetPath())
        for target in map(str, relationship.GetTargets()):
            if source != target:
                pairs.add((source, target) if source < target else (target, source))
    return frozenset(pairs)


def collider_shape_map(
    builder: ModelBuilder,
    path_shape_map: Mapping[str, int],
    shape_range: range | None = None,
) -> ColliderShapeMap:
    """Map authored collider paths to all colliding shapes created by one import.

    Newton's USD importer maps an authored path only to the first generated shape.
    Convex decomposition can add more colliding shapes with ``_convex_N`` labels;
    this function folds those pieces back into the authored path. Colliding shapes
    created by schema resolvers are retained by their absolute labels when the
    importer did not include them in ``path_shape_map``.

    Args:
        builder: Builder populated by the USD import.
        path_shape_map: Importer's authored-path to primary-shape mapping.
        shape_range: Shape indices produced by this import. Defaults to all shapes.

    Returns:
        Authored collider paths mapped to all of their colliding shape indices.
    """
    indices = shape_range if shape_range is not None else range(builder.shape_count)
    candidate_indices = set(indices)
    result: dict[str, list[int]] = defaultdict(list)
    claimed: set[int] = set()
    primary_labels: dict[str, str] = {}

    for path, index in path_shape_map.items():
        if index not in candidate_indices or not _shape_collides(builder, index):
            continue
        result[path].append(index)
        claimed.add(index)
        label = builder.shape_label[index]
        if isinstance(label, str):
            primary_labels[label] = path

    for index in sorted(candidate_indices):
        if index in claimed or not _shape_collides(builder, index):
            continue
        label = builder.shape_label[index]
        if not isinstance(label, str) or not label.startswith("/"):
            continue
        match = _CONVEX_PART_SUFFIX.fullmatch(label)
        path = primary_labels.get(match.group("label")) if match is not None else None
        if path is not None:
            primary = result[path][0]
            if builder.shape_body[index] == builder.shape_body[primary]:
                result[path].append(index)
                claimed.add(index)

    for index in sorted(candidate_indices - claimed):
        if not _shape_collides(builder, index):
            continue
        label = builder.shape_label[index]
        if isinstance(label, str) and label.startswith("/"):
            result[label].append(index)

    return {path: tuple(shape_indices) for path, shape_indices in result.items()}


def merge_collider_shape_maps(*shape_maps: Mapping[str, Sequence[int]]) -> ColliderShapeMap:
    """Merge collider maps while preserving each path's unique shape order."""
    merged: ColliderShapeMap = {}
    for shape_map in shape_maps:
        _merge_collider_shape_map_into(merged, shape_map)
    return merged


def _merge_collider_shape_map_into(
    target: ColliderShapeMap,
    incoming: Mapping[str, Sequence[int]],
) -> None:
    """Merge one shape map without copying unrelated accumulated paths."""
    for path, indices in incoming.items():
        existing = target.get(path, ())
        seen = set(existing)
        target[path] = (*existing, *(index for index in indices if index not in seen))


def collision_endpoint_shape_map(
    builder: ModelBuilder,
    collider_shapes: Mapping[str, Sequence[int]],
    path_body_map: Mapping[str, int],
    shape_range: range | None = None,
    articulation_paths: Iterable[str] = (),
) -> CollisionEndpointShapeMap:
    """Map collider, rigid-body, and articulation endpoints to their colliding shapes."""
    candidate_indices = set(shape_range if shape_range is not None else range(builder.shape_count))
    articulation_paths = set(articulation_paths)
    result = {path: list(indices) for path, indices in collider_shapes.items()}
    for path, body_index in path_body_map.items():
        owned_shapes = (
            index
            for index in builder.body_shapes.get(body_index, ())
            if index in candidate_indices and _shape_collides(builder, index)
        )
        existing = result.setdefault(path, [])
        seen = set(existing)
        existing.extend(index for index in owned_shapes if index not in seen)
    for articulation_index, path in enumerate(builder.articulation_label):
        if path not in articulation_paths:
            continue
        joint_start = builder.articulation_start[articulation_index]
        joint_end = builder.articulation_end[articulation_index]
        body_indices = {
            body
            for joint_index in range(joint_start, joint_end)
            for body in (builder.joint_parent[joint_index], builder.joint_child[joint_index])
            if body >= 0
        }
        owned_shapes = (
            shape
            for body in body_indices
            for shape in builder.body_shapes.get(body, ())
            if shape in candidate_indices and _shape_collides(builder, shape)
        )
        existing = result.setdefault(path, [])
        seen = set(existing)
        existing.extend(index for index in owned_shapes if index not in seen)
    return {path: tuple(indices) for path, indices in result.items()}


class _CollisionFilterCompiler:
    """Expand path-group policy into canonical exact shape pairs."""

    def __init__(self, cfg: CollisionFilterCfg, env_template: str):
        self._policy = CompiledCollisionFilter(cfg, env_template)

    def pairs(
        self,
        collider_shapes: Mapping[str, Sequence[int]],
        shape_worlds: Sequence[int] | None = None,
        universe: Iterable[int] | None = None,
        shape_owners: Mapping[int, int] | None = None,
        preapplied_owners: set[int] | None = None,
    ) -> frozenset[tuple[int, int]]:
        """Return exact deny pairs for one collider universe."""
        if universe is None:
            universe = (index for indices in collider_shapes.values() for index in indices)
        universe = set(universe)
        memberships_by_shape: dict[int, set[str]] = defaultdict(set)
        for path, shape_indices in collider_shapes.items():
            memberships = self._policy.memberships(path)
            for shape_index in shape_indices:
                if shape_index in universe:
                    memberships_by_shape[shape_index].update(memberships)

        preapplied_owners = preapplied_owners or set()
        shapes_by_profile: dict[tuple[int | None, tuple[str, ...], int | None], set[int]] = defaultdict(set)
        for shape_index in sorted(universe):
            memberships = memberships_by_shape[shape_index]
            group_profile = tuple(name for name in self._policy.groups if name in memberships)
            collision_world = None
            if shape_worlds is not None and int(shape_worlds[shape_index]) >= 0:
                collision_world = int(shape_worlds[shape_index])
            owner = None if shape_owners is None else shape_owners.get(shape_index)
            if owner not in preapplied_owners:
                owner = None
            shapes_by_profile[(collision_world, group_profile, owner)].add(shape_index)

        pairs: set[tuple[int, int]] = set()
        profiles = tuple(shapes_by_profile.items())
        indices_by_world: dict[int | None, list[int]] = defaultdict(list)
        for index, ((collision_world, _, _), _) in enumerate(profiles):
            indices_by_world[collision_world].append(index)
        global_indices = indices_by_world.get(None, [])
        for index, ((collision_world, first_memberships, first_owner), first_shapes) in enumerate(profiles):
            candidates = (
                range(index, len(profiles))
                if collision_world is None
                else (*global_indices, *indices_by_world[collision_world])
            )
            for second_index in candidates:
                if second_index < index:
                    continue
                (_, second_memberships, second_owner), second_shapes = profiles[second_index]
                if first_owner is not None and first_owner == second_owner:
                    continue
                if not self._policy.filters(first_memberships, second_memberships):
                    continue
                for shape_a in first_shapes:
                    for shape_b in second_shapes:
                        if shape_a == shape_b:
                            continue
                        pairs.add((min(shape_a, shape_b), max(shape_a, shape_b)))
        return frozenset(pairs)


class NewtonCollisionFilter:
    """Apply a collision policy around Newton's prototype replication seam.

    Source-local filters that are identical in every selected environment are
    authored on the source builder so :meth:`ModelBuilder.replicate` stores them
    compactly. Environment-specific and cross-source/global pairs are added to
    the assembled builder, still before model finalization.
    """

    def __init__(
        self,
        cfg: CollisionFilterCfg | None,
        sources: Sequence[str],
        destinations: Sequence[str],
        env_ids: np.ndarray,
        mapping: np.ndarray,
        env_template: str,
        authored_filtered_pairs: FilteredPathPairs = frozenset(),
    ):
        self._compiler = None if cfg is None or not cfg.groups else _CollisionFilterCompiler(cfg, env_template)
        self._sources = tuple(sources)
        self._destinations = tuple(destinations)
        self._env_ids = env_ids
        self._mapping = mapping
        self._authored_filtered_pairs = authored_filtered_pairs
        self._residual_authored_pairs = authored_filtered_pairs
        self._global_shapes: ColliderShapeMap = {}
        self._global_endpoint_shapes: CollisionEndpointShapeMap = {}
        self._global_shape_indices: frozenset[int] = frozenset()
        self._source_shapes: dict[str, ColliderShapeMap] = {}
        self._source_endpoint_shapes: dict[str, CollisionEndpointShapeMap] = {}
        self._source_builders: dict[str, ModelBuilder] = {}
        self._source_existing_pairs: dict[str, frozenset[tuple[int, int]]] = {}
        self._preapplied_rows: set[int] = set()

    def prepare(
        self,
        builder: ModelBuilder,
        global_shapes: ColliderShapeMap,
        source_builders: Mapping[str, ModelBuilder],
        source_shapes: Mapping[str, ColliderShapeMap],
        global_endpoint_shapes: CollisionEndpointShapeMap | None = None,
        source_endpoint_shapes: Mapping[str, CollisionEndpointShapeMap] | None = None,
    ) -> None:
        """Author reusable filters before source builders are replicated."""
        self._global_shapes = global_shapes
        self._global_endpoint_shapes = global_endpoint_shapes or global_shapes
        self._global_shape_indices = frozenset(_colliding_shape_indices(builder))
        self._source_builders = dict(source_builders)
        self._source_shapes = dict(source_shapes)
        self._source_endpoint_shapes = dict(source_endpoint_shapes or source_shapes)
        preapplied_path_pairs = {
            pair
            for pair in self._authored_filtered_pairs
            if self._global_endpoint_shapes.get(pair[0]) and self._global_endpoint_shapes.get(pair[1])
        }

        global_pairs = (
            frozenset()
            if self._compiler is None
            else self._compiler.pairs(global_shapes, universe=self._global_shape_indices)
        )
        global_pairs |= _shape_pairs_for_path_pairs(self._global_endpoint_shapes, preapplied_path_pairs)
        _add_unique_pairs(builder, global_pairs, _existing_pairs(builder))

        rows_by_source: dict[str, list[int]] = defaultdict(list)
        for row, source in enumerate(self._sources):
            rows_by_source[source].append(row)

        for source, rows in rows_by_source.items():
            source_builder = source_builders[source]
            existing_pairs = _existing_pairs(source_builder)
            templates: list[frozenset[tuple[int, int]]] = []
            if self._compiler is not None:
                for row in rows:
                    for column in np.flatnonzero(self._mapping[row]):
                        concrete_shapes = _rebase_collider_paths(
                            source_shapes[source],
                            source,
                            self._destinations[row].format(int(self._env_ids[column])),
                        )
                        templates.append(
                            self._compiler.pairs(
                                concrete_shapes,
                                universe=_colliding_shape_indices(source_builder),
                            )
                        )

            if templates and all(template == templates[0] for template in templates[1:]):
                _add_unique_pairs(source_builder, templates[0], existing_pairs)
                self._preapplied_rows.update(rows)
            source_path_pairs = {
                pair
                for pair in self._authored_filtered_pairs
                if self._source_endpoint_shapes[source].get(pair[0])
                and self._source_endpoint_shapes[source].get(pair[1])
            }
            _add_unique_pairs(
                source_builder,
                _shape_pairs_for_path_pairs(self._source_endpoint_shapes[source], source_path_pairs),
                _existing_pairs(source_builder),
            )
            preapplied_path_pairs.update(source_path_pairs)
            self._source_existing_pairs[source] = _existing_pairs(source_builder)
        self._residual_authored_pairs = self._authored_filtered_pairs - preapplied_path_pairs

    def apply_to_replicated_builder(
        self,
        builder: ModelBuilder,
        source_shape_offsets: Mapping[tuple[int, int], int],
    ) -> None:
        """Author environment-specific and cross-owner filters after replication."""
        if self._compiler is None and not self._residual_authored_pairs:
            return

        final_shapes = dict(self._global_shapes)
        final_endpoint_shapes = dict(self._global_endpoint_shapes)
        origins: dict[int, tuple[int, int, int]] = {}
        for row, source in enumerate(self._sources):
            source_builder = self._source_builders[source]
            for column in np.flatnonzero(self._mapping[row]):
                column = int(column)
                destination = self._destinations[row].format(int(self._env_ids[column]))
                offset = source_shape_offsets[row, column]
                local_to_final = {
                    local_index: offset + local_index for local_index in range(source_builder.shape_count)
                }
                for local_index in _colliding_shape_indices(source_builder):
                    final_index = local_to_final[local_index]
                    origins[final_index] = (row, column, local_index)

                concrete_shapes = {
                    _rebase_path(path, source, destination): tuple(local_to_final[index] for index in indices)
                    for path, indices in self._source_shapes[source].items()
                }
                _merge_collider_shape_map_into(final_shapes, concrete_shapes)
                concrete_endpoint_shapes = {
                    _rebase_path(path, source, destination): tuple(local_to_final[index] for index in indices)
                    for path, indices in self._source_endpoint_shapes[source].items()
                }
                _merge_collider_shape_map_into(final_endpoint_shapes, concrete_endpoint_shapes)

        mapped_shapes = {index for indices in final_shapes.values() for index in indices}
        supplement = {
            path: tuple(index for index in indices if index not in mapped_shapes)
            for path, indices in collider_shape_map(builder, {}).items()
        }
        final_shapes = merge_collider_shape_maps(
            final_shapes, {path: indices for path, indices in supplement.items() if indices}
        )
        final_endpoint_shapes = merge_collider_shape_maps(
            final_endpoint_shapes, {path: indices for path, indices in supplement.items() if indices}
        )

        policy_pairs = (
            frozenset()
            if self._compiler is None
            else self._compiler.pairs(
                final_shapes,
                builder.shape_world,
                universe=_colliding_shape_indices(builder),
                shape_owners={shape: origin[0] for shape, origin in origins.items()},
                preapplied_owners=self._preapplied_rows,
            )
        )
        authored_pairs = self._authored_shape_pairs(final_endpoint_shapes, builder.shape_world)
        new_pairs = set()
        for shape_a, shape_b in policy_pairs | authored_pairs:
            if shape_a in self._global_shape_indices and shape_b in self._global_shape_indices:
                continue
            origin_a = origins.get(shape_a)
            origin_b = origins.get(shape_b)
            if origin_a is not None and origin_b is not None and origin_a[:2] == origin_b[:2]:
                row = origin_a[0]
                local_pair = (min(origin_a[2], origin_b[2]), max(origin_a[2], origin_b[2]))
                if local_pair in self._source_existing_pairs[self._sources[row]]:
                    continue
            new_pairs.add((shape_a, shape_b))
        _add_unique_pairs(builder, new_pairs)

    def _authored_shape_pairs(
        self,
        collider_shapes: Mapping[str, Sequence[int]],
        shape_worlds: Sequence[int] | None = None,
    ) -> frozenset[tuple[int, int]]:
        endpoint_instances = {
            path: self._path_instances(path) for pair in self._residual_authored_pairs for path in pair
        }
        path_pairs = set()
        for first, second in self._residual_authored_pairs:
            first_instances = endpoint_instances[first]
            second_instances = endpoint_instances[second]
            if None in first_instances:
                worlds = second_instances
            elif None in second_instances:
                worlds = first_instances
            else:
                worlds = first_instances.keys() & second_instances.keys()
            for world in worlds:
                first_paths = first_instances.get(None, first_instances.get(world, ()))
                second_paths = second_instances.get(None, second_instances.get(world, ()))
                path_pairs.update((first_path, second_path) for first_path in first_paths for second_path in second_paths)
        return _shape_pairs_for_path_pairs(collider_shapes, path_pairs, shape_worlds)

    def _path_instances(self, path: str) -> dict[int | None, tuple[str, ...]]:
        instances: dict[int, set[str]] = defaultdict(set)
        for row, source in enumerate(self._sources):
            if clone_path.relative_to(path, source.rstrip("/") or "/") is None:
                continue
            for column in np.flatnonzero(self._mapping[row]):
                column = int(column)
                destination = self._destinations[row].format(int(self._env_ids[column]))
                instances[column].add(_rebase_path(path, source, destination))
        if not instances:
            return {None: (path,)}
        return {world: tuple(sorted(paths)) for world, paths in instances.items()}


def _shape_pairs_for_path_pairs(
    endpoint_shapes: Mapping[str, Sequence[int]],
    path_pairs: Iterable[tuple[str, str]],
    shape_worlds: Sequence[int] | None = None,
) -> frozenset[tuple[int, int]]:
    """Resolve collider/body endpoints to exact native shape pairs."""
    pairs = set()
    for first, second in path_pairs:
        for shape_a in endpoint_shapes.get(first, ()):
            for shape_b in endpoint_shapes.get(second, ()):
                if shape_a == shape_b or not _same_collision_world(shape_a, shape_b, shape_worlds):
                    continue
                pairs.add((min(shape_a, shape_b), max(shape_a, shape_b)))
    return frozenset(pairs)


def _shape_collides(builder: ModelBuilder, index: int) -> bool:
    return bool(builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES)


def _colliding_shape_indices(builder: ModelBuilder) -> tuple[int, ...]:
    return tuple(index for index in range(builder.shape_count) if _shape_collides(builder, index))


def _same_collision_world(shape_a: int, shape_b: int, shape_worlds: Sequence[int] | None) -> bool:
    if shape_worlds is None:
        return True
    world_a, world_b = int(shape_worlds[shape_a]), int(shape_worlds[shape_b])
    return world_a < 0 or world_b < 0 or world_a == world_b


def _rebase_collider_paths(collider_shapes: ColliderShapeMap, source: str, destination: str) -> ColliderShapeMap:
    return {_rebase_path(path, source, destination): indices for path, indices in collider_shapes.items()}


def _rebase_path(path: str, source: str, destination: str) -> str:
    suffix = clone_path.relative_to(path, source.rstrip("/") or "/")
    if suffix is None:
        raise ValueError(f"Newton collider path {path!r} is outside clone source {source!r}.")
    return (destination.rstrip("/") or "/") + suffix


def _is_at_or_below(path: str, root: str) -> bool:
    root = root.rstrip("/") or "/"
    return path == root or path.startswith(root.rstrip("/") + "/")


def _existing_pairs(builder: ModelBuilder) -> frozenset[tuple[int, int]]:
    # Reading the public property materializes Newton's compact replicated blocks.
    pairs = builder._shape_collision_filter_pairs  # pyright: ignore[reportPrivateUsage]
    return frozenset((min(a, b), max(a, b)) for a, b in pairs)


def _add_unique_pairs(
    builder: ModelBuilder,
    pairs: Sequence[tuple[int, int]] | frozenset[tuple[int, int]] | set[tuple[int, int]],
    existing_pairs: frozenset[tuple[int, int]] = frozenset(),
) -> None:
    seen = set(existing_pairs)
    for shape_a, shape_b in sorted(pairs):
        pair = (min(shape_a, shape_b), max(shape_a, shape_b))
        if pair not in seen:
            builder.add_shape_collision_filter_pair(*pair)
            seen.add(pair)
