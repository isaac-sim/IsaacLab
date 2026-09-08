# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compile backend-neutral collision groups into Newton shape filter pairs."""

from __future__ import annotations

import re
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
"""Directed USD source-to-target paths connected by ``physics:filteredPairs``."""


@dataclass(frozen=True)
class AuthoredCollisionFilterSnapshot:
    """Authored collision metadata needed after Newton partitions the USD import."""

    filtered_pairs: FilteredPathPairs = frozenset()
    articulation_paths: frozenset[str] = frozenset()
    deformable_owner_simulations: tuple[tuple[str, tuple[str, ...]], ...] = ()
    unsupported_endpoint_reasons: tuple[tuple[str, str], ...] = ()


def snapshot_authored_collision_filter(stage: Usd.Stage, roots: Iterable[str]) -> AuthoredCollisionFilterSnapshot:
    """Snapshot collision metadata below clone sources and explicitly imported globals.

    Relationship targets remain unrestricted: only the prim carrying the relationship
    must be below a selected root. This avoids walking an already expanded destination
    stage while preserving cross-source and source-to-global relationships.
    """
    from pxr import Usd, UsdPhysics  # noqa: PLC0415

    pairs = set()
    articulation_paths = set()
    deformable_body_paths = set()
    simulation_families: dict[str, str] = {}
    for root_path in _minimal_stage_roots(roots):
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            path = str(prim.GetPath())
            relationship = prim.GetRelationship("physics:filteredPairs")
            if relationship:
                pairs.update((path, target) for target in map(str, relationship.GetTargets()) if path != target)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                articulation_paths.add(path)
            schemas = prim.GetPrimTypeInfo().GetAppliedAPISchemas()
            if "PhysicsDeformableBodyAPI" in schemas:
                deformable_body_paths.add(path)
            if "PhysicsCurvesDeformableSimAPI" in schemas:
                simulation_families[path] = "cable"
            elif "PhysicsSurfaceDeformableSimAPI" in schemas:
                simulation_families[path] = "cloth"
            elif "PhysicsVolumeDeformableSimAPI" in schemas:
                simulation_families[path] = "volume"

    owner_simulations: dict[str, list[str]] = defaultdict(list)
    for simulation_path in simulation_families:
        owners = [path for path in deformable_body_paths if _is_at_or_below(simulation_path, path)]
        if owners:
            owner_simulations[max(owners, key=len)].append(simulation_path)

    unsupported_reasons = {}
    for path, family in simulation_families.items():
        if family != "cable":
            unsupported_reasons[path] = f"{family} particle deformables cannot be represented by shape filter pairs"
    for owner, simulations in owner_simulations.items():
        unsupported = [simulation_families[path] for path in simulations if simulation_families[path] != "cable"]
        if unsupported:
            families = ", ".join(sorted(set(unsupported)))
            unsupported_reasons[owner] = (
                f"its {families} particle simulation cannot be represented by shape filter pairs"
            )

    for path in {endpoint for pair in pairs for endpoint in pair}:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            unsupported_reasons[path] = "the path does not exist"

    return AuthoredCollisionFilterSnapshot(
        filtered_pairs=frozenset(pairs),
        articulation_paths=frozenset(articulation_paths),
        deformable_owner_simulations=tuple(
            (owner, tuple(sorted(simulations))) for owner, simulations in sorted(owner_simulations.items())
        ),
        unsupported_endpoint_reasons=tuple(sorted(unsupported_reasons.items())),
    )


@contextmanager
def defer_importer_filtered_pairs_warnings(enabled: bool):
    """Suppress partition-local importer diagnostics until final assembly resolves them."""
    if not enabled:
        yield
        return
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*: physics:filteredPairs was not imported because .*",
            category=UserWarning,
        )
        yield


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
    path_cable_map: Mapping[str, tuple[Sequence[int], Sequence[int]]] | None = None,
    deformable_owner_simulations: Mapping[str, Sequence[str]] | None = None,
) -> CollisionEndpointShapeMap:
    """Map supported authored physics endpoints to their colliding Newton shapes."""
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
    for path, (body_indices, _) in (path_cable_map or {}).items():
        owned_shapes = (
            shape
            for body in body_indices
            for shape in builder.body_shapes.get(body, ())
            if shape in candidate_indices and _shape_collides(builder, shape)
        )
        existing = result.setdefault(path, [])
        seen = set(existing)
        existing.extend(index for index in owned_shapes if index not in seen)
    for owner, simulations in (deformable_owner_simulations or {}).items():
        owned_shapes = tuple(shape for simulation in simulations for shape in result.get(simulation, ()))
        if not owned_shapes:
            continue
        existing = result.setdefault(owner, [])
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

    def membership_signature(
        self,
        collider_shapes: Mapping[str, Sequence[int]],
        universe: Iterable[int],
    ) -> tuple[tuple[int, tuple[str, ...]], ...]:
        """Return the local shape memberships that fully determine policy pairs."""
        selected = set(universe)
        memberships_by_shape: dict[int, set[str]] = defaultdict(set)
        for path, shape_indices in collider_shapes.items():
            memberships = self._policy.memberships(path)
            for shape_index in shape_indices:
                if shape_index in selected:
                    memberships_by_shape[shape_index].update(memberships)
        return tuple(
            (shape, tuple(name for name in self._policy.groups if name in memberships_by_shape[shape]))
            for shape in sorted(selected)
        )


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
        authored: AuthoredCollisionFilterSnapshot = AuthoredCollisionFilterSnapshot(),
    ):
        self._compiler = None if cfg is None or not cfg.groups else _CollisionFilterCompiler(cfg, env_template)
        self._sources = tuple(sources)
        self._destinations = tuple(destinations)
        self._env_ids = env_ids
        self._mapping = mapping
        self._env_template = env_template
        self._prototype_env_roots = frozenset(
            parts[0] for source in sources if (parts := _environment_path_parts(source, env_template)) is not None
        )
        self._authored_filtered_pairs = authored.filtered_pairs
        self._residual_authored_pairs = authored.filtered_pairs
        self._unsupported_endpoint_reasons = dict(authored.unsupported_endpoint_reasons)
        self._global_shapes: ColliderShapeMap = {}
        self._global_endpoint_shapes: CollisionEndpointShapeMap = {}
        self._global_shape_indices: frozenset[int] = frozenset()
        self._source_shapes: dict[str, ColliderShapeMap] = {}
        self._source_endpoint_shapes: dict[str, CollisionEndpointShapeMap] = {}
        self._source_builders: dict[str, ModelBuilder] = {}
        self._source_existing_pairs: dict[str, frozenset[tuple[int, int]]] = {}
        self._source_colliding_shape_counts: dict[str, int] = {}
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
        self._source_colliding_shape_counts = {
            source: len(_colliding_shape_indices(source_builder)) for source, source_builder in source_builders.items()
        }
        preapplied_path_pairs = set()

        global_pairs = (
            frozenset()
            if self._compiler is None or not self._global_shape_indices
            else self._compiler.pairs(global_shapes, universe=self._global_shape_indices)
        )
        global_pairs |= _shape_pairs_for_path_pairs(self._global_endpoint_shapes, self._authored_filtered_pairs)
        _add_unique_pairs(builder, global_pairs, _existing_pairs(builder))

        rows_by_source: dict[str, list[int]] = defaultdict(list)
        for row, source in enumerate(self._sources):
            rows_by_source[source].append(row)

        for source, rows in rows_by_source.items():
            source_builder = source_builders[source]
            existing_pairs = _existing_pairs(source_builder)
            first_template = None
            homogeneous = True
            if self._compiler is not None:
                template_cache = {}
                universe = _colliding_shape_indices(source_builder)
                for row in rows:
                    for column in np.flatnonzero(self._mapping[row]):
                        concrete_shapes = _rebase_collider_paths(
                            source_shapes[source],
                            source,
                            self._destinations[row].format(int(self._env_ids[column])),
                        )
                        signature = self._compiler.membership_signature(concrete_shapes, universe)
                        template = template_cache.get(signature)
                        if template is None:
                            template = self._compiler.pairs(concrete_shapes, universe=universe)
                            template_cache[signature] = template
                        if first_template is None:
                            first_template = template
                        elif template != first_template:
                            homogeneous = False

            if first_template is not None and homogeneous:
                _add_unique_pairs(source_builder, first_template, existing_pairs)
                self._preapplied_rows.update(rows)
            source_path_pairs = {
                pair
                for pair in self._authored_filtered_pairs
                if _is_at_or_below(pair[0], source)
                and _is_at_or_below(pair[1], source)
                and not any(other != source and _is_at_or_below(other, pair[1]) for other in self._sources)
                and _path_pair_has_endpoints(self._source_endpoint_shapes[source], pair)
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
        if self._can_skip_final_policy_expansion(builder):
            self._warn_authored_pair_diagnostics({})
            return
        if self._compiler is None and not self._residual_authored_pairs:
            self._warn_authored_pair_diagnostics({})
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
        authored_pairs, unresolved_pairs = self._authored_shape_pairs(final_endpoint_shapes, builder.shape_world)
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
        _add_unique_pairs(builder, new_pairs, _existing_pairs(builder))
        self._warn_authored_pair_diagnostics(unresolved_pairs)

    def _can_skip_final_policy_expansion(self, builder: ModelBuilder) -> bool:
        if self._compiler is None or self._residual_authored_pairs or self._global_shape_indices:
            return False
        active_rows = {row for row in range(len(self._sources)) if np.any(self._mapping[row])}
        if len(active_rows) != 1 or not active_rows.issubset(self._preapplied_rows):
            return False
        expected_shapes = sum(
            self._source_colliding_shape_counts[self._sources[row]] * int(np.count_nonzero(self._mapping[row]))
            for row in active_rows
        )
        return expected_shapes == len(_colliding_shape_indices(builder))

    def _authored_shape_pairs(
        self,
        collider_shapes: Mapping[str, Sequence[int]],
        shape_worlds: Sequence[int] | None = None,
    ) -> tuple[frozenset[tuple[int, int]], dict[tuple[str, str], str]]:
        endpoint_instances = {
            path: self._path_instances(path) for pair in self._residual_authored_pairs for path in pair
        }
        pairs = set()
        unresolved = {}
        for first, second in self._residual_authored_pairs:
            first_instances = endpoint_instances[first]
            second_instances = endpoint_instances[second]
            if None in first_instances:
                worlds = second_instances
            elif None in second_instances:
                worlds = first_instances
            else:
                worlds = first_instances.keys() & second_instances.keys()
            concrete_pairs = []
            for world in worlds:
                first_paths = first_instances.get(None, first_instances.get(world, ()))
                second_paths = second_instances.get(None, second_instances.get(world, ()))
                concrete_pairs.extend(
                    (first_path, second_path) for first_path in first_paths for second_path in second_paths
                )
            if not concrete_pairs:
                continue
            relation_pairs = _shape_pairs_for_path_pairs(collider_shapes, concrete_pairs, shape_worlds)
            pairs.update(relation_pairs)
            if not relation_pairs and not any(
                _path_pair_has_endpoints(collider_shapes, pair) for pair in concrete_pairs
            ):
                unresolved[(first, second)] = self._unresolved_pair_reason(
                    first, second, collider_shapes, concrete_pairs
                )
        return frozenset(pairs), unresolved

    def _unresolved_pair_reason(
        self,
        source: str,
        target: str,
        endpoint_shapes: Mapping[str, Sequence[int]],
        concrete_pairs: Sequence[tuple[str, str]],
    ) -> str:
        source_resolved = any(endpoint_shapes.get(concrete_source) for concrete_source, _ in concrete_pairs)
        if not source_resolved:
            reason = self._unsupported_endpoint_reasons.get(source)
            return reason or "the source did not produce a supported collider, rigid body, articulation, or cable"
        target_resolved = any(
            _target_shape_indices(endpoint_shapes, concrete_target) for _, concrete_target in concrete_pairs
        )
        if not target_resolved:
            reasons = {
                reason for path, reason in self._unsupported_endpoint_reasons.items() if _is_at_or_below(path, target)
            }
            return next(iter(reasons)) if len(reasons) == 1 else "the target hierarchy produced no supported collider"
        return "the resolved endpoints have no common collision world"

    def _warn_authored_pair_diagnostics(self, unresolved: Mapping[tuple[str, str], str]) -> None:
        for source, target in sorted(self._authored_filtered_pairs):
            unsupported = []
            if source in self._unsupported_endpoint_reasons:
                unsupported.append(f"{source}: {self._unsupported_endpoint_reasons[source]}")
            unsupported.extend(
                f"{path}: {reason}"
                for path, reason in self._unsupported_endpoint_reasons.items()
                if path != source and _is_at_or_below(path, target)
            )
            reason = unresolved.get((source, target))
            if reason is not None:
                warnings.warn(
                    f"{source} -> {target}: physics:filteredPairs was not applied by Newton because {reason}.",
                    stacklevel=3,
                )
            elif unsupported:
                warnings.warn(
                    f"{source} -> {target}: physics:filteredPairs was only partially applied by Newton; "
                    + "; ".join(sorted(set(unsupported)))
                    + ".",
                    stacklevel=3,
                )

    def _path_instances(self, path: str) -> dict[int | None, tuple[str, ...]]:
        instances: dict[int, set[str]] = defaultdict(set)
        for row, source in enumerate(self._sources):
            if clone_path.relative_to(path, source.rstrip("/") or "/") is None:
                continue
            for column in np.flatnonzero(self._mapping[row]):
                column = int(column)
                destination = self._destinations[row].format(int(self._env_ids[column]))
                instances[column].add(_rebase_path(path, source, destination))
        if instances:
            return {world: tuple(sorted(paths)) for world, paths in instances.items()}
        prototype_root = next((root for root in self._prototype_env_roots if _is_at_or_below(path, root)), None)
        if prototype_root is not None:
            suffix = path[len(prototype_root) :]
            return {
                column: (self._env_template.format(int(env_id)) + suffix,)
                for column, env_id in enumerate(self._env_ids)
            }
        return {None: (path,)}


def _shape_pairs_for_path_pairs(
    endpoint_shapes: Mapping[str, Sequence[int]],
    path_pairs: Iterable[tuple[str, str]],
    shape_worlds: Sequence[int] | None = None,
) -> frozenset[tuple[int, int]]:
    """Resolve typed sources against all physics participants below each target."""
    pairs = set()
    for first, second in path_pairs:
        for shape_a in endpoint_shapes.get(first, ()):
            for shape_b in _target_shape_indices(endpoint_shapes, second):
                if shape_a == shape_b or not _same_collision_world(shape_a, shape_b, shape_worlds):
                    continue
                pairs.add((min(shape_a, shape_b), max(shape_a, shape_b)))
    return frozenset(pairs)


def _target_shape_indices(endpoint_shapes: Mapping[str, Sequence[int]], target: str) -> tuple[int, ...]:
    """Return unique shapes owned by physics endpoints at or below a relationship target."""
    return tuple(
        sorted({shape for path, shapes in endpoint_shapes.items() if _is_at_or_below(path, target) for shape in shapes})
    )


def _path_pair_has_endpoints(endpoint_shapes: Mapping[str, Sequence[int]], pair: tuple[str, str]) -> bool:
    return bool(endpoint_shapes.get(pair[0])) and bool(_target_shape_indices(endpoint_shapes, pair[1]))


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


def _minimal_stage_roots(roots: Iterable[str]) -> tuple[str, ...]:
    """Drop duplicate and nested roots before traversing authored policy metadata."""
    selected = []
    for path in sorted({path.rstrip("/") or "/" for path in roots}, key=lambda value: (value.count("/"), value)):
        if not any(_is_at_or_below(path, root) for root in selected):
            selected.append(path)
    return tuple(selected)


def _environment_path_parts(path: str, env_template: str) -> tuple[str, str] | None:
    """Split a concrete path into the environment root and its descendant suffix."""
    prefix, marker, suffix = env_template.partition("{}")
    if not marker:
        return None
    match = re.match(re.escape(prefix) + r"[^/]+" + re.escape(suffix), path)
    if match is None or (match.end() < len(path) and path[match.end()] != "/"):
        return None
    return path[: match.end()], path[match.end() :]


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
