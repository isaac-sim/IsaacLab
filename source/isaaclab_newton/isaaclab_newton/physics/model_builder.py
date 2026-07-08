# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton :class:`~newton.ModelBuilder` subclass with a vectorized finalize step.

Newton's ``ModelBuilder.find_shape_contact_pairs`` enumerates candidate shape
pairs in a Python double loop and materializes the result from a list of tuples.
For scenes with many cloned worlds this is one of the largest finalize costs
(``O(worlds * shapes_per_world^2)`` Python iterations). This module reimplements
the pair search with numpy over the same candidate structure — global shapes
against everything, plus the upper triangle of each world — producing exactly
the same pairs in the same order.
"""

from __future__ import annotations

from itertools import chain

import numpy as np
import warp as wp
from newton import Model, ModelBuilder, ShapeFlags

# Numpy mirror of ``ModelBuilder.shape_collision_filter_pairs``, maintained by the
# batched replication path. Converting millions of filter tuples back to an array at
# finalize costs seconds; the mirror is used instead when its length still matches
# the list (any pairs appended outside the batched path invalidate it).
_FILTER_PAIRS_CACHE_ATTR = "_isaaclab_collision_filter_pairs_np"


def cache_collision_filter_pairs(builder: ModelBuilder, new_pairs: np.ndarray) -> None:
    """Record filter pairs just appended to ``builder.shape_collision_filter_pairs``.

    Args:
        builder: Builder whose filter-pair list was extended with ``new_pairs``.
        new_pairs: The appended pairs, shape ``[count, 2]``, int64. Must be called
            after the list extension so the cache covers the full list.
    """
    cached = getattr(builder, _FILTER_PAIRS_CACHE_ATTR, None)
    if cached is None:
        prior_count = len(builder.shape_collision_filter_pairs) - len(new_pairs)
        cached = np.asarray(builder.shape_collision_filter_pairs[:prior_count], dtype=np.int64).reshape(-1, 2)
    setattr(builder, _FILTER_PAIRS_CACHE_ATTR, np.concatenate((cached, new_pairs)))


def _filter_pairs_array(builder: ModelBuilder) -> np.ndarray:
    """Return the builder's filter pairs as an ``[count, 2]`` int64 array."""
    pairs = builder.shape_collision_filter_pairs
    cached = getattr(builder, _FILTER_PAIRS_CACHE_ATTR, None)
    if cached is not None and len(cached) == len(pairs):
        return cached
    flat = np.fromiter(chain.from_iterable(pairs), dtype=np.int64, count=2 * len(pairs))
    return flat.reshape(-1, 2)


# Metadata recorded by the batched replication path when all worlds were appended from
# identical piece sequences onto a world-less builder. Under that structure the
# per-world shape blocks (flags, groups, filter pairs) are identical modulo a constant
# index offset, so contact pairs can be computed for one world and tiled.
_HOMOGENEOUS_META_ATTR = "_isaaclab_homogeneous_worlds"


def mark_homogeneous_worlds(builder: ModelBuilder, world_count: int, prior_shape_count: int, prior_filter_count: int) -> None:
    """Record that the builder's worlds are identical copies laid out in blocks.

    Called by the batched replication path. The recorded totals invalidate the
    metadata if any shapes or filter pairs are appended afterwards.

    Args:
        builder: Builder whose worlds were just appended.
        world_count: Number of identical worlds appended.
        prior_shape_count: Shape count before the worlds were appended (all global).
        prior_filter_count: Filter-pair count before the worlds were appended.
    """
    setattr(
        builder,
        _HOMOGENEOUS_META_ATTR,
        {
            "world_count": world_count,
            "prior_shape_count": prior_shape_count,
            "prior_filter_count": prior_filter_count,
            "total_shape_count": len(builder.shape_flags),
            "total_filter_count": len(builder.shape_collision_filter_pairs),
        },
    )


def _find_pairs_homogeneous(builder: ModelBuilder, meta: dict) -> np.ndarray | None:
    """Compute contact pairs for one template world and tile them across all worlds.

    Requires the layout recorded by :func:`mark_homogeneous_worlds`: global shapes
    first, then ``world_count`` identical shape blocks. World layout, flags, and
    groups are re-verified numerically (cheap); the per-world periodicity of the
    filter pairs follows from the batched construction and the recorded totals.

    Returns:
        Pairs identical (values and order) to the general search, or ``None`` when
        any invariant does not hold (caller falls back to the general path).
    """
    num_shapes = len(builder.shape_flags)
    prior = meta["prior_shape_count"]
    world_count = meta["world_count"]
    if (
        meta["total_shape_count"] != num_shapes
        or meta["total_filter_count"] != len(builder.shape_collision_filter_pairs)
        or builder.world_count != world_count
        or world_count < 1
        or (num_shapes - prior) % world_count
    ):
        return None
    shapes_per_world = (num_shapes - prior) // world_count

    flags = np.asarray(builder.shape_flags, dtype=np.int64)
    worlds = np.asarray(builder.shape_world, dtype=np.int64)
    groups = np.asarray(builder.shape_collision_group, dtype=np.int64)

    # Verify the block layout and per-world periodicity of flags/groups.
    if not (worlds[:prior] == -1).all():
        return None
    if shapes_per_world:
        world_blocks = worlds[prior:].reshape(world_count, shapes_per_world)
        if not (world_blocks == np.arange(world_count)[:, None]).all():
            return None
        for arr in (flags, groups):
            blocks = arr[prior:].reshape(world_count, shapes_per_world)
            if not (blocks == blocks[0]).all():
                return None

    num_filters = len(builder.shape_collision_filter_pairs)
    prior_filters = meta["prior_filter_count"]
    if (num_filters - prior_filters) % world_count:
        return None
    filters_per_world = (num_filters - prior_filters) // world_count
    filter_pairs = _filter_pairs_array(builder)
    template_filters = filter_pairs[prior_filters : prior_filters + filters_per_world]
    if len(template_filters) and not (
        (template_filters >= prior) & (template_filters < prior + shapes_per_world)
    ).all():
        return None

    def keep_unfiltered(lo: np.ndarray, hi: np.ndarray, pairs: np.ndarray) -> np.ndarray | None:
        if not len(pairs):
            return None
        keys = np.minimum(pairs[:, 0], pairs[:, 1]) * num_shapes + np.maximum(pairs[:, 0], pairs[:, 1])
        keys.sort()
        candidate_keys = lo * num_shapes + hi
        positions = np.minimum(np.searchsorted(keys, candidate_keys), len(keys) - 1)
        return keys[positions] != candidate_keys

    def group_mask(group_a, group_b):
        return (group_a != 0) & (group_b != 0) & np.where(group_a > 0, (group_a == group_b) | (group_b < 0), group_a != group_b)

    world_offsets = np.arange(world_count, dtype=np.int64) * shapes_per_world

    # Template-world candidates (colliding shapes, upper triangle, group test, filters).
    template_colliding = np.flatnonzero(flags[prior : prior + shapes_per_world] & int(ShapeFlags.COLLIDE_SHAPES))
    template_groups = groups[prior + template_colliding]
    row, col = np.triu_indices(len(template_colliding), k=1)
    mask = group_mask(template_groups[row], template_groups[col])
    a0 = prior + template_colliding[row][mask]
    b0 = prior + template_colliding[col][mask]
    keep = keep_unfiltered(a0, b0, template_filters)
    if keep is not None:
        a0, b0 = a0[keep], b0[keep]

    blocks_a: list[np.ndarray] = []
    blocks_b: list[np.ndarray] = []

    # Global shapes pair with the later globals and with every world's block; the
    # construction guarantees no global-to-world filter pairs exist.
    global_colliding = np.flatnonzero(flags[:prior] & int(ShapeFlags.COLLIDE_SHAPES))
    global_filters = filter_pairs[:prior_filters]
    for i, g in enumerate(global_colliding.tolist()):
        later = global_colliding[i + 1 :]
        gg = later[group_mask(groups[g], groups[later])]
        keep = keep_unfiltered(np.full(len(gg), g, dtype=np.int64), gg, global_filters)
        if keep is not None:
            gg = gg[keep]
        gw_template = prior + template_colliding[group_mask(groups[g], template_groups)]
        gw = (gw_template[None, :] + world_offsets[:, None]).reshape(-1)
        blocks_a.append(np.full(len(gg) + len(gw), g, dtype=np.int64))
        blocks_b.append(np.concatenate((gg, gw)))

    # World blocks: tile the template pairs across all worlds (world-major order).
    blocks_a.append((a0[None, :] + world_offsets[:, None]).reshape(-1))
    blocks_b.append((b0[None, :] + world_offsets[:, None]).reshape(-1))

    first = np.concatenate(blocks_a)
    second = np.concatenate(blocks_b)
    return np.stack((first, second), axis=1).astype(np.int32)


def find_shape_contact_pairs_vectorized(builder: ModelBuilder, model: Model) -> None:
    """Vectorized drop-in for :meth:`newton.ModelBuilder.find_shape_contact_pairs`.

    Produces bit-identical ``model.shape_contact_pairs`` /
    ``model.shape_contact_pair_count`` (same pairs, same order) as Newton's
    reference implementation.

    Args:
        builder: Builder holding the accumulated shape and collision-filter data.
        model: Model that receives the contact-pair arrays.
    """
    meta = getattr(builder, _HOMOGENEOUS_META_ATTR, None)
    if meta is not None:
        pairs = _find_pairs_homogeneous(builder, meta)
        if pairs is not None:
            model.shape_contact_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)
            model.shape_contact_pair_count = len(pairs)
            return

    flags = np.asarray(builder.shape_flags, dtype=np.int64)
    colliding = np.flatnonzero(flags & int(ShapeFlags.COLLIDE_SHAPES))
    worlds = np.asarray(builder.shape_world, dtype=np.int64)[colliding]
    # Stable world sort mirrors the reference implementation's iteration order.
    order = np.argsort(worlds, kind="stable")
    shapes = colliding[order]
    worlds = worlds[order]
    groups = np.asarray(builder.shape_collision_group, dtype=np.int64)[shapes]

    def group_mask(group_a: np.ndarray | int, group_b: np.ndarray) -> np.ndarray:
        # Mirrors broad_phase_common.test_group_pair: group 0 collides with nothing;
        # positive groups collide with themselves and any negative group; negative
        # groups collide with any different group.
        return (group_a != 0) & (group_b != 0) & np.where(group_a > 0, (group_a == group_b) | (group_b < 0), group_a != group_b)

    blocks_a: list[np.ndarray] = []
    blocks_b: list[np.ndarray] = []

    # Global shapes (world -1) sort first and pair with every later shape; the
    # world test always passes for them.
    num_global = int(np.searchsorted(worlds, -1, side="right"))
    for i in range(num_global):
        candidates = shapes[i + 1 :][group_mask(int(groups[i]), groups[i + 1 :])]
        blocks_a.append(np.full(len(candidates), shapes[i], dtype=np.int64))
        blocks_b.append(candidates)

    # Non-global shapes only pair within their own world (the reference loop
    # breaks at the first different world thanks to the sort).
    if num_global < len(shapes):
        world_slice = worlds[num_global:]
        boundaries = np.flatnonzero(np.diff(world_slice)) + 1
        starts = np.concatenate(([0], boundaries)) + num_global
        ends = np.concatenate((boundaries, [len(world_slice)])) + num_global
        triu_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for start, end in zip(starts.tolist(), ends.tolist()):
            count = end - start
            if count < 2:
                continue
            tri = triu_cache.get(count)
            if tri is None:
                tri = np.triu_indices(count, k=1)
                triu_cache[count] = tri
            segment = shapes[start:end]
            segment_groups = groups[start:end]
            row, col = tri
            mask = group_mask(segment_groups[row], segment_groups[col])
            blocks_a.append(segment[row][mask])
            blocks_b.append(segment[col][mask])

    if blocks_a:
        first = np.concatenate(blocks_a)
        second = np.concatenate(blocks_b)
        lo = np.minimum(first, second)
        hi = np.maximum(first, second)
        if builder.shape_collision_filter_pairs:
            # Membership test against the filter pairs via sorted encoded keys.
            # Duplicates in the list are harmless for searchsorted membership, so the
            # builder list (or its numpy mirror) is used instead of the model's set —
            # converting millions of tuples at finalize would dominate the search.
            shape_count = len(builder.shape_flags)
            filter_pairs = _filter_pairs_array(builder)
            filter_keys = (
                np.minimum(filter_pairs[:, 0], filter_pairs[:, 1]) * shape_count
                + np.maximum(filter_pairs[:, 0], filter_pairs[:, 1])
            )
            filter_keys.sort()
            keys = lo * shape_count + hi
            positions = np.minimum(np.searchsorted(filter_keys, keys), len(filter_keys) - 1)
            keep = filter_keys[positions] != keys
            lo = lo[keep]
            hi = hi[keep]
        pairs = np.stack((lo, hi), axis=1).astype(np.int32)
    else:
        pairs = np.empty((0, 2), dtype=np.int32)

    model.shape_contact_pairs = wp.array(pairs, dtype=wp.vec2i, device=model.device)
    model.shape_contact_pair_count = len(pairs)


class NewtonModelBuilder(ModelBuilder):
    """:class:`newton.ModelBuilder` with a vectorized contact-pair finalize step.

    Behaves identically to the base class; only
    :meth:`~newton.ModelBuilder.find_shape_contact_pairs` is replaced with
    :func:`find_shape_contact_pairs_vectorized`.
    """

    def find_shape_contact_pairs(self, model: Model) -> None:
        """Identify and store all potential shape contact pairs for collision detection."""
        find_shape_contact_pairs_vectorized(self, model)
