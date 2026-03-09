# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contact filter resolution for replicated Newton scenes.

When scenes are built by replication (e.g. :func:`newton_replicate`), bodies and
shapes are laid out per world; worlds can be heterogeneous (e.g. object in envs 0–26,
banana in envs 26–128). Contact sensor patterns are resolved by matching in each
world's slice and concatenating global indices, so the result respects which
entities exist in which world.

This module owns:
  * pattern → index resolution (regex, fnmatch, per-world matching)
  * :class:`_ReplicatedContactSensor` — a replicated-kernel sensor for multi-world
    models that avoids the O(n_worlds²) setup cost of building a global shape-pair
    table (see class docstring for details)
  * the single public entry point :func:`build_contact_sensor`
"""

from __future__ import annotations

import fnmatch
import logging

import numpy as np
import warp as wp
from newton.sensors import SensorContact as NewtonContactSensor

from newton import Model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_fnmatch(expr: str | list[str] | None) -> str | list[str] | None:
    """Convert Isaac Lab regex expressions (``.*``) to fnmatch glob (``*``)."""
    if expr is None:
        return None
    if isinstance(expr, str):
        return expr.replace(".*", "*")
    return [p.replace(".*", "*") for p in expr]


def _normalize_for_labels(expr: str | list[str] | None, labels: list[str]) -> str | list[str] | None:
    """Strip leading path components from *expr* when labels are bare names.

    Model labels may be full USD paths (``/World/envs/env_0/Robot/base``) or bare
    names (``base``).  When the labels are bare names but the user expression
    contains slashes, we strip everything up to the last ``/``.
    """
    if expr is None or not labels:
        return expr
    label_has_paths = any("/" in lbl for lbl in labels)
    items = [expr] if isinstance(expr, str) else list(expr)
    expr_uses_paths = any("/" in p for p in items)
    if label_has_paths or not expr_uses_paths:
        return expr
    normalized = [p.rsplit("/", 1)[-1] for p in items]
    return normalized[0] if isinstance(expr, str) else normalized


# ---------------------------------------------------------------------------
# Per-world matching
# ---------------------------------------------------------------------------


def _match_labels_in_world(
    labels: list[str],
    pattern: str | list[str],
    world_id: int,
    world_start: np.ndarray,
) -> list[int]:
    """Match *pattern* in a single world's label slice and return global indices."""
    patterns = [pattern] if isinstance(pattern, str) else pattern
    start, end = int(world_start[world_id]), int(world_start[world_id + 1])
    return [start + i for i, lbl in enumerate(labels[start:end]) if any(fnmatch.fnmatch(lbl, p) for p in patterns)]


def _resolve_pattern(
    pattern: str | list[str] | None,
    labels: list[str],
    world_start: np.ndarray | None,
    world_count: int,
) -> list[int] | None:
    """Resolve a single fnmatch pattern to global indices, per-world when applicable."""
    if pattern is None:
        return None
    # Single-world models without begin_world()/end_world() have bogus world_start
    # arrays (e.g. [1, 1, 1]). Fall back to treating all labels as one world.
    if world_start is None or world_start.size < 2 or int(world_start[0]) >= len(labels):
        world_start = np.array([0, len(labels)])
        world_count = 1
    indices: list[int] = []
    for w in range(world_count):
        indices.extend(_match_labels_in_world(labels, pattern, w, world_start))
    return indices


# ---------------------------------------------------------------------------
# Replicated-kernel sensor
# ---------------------------------------------------------------------------
#
# Why this exists:
#
# Newton's SensorContact.__init__ builds a global shape-pair table via
# _assemble_sensor_mappings, which is O(n_sensing × n_counterparts).  When
# indices are replicated across N worlds (e.g. 4096 envs), that becomes
# O(N² × per_world²) — 70 s for 4096 worlds with a 4-body robot.
#
# _ReplicatedContactSensor avoids this by building a SensorContact template
# from world 0 only (fast), then using a custom Warp kernel at runtime that
# translates each contact's global shape indices to world-local indices and
# looks up the world-0 shape-pair table.
#
# Setup cost: O(per_world²) regardless of N.
# Runtime cost: same as Newton's kernel — O(contacts).
#
# The external interface (.shape, .sensing_objs, .counterparts,
# .reading_indices, .net_force, .update()) matches SensorContact exactly
# so callers don't need to know which implementation is used.
#
# Note: _bisect_shape_pairs duplicates Newton's bisect_shape_pairs because
# Warp kernels cannot call functions defined in other modules at compile time.


@wp.func
def _bisect_shape_pairs(
    shape_pairs_sorted: wp.array(dtype=wp.vec2i),
    n_shape_pairs: wp.int32,
    value: wp.vec2i,
) -> wp.int32:
    lo = wp.int32(0)
    hi = n_shape_pairs
    while lo < hi:
        mid = (lo + hi) // 2
        pair_mid = shape_pairs_sorted[mid]
        if pair_mid[0] < value[0] or (pair_mid[0] == value[0] and pair_mid[1] < value[1]):
            lo = mid + 1
        else:
            hi = mid
    return lo


@wp.kernel(enable_backward=False)
def _replicate_net_force_kernel(
    num_contacts: wp.array(dtype=wp.int32),
    contact_shape0: wp.array(dtype=wp.int32),
    contact_shape1: wp.array(dtype=wp.int32),
    contact_force: wp.array(dtype=wp.spatial_vector),
    shape_world: wp.array(dtype=wp.int32),
    shape_world_start: wp.array(dtype=wp.int32),
    sp_sorted_local: wp.array(dtype=wp.vec2i),
    num_sp: wp.int32,
    sp_ep: wp.array(dtype=wp.vec2i),
    sp_ep_offset: wp.array(dtype=wp.int32),
    sp_ep_count: wp.array(dtype=wp.int32),
    n_sensors_per_env: wp.int32,
    n_readings: wp.int32,
    net_force: wp.array(dtype=wp.vec3),
):
    """Scatter contact forces into per-world net_force using local shape-pair lookup."""
    con_idx = wp.tid()
    if con_idx >= num_contacts[0]:
        return

    shape0 = contact_shape0[con_idx]
    shape1 = contact_shape1[con_idx]
    if shape0 < 0 or shape1 < 0:
        return

    world_id = shape_world[shape0]
    if world_id < 0:
        world_id = shape_world[shape1]
    if world_id < 0:
        return

    base = shape_world_start[world_id]
    is_global0 = shape_world[shape0] < 0
    is_global1 = shape_world[shape1] < 0
    local0 = wp.where(is_global0, wp.int32(-1), shape0 - base)
    local1 = wp.where(is_global1, wp.int32(-1), shape1 - base)

    force = wp.spatial_top(contact_force[con_idx])
    world_row_offset = world_id * n_sensors_per_env * n_readings

    # Bilateral shape-pair lookup (skip when either shape is global;
    # global contacts are handled exclusively by the mono path below)
    if not is_global0 and not is_global1:
        smin = wp.min(local0, local1)
        smax = wp.max(local0, local1)
        normalized_pair = wp.vec2i(smin, smax)
        sp_flip = normalized_pair[0] != local0
        sp_ord = _bisect_shape_pairs(sp_sorted_local, num_sp, normalized_pair)

        if sp_ord < num_sp:
            if sp_sorted_local[sp_ord][0] == normalized_pair[0] and sp_sorted_local[sp_ord][1] == normalized_pair[1]:
                offset = sp_ep_offset[sp_ord]
                for i in range(sp_ep_count[sp_ord]):
                    ep = sp_ep[offset + i]
                    force_acc = ep[0]
                    flip = ep[1]
                    base_acc = world_row_offset + force_acc
                    wp.atomic_add(net_force, base_acc, wp.where(sp_flip != flip, -force, force))

    for i in range(2):
        local_s = wp.where(i == 0, local0, local1)
        mono_sp = wp.vec2i(-1, local_s)
        mono_ord = _bisect_shape_pairs(sp_sorted_local, num_sp, mono_sp)
        if mono_ord < num_sp:
            if sp_sorted_local[mono_ord][0] == -1 and sp_sorted_local[mono_ord][1] == local_s:
                force_acc = sp_ep[sp_ep_offset[mono_ord]][0]
                base_acc = world_row_offset + force_acc
                wp.atomic_add(net_force, base_acc, wp.where(bool(i), -force, force))


class _ReplicatedContactSensor:
    """Contact sensor for replicated (multi-world) models.

    Built from a world-0 ``SensorContact`` template.  At runtime, a custom
    Warp kernel translates each contact's global shape indices to world-local
    indices and looks up the world-0 shape-pair table.

    This avoids the O(n_worlds²) setup cost of ``SensorContact.__init__``
    while keeping O(contacts) runtime — identical to Newton's built-in kernel.
    """

    def __init__(
        self,
        template: NewtonContactSensor,
        n_worlds: int,
        shape_world: wp.array,
        shape_world_start: wp.array,
        body_world_start: wp.array,
        sp_sorted_local: wp.array,
        device,
    ):
        self._n_worlds = n_worlds
        self._shape_world = shape_world
        self._shape_world_start = shape_world_start
        self.device = device

        n_sensors_per_env = template.shape[0]
        n_readings = template.shape[1]
        self.shape = (n_worlds * n_sensors_per_env, n_readings)
        self.reading_indices = template.reading_indices
        self.counterparts = template.counterparts

        total = n_worlds * n_sensors_per_env * n_readings
        self._net_force = wp.zeros(total, dtype=wp.vec3, device=device)
        self.net_force = self._net_force.reshape(self.shape)

        body_start_np = body_world_start.numpy()
        shape_start_np = shape_world_start.numpy()
        self.sensing_objs = []
        for w in range(n_worlds):
            for i in range(n_sensors_per_env):
                idx0, kind = template.sensing_objs[i]
                is_body = kind == NewtonContactSensor.ObjectType.BODY
                start_np = body_start_np if is_body else shape_start_np
                self.sensing_objs.append((int(start_np[w]) + idx0 - int(start_np[0]), kind))

        self._sp_sorted_local = sp_sorted_local
        self._n_sp = template._n_shape_pairs
        self._n_sensors_per_env = n_sensors_per_env
        self._n_readings = n_readings
        self._sp_ep = template._sp_reading
        self._sp_ep_offset = template._sp_ep_offset
        self._sp_ep_count = template._sp_ep_count

    def update(self, state, contacts):
        """Update net_force from contacts using per-world local lookup."""
        if contacts.force is None:
            return
        self._net_force.zero_()
        wp.launch(
            _replicate_net_force_kernel,
            dim=contacts.rigid_contact_max,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.force,
                self._shape_world,
                self._shape_world_start,
                self._sp_sorted_local,
                self._n_sp,
                self._sp_ep,
                self._sp_ep_offset,
                self._sp_ep_count,
                self._n_sensors_per_env,
                self._n_readings,
            ],
            outputs=[self._net_force],
            device=self.device,
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_contact_sensor(
    model: Model,
    *,
    body_names_expr: str | list[str] | None = None,
    shape_names_expr: str | list[str] | None = None,
    contact_partners_body_expr: str | list[str] | None = None,
    contact_partners_shape_expr: str | list[str] | None = None,
    prune_noncolliding: bool = True,
    verbose: bool = False,
) -> NewtonContactSensor | _ReplicatedContactSensor:
    """Resolve patterns and build a contact sensor for the given model.

    This is the single entry point for all contact sensor construction.  It
    handles single-world and multi-world (replicated) models uniformly:

    1. Converts Isaac Lab regex expressions (``.*``) to fnmatch globs (``*``).
    2. Normalises path-based patterns when model labels are bare names.
    3. Resolves patterns to global ``list[int]`` indices (per-world aware).
    4. For multi-world models, builds a world-0 template and wraps it with
       :class:`_ReplicatedContactSensor` to avoid O(n_worlds²) setup cost.
    5. For single-world models, passes pre-resolved indices directly to
       :class:`newton.sensors.SensorContact`.

    Returns:
        A sensor with the standard ``SensorContact`` interface.
    """
    body_labels = list(model.body_label)
    shape_labels = list(model.shape_label)

    sensing_bodies = _normalize_for_labels(_to_fnmatch(body_names_expr), body_labels)
    sensing_shapes = _normalize_for_labels(_to_fnmatch(shape_names_expr), shape_labels)
    counter_bodies = _normalize_for_labels(_to_fnmatch(contact_partners_body_expr), body_labels)
    counter_shapes = _normalize_for_labels(_to_fnmatch(contact_partners_shape_expr), shape_labels)

    world_count = getattr(model, "world_count", 1)
    body_world_start_wp = getattr(model, "body_world_start", None)
    shape_world_start_wp = getattr(model, "shape_world_start", None)

    can_replicate = world_count > 1 and shape_world_start_wp is not None and body_world_start_wp is not None

    if can_replicate:
        return _build_replicated(
            model,
            body_labels,
            shape_labels,
            sensing_bodies,
            sensing_shapes,
            counter_bodies,
            counter_shapes,
            body_world_start_wp,
            shape_world_start_wp,
            world_count,
            verbose,
        )

    body_start_np = body_world_start_wp.numpy() if body_world_start_wp is not None else None
    shape_start_np = shape_world_start_wp.numpy() if shape_world_start_wp is not None else None

    return NewtonContactSensor(
        model,
        sensing_obj_bodies=_resolve_pattern(sensing_bodies, body_labels, body_start_np, world_count),
        sensing_obj_shapes=_resolve_pattern(sensing_shapes, shape_labels, shape_start_np, world_count),
        counterpart_bodies=_resolve_pattern(counter_bodies, body_labels, body_start_np, world_count),
        counterpart_shapes=_resolve_pattern(counter_shapes, shape_labels, shape_start_np, world_count),
        include_total=True,
        prune_noncolliding=prune_noncolliding,
        verbose=verbose,
    )


def _build_replicated(
    model: Model,
    body_labels: list[str],
    shape_labels: list[str],
    sensing_bodies,
    sensing_shapes,
    counter_bodies,
    counter_shapes,
    body_world_start_wp,
    shape_world_start_wp,
    world_count: int,
    verbose: bool,
) -> _ReplicatedContactSensor:
    """Build a replicated contact sensor from a world-0 template."""
    body_start = body_world_start_wp.numpy()
    shape_start = shape_world_start_wp.numpy()

    sb0 = _match_labels_in_world(body_labels, sensing_bodies, 0, body_start) if sensing_bodies else None
    ss0 = _match_labels_in_world(shape_labels, sensing_shapes, 0, shape_start) if sensing_shapes else None
    cb0 = _match_labels_in_world(body_labels, counter_bodies, 0, body_start) if counter_bodies else None
    cs0 = _match_labels_in_world(shape_labels, counter_shapes, 0, shape_start) if counter_shapes else None

    template = NewtonContactSensor(
        model,
        sensing_obj_bodies=sb0 if sensing_bodies else None,
        sensing_obj_shapes=ss0 if sensing_shapes else None,
        counterpart_bodies=cb0 if counter_bodies else None,
        counterpart_shapes=cs0 if counter_shapes else None,
        include_total=True,
        prune_noncolliding=False,
        verbose=verbose,
    )

    sp_np = template._sp_sorted.numpy()
    sw0 = int(shape_start[0])
    sp_local = np.empty_like(sp_np)
    for i in range(sp_np.shape[0]):
        a, b = int(sp_np[i][0]), int(sp_np[i][1])
        sp_local[i][0] = -1 if a == -1 else a - sw0
        sp_local[i][1] = -1 if b == -1 else b - sw0
    sp_sorted_local = wp.array(sp_local, dtype=wp.vec2i, device=model.device)

    return _ReplicatedContactSensor(
        template,
        world_count,
        model.shape_world,
        shape_world_start_wp,
        body_world_start_wp,
        sp_sorted_local,
        model.device,
    )
