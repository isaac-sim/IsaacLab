# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Batched Newton model replication for IsaacLab clone plans.

The legacy replication path (:func:`~isaaclab_newton.cloner.newton_clone_utils.replicate_builder_mapping`)
calls :meth:`newton.ModelBuilder.add_builder` once per environment and then rewrites
entity labels in a second pass over the whole builder
(:func:`~isaaclab_newton.cloner.newton_clone_utils.rename_builder_labels`). Both steps
are Python loops whose cost grows with the number of environments and dominate startup
time for large scenes.

This module provides :class:`BatchedModelBuilder`, which appends all replicated worlds
to a destination :class:`newton.ModelBuilder` in one vectorized pass, and
:func:`replicate_builder_mapping_batched`, a drop-in alternative to the legacy function
that consumes the same clone-plan mapping inputs and produces an equivalent builder
state. Final per-environment labels are written directly, so no rename pass is needed;
the Fabric body bindings that the legacy rename pass produced are returned instead.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from itertools import chain
from typing import Any

import numpy as np
import torch
import warp as wp
from newton import JointType, Model, ModelBuilder
from newton._src.sim.graph_coloring import combine_independent_particle_coloring

from isaaclab_newton.physics.model_builder import cache_collision_filter_pairs, mark_homogeneous_worlds

# Attributes that ``ModelBuilder.add_builder`` copies verbatim (no index remapping).
# Mirrors ``more_builder_attrs`` in newton's ``ModelBuilder.add_builder``, plus
# ``joint_target_q`` and ``shape_collision_group`` which it also extends verbatim.
# ``shape_transform``, ``joint_X_p``, and ``joint_q`` are extended verbatim first and
# then partially overwritten with transformed values (see ``_apply_root_transforms``).
_VERBATIM_ATTRS: tuple[str, ...] = (
    "body_inertia",
    "body_mass",
    "body_inv_inertia",
    "body_inv_mass",
    "body_com",
    "body_lock_inertia",
    "body_flags",
    "body_qd",
    "joint_type",
    "joint_enabled",
    "joint_collision_filter_parent",
    "joint_X_c",
    "joint_armature",
    "joint_axis",
    "joint_dof_dim",
    "joint_qd",
    "joint_cts",
    "joint_f",
    "joint_act",
    "joint_target_q",
    "joint_target_qd",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_ke",
    "joint_limit_kd",
    "joint_target_ke",
    "joint_target_kd",
    "joint_damping",
    "joint_target_mode",
    "joint_effort_limit",
    "joint_velocity_limit",
    "joint_friction",
    "shape_flags",
    "shape_type",
    "shape_scale",
    "shape_source",
    "shape_color",
    "shape_is_solid",
    "shape_margin",
    "shape_material_ke",
    "shape_material_kd",
    "shape_material_kf",
    "shape_material_ka",
    "shape_material_mu",
    "shape_material_restitution",
    "shape_material_mu_torsional",
    "shape_material_mu_rolling",
    "shape_material_kh",
    "shape_collision_radius",
    "shape_collision_group",
    "shape_gap",
    "shape_sdf_narrow_band_range",
    "shape_sdf_max_resolution",
    "shape_sdf_target_voxel_size",
    "shape_sdf_texture_format",
    "shape_sdf_padding",
    "shape_transform",
    "joint_X_p",
    "joint_q",
    "particle_qd",
    "particle_mass",
    "particle_radius",
    "particle_flags",
    "edge_rest_angle",
    "edge_rest_length",
    "edge_bending_properties",
    "spring_rest_length",
    "spring_stiffness",
    "spring_damping",
    "spring_control",
    "tri_poses",
    "tri_activations",
    "tri_materials",
    "tri_areas",
    "tet_poses",
    "tet_activations",
    "tet_materials",
    "constraint_mimic_coef0",
    "constraint_mimic_coef1",
    "constraint_mimic_enabled",
)

# Label attributes rewritten from the source root to the per-world destination root.
_LABEL_ATTRS: tuple[str, ...] = (
    "body_label",
    "joint_label",
    "shape_label",
    "articulation_label",
    "constraint_mimic_label",
)


# region float32 transform math (mirrors Warp's native formulas)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply quaternions (xyzw), mirroring Warp's float32 ``quat`` product."""
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bx + bw * ax + ay * bz - by * az,
            aw * by + bw * ay + az * bx - bz * ax,
            aw * bz + bw * az + ax * by - bx * ay,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        axis=-1,
    )


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors by quaternions (xyzw), mirroring Warp's float32 ``quat_rotate``."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    f2 = np.float32(2.0)
    c = f2 * qw * qw - np.float32(1.0)
    d = f2 * (qx * x + qy * y + qz * z)
    return np.stack(
        (
            x * c + qx * d + (qy * z - qz * y) * qw * f2,
            y * c + qy * d + (qz * x - qx * z) * qw * f2,
            z * c + qz * d + (qx * y - qy * x) * qw * f2,
        ),
        axis=-1,
    )


def _transform_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compose transforms ``a * b`` ([..., 7] as pos xyz + quat xyzw), mirroring Warp."""
    out = np.empty(np.broadcast_shapes(a.shape, b.shape), dtype=np.float32)
    out[..., :3] = _quat_rotate(a[..., 3:], b[..., :3]) + a[..., :3]
    out[..., 3:] = _quat_mul(a[..., 3:], b[..., 3:])
    return out


# endregion


def _label_suffixes(labels: Sequence[str], source_root: str) -> tuple[list[str | None], bool]:
    """Compute per-label suffixes below ``source_root``.

    Returns:
        Tuple of the suffix list (``None`` for labels not under the root, matching
        :func:`~isaaclab.cloner.cloner_utils.replace_path_prefix` semantics) and
        whether every label is renamable.
    """
    prefix = source_root + "/"
    cut = len(source_root)
    suffixes: list[str | None] = [
        lbl[cut:] if isinstance(lbl, str) and (lbl == source_root or lbl.startswith(prefix)) else None for lbl in labels
    ]
    return suffixes, all(s is not None for s in suffixes)


@dataclass
class _AttrTemplate:
    """Precomputed merge plan for one custom attribute of a source builder."""

    key: str
    attr: Any  # ModelBuilder.CustomAttribute
    is_string_freq: bool
    freq_key: str | None  # custom-frequency key for string frequencies
    index_freq: Any  # Model.AttributeFrequency for enum frequencies, else None
    ref: str | None  # references entity/frequency name, or "world", or None
    is_eq_target: bool
    keys: np.ndarray | None = None  # enum-frequency entity indices
    values: list[Any] = field(default_factory=list)  # raw values in key order
    int_values: np.ndarray | None = None  # fast path when all values are plain ints
    target_kinds: list[int] | None = None  # for the equality-target remap
    rename_suffixes: list[str | None] | None = None  # world-referenced string labels


class _SourceView:
    """Cached numpy views and merge metadata for one source :class:`newton.ModelBuilder`.

    Computed once per source and reused for every replicated world, so all per-world
    work reduces to array slicing and offset arithmetic.
    """

    def __init__(self, builder: ModelBuilder, source_root: str | None):
        self.builder = builder
        self.source_root = source_root

        self.counts: dict[str, int] = {
            "body": builder.body_count,
            "shape": builder.shape_count,
            "joint": builder.joint_count,
            "articulation": builder.articulation_count,
            "constraint_mimic": len(builder.constraint_mimic_joint0),
            "particle": builder.particle_count,
            "spring": builder.spring_count,
            "edge": builder.edge_count,
            "triangle": builder.tri_count,
            "tetrahedron": builder.tet_count,
            "joint_dof": builder.joint_dof_count,
            "joint_coord": builder.joint_coord_count,
            "joint_constraint": builder.joint_constraint_count,
        }
        self.freq_counts: dict[str, int] = dict(builder._custom_frequency_counts)

        self.body_q = np.array(builder.body_q, dtype=np.float32).reshape(-1, 7)
        self.shape_body = np.asarray(builder.shape_body, dtype=np.int64)
        static = np.flatnonzero(self.shape_body == -1)
        self.static_shape_idx = static
        shape_transform = np.array(builder.shape_transform, dtype=np.float32).reshape(-1, 7)
        self.static_shape_xf = shape_transform[static]

        self.joint_parent = np.asarray(builder.joint_parent, dtype=np.int64)
        self.joint_child = np.asarray(builder.joint_child, dtype=np.int64)
        self.joint_q_start = np.asarray(builder.joint_q_start, dtype=np.int64)
        self.joint_qd_start = np.asarray(builder.joint_qd_start, dtype=np.int64)
        self.joint_cts_start = np.asarray(builder.joint_cts_start, dtype=np.int64)
        self.joint_articulation = np.asarray(builder.joint_articulation, dtype=np.int64)
        self.articulation_start = np.asarray(builder.articulation_start, dtype=np.int64)
        self.articulation_end = np.asarray(builder.articulation_end, dtype=np.int64)
        self.mimic_joint0 = np.asarray(builder.constraint_mimic_joint0, dtype=np.int64)
        self.mimic_joint1 = np.asarray(builder.constraint_mimic_joint1, dtype=np.int64)
        self.filter_pairs = np.asarray(builder.shape_collision_filter_pairs, dtype=np.int64).reshape(-1, 2)

        joint_type = np.asarray(builder.joint_type, dtype=np.int64)
        free = joint_type == int(JointType.FREE)
        self.free_joint_q_start = self.joint_q_start[free]
        joint_X_p = np.array(builder.joint_X_p, dtype=np.float32).reshape(-1, 7)
        joint_q = np.asarray(builder.joint_q, dtype=np.float32)
        if len(self.free_joint_q_start):
            self.free_joint_q = np.stack([joint_q[qs : qs + 7] for qs in self.free_joint_q_start], axis=0)
        else:
            self.free_joint_q = np.empty((0, 7), dtype=np.float32)
        root_joints = np.flatnonzero((self.joint_parent == -1) & ~free)
        self.root_joint_idx = root_joints
        self.root_joint_xf = joint_X_p[root_joints]

        self.particle_q = np.array(builder.particle_q, dtype=np.float32).reshape(-1, 3)
        self.spring_indices = np.asarray(builder.spring_indices, dtype=np.int64)
        self.edge_indices = np.asarray(builder.edge_indices, dtype=np.int64).reshape(-1, 4)
        self.tri_indices = np.asarray(builder.tri_indices, dtype=np.int64).reshape(-1, 3)
        self.tet_indices = np.asarray(builder.tet_indices, dtype=np.int64).reshape(-1, 4)

        # body_shapes: -1 entries are appended to the destination's global list; other
        # bodies get their own entry, preserving the source dict order.
        self.global_body_shapes: list[int] = list(builder.body_shapes.get(-1, ()))
        self.body_shapes_items: list[tuple[int, list[int]]] = [
            (b, shapes) for b, shapes in builder.body_shapes.items() if b != -1
        ]

        self.labels: dict[str, tuple[list[str], list[str | None], bool]] = {}
        for attr in _LABEL_ATTRS:
            labels = getattr(builder, attr)
            if source_root is None:
                self.labels[attr] = (labels, [None] * len(labels), False)
            else:
                suffixes, all_renamable = _label_suffixes(labels, source_root)
                self.labels[attr] = (labels, suffixes, all_renamable)

        self.attr_templates = self._build_attr_templates()
        self.declared_in_destination = False

        # Local indices of shapes whose color must be recomputed from the destination
        # builder's palette (used by the injected env-root site shapes; USD-parsed
        # sources keep their colors verbatim, matching ``add_builder``).
        self.palette_color_shapes: list[int] = []

    def _build_attr_templates(self) -> list[_AttrTemplate]:
        builder = self.builder
        world_value_freqs = {
            attr.frequency for attr in builder.custom_attributes.values() if attr.references == "world"
        }
        templates = []
        for key, attr in builder.custom_attributes.items():
            is_string_freq = isinstance(attr.frequency, str)
            tmpl = _AttrTemplate(
                key=key,
                attr=attr,
                is_string_freq=is_string_freq,
                freq_key=attr.frequency if is_string_freq else None,
                index_freq=None if is_string_freq else attr.frequency,
                ref=attr.references,
                is_eq_target=(key == "mujoco:equality_constraint_target"),
            )
            if attr.values:
                if is_string_freq:
                    tmpl.values = list(attr.values)
                else:
                    tmpl.keys = np.fromiter(attr.values.keys(), dtype=np.int64, count=len(attr.values))
                    tmpl.values = list(attr.values.values())
                if tmpl.ref is not None and tmpl.ref != "world":
                    if all(type(v) is int for v in tmpl.values):
                        tmpl.int_values = np.asarray(tmpl.values, dtype=np.int64)
                if tmpl.is_eq_target:
                    kind_attr = builder.custom_attributes.get("mujoco:equality_constraint_target_kind")
                    kinds = []
                    kind_values = kind_attr.values if kind_attr is not None and kind_attr.values else []
                    for idx in range(len(tmpl.values)):
                        kind = 0
                        if idx < len(kind_values) and kind_values[idx] is not None:
                            try:
                                kind = int(kind_values[idx])
                            except (TypeError, ValueError):
                                kind = 0
                        kinds.append(kind)
                    tmpl.target_kinds = kinds
                # String-valued attributes whose frequency also carries a world reference
                # are entity labels that the legacy rename pass rewrites per world.
                if attr.dtype is str and attr.frequency in world_value_freqs and self.source_root is not None:
                    tmpl.rename_suffixes, _ = _label_suffixes(
                        [v if isinstance(v, str) else "" for v in tmpl.values], self.source_root
                    )
            templates.append(tmpl)
        return templates

    @property
    def gravity_vector(self) -> tuple[float, float, float]:
        up = self.builder.up_vector
        g = self.builder.gravity
        return (up[0] * g, up[1] * g, up[2] * g)


@dataclass
class _WorldPiece:
    """One source builder placed into one destination world."""

    view: _SourceView
    world: int
    xform: np.ndarray  # combined root transform [7], float32
    dest_root: str | None  # per-world label root, None keeps labels verbatim
    overrides_gravity: bool = True


@dataclass
class _BatchOffsets:
    """Per-piece entity start indices in the destination builder after a batch append."""

    body: np.ndarray
    shape: np.ndarray
    joint: np.ndarray


class BatchedModelBuilder:
    """Appends replicated worlds to a Newton :class:`~newton.ModelBuilder` in one batch.

    Produces the same builder state as calling :meth:`~newton.ModelBuilder.begin_world`,
    :meth:`~newton.ModelBuilder.add_builder`, and :meth:`~newton.ModelBuilder.end_world`
    once per world, but replaces the per-world Python loops with vectorized numpy
    operations over cached per-source arrays. Labels are written with their final
    per-world destination roots, so no post-hoc rename pass is required.

    Not supported (callers must fall back to the legacy path): per-world builder hooks
    that mutate the destination builder between worlds, and muscle data (which
    ``add_builder`` itself does not copy either).
    """

    def __init__(self, builder: ModelBuilder):
        """Initialize the batched builder.

        Args:
            builder: Destination builder that receives the replicated worlds. It may
                already contain global (world ``-1``) entities.
        """
        self.builder = builder
        self._views: dict[tuple[int, str | None], _SourceView] = {}

    def source_view(self, source: ModelBuilder, source_root: str | None) -> _SourceView:
        """Return the cached :class:`_SourceView` for a source builder and label root."""
        key = (id(source), source_root)
        view = self._views.get(key)
        if view is None:
            view = _SourceView(source, source_root)
            self._views[key] = view
        return view

    def append_worlds(
        self,
        pieces: list[_WorldPiece],
        world_count: int,
        world_gravity: list[tuple[float, float, float]],
    ) -> _BatchOffsets:
        """Append all pieces to the destination builder.

        Args:
            pieces: Pieces in final entity order (world-major, rows in mapping order).
            world_count: Number of new worlds to register.
            world_gravity: Gravity vector per new world.

        Returns:
            Per-piece entity start offsets for caller-side bookkeeping.
        """
        b = self.builder

        views = [p.view for p in pieces]
        num_pieces = len(pieces)
        piece_worlds = np.fromiter((p.world for p in pieces), dtype=np.int64, count=num_pieces)
        xforms = np.stack([p.xform for p in pieces], axis=0) if pieces else np.empty((0, 7), dtype=np.float32)

        counts = {
            kind: np.fromiter((v.counts[kind] for v in views), dtype=np.int64, count=num_pieces)
            for kind in (
                "body",
                "shape",
                "joint",
                "articulation",
                "constraint_mimic",
                "particle",
                "spring",
                "edge",
                "triangle",
                "tetrahedron",
                "joint_dof",
                "joint_coord",
                "joint_constraint",
            )
        }
        bases = {
            "body": b.body_count,
            "shape": b.shape_count,
            "joint": b.joint_count,
            "articulation": b.articulation_count,
            "constraint_mimic": len(b.constraint_mimic_joint0),
            "particle": b.particle_count,
            "spring": b.spring_count,
            "edge": b.edge_count,
            "triangle": b.tri_count,
            "tetrahedron": b.tet_count,
            "joint_dof": b.joint_dof_count,
            "joint_coord": b.joint_coord_count,
            "joint_constraint": b.joint_constraint_count,
        }
        starts = {kind: bases[kind] + _exclusive_cumsum(counts[kind]) for kind in counts}

        self._extend_verbatim(views)
        self._extend_world_indices(piece_worlds, counts)
        self._extend_indexed(views, starts, counts)
        self._apply_root_transforms(views, starts, counts, xforms)
        self._extend_labels(pieces, starts)
        self._extend_topology(counts)
        self._merge_custom_attributes(pieces, starts)
        self._merge_actuators(views, starts)
        self._merge_particles(views, starts, counts, xforms)

        seen: set[int] = set()
        for view in views:
            if id(view) in seen:
                continue
            seen.add(id(view))
            src = view.builder
            b._requested_contact_attributes.update(src._requested_contact_attributes)
            b._requested_state_attributes.update(src._requested_state_attributes)
            for freq_key, freq_obj in src.custom_frequencies.items():
                if freq_key not in b.custom_frequencies:
                    b.custom_frequencies[freq_key] = freq_obj
            for key, finalizer in src._custom_attribute_model_finalizers.items():
                b._add_custom_attribute_model_finalizer(key, finalizer)

        b.joint_dof_count += int(counts["joint_dof"].sum())
        b.joint_coord_count += int(counts["joint_coord"].sum())
        b.joint_constraint_count += int(counts["joint_constraint"].sum())

        b.world_count += world_count
        b.world_gravity.extend(world_gravity)

        return _BatchOffsets(body=starts["body"], shape=starts["shape"], joint=starts["joint"])

    # region assembly stages

    def _extend_verbatim(self, views: list[_SourceView]) -> None:
        b = self.builder
        for attr in _VERBATIM_ATTRS:
            getattr(b, attr).extend(chain.from_iterable(getattr(v.builder, attr) for v in views))

    def _extend_world_indices(self, piece_worlds: np.ndarray, counts: dict[str, np.ndarray]) -> None:
        b = self.builder
        for attr, kind in (
            ("body_world", "body"),
            ("shape_world", "shape"),
            ("joint_world", "joint"),
            ("articulation_world", "articulation"),
            ("constraint_mimic_world", "constraint_mimic"),
            ("particle_world", "particle"),
        ):
            getattr(b, attr).extend(np.repeat(piece_worlds, counts[kind]).tolist())

    def _extend_indexed(
        self, views: list[_SourceView], starts: dict[str, np.ndarray], counts: dict[str, np.ndarray]
    ) -> None:
        b = self.builder

        def offset_concat(arrays: list[np.ndarray], offs: np.ndarray, lens: np.ndarray, keep_negative: bool):
            if not len(arrays) or not int(lens.sum()):
                return None
            arr = np.concatenate(arrays)
            off = np.repeat(offs, lens)
            if keep_negative:
                return np.where(arr >= 0, arr + off, arr)
            return arr + off

        def extend(dst: list, arrays: list[np.ndarray], offs, lens, keep_negative=False):
            out = offset_concat(arrays, offs, lens, keep_negative)
            if out is not None:
                dst.extend(out.tolist())

        extend(b.shape_body, [v.shape_body for v in views], starts["body"], counts["shape"], keep_negative=True)
        extend(b.joint_parent, [v.joint_parent for v in views], starts["body"], counts["joint"], keep_negative=True)
        extend(b.joint_child, [v.joint_child for v in views], starts["body"], counts["joint"])
        extend(b.joint_q_start, [v.joint_q_start for v in views], starts["joint_coord"], counts["joint"])
        extend(b.joint_qd_start, [v.joint_qd_start for v in views], starts["joint_dof"], counts["joint"])
        extend(b.joint_cts_start, [v.joint_cts_start for v in views], starts["joint_constraint"], counts["joint"])
        extend(
            b.joint_articulation,
            [v.joint_articulation for v in views],
            starts["articulation"],
            counts["joint"],
            keep_negative=True,
        )
        extend(b.articulation_start, [v.articulation_start for v in views], starts["joint"], counts["articulation"])
        extend(b.articulation_end, [v.articulation_end for v in views], starts["joint"], counts["articulation"])
        extend(
            b.constraint_mimic_joint0,
            [v.mimic_joint0 for v in views],
            starts["joint"],
            counts["constraint_mimic"],
            keep_negative=True,
        )
        extend(
            b.constraint_mimic_joint1,
            [v.mimic_joint1 for v in views],
            starts["joint"],
            counts["constraint_mimic"],
            keep_negative=True,
        )

        # Collision filter pairs are (int, int) tuples; groups are copied verbatim.
        # The numpy form is also cached for the vectorized contact-pair finalize step.
        pair_counts = np.fromiter((len(v.filter_pairs) for v in views), dtype=np.int64, count=len(views))
        if int(pair_counts.sum()):
            pairs = np.concatenate([v.filter_pairs for v in views])
            pairs = pairs + np.repeat(starts["shape"], pair_counts)[:, None]
            b.shape_collision_filter_pairs.extend(map(tuple, pairs.tolist()))
            cache_collision_filter_pairs(b, pairs)

        # Deformable element indices reference particles. Springs are a flat index
        # list; triangles/tetrahedra/edges keep per-element rows; edges keep -1
        # sentinels, all matching ``add_builder``.
        extend(b.spring_indices, [v.spring_indices for v in views], starts["particle"], 2 * counts["spring"])

        def extend_rows(dst: list, arrays: list[np.ndarray], lens: np.ndarray, keep_negative: bool = False):
            if not len(arrays) or not int(lens.sum()):
                return
            rows = np.concatenate(arrays)
            off = np.repeat(starts["particle"], lens)[:, None]
            out = np.where(rows >= 0, rows + off, rows) if keep_negative else rows + off
            dst.extend(out.tolist())

        extend_rows(b.tri_indices, [v.tri_indices for v in views], counts["triangle"])
        extend_rows(b.tet_indices, [v.tet_indices for v in views], counts["tetrahedron"])
        extend_rows(b.edge_indices, [v.edge_indices for v in views], counts["edge"], keep_negative=True)

    def _apply_root_transforms(
        self,
        views: list[_SourceView],
        starts: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        xforms: np.ndarray,
    ) -> None:
        """Apply per-piece world transforms to root entities, mirroring ``add_builder``.

        ``add_builder`` composes its ``xform`` argument with every body pose, with the
        parent transform of world-root joints, with the initial coordinates of free
        joints, and with the local transform of static (body ``-1``) shapes.
        """
        b = self.builder

        # All body poses. Stored as plain 7-float lists rather than wp.transform views:
        # per-row numpy view creation costs ~1s per 250k bodies, the float32 values are
        # identical, and downstream consumers (finalize's wp.array conversion, add_joint
        # helpers) accept any 7-sequence.
        if int(counts["body"].sum()):
            body_xf = np.concatenate([v.body_q for v in views])
            t = np.repeat(xforms, counts["body"], axis=0)
            new_body_q = _transform_mul(t, body_xf)
            b.body_q.extend(new_body_q.tolist())

        # Static shapes: overwrite the verbatim-copied local transforms.
        static_counts = np.fromiter((len(v.static_shape_idx) for v in views), dtype=np.int64, count=len(views))
        if int(static_counts.sum()):
            local = np.concatenate([v.static_shape_idx for v in views])
            pos = local + np.repeat(starts["shape"], static_counts)
            src = np.concatenate([v.static_shape_xf for v in views])
            t = np.repeat(xforms, static_counts, axis=0)
            out = np.ascontiguousarray(_transform_mul(t, src))
            shape_transform = b.shape_transform
            for p, row in zip(pos.tolist(), out):
                shape_transform[p] = wp.transform.from_buffer(row)

        # World-root joints (non-free): overwrite joint_X_p.
        root_counts = np.fromiter((len(v.root_joint_idx) for v in views), dtype=np.int64, count=len(views))
        if int(root_counts.sum()):
            local = np.concatenate([v.root_joint_idx for v in views])
            pos = local + np.repeat(starts["joint"], root_counts)
            src = np.concatenate([v.root_joint_xf for v in views])
            t = np.repeat(xforms, root_counts, axis=0)
            out = np.ascontiguousarray(_transform_mul(t, src))
            joint_X_p = b.joint_X_p
            for p, row in zip(pos.tolist(), out):
                joint_X_p[p] = wp.transform.from_buffer(row)

        # Free joints: overwrite the 7 initial coordinates.
        free_counts = np.fromiter((len(v.free_joint_q_start) for v in views), dtype=np.int64, count=len(views))
        if int(free_counts.sum()):
            local = np.concatenate([v.free_joint_q_start for v in views])
            pos = local + np.repeat(starts["joint_coord"], free_counts)
            src = np.concatenate([v.free_joint_q for v in views])
            t = np.repeat(xforms, free_counts, axis=0)
            out = _transform_mul(t, src)
            joint_q = b.joint_q
            for p, row in zip(pos.tolist(), out.tolist()):
                joint_q[p : p + 7] = row

    def _extend_labels(self, pieces: list[_WorldPiece], starts: dict[str, np.ndarray]) -> None:
        b = self.builder
        for attr in _LABEL_ATTRS:
            dst = getattr(b, attr)
            for piece in pieces:
                originals, suffixes, all_renamable = piece.view.labels[attr]
                root = piece.dest_root
                if root is None:
                    dst.extend(originals)
                elif all_renamable:
                    dst.extend(root + s for s in suffixes)
                else:
                    dst.extend(o if s is None else root + s for o, s in zip(originals, suffixes))

        # Env-root site shapes take the destination builder's palette color at their
        # final global index, matching per-world ``add_site`` on the main builder.
        palette = ModelBuilder._shape_palette_color
        shape_color = b.shape_color
        shape_starts = starts["shape"].tolist()
        for i, piece in enumerate(pieces):
            for local_idx in piece.view.palette_color_shapes:
                global_idx = shape_starts[i] + local_idx
                shape_color[global_idx] = palette(global_idx)

    def _extend_topology(self, counts: dict[str, np.ndarray]) -> None:
        """Rebuild the body-shape and joint adjacency dicts for the appended entities.

        Grouping the final ``shape_body`` / ``joint_parent`` / ``joint_child`` arrays
        reproduces the per-piece insertion result: within a piece both shapes and
        joints are appended in ascending index order, so a stable sort by owner gives
        the same per-owner lists.
        """
        b = self.builder

        total_shapes = int(counts["shape"].sum())
        total_bodies = int(counts["body"].sum())
        if total_shapes:
            shape_base = len(b.shape_body) - total_shapes
            shape_body = np.asarray(b.shape_body[shape_base:], dtype=np.int64)
            shape_idx = np.arange(shape_base, shape_base + total_shapes, dtype=np.int64)
            static = shape_body == -1
            if static.any():
                b.body_shapes[-1].extend(shape_idx[static].tolist())
            owned_body = shape_body[~static]
            owned_shape = shape_idx[~static]
            order = np.argsort(owned_body, kind="stable")
            owned_body = owned_body[order]
            owned_shape = owned_shape[order]
            bounds = np.flatnonzero(np.diff(owned_body)) + 1
            grouped = dict(
                zip(
                    owned_body[np.concatenate(([0], bounds))].tolist() if len(owned_body) else [],
                    (v.tolist() for v in np.split(owned_shape, bounds)),
                )
            )
            body_base = b.body_count - total_bodies
            b.body_shapes.update((body, grouped.get(body, [])) for body in range(body_base, body_base + total_bodies))

        total_joints = int(counts["joint"].sum())
        if total_joints:
            joint_base = b.joint_count - total_joints
            parents = b.joint_parent[joint_base:]
            children = b.joint_child[joint_base:]
            joint_parents = b.joint_parents
            joint_children = b.joint_children
            for j, (p, c) in enumerate(zip(parents, children), start=joint_base):
                entry = joint_parents.get(c)
                if entry is None:
                    joint_parents[c] = [(p, j)]
                else:
                    entry.append((p, j))
                entry = joint_children.get(p)
                if entry is None:
                    joint_children[p] = [(c, j)]
                else:
                    entry.append((c, j))

    _ENTITY_START_KINDS: tuple[str, ...] = (
        "body",
        "shape",
        "joint",
        "joint_dof",
        "joint_coord",
        "joint_constraint",
        "articulation",
        "constraint_mimic",
        "particle",
        "edge",
        "triangle",
        "tetrahedron",
        "spring",
    )

    def _merge_custom_attributes(self, pieces: list[_WorldPiece], starts: dict[str, np.ndarray]) -> None:
        """Merge custom attributes per piece, mirroring ``add_builder``'s merge pass."""
        b = self.builder
        freq_counts = b._custom_frequency_counts
        entity_starts = {kind: starts[kind].tolist() for kind in self._ENTITY_START_KINDS}

        for i, piece in enumerate(pieces):
            view = piece.view
            if not view.declared_in_destination:
                self._declare_attributes(view)
                view.declared_in_destination = True
            if not any(tmpl.values for tmpl in view.attr_templates):
                self._bump_freq_counts(view, freq_counts)
                continue

            freq_snapshot = dict(freq_counts)
            piece_starts = {kind: starts_list[i] for kind, starts_list in entity_starts.items()}

            for tmpl in view.attr_templates:
                if not tmpl.values:
                    continue
                merged = b.custom_attributes[tmpl.key]
                if tmpl.is_string_freq:
                    index_offset = freq_snapshot.get(tmpl.freq_key, 0)
                elif tmpl.index_freq == Model.AttributeFrequency.ONCE:
                    index_offset = 0
                elif tmpl.index_freq == Model.AttributeFrequency.WORLD:
                    index_offset = piece.world
                else:
                    index_offset = piece_starts[tmpl.index_freq.name.lower()]

                mapped = self._mapped_values(tmpl, piece, piece_starts, freq_snapshot, view.freq_counts)

                if merged.values is None:
                    merged.values = [] if tmpl.is_string_freq else {}
                if tmpl.is_string_freq:
                    if len(merged.values) < index_offset:
                        merged.values.extend([None] * (index_offset - len(merged.values)))
                    merged.values.extend(mapped)
                else:
                    merged.values.update(zip((tmpl.keys + index_offset).tolist(), mapped))

            self._bump_freq_counts(view, freq_counts)

    @staticmethod
    def _mapped_values(
        tmpl: _AttrTemplate,
        piece: _WorldPiece,
        piece_starts: dict[str, int],
        freq_snapshot: dict[str, int],
        source_freq_counts: dict[str, int],
    ) -> list[Any]:
        """Transform one template's values for one piece."""
        if tmpl.is_eq_target:
            joint_off = piece_starts["joint"]
            mimic_off = piece_starts["constraint_mimic"]
            out = []
            for value, kind in zip(tmpl.values, tmpl.target_kinds):
                try:
                    target = int(value)
                except (TypeError, ValueError):
                    out.append(value)
                    continue
                if target < 0:
                    out.append(value)
                elif kind == 1:
                    out.append(target + joint_off)
                elif kind == 2:
                    out.append(target + mimic_off)
                else:
                    out.append(value)
            return out

        if tmpl.rename_suffixes is not None and piece.dest_root is not None:
            return [v if s is None else piece.dest_root + s for v, s in zip(tmpl.values, tmpl.rename_suffixes)]

        if tmpl.ref is None:
            return tmpl.values
        if tmpl.ref == "world":
            return [piece.world] * len(tmpl.values)

        if tmpl.ref in piece_starts:
            offset = piece_starts[tmpl.ref]
        elif tmpl.ref in freq_snapshot:
            offset = freq_snapshot[tmpl.ref]
        elif tmpl.ref in source_freq_counts:
            offset = 0
        else:
            raise ValueError(
                f"Unknown references value '{tmpl.ref}'. "
                f"Valid values are: {list(piece_starts.keys())} or custom frequencies."
            )
        if offset == 0:
            return tmpl.values
        if tmpl.int_values is not None:
            return np.where(tmpl.int_values >= 0, tmpl.int_values + offset, tmpl.int_values).tolist()
        out = []
        for v in tmpl.values:
            if isinstance(v, int):
                out.append(v + offset if v >= 0 else v)
            elif isinstance(v, (list, tuple)):
                out.append(type(v)(x + offset if isinstance(x, int) and x >= 0 else x for x in v))
            else:
                try:
                    out.append(v + offset)
                except TypeError:
                    out.append(v)
        return out

    def _declare_attributes(self, view: _SourceView) -> None:
        """Declare the view's custom attributes on the destination, checking defaults."""
        b = self.builder
        for tmpl in view.attr_templates:
            merged = b.custom_attributes.get(tmpl.key)
            if merged is None:
                empty = [] if tmpl.is_string_freq else {}
                b.custom_attributes[tmpl.key] = dataclass_replace(tmpl.attr, values=empty)
                continue
            try:
                defaults_match = merged.default == tmpl.attr.default
                if hasattr(defaults_match, "__iter__") and not isinstance(defaults_match, (str, bytes)):
                    defaults_match = all(defaults_match)
            except (ValueError, TypeError):
                defaults_match = False
            if not defaults_match:
                raise ValueError(
                    f"Custom attribute '{tmpl.key}' default mismatch when merging builders: "
                    f"existing={merged.default}, incoming={tmpl.attr.default}"
                )

    @staticmethod
    def _bump_freq_counts(view: _SourceView, freq_counts: dict[str, int]) -> None:
        for freq_key, count in view.freq_counts.items():
            freq_counts[freq_key] = freq_counts.get(freq_key, 0) + count

    def _merge_actuators(self, views: list[_SourceView], starts: dict[str, np.ndarray]) -> None:
        b = self.builder
        dof_starts = starts["joint_dof"].tolist()
        coord_starts = starts["joint_coord"].tolist()
        for i, view in enumerate(views):
            for entry_key, sub_entry in view.builder.actuator_entries.items():
                entry = b.actuator_entries.setdefault(
                    entry_key,
                    ModelBuilder.ActuatorEntry(
                        controller_class=sub_entry.controller_class,
                        clamping_classes=sub_entry.clamping_classes,
                        clamping_shared_kwargs=sub_entry.clamping_shared_kwargs,
                        controller_shared_kwargs=sub_entry.controller_shared_kwargs,
                        indices=[],
                        pos_indices=[],
                        controller_args=[],
                        delay_args=[],
                        clamping_args=[],
                    ),
                )
                dof_off = dof_starts[i]
                coord_off = coord_starts[i]
                entry.indices.extend(idx + dof_off for idx in sub_entry.indices)
                entry.pos_indices.extend(idx + coord_off for idx in sub_entry.pos_indices)
                entry.controller_args.extend(sub_entry.controller_args)
                entry.delay_args.extend(sub_entry.delay_args)
                entry.clamping_args.extend(sub_entry.clamping_args)

    def _merge_particles(
        self,
        views: list[_SourceView],
        starts: dict[str, np.ndarray],
        counts: dict[str, np.ndarray],
        xforms: np.ndarray,
    ) -> None:
        b = self.builder
        if not int(counts["particle"].sum()):
            return
        # Particle positions are translated (not rotated), matching ``add_builder``.
        positions = np.concatenate([v.particle_q for v in views])
        offsets = np.repeat(xforms[:, :3], counts["particle"], axis=0)
        b.particle_q.extend((positions + offsets).tolist())

        particle_starts = starts["particle"].tolist()
        for i, view in enumerate(views):
            if view.counts["particle"]:
                b.particle_max_velocity = view.builder.particle_max_velocity
            groups = view.builder.particle_color_groups
            if groups:
                translated = [group + particle_starts[i] for group in groups]
                b.particle_color_groups = combine_independent_particle_coloring(b.particle_color_groups, translated)

    # endregion


def _exclusive_cumsum(counts: np.ndarray) -> np.ndarray:
    out = np.zeros_like(counts)
    np.cumsum(counts[:-1], out=out[1:])
    return out


def _build_env_root_sites_builder(builder: ModelBuilder, env_root_sites: dict[str, wp.transform]) -> ModelBuilder:
    """Build a scratch source builder holding the per-world env-root sites.

    Matches per-world :meth:`~newton.ModelBuilder.add_site` calls on the main builder:
    ``rigid_gap`` is copied so gap resolution is identical, and the caller recomputes
    palette colors at the final global shape indices (see ``palette_color_shapes``).
    """
    sites_builder = ModelBuilder(up_axis=builder.up_axis, gravity=builder.gravity)
    sites_builder.rigid_gap = builder.rigid_gap
    for label, xform in env_root_sites.items():
        sites_builder.add_site(body=-1, xform=xform, label=label)
    return sites_builder


def replicate_builder_mapping_batched(
    builder: ModelBuilder,
    sources: Sequence[str],
    destinations: Sequence[str],
    env_ids: torch.Tensor,
    mapping: torch.Tensor,
    positions: torch.Tensor,
    quaternions: torch.Tensor,
    source_builders: dict[str, ModelBuilder],
    *,
    source_site_indices: dict[int, dict[str, list[int]]] | None = None,
    env_root_sites: dict[str, wp.transform] | None = None,
    post_replicate_hooks: Sequence[Callable[[ModelBuilder], None]] = (),
) -> tuple[dict[str, list[list[int]]], list[wp.transform], list[tuple[str, int]]]:
    """Replicate source builders into per-env Newton worlds using batched appends.

    Batched drop-in for :func:`~isaaclab_newton.cloner.newton_clone_utils.replicate_builder_mapping`
    followed by :func:`~isaaclab_newton.cloner.newton_clone_utils.rename_builder_labels`:
    it produces an equivalent builder (final labels included) and returns the Fabric
    body bindings the rename pass would have computed.

    Args:
        builder: Destination builder (may already contain global entities).
        sources: Source prim paths, one per clone-plan row.
        destinations: Destination path templates with ``"{}"`` per row.
        env_ids: Environment ids for destination worlds.
        mapping: Boolean source-to-environment mapping matrix ``[num_rows, num_worlds]``.
        positions: Per-environment world positions [m], shape ``[num_worlds, 3]``.
        quaternions: Per-environment orientations (xyzw), shape ``[num_worlds, 4]``.
        source_builders: Mapping from source prim path to its parsed builder.
        source_site_indices: ``{id(source_builder): {label: [local_shape_idx, ...]}}``
            for sites injected into source builders.
        env_root_sites: ``{label: transform}`` bodyless sites added once per world.
        post_replicate_hooks: Callables run on the builder after replication.

    Returns:
        Tuple of ``(local_site_map, world_xforms, fabric_body_bindings)``.
    """
    source_site_indices = source_site_indices or {}
    env_root_sites = env_root_sites or {}

    mapping_np = mapping.detach().cpu().numpy().astype(bool)
    positions_np = positions.detach().cpu().numpy().astype(np.float32).reshape(-1, 3)
    quaternions_np = quaternions.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
    env_id_list = env_ids.detach().cpu().tolist()
    num_rows, num_worlds = mapping_np.shape

    world_xf = np.concatenate([positions_np, quaternions_np], axis=1)
    world_xforms = [wp.transform(positions_np[col], quaternions_np[col]) for col in range(num_worlds)]

    # Combined per-(row, world) root transforms: world_xform * inverse(source_xform),
    # with the inverse computed by Warp exactly as in the legacy path.
    source_world_indices = mapping_np.argmax(axis=1)
    if num_rows:
        inv_source_xf = np.stack(
            [
                np.array(wp.transform_inverse(wp.transform(positions_np[col], quaternions_np[col])), dtype=np.float32)
                for col in source_world_indices
            ],
            axis=0,
        )
        row_world_xf = _transform_mul(world_xf[None, :, :], inv_source_xf[:, None, :])  # [R, N, 7]
    else:
        row_world_xf = np.empty((0, num_worlds, 7), dtype=np.float32)

    batched = BatchedModelBuilder(builder)

    sites_view: _SourceView | None = None
    if env_root_sites:
        sites_builder = _build_env_root_sites_builder(builder, env_root_sites)
        sites_view = batched.source_view(sites_builder, None)
        sites_view.palette_color_shapes = list(range(sites_builder.shape_count))

    row_views: list[_SourceView] = []
    row_source_roots: list[str] = []
    for row in range(num_rows):
        source_root = sources[row].rstrip("/") or "/"
        row_views.append(batched.source_view(source_builders[sources[row]], source_root))
        row_source_roots.append(source_root)
    row_dest_roots: list[list[str]] = [
        [destinations[row].format(env_id).rstrip("/") or "/" for env_id in env_id_list] for row in range(num_rows)
    ]

    base_world = builder.world_count
    prior_shape_count = builder.shape_count
    prior_filter_count = len(builder.shape_collision_filter_pairs)
    default_gravity = tuple(u * builder.gravity for u in builder.up_vector)

    pieces: list[_WorldPiece] = []
    piece_index: dict[tuple[int, int], int] = {}  # (row, world_col) -> piece position
    world_gravity: list[tuple[float, float, float]] = []
    active_rows_per_world = [np.flatnonzero(mapping_np[:, col]) for col in range(num_worlds)]
    for col in range(num_worlds):
        world = base_world + col
        if sites_view is not None:
            piece_index[(-1, col)] = len(pieces)
            pieces.append(
                _WorldPiece(view=sites_view, world=world, xform=world_xf[col], dest_root=None, overrides_gravity=False)
            )
        gravity = default_gravity
        for row in active_rows_per_world[col].tolist():
            piece_index[(row, col)] = len(pieces)
            pieces.append(
                _WorldPiece(
                    view=row_views[row],
                    world=world,
                    xform=row_world_xf[row, col],
                    dest_root=row_dest_roots[row][col],
                )
            )
            gravity = row_views[row].gravity_vector
        world_gravity.append(gravity)

    offsets = batched.append_worlds(pieces, num_worlds, world_gravity)

    # All worlds built from the same rows onto a world-less builder are identical
    # modulo a constant index offset; record that for the contact-pair fast path.
    if base_world == 0 and num_worlds > 0:
        first_rows = active_rows_per_world[0]
        if all(np.array_equal(rows, first_rows) for rows in active_rows_per_world[1:]):
            mark_homogeneous_worlds(builder, num_worlds, prior_shape_count, prior_filter_count)

    # Site bookkeeping: env-root sites and injected source sites, per world.
    local_site_map: dict[str, list[list[int]]] = {}
    shape_starts = offsets.shape.tolist()
    if sites_view is not None:
        for local_idx, label in enumerate(env_root_sites):
            per_world = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])
            for col in range(num_worlds):
                per_world[col].append(shape_starts[piece_index[(-1, col)]] + local_idx)
    for row in range(num_rows):
        source_builder = source_builders[sources[row]]
        for label, source_shape_indices in source_site_indices.get(id(source_builder), {}).items():
            per_world = local_site_map.setdefault(label, [[] for _ in range(num_worlds)])
            for col in np.flatnonzero(mapping_np[row]).tolist():
                start = shape_starts[piece_index[(row, col)]]
                per_world[col].extend(start + shape_idx for shape_idx in source_shape_indices)

    fabric_body_bindings = _fabric_body_bindings(builder, pieces, offsets, piece_index, row_source_roots, mapping_np)

    for hook in post_replicate_hooks:
        hook(builder)
    return local_site_map, world_xforms, fabric_body_bindings


def _fabric_body_bindings(
    builder: ModelBuilder,
    pieces: list[_WorldPiece],
    offsets: _BatchOffsets,
    piece_index: dict[tuple[int, int], int],
    row_source_roots: list[str],
    mapping_np: np.ndarray,
) -> list[tuple[str, int]]:
    """Compute Fabric body bindings, mirroring the legacy rename pass.

    The legacy pass binds a body when its label was actually rewritten, i.e. the label
    is under the row's source root and the destination root differs. Remaining bodies
    (globals and same-root worlds such as ``env_0``) are appended afterwards in index
    order.
    """
    bindings: list[tuple[str, int]] = []
    bound: set[int] = set()
    body_label = builder.body_label
    body_starts = offsets.body.tolist()

    num_rows = mapping_np.shape[0]
    for row in range(num_rows):
        source_root = row_source_roots[row]
        for col in np.flatnonzero(mapping_np[row]).tolist():
            idx = piece_index[(row, col)]
            piece = pieces[idx]
            if piece.dest_root == source_root:
                continue
            start = body_starts[idx]
            _, suffixes, _ = piece.view.labels["body_label"]
            for local_idx, suffix in enumerate(suffixes):
                if suffix is None:
                    continue
                global_idx = start + local_idx
                bindings.append((body_label[global_idx], global_idx))
                bound.add(global_idx)

    bindings.extend((label, index) for index, label in enumerate(body_label) if index not in bound)
    return bindings
