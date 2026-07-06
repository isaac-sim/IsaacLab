# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Segmentation-mapping helpers for the Newton Warp renderer.

Newton's ray tracer emits a single per-pixel *shape index* (``shape_index_image``), i.e. the global
index of the model shape hit by each ray. Isaac Lab's camera contract instead exposes three
segmentation outputs — ``semantic_segmentation``, ``instance_segmentation_fast`` and
``instance_id_segmentation_fast`` — each with its own id space and an accompanying
``idToLabels`` / ``idToSemantics`` mapping (see
:class:`~isaaclab.sensors.camera.CameraData`).

This module reconstructs those outputs on the host from the Newton model's per-shape prim paths
(``model.shape_label``) and the USD stage's :class:`UsdSemantics.LabelsAPI` labels (authored by
:attr:`~isaaclab.sim.spawners.SpawnerCfg.semantic_tags`), then remaps the shape-index image into each
requested output with a Warp kernel. Colors match the Isaac RTX / OVRTX palette so colorized outputs
are visually consistent across renderers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import warp as wp

# Colorization (host ``random_color_from_id`` / ``pack_rgba``) and the reserved BACKGROUND / UNLABELLED
# ids are shared with the RTX and OVRTX renderers to keep colorized segmentation visually consistent.
from isaaclab.renderers.segmentation_colors import BACKGROUND_ID, UNLABELLED_ID, pack_rgba, random_color_from_id

if TYPE_CHECKING:
    import newton

    from pxr import Usd

_FIRST_ID: int = 2
"""First id assigned to a real semantic/instance group (0 and 1 are reserved)."""

SegKind = Literal["semantic_segmentation", "instance_segmentation_fast", "instance_id_segmentation_fast"]


# ------------------------------------------------------------------------------------------------
# Semantic-filter predicate — RTX parity.
#
# The Isaac RTX renderer evaluates :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter`
# inside Replicator's C++ ``SyntheticData`` instance-mapping filter (see
# :func:`isaaclab_physx.renderers.isaac_rtx_renderer._camera_semantic_filter_predicate`). The kit-less
# Newton path has no Replicator, so the three functions below re-implement the *same* predicate grammar
# on the host so :attr:`~isaaclab_newton.renderers.NewtonWarpRendererCfg.semantic_filter` selects
# exactly the labels the RTX renderer would. The grammar is documented on
# :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter`.
# ------------------------------------------------------------------------------------------------


def _eval_label_expr(expr: str, labels: set[str]) -> bool:
    """Evaluate one clause's label expression against a prim's label set (RTX-parity grammar).

    Grammar (matching the Isaac RTX / Replicator ``semantic_filter`` predicate): an OR of ``|`` / ``,``
    separated terms; each term is an AND of ``&`` separated factors; each factor is ``label``,
    ``!label`` (negation) or ``*`` (any).
    """
    expr = expr.strip()
    if not expr:
        return False
    for term in expr.replace(",", "|").split("|"):
        term = term.strip()
        if not term:
            continue
        if all(_eval_label_factor(factor.strip(), labels) for factor in term.split("&")):
            return True
    return False


def _eval_label_factor(factor: str, labels: set[str]) -> bool:
    """Evaluate a single filter factor: ``*`` (any), ``!label`` (absent) or ``label`` (present)."""
    if factor == "*":
        return True
    if factor.startswith("!"):
        return factor[1:].strip() not in labels
    return factor in labels


def _parse_semantic_filter(semantic_filter: str | list[str]) -> list[tuple[str, str]]:
    """Normalize a semantic filter into a list of ``(type, label_expr)`` clauses (RTX parity).

    Reproduces the normalization in
    :func:`isaaclab_physx.renderers.isaac_rtx_renderer._camera_semantic_filter_predicate` so the same
    :attr:`~isaaclab_physx.renderers.IsaacRtxRendererCfg.semantic_filter` value behaves identically on
    the Newton Warp backend: a list is treated as bare semantic types (``["class"] -> "class:*"``); a
    string is a ``;``-separated disjunction of ``type:label_expr`` clauses (e.g. ``"class:* ; *:shelf"``).
    """
    if isinstance(semantic_filter, list):
        text = "; ".join(f"{t}:*" for t in semantic_filter)
    else:
        text = semantic_filter
    clauses: list[tuple[str, str]] = []
    for clause in text.split(";"):
        clause = clause.strip()
        if not clause:
            continue
        sem_type, _, label_expr = clause.partition(":")
        clauses.append((sem_type.strip(), label_expr.strip() or "*"))
    return clauses


@wp.kernel(enable_backward=False)
def _remap_shape_index_to_id_kernel(
    shape_index: wp.array(dtype=wp.uint32, ndim=4),
    shape_to_id: wp.array(dtype=wp.int32),
    shape_count: wp.int32,
    out_id: wp.array(dtype=wp.int32, ndim=4),
):
    """Write the segmentation id of each pixel's shape; ray-miss / sentinel pixels get BACKGROUND (0)."""
    w, c, y, x = wp.tid()
    idx = shape_index[w, c, y, x]
    if idx < wp.uint32(shape_count):
        out_id[w, c, y, x] = shape_to_id[wp.int32(idx)]
    else:
        out_id[w, c, y, x] = wp.int32(0)


@wp.kernel(enable_backward=False)
def _remap_shape_index_to_color_kernel(
    shape_index: wp.array(dtype=wp.uint32, ndim=4),
    shape_to_color: wp.array(dtype=wp.uint32),
    shape_count: wp.int32,
    out_color: wp.array(dtype=wp.uint32, ndim=4),
):
    """Write the packed RGBA color of each pixel's shape; ray-miss / sentinel pixels get (0, 0, 0, 0)."""
    w, c, y, x = wp.tid()
    idx = shape_index[w, c, y, x]
    if idx < wp.uint32(shape_count):
        out_color[w, c, y, x] = shape_to_color[wp.int32(idx)]
    else:
        out_color[w, c, y, x] = wp.uint32(0)


@dataclass
class SegmentationPlan:
    """Device lookup tables and host metadata for one segmentation output.

    Built once per (kind, colorize) by :meth:`SegmentationMapper.plan` and reused across frames since
    the scene geometry and semantics are static after construction.
    """

    kind: SegKind
    colorize: bool
    shape_count: int
    shape_to_id: wp.array  # int32, shape [shape_count]
    shape_to_color: wp.array | None  # uint32, shape [shape_count]; None when not colorized
    info: dict[str, dict]

    def apply(self, shape_index: wp.array, out_view: wp.array) -> None:
        """Remap Newton's shape-index buffer into ``out_view`` (colorized RGBA or raw int32 ids).

        Args:
            shape_index: Newton ``shape_index_image`` scratch, ``(world_count, 1, H, W)`` uint32.
            out_view: Destination view aliasing the camera output buffer as ``(world_count, 1, H, W)``,
                dtype uint32 when colorized, else int32.
        """
        if self.shape_count == 0:
            out_view.zero_()
            return
        if self.colorize:
            wp.launch(
                _remap_shape_index_to_color_kernel,
                dim=shape_index.shape,
                inputs=[shape_index, self.shape_to_color, self.shape_count],
                outputs=[out_view],
                device=out_view.device,
            )
        else:
            wp.launch(
                _remap_shape_index_to_id_kernel,
                dim=shape_index.shape,
                inputs=[shape_index, self.shape_to_id, self.shape_count],
                outputs=[out_view],
                device=out_view.device,
            )


class SegmentationMapper:
    """Builds per-shape segmentation lookup tables from a Newton model and its USD stage.

    Construction is cheap (references only). The per-shape resolution runs lazily in :meth:`plan`, is
    cached per (kind, colorize), and is shared across all cameras of a renderer since it depends only
    on the scene geometry and its authored semantics.
    """

    def __init__(self, model: newton.Model, stage: Usd.Stage | None, cfg) -> None:
        self._model = model
        self._stage = stage
        self._cfg = cfg
        self._shape_labels: list[str] = list(getattr(model, "shape_label", []) or [])
        self._shape_count = len(self._shape_labels)
        self._device = str(model.device)
        self._filter_clauses = _parse_semantic_filter(cfg.semantic_filter)
        # Cache of prim path -> (matched_labels or None); labels resolved with ancestor inheritance.
        self._matched_cache: dict[str, dict[str, list[str]] | None] = {}
        self._plans: dict[tuple[str, bool], SegmentationPlan] = {}

    def plan(self, kind: SegKind, colorize: bool) -> SegmentationPlan:
        """Return the (cached) :class:`SegmentationPlan` for ``kind`` at the requested colorization."""
        key = (kind, colorize)
        if key not in self._plans:
            self._plans[key] = self._build_plan(kind, colorize)
        return self._plans[key]

    # -- host resolution ---------------------------------------------------------------------------

    def _matched_labels(self, prim_path: str) -> dict[str, list[str]] | None:
        """Return the filter-matched semantic labels for ``prim_path``, inheriting from ancestors.

        Walks from the prim up to the stage root, returning the labels of the *nearest* prim that
        carries a :class:`UsdSemantics.LabelsAPI` label satisfying :attr:`semantic_filter`. Returns
        ``None`` when no ancestor matches (i.e. the prim is unlabelled for this filter).
        """
        if prim_path in self._matched_cache:
            return self._matched_cache[prim_path]

        result: dict[str, list[str]] | None = None
        matched_path: str | None = None
        if self._stage is not None and prim_path.startswith("/"):
            from isaaclab.sim.utils.semantics import get_labels  # noqa: PLC0415

            prim = self._stage.GetPrimAtPath(prim_path)
            while prim is not None and prim.IsValid() and prim.GetPath().pathString != "/":
                labels = get_labels(prim)
                if labels:
                    kept = self._apply_filter(labels)
                    if kept:
                        result = kept
                        matched_path = prim.GetPath().pathString
                        break
                prim = prim.GetParent()

        # Cache the resolved match against the matched prim path too, so sibling shapes under the same
        # labelled ancestor share the lookup without re-walking the hierarchy.
        self._matched_cache[prim_path] = result
        if matched_path is not None and matched_path not in self._matched_cache:
            self._matched_cache[matched_path] = result
        return result

    def _matched_ancestor_path(self, prim_path: str) -> str | None:
        """Return the prim path of the nearest labelled ancestor (inclusive), or ``None``."""
        if self._stage is None or not prim_path.startswith("/"):
            return None
        from isaaclab.sim.utils.semantics import get_labels  # noqa: PLC0415

        prim = self._stage.GetPrimAtPath(prim_path)
        while prim is not None and prim.IsValid() and prim.GetPath().pathString != "/":
            labels = get_labels(prim)
            if labels and self._apply_filter(labels):
                return prim.GetPath().pathString
            prim = prim.GetParent()
        return None

    def _apply_filter(self, labels: dict[str, list[str]]) -> dict[str, list[str]]:
        """Restrict ``labels`` (``{type: [labels]}``) to the types/labels passing the semantic filter.

        Applies the RTX-parity predicate parsed by :func:`_parse_semantic_filter`: a semantic type is
        kept when some filter clause matches its type (or ``*``) and its label set satisfies the
        clause's expression, mirroring the Isaac RTX / Replicator ``semantic_filter`` behavior. Returns
        an empty dict when the prim matches no clause (i.e. it is UNLABELLED for this filter).
        """
        kept: dict[str, list[str]] = {}
        for sem_type, sem_labels in labels.items():
            label_set = set(sem_labels)
            for clause_type, clause_expr in self._filter_clauses:
                if clause_type not in ("*", sem_type):
                    continue
                if _eval_label_expr(clause_expr, label_set):
                    kept[sem_type] = list(sem_labels)
                    break
        return kept

    @staticmethod
    def _semantics_payload(labels: dict[str, list[str]]) -> dict[str, str]:
        """Collapse ``{type: [labels]}`` into the ``{type: "l1,l2"}`` payload used in info dicts."""
        return {sem_type: ",".join(sem_labels) for sem_type, sem_labels in labels.items()}

    def _build_plan(self, kind: SegKind, colorize: bool) -> SegmentationPlan:
        shape_to_id = np.zeros(self._shape_count, dtype=np.int32)
        # id -> (idToLabels value, idToSemantics value or None)
        id_labels: dict[int, object] = {}
        id_semantics: dict[int, dict[str, str]] = {}
        # dedup group key -> id
        group_ids: dict[object, int] = {}
        next_id = _FIRST_ID

        for shape_index, prim_path in enumerate(self._shape_labels):
            if kind == "instance_id_segmentation_fast":
                # Every shape is its own instance keyed by its (leaf) prim path.
                group_key: object = prim_path
                label_value: object = prim_path
                semantics_value: dict[str, str] | None = None
            elif kind == "instance_segmentation_fast":
                matched = self._matched_labels(prim_path)
                if matched is None:
                    shape_to_id[shape_index] = UNLABELLED_ID
                    continue
                ancestor = self._matched_ancestor_path(prim_path) or prim_path
                group_key = ancestor
                label_value = ancestor
                semantics_value = self._semantics_payload(matched)
            else:  # semantic_segmentation
                matched = self._matched_labels(prim_path)
                if matched is None:
                    shape_to_id[shape_index] = UNLABELLED_ID
                    continue
                payload = self._semantics_payload(matched)
                group_key = tuple(sorted(payload.items()))
                label_value = payload
                semantics_value = None

            seg_id = group_ids.get(group_key)
            if seg_id is None:
                seg_id = next_id
                next_id += 1
                group_ids[group_key] = seg_id
                id_labels[seg_id] = label_value
                if semantics_value is not None:
                    id_semantics[seg_id] = semantics_value
            shape_to_id[shape_index] = seg_id

        info = self._build_info(kind, colorize, id_labels, id_semantics)
        shape_to_color = self._build_color_palette(kind, colorize, shape_to_id, id_labels)

        return SegmentationPlan(
            kind=kind,
            colorize=colorize,
            shape_count=self._shape_count,
            shape_to_id=wp.array(shape_to_id, dtype=wp.int32, device=self._device),
            shape_to_color=(
                wp.array(shape_to_color, dtype=wp.uint32, device=self._device) if shape_to_color is not None else None
            ),
            info=info,
        )

    # -- info + color assembly ---------------------------------------------------------------------

    def _id_to_color(self, seg_id: int, label_value: object) -> tuple[int, int, int, int]:
        """Resolve the RGBA color for an id, honoring ``semantic_segmentation_mapping`` overrides."""
        mapping = self._cfg.semantic_segmentation_mapping
        if mapping and isinstance(label_value, dict):
            for sem_type, sem_labels in label_value.items():
                for lbl in sem_labels.split(","):
                    override = mapping.get(f"{sem_type}:{lbl}")
                    if override is not None:
                        return tuple(int(component) for component in override)  # type: ignore[return-value]
        return random_color_from_id(seg_id)

    def _reserved_semantics_label(self, name: str) -> dict[str, str]:
        return {"class": name}

    def _build_info(
        self,
        kind: SegKind,
        colorize: bool,
        id_labels: dict[int, object],
        id_semantics: dict[int, dict[str, str]],
    ) -> dict[str, dict]:
        """Assemble the Replicator-compatible ``idToLabels`` (and ``idToSemantics``) info dict."""
        # Reserved entries present for every segmentation output.
        reserved_labels: dict[int, object]
        reserved_semantics: dict[int, dict[str, str]]
        if kind == "semantic_segmentation":
            reserved_labels = {
                BACKGROUND_ID: self._reserved_semantics_label("BACKGROUND"),
                UNLABELLED_ID: self._reserved_semantics_label("UNLABELLED"),
            }
            reserved_semantics = {}
        else:
            reserved_labels = {BACKGROUND_ID: "BACKGROUND", UNLABELLED_ID: "UNLABELLED"}
            reserved_semantics = {
                BACKGROUND_ID: self._reserved_semantics_label("BACKGROUND"),
                UNLABELLED_ID: self._reserved_semantics_label("UNLABELLED"),
            }

        all_labels = {**reserved_labels, **id_labels}

        def key_for(seg_id: int) -> str:
            if colorize:
                return str(self._id_to_color(seg_id, id_labels.get(seg_id, all_labels[seg_id])))
            return str(seg_id)

        id_to_labels = {key_for(seg_id): value for seg_id, value in all_labels.items()}
        info: dict[str, dict] = {"idToLabels": id_to_labels}

        if kind == "instance_segmentation_fast":
            all_semantics = {**reserved_semantics, **id_semantics}
            info["idToSemantics"] = {key_for(seg_id): value for seg_id, value in all_semantics.items()}
        return info

    def _build_color_palette(
        self,
        kind: SegKind,
        colorize: bool,
        shape_to_id: np.ndarray,
        id_labels: dict[int, object],
    ) -> np.ndarray | None:
        """Build the per-shape packed-RGBA palette used by the colorize kernel, or ``None``."""
        if not colorize or self._shape_count == 0:
            return None
        color_cache: dict[int, int] = {}
        shape_to_color = np.zeros(self._shape_count, dtype=np.uint32)
        for shape_index in range(self._shape_count):
            seg_id = int(shape_to_id[shape_index])
            packed = color_cache.get(seg_id)
            if packed is None:
                packed = pack_rgba(self._id_to_color(seg_id, id_labels.get(seg_id)))
                color_cache[seg_id] = packed
            shape_to_color[shape_index] = packed
        return shape_to_color
