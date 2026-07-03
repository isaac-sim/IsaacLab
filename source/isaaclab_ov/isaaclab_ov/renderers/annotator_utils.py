# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Decode OVRTX's renderer-internal id/label AOVs into annotator ``info`` dicts.

OVRTX does not run omni.replicator's OGN annotator graph, so the ``idToLabels``/
``idToSemantics`` dicts that graph produces CPU-side for ``semantic_segmentation``,
``instance_segmentation_fast``, and ``instance_id_segmentation_fast`` have to be
reconstructed here from OVRTX's raw render vars:

* ``SemanticIdMap``            -- ``{semantic_id: label}``.
* ``StableIdMap``              -- ``{stable_id: prim_path}``, keyed on the full
  128-bit stable id, not just its first 32-bit word.
* ``StableIdSemanticIdMap``    -- ``{stable_id: semantic_id}`` entries, stored in
  the same order as the compacted per-frame ids used by
  ``NonStableInstanceSegmentation`` (see :func:`resolve_instance_segmentation_labels`).
* ``InstanceMap``              -- for each renderer instance id used by
  ``InstanceSegmentationSD``, its position in ``StableIdSemanticIdMap`` (see
  :func:`resolve_instance_id_segmentation_labels`).

The ``instance_segmentation_fast`` join mirrors
``OgnInstanceSegmentationPostRender::computeCuda`` (``omni.replicator.nv``)
exactly, verified against its source. The ``instance_id_segmentation_fast`` join
(via ``InstanceMap``/``InstanceSegmentationSD``) was reverse-engineered
empirically -- no OGN source for the corresponding post-render node was
available -- by cross-referencing co-located pixels against the
source-verified ``instance_segmentation_fast`` join, across render passes with
different renderer-assigned id orderings.

These render vars are not part of ovrtx's documented/supported output table --
they are internal renderer AOVs that happen to use the same RenderVar/sourceName
mechanism as the supported ones. Treat their binary layout as unstable.
"""

from __future__ import annotations

import numpy as np

# A 128-bit stable id, stored as four little-endian uint32 words. Only the
# first word is unique enough for casual inspection, but the full tuple must
# be used as a dict key -- two different stable ids can share their first word.
StableId = tuple[int, int, int, int]

_INSTANCE_MAP_SENTINEL = 0xFFFFFFFF
"""``InstanceMap`` entry value marking a renderer instance id with no semantic instance (e.g. UNLABELLED)."""

_BACKGROUND_ID = 0
_UNLABELLED_ID = 1

# Shared binary layout for SemanticIdMap (id -> label) and StableIdMap
# (id -> prim path): a packed array of {id[4], label_length, label_offset}
# entries (128-bit id as four uint32 words), followed by a string table,
# followed by a trailing 4-byte entry count.
_ID_STRING_ENTRY_DTYPE = np.dtype(
    [("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
)

# StableIdSemanticIdMap entries are fixed-size 32-byte {stable_id[4], semantic_id[4]}
# records with no trailing count or string table -- the entry count is simply
# buffer_size / 32.
_STABLE_ID_SEMANTIC_ID_ENTRY_DTYPE = np.dtype(
    [("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))]
)


def clean_label(label: str) -> str:
    """Match ``OgnInstanceSegmentationPostRender::cleanLabel``.

    Drops semicolons and the whitespace immediately following ``:``/``,``, then
    trims trailing whitespace, matching the label formatting
    ``instance_segmentation_fast`` applies before it lands in ``idToSemantics``.

    Args:
        label: Raw semantic label string, as decoded from ``SemanticIdMap``.

    Returns:
        The cleaned label.
    """
    result = []
    skip_whitespace = False
    for c in label:
        if c == ";":
            continue
        if skip_whitespace:
            if c in (" ", "\t"):
                continue
            skip_whitespace = False
        if c in (":", ","):
            result.append(c)
            skip_whitespace = True
            continue
        result.append(c)
    return "".join(result).rstrip(" \t\r\n")


def decode_id_string_entries(tensor: np.ndarray) -> list[tuple[StableId, str]]:
    """Decode a ``SemanticIdMap``/``StableIdMap`` payload into ``[(id, label), ...]``.

    Order-preserving: entry position in the returned list matches its position
    in the underlying buffer, which :func:`resolve_instance_segmentation_labels`
    and :func:`resolve_instance_id_segmentation_labels` index into directly.

    Args:
        tensor: Raw ``SemanticIdMap``/``StableIdMap`` render var, mapped to host memory.

    Returns:
        A list of ``(id, label)`` pairs in buffer order.
    """
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size < 4:
        return []

    num_entries = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    if num_entries <= 0:
        return []
    entries_size = num_entries * _ID_STRING_ENTRY_DTYPE.itemsize
    if entries_size > data.size - 4:
        return []

    entries = data[:entries_size].view(_ID_STRING_ENTRY_DTYPE).reshape(num_entries)
    result: list[tuple[StableId, str]] = []
    for entry in entries:
        entry_id = tuple(int(w) for w in entry["id"])
        label_offset = int(entry["label_offset"])
        label_length = int(entry["label_length"])
        label_end = label_offset + label_length
        if label_end > data.size:
            continue
        label = data[label_offset:label_end].tobytes().decode("utf-8", "replace")
        result.append((entry_id, label.rstrip("\x00").rstrip()))
    return result


def decode_id_string_map(tensor: np.ndarray) -> dict[int, str]:
    """Decode a ``SemanticIdMap``/``StableIdMap`` payload into ``{id[0]: label}``.

    Args:
        tensor: Raw ``SemanticIdMap``/``StableIdMap`` render var, mapped to host memory.

    Returns:
        A dict keyed on each entry's first 32-bit id word. Sufficient for
        ``SemanticIdMap`` (semantic ids fit in 32 bits), but collapses distinct
        ``StableIdMap`` entries that happen to share their first word -- use
        :func:`decode_id_string_entries` there instead.
    """
    return {entry_id[0]: label for entry_id, label in decode_id_string_entries(tensor)}


def decode_stable_id_semantic_id_entries(tensor: np.ndarray) -> list[tuple[StableId, int]]:
    """Decode a ``StableIdSemanticIdMap`` payload into ``[(stable_id, semantic_id), ...]``.

    Order-preserving, see :func:`decode_id_string_entries`.

    Args:
        tensor: Raw ``StableIdSemanticIdMap`` render var, mapped to host memory.

    Returns:
        A list of ``(stable_id, semantic_id)`` pairs in buffer order.
    """
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    itemsize = _STABLE_ID_SEMANTIC_ID_ENTRY_DTYPE.itemsize
    num_entries = data.size // itemsize
    if num_entries <= 0:
        return []

    entries = data[: num_entries * itemsize].view(_STABLE_ID_SEMANTIC_ID_ENTRY_DTYPE)
    return [(tuple(int(w) for w in e["stable_id"]), int(e["semantic_id"][0])) for e in entries]


def decode_instance_map(tensor: np.ndarray) -> list[int]:
    """Decode an ``InstanceMap`` payload into a flat list of ``StableIdSemanticIdMap`` positions.

    ``InstanceMap`` is a flat ``uint32`` array with no header or trailing count:
    entry ``i`` gives the position in ``StableIdSemanticIdMap`` for renderer
    instance id ``i + 1`` as used by ``InstanceSegmentationSD`` (id ``0`` is
    always BACKGROUND and never indexes into it). A value of
    :data:`_INSTANCE_MAP_SENTINEL` marks an id with no semantic instance (e.g.
    the reserved UNLABELLED id ``1``).

    Args:
        tensor: Raw ``InstanceMap`` render var, mapped to host memory.

    Returns:
        The flat list of ``uint32`` entries, indexed by ``renderer_instance_id - 1``.
    """
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size % 4 != 0:
        return []
    return data.view(np.uint32).tolist()


def resolve_semantic_segmentation_labels(semantic_id_map_tensor: np.ndarray) -> dict[int, str]:
    """Build ``idToLabels`` for ``semantic_segmentation`` from ``SemanticIdMap``.

    ``SemanticSegmentation`` pixel values are semantic ids directly (no
    positional indirection): ``SemanticIdMap``'s ``id`` field already equals
    the pixel value it labels.

    Args:
        semantic_id_map_tensor: Raw ``SemanticIdMap`` render var, mapped to host memory.

    Returns:
        ``{semantic_id: label}``, including the reserved ``0: "BACKGROUND"`` and
        ``1: "UNLABELLED"`` entries.
    """
    id_to_labels = {_BACKGROUND_ID: "BACKGROUND", _UNLABELLED_ID: "UNLABELLED"}
    for semantic_id, label in decode_id_string_entries(semantic_id_map_tensor):
        id_to_labels[semantic_id[0]] = clean_label(label)
    return id_to_labels


def resolve_instance_segmentation_labels(
    pixel_ids: set[int],
    stable_id_semantic_id_map_tensor: np.ndarray,
    semantic_id_map_tensor: np.ndarray,
    stable_id_map_tensor: np.ndarray,
) -> tuple[dict[int, str], dict[int, str]]:
    """Build ``idToLabels``/``idToSemantics`` for ``instance_segmentation_fast``.

    Replicates ``OgnInstanceSegmentationPostRender::computeCuda``
    (``omni.replicator.nv``): for each ``NonStableInstanceSegmentation`` pixel
    value ``v >= 2``, ``v - 2`` is a direct index (not a value match) into
    ``StableIdSemanticIdMap``; that entry's ``semantic_id - 2`` is, in turn, a
    direct index into ``SemanticIdMap``. Only the final ``stable_id -> prim_path``
    step is a real hashed lookup, keyed on the full 128-bit stable id.

    Args:
        pixel_ids: Unique pixel values observed in ``NonStableInstanceSegmentation``.
        stable_id_semantic_id_map_tensor: Raw ``StableIdSemanticIdMap`` render var, mapped to host memory.
        semantic_id_map_tensor: Raw ``SemanticIdMap`` render var, mapped to host memory.
        stable_id_map_tensor: Raw ``StableIdMap`` render var, mapped to host memory.

    Returns:
        A ``(id_to_labels, id_to_semantics)`` tuple, each covering exactly the
        ids present in ``pixel_ids`` plus the reserved ``0``/``1`` entries.
    """
    stable_id_semantic_id_entries = decode_stable_id_semantic_id_entries(stable_id_semantic_id_map_tensor)
    semantic_id_entries = decode_id_string_entries(semantic_id_map_tensor)
    stable_id_to_prim_path = dict(decode_id_string_entries(stable_id_map_tensor))

    id_to_labels = {_BACKGROUND_ID: "BACKGROUND", _UNLABELLED_ID: "UNLABELLED"}
    id_to_semantics = {_BACKGROUND_ID: "class:BACKGROUND", _UNLABELLED_ID: "class:UNLABELLED"}

    for pixel_id in pixel_ids:
        if pixel_id < 2:
            continue

        stable_id_semantic_idx = pixel_id - 2
        if stable_id_semantic_idx >= len(stable_id_semantic_id_entries):
            continue
        stable_id, semantic_id = stable_id_semantic_id_entries[stable_id_semantic_idx]

        semantic_idx = semantic_id - 2
        if semantic_id < 2 or semantic_idx >= len(semantic_id_entries):
            continue
        _, raw_label = semantic_id_entries[semantic_idx]

        id_to_labels[pixel_id] = stable_id_to_prim_path.get(stable_id, "UNLABELLED")
        id_to_semantics[pixel_id] = clean_label(raw_label)

    return id_to_labels, id_to_semantics


def resolve_instance_id_segmentation_labels(
    pixel_ids: set[int],
    instance_map_tensor: np.ndarray,
    stable_id_semantic_id_map_tensor: np.ndarray,
    stable_id_map_tensor: np.ndarray,
) -> dict[int, str]:
    """Build ``idToLabels`` for ``instance_id_segmentation_fast``.

    For each ``InstanceSegmentationSD`` pixel value ``v >= 2``,
    ``InstanceMap[v - 1]`` gives the position of that instance in
    ``StableIdSemanticIdMap`` (see :func:`decode_instance_map`); the entry's
    stable id is then looked up in ``StableIdMap`` (hashed on the full 128-bit
    id) to get the prim path.

    Args:
        pixel_ids: Unique pixel values observed in ``InstanceSegmentationSD``.
        instance_map_tensor: Raw ``InstanceMap`` render var, mapped to host memory.
        stable_id_semantic_id_map_tensor: Raw ``StableIdSemanticIdMap`` render var, mapped to host memory.
        stable_id_map_tensor: Raw ``StableIdMap`` render var, mapped to host memory.

    Returns:
        ``{instance_id: prim_path}``, covering exactly the ids present in
        ``pixel_ids`` plus the reserved ``0``/``1`` entries.
    """
    instance_map = decode_instance_map(instance_map_tensor)
    stable_id_semantic_id_entries = decode_stable_id_semantic_id_entries(stable_id_semantic_id_map_tensor)
    stable_id_to_prim_path = dict(decode_id_string_entries(stable_id_map_tensor))

    id_to_labels = {_BACKGROUND_ID: "BACKGROUND", _UNLABELLED_ID: "UNLABELLED"}

    for pixel_id in pixel_ids:
        if pixel_id < 2:
            continue

        instance_map_idx = pixel_id - 1
        if instance_map_idx >= len(instance_map):
            continue
        stable_id_semantic_idx = instance_map[instance_map_idx]
        if stable_id_semantic_idx == _INSTANCE_MAP_SENTINEL:
            id_to_labels[pixel_id] = "UNLABELLED"
            continue
        if stable_id_semantic_idx >= len(stable_id_semantic_id_entries):
            continue

        stable_id, _ = stable_id_semantic_id_entries[stable_id_semantic_idx]
        id_to_labels[pixel_id] = stable_id_to_prim_path.get(stable_id, "UNLABELLED")

    return id_to_labels
