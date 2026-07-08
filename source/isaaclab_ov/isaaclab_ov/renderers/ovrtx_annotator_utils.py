# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Semantic annotator utilities for the OVRTX renderer.

Decodes the OVRTX ``SemanticIdMap`` render var into an ``idToLabels`` mapping compatible with the
Isaac RTX / Replicator contract exposed through ``camera.data.info["semantic_segmentation"]``.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from .ovrtx_renderer_kernels import generate_random_colors_from_ids_kernel

# Reserved semantic IDs shared by the OVRTX SemanticSegmentation AOV and Isaac RTX / Replicator:
# ID 0 is BACKGROUND (no prim), ID 1 is UNLABELLED (a prim with no matching semantic label). Entries
# with ID >= 2 are decoded from the SemanticIdMap render var.
SEMANTIC_ID_BACKGROUND = 0
SEMANTIC_ID_UNLABELLED = 1
RESERVED_SEMANTIC_LABELS: dict[int, dict[str, str]] = {
    SEMANTIC_ID_BACKGROUND: {"class": "BACKGROUND"},
    SEMANTIC_ID_UNLABELLED: {"class": "UNLABELLED"},
}


def parse_semantic_label(raw_label: str) -> dict[str, str]:
    """Parse a raw OVRTX semantic label string into a ``{semantic_type: label}`` dict.

    OVRTX encodes labels as ``"<type>: <label>;"`` segments (e.g. ``"class: cone;"``), matching the
    ``{"class": "cone"}`` form Isaac RTX / Replicator expose in ``idToLabels``. Multiple segments are
    separated by ``";"``. Segments without a ``":"`` are ignored.

    Args:
        raw_label: Raw label string decoded from the SemanticIdMap buffer.

    Returns:
        Mapping from semantic type to label, e.g. ``{"class": "cone"}``.
    """
    labels: dict[str, str] = {}
    for segment in raw_label.split(";"):
        segment = segment.strip()
        if not segment or ":" not in segment:
            continue
        semantic_type, semantic_label = segment.split(":", 1)
        labels[semantic_type.strip()] = semantic_label.strip()
    return labels


def decode_semantic_id_map(id_map: np.ndarray) -> dict[int, dict[str, str]]:
    """Decode the OVRTX ``SemanticIdMap`` render var buffer into ``{semantic_id: {type: label}}``.

    The buffer packs, in order: an array of ``SemanticIdentifierMap`` entries, the label string blob, and a
    trailing ``uint32`` entry count. Each entry is ``{uint32 id[4]; uint32 labelLength; uint32 labelOffset}``;
    the ``id[0]`` field is the semantic ID and the label bytes live at ``[labelOffset, labelOffset + labelLength)``.
    Mirrors the reference decode shipped with the ovrtx runtime.

    Args:
        id_map: Raw SemanticIdMap buffer mapped to host memory (any dtype; viewed as bytes).

    Returns:
        Mapping from semantic ID (>= 2) to its parsed ``{type: label}`` dict. Reserved IDs (0/1) are not
        included; callers add BACKGROUND/UNLABELLED separately (see :data:`RESERVED_SEMANTIC_LABELS`).
    """
    data = np.ascontiguousarray(id_map).view(np.uint8).reshape(-1)
    if data.size < 4:
        return {}

    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    num_entries = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    entries_size = num_entries * entry_dtype.itemsize
    if entries_size > data.size - 4:
        raise ValueError(
            f"Corrupt SemanticIdMap: {num_entries} entries ({entries_size} bytes) exceed buffer size {data.size}."
        )

    # Labels live between the entries and the trailing 4-byte entry count, so a valid label must end at or
    # before ``data.size - 4`` (never spilling into the count field), consistent with the check above.
    labels_end = data.size - 4
    entries = data[:entries_size].view(entry_dtype).reshape(num_entries)
    labels_by_id: dict[int, dict[str, str]] = {}
    for entry in entries:
        semantic_id = int(entry["id"][0])
        label_offset = int(entry["label_offset"])
        label_end = label_offset + int(entry["label_length"])
        if label_end > labels_end:
            raise ValueError(
                f"Corrupt SemanticIdMap: label for id {semantic_id} spans [{label_offset}, {label_end}) beyond"
                f" the label region end {labels_end} (buffer size {data.size}, minus the 4-byte entry count)."
            )
        raw_label = data[label_offset:label_end].tobytes().decode("utf-8").rstrip("\x00").rstrip()
        labels_by_id[semantic_id] = parse_semantic_label(raw_label)
    return labels_by_id


def semantic_color_keys(semantic_ids: list[int], device: str) -> list[str]:
    """Return the ``"(r, g, b, a)"`` color-tuple key for each semantic ID.

    Colors are computed with the same kernel that colorizes the segmentation buffer
    (:func:`generate_random_colors_from_ids_kernel`), so the returned keys match the pixel colors.

    Args:
        semantic_ids: Semantic IDs to colorize, in the order the keys are returned.
        device: Warp device on which to run the colorization kernel (e.g. ``"cuda:0"``).

    Returns:
        One ``"(r, g, b, a)"`` string per input ID, in the same order.
    """
    ids_wp = wp.array(np.asarray(semantic_ids, dtype=np.uint32).reshape(1, -1, 1), dtype=wp.uint32, device=device)
    colors_wp = wp.zeros(shape=ids_wp.shape, dtype=wp.uint32, device=device)
    wp.launch(
        kernel=generate_random_colors_from_ids_kernel,
        dim=ids_wp.shape,
        inputs=[ids_wp, colors_wp],
        device=device,
    )
    keys: list[str] = []
    for color in colors_wp.numpy().reshape(-1):
        color = int(color)
        rgba = (color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF, (color >> 24) & 0xFF)
        keys.append(str(rgba))
    return keys


def build_semantic_id_to_labels(
    labels_by_id: dict[int, dict[str, str]], colorize: bool, device: str
) -> dict[str, dict[str, str]]:
    """Build the ``idToLabels`` mapping from decoded semantic IDs, keyed by ID or RGBA color.

    Args:
        labels_by_id: Decoded ``{semantic_id: {type: label}}`` for IDs >= 2 from the SemanticIdMap.
        colorize: If True, keys are ``"(r, g, b, a)"`` color tuples matching the colorized segmentation
            buffer; otherwise keys are the decimal semantic IDs (matching the raw ``int32`` output).
        device: Warp device used to compute colors when ``colorize`` is True.

    Returns:
        Mapping from string key to ``{type: label}``, always including the reserved BACKGROUND (ID 0) and
        UNLABELLED (ID 1) entries.
    """
    id_labels: dict[int, dict[str, str]] = {**RESERVED_SEMANTIC_LABELS, **labels_by_id}
    sorted_ids = sorted(id_labels)

    if not colorize:
        return {str(semantic_id): id_labels[semantic_id] for semantic_id in sorted_ids}

    color_keys = semantic_color_keys(sorted_ids, device)
    return {color_keys[idx]: id_labels[semantic_id] for idx, semantic_id in enumerate(sorted_ids)}
