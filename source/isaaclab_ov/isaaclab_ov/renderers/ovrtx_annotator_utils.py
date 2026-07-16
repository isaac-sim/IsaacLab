# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Host-side stand-in for the not-yet-released ``ovannotators`` library.  This module will be replaced
by the library once released.

Until ``ovannotators`` ships, these helpers decode the OVRTX segmentation "map" render vars and rebuild the
``idToLabels`` / ``idToSemantics`` info dicts that Replicator / Isaac RTX expose through ``camera.data.info``:

- ``semantic_segmentation``: from ``SemanticIdMap`` -> ``idToLabels`` (id -> ``{type: label}``).
- ``instance_segmentation_fast``: from ``StableIdSemanticIdMap`` + ``StableIdMap`` + ``SemanticIdMap`` ->
  ``idToLabels`` (instance -> USD prim path) and ``idToSemantics`` (instance -> ``{type: label}``).

For colorized outputs the keys are raw ``(r, g, b, a)`` tuples, matching Replicator's fast segmentation nodes.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
import warp as wp

from .ovrtx_renderer_kernels import generate_random_colors_from_ids_kernel

# A 128-bit identifier as four ``uint32`` words, low word first: the stable ID key of the StableIdMap /
# StableIdSemanticIdMap, and the raw ``id`` field of every IdentifierMap entry.
StableId: TypeAlias = tuple[int, int, int, int]
# A colorized segmentation key: the ``(r, g, b, a)`` value of a colorized instance/semantic pixel.
RgbaColor: TypeAlias = tuple[int, int, int, int]
# A decoded, right-stripped UTF-8 label from an IdentifierMap entry (a semantic label or a USD prim path).
RawLabel: TypeAlias = str
# One decoded IdentifierMap entry: its 128-bit id paired with its label.
StableIdLabelPair: TypeAlias = tuple[StableId, RawLabel]

# Reserved semantic IDs shared by the OVRTX SemanticSegmentation AOV and Isaac RTX / Replicator:
# ID 0 is BACKGROUND (no prim), ID 1 is UNLABELLED (a prim with no matching semantic label). Entries
# with ID >= 2 are decoded from the SemanticIdMap render var.
_SEMANTIC_ID_BACKGROUND = 0
_SEMANTIC_ID_UNLABELLED = 1
# Single source of truth for the reserved ID 0/1 label names, shared across every segmentation map and
# matching the ``"BACKGROUND"`` / ``"UNLABELLED"`` strings Replicator writes for pixel IDs 0 and 1.
_RESERVED_LABEL_NAMES: dict[int, str] = {
    _SEMANTIC_ID_BACKGROUND: "BACKGROUND",
    _SEMANTIC_ID_UNLABELLED: "UNLABELLED",
}
# Number of reserved IDs (BACKGROUND, UNLABELLED). Real per-instance entries are offset by this count: entry
# ``i`` of the StableIdSemanticIdMap corresponds to pixel ID ``i + _NUM_RESERVED_IDS``, and semantic IDs
# ``>= _NUM_RESERVED_IDS`` are real (see the ``+ 2`` offset in the ovrtx / Replicator contract).
_NUM_RESERVED_IDS = len(_RESERVED_LABEL_NAMES)
# Reserved prim-path labels (plain strings) for the instance-segmentation ``idToLabels`` mapping.
_RESERVED_INSTANCE_LABELS: dict[int, str] = dict(_RESERVED_LABEL_NAMES)
# Reserved ``{type: label}`` dicts for the semantic ``idToLabels`` and instance ``idToSemantics`` mappings.
_RESERVED_SEMANTIC_LABELS: dict[int, dict[str, str]] = {
    reserved_id: {"class": name} for reserved_id, name in _RESERVED_LABEL_NAMES.items()
}


def _parse_semantic_label(raw_label: str) -> dict[str, str]:
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


def _decode_identifier_map(id_map: np.ndarray, map_name: str) -> list[StableIdLabelPair]:
    """Decode an OVRTX ``IdentifierMap``-layout render var buffer into ``(id, raw_label)`` pairs.

    Shared layout for the ``SemanticIdMap`` and ``StableIdMap`` render vars: the buffer packs, in order,
    an array of ``IdentifierMap`` entries, the label string blob, and a trailing ``uint32`` entry count.
    Each entry is ``{uint32 id[4]; uint32 labelLength; uint32 labelOffset}``; the label bytes live at
    ``[labelOffset, labelOffset + labelLength)`` measured from the buffer base. Mirrors the reference decode
    shipped with the ovrtx runtime.

    Args:
        id_map: Raw map buffer mapped to host memory (any dtype; viewed as bytes).
        map_name: Buffer name used in corruption error messages (e.g. ``"SemanticIdMap"``).

    Returns:
        One ``(id, raw_label)`` pair per entry in buffer order. ``id`` is the four ``uint32`` identifier
        words; ``raw_label`` is the decoded, right-stripped UTF-8 label string.

    Raises:
        ValueError: If the entry count or any label range exceeds the buffer bounds.
    """
    data = np.ascontiguousarray(id_map).view(np.uint8).reshape(-1)
    if data.size < 4:
        return []

    # ``<u4`` (little-endian): matches the ovannotators reference decoder. kit writes native ``uint32_t``
    # struct fields with no byte-swap, so this is correct on the (little-endian) platforms it targets.
    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    num_entries = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    entries_size = num_entries * entry_dtype.itemsize
    if entries_size > data.size - 4:
        raise ValueError(
            f"Corrupt {map_name}: {num_entries} entries ({entries_size} bytes) exceed buffer size {data.size}."
        )

    # Labels live between the entries and the trailing 4-byte entry count, so a valid label must end at or
    # before ``data.size - 4`` (never spilling into the count field), consistent with the check above.
    labels_end = data.size - 4
    entries = data[:entries_size].view(entry_dtype).reshape(num_entries)
    decoded: list[StableIdLabelPair] = []
    for entry in entries:
        id_words = tuple(int(word) for word in entry["id"])
        label_offset = int(entry["label_offset"])
        label_end = label_offset + int(entry["label_length"])
        if label_end > labels_end:
            raise ValueError(
                f"Corrupt {map_name}: label for id {id_words} spans [{label_offset}, {label_end}) beyond"
                f" the label region end {labels_end} (buffer size {data.size}, minus the 4-byte entry count)."
            )
        raw_label = data[label_offset:label_end].tobytes().decode("utf-8").rstrip("\x00").rstrip()
        decoded.append((id_words, raw_label))
    return decoded


def decode_semantic_id_map(id_map: np.ndarray) -> dict[int, dict[str, str]]:
    """Decode the OVRTX ``SemanticIdMap`` render var buffer into ``{semantic_id: {type: label}}``.

    The ``id[0]`` field of each :func:`_decode_identifier_map` entry is the semantic ID, and the label is
    parsed with :func:`_parse_semantic_label`.

    Args:
        id_map: Raw SemanticIdMap buffer mapped to host memory (any dtype; viewed as bytes).

    Returns:
        Mapping from semantic ID (>= 2) to its parsed ``{type: label}`` dict. Reserved IDs (0/1) are not
        included; callers add BACKGROUND/UNLABELLED separately (see :data:`_RESERVED_SEMANTIC_LABELS`).
    """
    return {
        id_words[0]: _parse_semantic_label(raw_label)
        for id_words, raw_label in _decode_identifier_map(id_map, "SemanticIdMap")
    }


def decode_stable_id_map(id_map: np.ndarray) -> dict[StableId, RawLabel]:
    """Decode the OVRTX ``StableIdMap`` render var buffer into ``{stable_id: prim_path}``.

    Same :func:`_decode_identifier_map` layout as the SemanticIdMap, but the four ``id`` words form the
    128-bit stable ID key and the label is a USD prim-path string (not a semantic label, so it is not parsed).

    Args:
        id_map: Raw StableIdMap buffer mapped to host memory (any dtype; viewed as bytes).

    Returns:
        Mapping from the four-word stable ID to its USD prim-path string.
    """
    return {id_words: raw_label for id_words, raw_label in _decode_identifier_map(id_map, "StableIdMap")}


def decode_stable_id_semantic_id_map(id_map: np.ndarray) -> list[tuple[StableId, int]]:
    """Decode the OVRTX ``StableIdSemanticIdMap`` render var buffer into per-instance ``(stable_id, semantic_id)``.

    Unlike the identifier maps, this buffer is a pure fixed-size array of ``StableIdSemanticIdMapValues``
    entries ``{uint32 stableId[4]; uint32 semanticId[4]}`` (32 bytes each) with no label blob and no trailing
    count; the entry count is ``buffer_bytes // 32``. Array position ``i`` corresponds to the instance-segmentation
    pixel ID ``i + _NUM_RESERVED_IDS`` (IDs 0 and 1 are the reserved BACKGROUND/UNLABELLED values).

    Args:
        id_map: Raw StableIdSemanticIdMap buffer mapped to host memory (any dtype; viewed as bytes).

    Returns:
        One ``(stable_id, semantic_id)`` per entry, indexed by ``pixel_id - _NUM_RESERVED_IDS``. ``stable_id``
        is the four ``uint32`` stable-ID words (the :func:`decode_stable_id_map` key); ``semantic_id`` is
        ``semanticId[0]``.
    """
    data = np.ascontiguousarray(id_map).view(np.uint8).reshape(-1)
    entry_dtype = np.dtype([("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))])
    num_entries = data.size // entry_dtype.itemsize
    if num_entries == 0:
        return []
    entries = data[: num_entries * entry_dtype.itemsize].view(entry_dtype).reshape(num_entries)
    return [(tuple(int(word) for word in entry["stable_id"]), int(entry["semantic_id"][0])) for entry in entries]


def _color_keys_for_ids(ids: list[int], device: str) -> list[RgbaColor]:
    """Return the ``(r, g, b, a)`` color-tuple key for each segmentation ID.

    Colors are computed with the same kernel that colorizes the segmentation buffer
    (:func:`generate_random_colors_from_ids_kernel`), so the returned keys match the pixel colors. Used for
    both semantic IDs and instance-segmentation pixel IDs, which share the same colorization. These raw tuples
    are the ``idToLabels`` / ``idToSemantics`` key form for the colorized outputs, matching Replicator's fast
    C++ segmentation nodes.

    Args:
        ids: Segmentation IDs to colorize, in the order the keys are returned.
        device: Warp device on which to run the colorization kernel (e.g. ``"cuda:0"``).

    Returns:
        One ``(r, g, b, a)`` tuple key per input ID, in the same order.
    """
    ids_wp = wp.array(np.asarray(ids, dtype=np.uint32).reshape(1, -1, 1), dtype=wp.uint32, device=device)
    colors_wp = wp.zeros(shape=ids_wp.shape, dtype=wp.uint32, device=device)
    wp.launch(
        kernel=generate_random_colors_from_ids_kernel,
        dim=ids_wp.shape,
        inputs=[ids_wp, colors_wp],
        device=device,
    )
    keys: list[RgbaColor] = []
    for color in colors_wp.numpy().reshape(-1):
        color = int(color)
        keys.append((color & 0xFF, (color >> 8) & 0xFF, (color >> 16) & 0xFF, (color >> 24) & 0xFF))
    return keys


def build_semantic_id_to_labels(
    labels_by_id: dict[int, dict[str, str]], colorize: bool, device: str
) -> dict[RgbaColor | str, dict[str, str]]:
    """Build the ``idToLabels`` mapping from decoded semantic IDs, keyed by RGBA color tuple or ID.

    Args:
        labels_by_id: Decoded ``{semantic_id: {type: label}}`` for IDs >= 2 from the SemanticIdMap.
        colorize: If True, keys are ``(r, g, b, a)`` color tuples matching the colorized segmentation buffer
            (the Replicator fast-node contract); otherwise keys are the decimal semantic ID strings (matching
            the raw ``int32`` output).
        device: Warp device used to compute colors when ``colorize`` is True.

    Returns:
        Mapping from key to ``{type: label}``, always including the reserved BACKGROUND (ID 0) and UNLABELLED
        (ID 1) entries.
    """
    id_labels: dict[int, dict[str, str]] = {**_RESERVED_SEMANTIC_LABELS, **labels_by_id}
    sorted_ids = sorted(id_labels)

    if not colorize:
        return {str(semantic_id): id_labels[semantic_id] for semantic_id in sorted_ids}

    color_keys = _color_keys_for_ids(sorted_ids, device)
    return {color_keys[idx]: id_labels[semantic_id] for idx, semantic_id in enumerate(sorted_ids)}


def build_instance_id_to_labels_and_semantics(
    stable_id_semantic_id_map: list[tuple[StableId, int]],
    stable_id_to_path: dict[StableId, RawLabel],
    semantic_id_to_labels: dict[int, dict[str, str]],
    colorize: bool,
    device: str,
) -> tuple[dict[RgbaColor | str, str], dict[RgbaColor | str, dict[str, str]]]:
    """Build the instance-segmentation ``idToLabels`` and ``idToSemantics`` mappings.

    Resolves each instance pixel ID through the three decoded maps, following the ovrtx / Replicator chain:
    pixel ID ``i + _NUM_RESERVED_IDS`` indexes ``stable_id_semantic_id_map[i]`` to get its stable ID and
    semantic ID; the stable ID resolves to a USD prim path via ``stable_id_to_path`` (``idToLabels``) and the
    semantic ID
    resolves to a ``{type: label}`` dict via ``semantic_id_to_labels`` (``idToSemantics``). Pixel IDs 0 and 1
    are the reserved BACKGROUND / UNLABELLED entries and are always included.

    Args:
        stable_id_semantic_id_map: Decoded ``StableIdSemanticIdMap`` entries from
            :func:`decode_stable_id_semantic_id_map`, indexed by ``pixel_id - _NUM_RESERVED_IDS``.
        stable_id_to_path: Decoded ``{stable_id: prim_path}`` from :func:`decode_stable_id_map`.
        semantic_id_to_labels: Decoded ``{semantic_id: {type: label}}`` from :func:`decode_semantic_id_map`.
        colorize: If True, keys are ``(r, g, b, a)`` color tuples matching the colorized segmentation buffer
            (the Replicator fast-node contract); otherwise keys are the decimal pixel ID strings (matching the
            raw ``uint32`` output).
        device: Warp device used to compute colors when ``colorize`` is True.

    Returns:
        A ``(idToLabels, idToSemantics)`` tuple. ``idToLabels`` maps each key to a USD prim-path string;
        ``idToSemantics`` maps each key to a ``{type: label}`` dict.
    """
    # Pixel IDs 0/1 are reserved; entry ``i`` corresponds to pixel ID ``i + _NUM_RESERVED_IDS``.
    pixel_ids: list[int] = [_SEMANTIC_ID_BACKGROUND, _SEMANTIC_ID_UNLABELLED]
    prim_paths: list[str] = [
        _RESERVED_INSTANCE_LABELS[_SEMANTIC_ID_BACKGROUND],
        _RESERVED_INSTANCE_LABELS[_SEMANTIC_ID_UNLABELLED],
    ]
    semantics: list[dict[str, str]] = [
        _RESERVED_SEMANTIC_LABELS[_SEMANTIC_ID_BACKGROUND],
        _RESERVED_SEMANTIC_LABELS[_SEMANTIC_ID_UNLABELLED],
    ]
    for index, (stable_id, semantic_id) in enumerate(stable_id_semantic_id_map):
        pixel_ids.append(index + _NUM_RESERVED_IDS)
        # Fall back to UNLABELLED when a stable/semantic ID is missing from its map (defensive; the producer
        # only emits pixel IDs >= 2 for labelled prims, so both lookups are expected to hit).
        prim_paths.append(stable_id_to_path.get(stable_id, _RESERVED_INSTANCE_LABELS[_SEMANTIC_ID_UNLABELLED]))
        semantics.append(semantic_id_to_labels.get(semantic_id, _RESERVED_SEMANTIC_LABELS[_SEMANTIC_ID_UNLABELLED]))

    keys = _color_keys_for_ids(pixel_ids, device) if colorize else [str(pixel_id) for pixel_id in pixel_ids]
    id_to_labels = {key: prim_paths[idx] for idx, key in enumerate(keys)}
    id_to_semantics = {key: semantics[idx] for idx, key in enumerate(keys)}
    return id_to_labels, id_to_semantics
