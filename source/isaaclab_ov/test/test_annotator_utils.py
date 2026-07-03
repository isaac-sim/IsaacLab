# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for OVRTX annotator id/label decoding, using synthetic render var buffers.

These construct the raw byte layouts by hand (no GPU/ovrtx runtime required) to
pin down the binary formats documented in ``annotator_utils.py``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

_REQUIRED_MODULES = ("isaaclab_ov",)
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    from isaaclab_ov.renderers.annotator_utils import (  # noqa: E402
        clean_label,
        decode_id_string_entries,
        decode_id_string_map,
        decode_instance_map,
        decode_stable_id_semantic_id_entries,
        resolve_instance_id_segmentation_labels,
        resolve_instance_segmentation_labels,
        resolve_semantic_segmentation_labels,
    )
else:
    clean_label = None
    decode_id_string_entries = None
    decode_id_string_map = None
    decode_instance_map = None
    decode_stable_id_semantic_id_entries = None
    resolve_instance_id_segmentation_labels = None
    resolve_instance_segmentation_labels = None
    resolve_semantic_segmentation_labels = None


def _build_id_string_buffer(entries: list[tuple[tuple[int, int, int, int], str]]) -> np.ndarray:
    """Build a raw ``SemanticIdMap``/``StableIdMap`` buffer from ``(id, label)`` entries."""
    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    header_size = len(entries) * entry_dtype.itemsize

    labels_bytes = b""
    header = np.zeros(len(entries), dtype=entry_dtype)
    for i, (entry_id, label) in enumerate(entries):
        label_bytes = label.encode("utf-8")
        header[i]["id"] = entry_id
        header[i]["label_length"] = len(label_bytes)
        header[i]["label_offset"] = header_size + len(labels_bytes)
        labels_bytes += label_bytes

    trailing_count = np.array([len(entries)], dtype="<u4")
    return np.frombuffer(header.tobytes() + labels_bytes + trailing_count.tobytes(), dtype=np.uint8)


def _build_stable_id_semantic_id_buffer(entries: list[tuple[tuple[int, int, int, int], int]]) -> np.ndarray:
    """Build a raw ``StableIdSemanticIdMap`` buffer from ``(stable_id, semantic_id)`` entries."""
    entry_dtype = np.dtype([("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))])
    array = np.zeros(len(entries), dtype=entry_dtype)
    for i, (stable_id, semantic_id) in enumerate(entries):
        array[i]["stable_id"] = stable_id
        array[i]["semantic_id"][0] = semantic_id
    return np.frombuffer(array.tobytes(), dtype=np.uint8)


def _build_instance_map_buffer(values: list[int]) -> np.ndarray:
    """Build a raw ``InstanceMap`` buffer: a flat ``uint32`` array, no header/trailing count."""
    return np.array(values, dtype="<u4").view(np.uint8)


def _sid(word0: int) -> tuple[int, int, int, int]:
    """Build a stable-id-shaped tuple, distinguishable by its first word only."""
    return (word0, 0, 0, 0)


class Test_CleanLabel:
    def test_strips_semicolons_and_post_delimiter_whitespace(self):
        assert clean_label("instance: cone_01; class: cone;") == "instance:cone_01 class:cone"

    def test_leaves_plain_label_unchanged(self):
        assert clean_label("class:cube") == "class:cube"


class Test_DecodeIdStringEntries:
    def test_round_trips_multiple_entries(self):
        buffer = _build_id_string_buffer([(_sid(2), "class:cube"), (_sid(3), "class:sphere")])
        entries = decode_id_string_entries(buffer)
        assert entries == [(_sid(2), "class:cube"), (_sid(3), "class:sphere")]

    def test_empty_buffer_returns_empty_list(self):
        assert decode_id_string_entries(np.zeros(0, dtype=np.uint8)) == []

    def test_decode_id_string_map_keys_on_first_word(self):
        buffer = _build_id_string_buffer([(_sid(2), "class:cube"), (_sid(3), "class:sphere")])
        assert decode_id_string_map(buffer) == {2: "class:cube", 3: "class:sphere"}


class Test_DecodeStableIdSemanticIdEntries:
    def test_round_trips_multiple_entries(self):
        buffer = _build_stable_id_semantic_id_buffer([(_sid(1000), 2), (_sid(2000), 3)])
        assert decode_stable_id_semantic_id_entries(buffer) == [(_sid(1000), 2), (_sid(2000), 3)]

    def test_empty_buffer_returns_empty_list(self):
        assert decode_stable_id_semantic_id_entries(np.zeros(0, dtype=np.uint8)) == []


class Test_DecodeInstanceMap:
    def test_returns_flat_uint32_list(self):
        buffer = _build_instance_map_buffer([0xFFFFFFFF, 0, 2, 1])
        assert decode_instance_map(buffer) == [0xFFFFFFFF, 0, 2, 1]

    def test_odd_byte_count_returns_empty_list(self):
        assert decode_instance_map(np.zeros(3, dtype=np.uint8)) == []


class Test_ResolveSemanticSegmentationLabels:
    def test_includes_reserved_and_decoded_ids(self):
        semantic_id_map = _build_id_string_buffer([(_sid(2), "class:cube"), (_sid(3), "class: sphere;")])
        id_to_labels = resolve_semantic_segmentation_labels(semantic_id_map)
        assert id_to_labels == {0: "BACKGROUND", 1: "UNLABELLED", 2: "class:cube", 3: "class:sphere"}


class Test_ResolveInstanceSegmentationLabels:
    def test_join_matches_ogn_instance_segmentation_post_render(self):
        # Two instances: pixel 2 -> StableIdSemanticIdMap[0] -> semantic id 2 -> SemanticIdMap[0]="class:cube".
        # pixel 3 -> StableIdSemanticIdMap[1] -> semantic id 3 -> SemanticIdMap[1]="class:sphere".
        stable_id_semantic_id_map = _build_stable_id_semantic_id_buffer([(_sid(1000), 2), (_sid(2000), 3)])
        semantic_id_map = _build_id_string_buffer([(_sid(2), "class:cube"), (_sid(3), "class: sphere;")])
        stable_id_map = _build_id_string_buffer([(_sid(1000), "/World/Cube"), (_sid(2000), "/World/Sphere")])

        id_to_labels, id_to_semantics = resolve_instance_segmentation_labels(
            {0, 1, 2, 3}, stable_id_semantic_id_map, semantic_id_map, stable_id_map
        )

        assert id_to_labels == {0: "BACKGROUND", 1: "UNLABELLED", 2: "/World/Cube", 3: "/World/Sphere"}
        assert id_to_semantics == {0: "class:BACKGROUND", 1: "class:UNLABELLED", 2: "class:cube", 3: "class:sphere"}

    def test_unmatched_stable_id_falls_back_to_unlabelled(self):
        # StableIdSemanticIdMap references a stable id absent from StableIdMap
        # (e.g. a second instance sharing another's semantic label, per the
        # real scene's "two prims, one class label" case).
        stable_id_semantic_id_map = _build_stable_id_semantic_id_buffer([(_sid(9999), 2)])
        semantic_id_map = _build_id_string_buffer([(_sid(2), "class:cube")])
        stable_id_map = _build_id_string_buffer([(_sid(1000), "/World/Cube")])

        id_to_labels, _ = resolve_instance_segmentation_labels(
            {2}, stable_id_semantic_id_map, semantic_id_map, stable_id_map
        )
        assert id_to_labels[2] == "UNLABELLED"

    def test_only_covers_requested_pixel_ids(self):
        stable_id_semantic_id_map = _build_stable_id_semantic_id_buffer([(_sid(1000), 2), (_sid(2000), 3)])
        semantic_id_map = _build_id_string_buffer([(_sid(2), "class:cube"), (_sid(3), "class:sphere")])
        stable_id_map = _build_id_string_buffer([(_sid(1000), "/World/Cube"), (_sid(2000), "/World/Sphere")])

        id_to_labels, id_to_semantics = resolve_instance_segmentation_labels(
            {0, 1, 2}, stable_id_semantic_id_map, semantic_id_map, stable_id_map
        )
        assert 3 not in id_to_labels
        assert 3 not in id_to_semantics


class Test_ResolveInstanceIdSegmentationLabels:
    def test_join_via_instance_map_indirection(self):
        # InstanceSegmentationSD ids 2 and 3 route through InstanceMap (indexed
        # by id - 1) to StableIdSemanticIdMap positions, in a different order
        # than instance_segmentation_fast's direct (id - 2) indexing.
        instance_map = _build_instance_map_buffer([0xFFFFFFFF, 1, 0])
        stable_id_semantic_id_map = _build_stable_id_semantic_id_buffer([(_sid(1000), 2), (_sid(2000), 3)])
        stable_id_map = _build_id_string_buffer([(_sid(1000), "/World/Cube"), (_sid(2000), "/World/Sphere")])

        id_to_labels = resolve_instance_id_segmentation_labels(
            {0, 1, 2, 3}, instance_map, stable_id_semantic_id_map, stable_id_map
        )

        assert id_to_labels == {
            0: "BACKGROUND",
            1: "UNLABELLED",
            2: "/World/Sphere",  # InstanceMap[2 - 1] == 1 -> StableIdSemanticIdMap[1] -> stable id 2000
            3: "/World/Cube",  # InstanceMap[3 - 1] == 0 -> StableIdSemanticIdMap[0] -> stable id 1000
        }

    def test_sentinel_entry_resolves_to_unlabelled(self):
        instance_map = _build_instance_map_buffer([0xFFFFFFFF])
        stable_id_semantic_id_map = _build_stable_id_semantic_id_buffer([(_sid(1000), 2)])
        stable_id_map = _build_id_string_buffer([(_sid(1000), "/World/Cube")])

        id_to_labels = resolve_instance_id_segmentation_labels(
            {0, 1}, instance_map, stable_id_semantic_id_map, stable_id_map
        )
        assert id_to_labels == {0: "BACKGROUND", 1: "UNLABELLED"}
