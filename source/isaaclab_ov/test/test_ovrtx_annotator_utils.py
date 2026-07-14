# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX semantic annotator utilities (SemanticIdMap decode -> idToLabels)."""

import importlib.util

import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.isaacsim_ci,
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import numpy as np  # noqa: E402
    from isaaclab_ov.renderers.ovrtx_annotator_utils import (  # noqa: E402
        build_semantic_id_to_labels,
        decode_semantic_id_map,
        parse_semantic_label,
    )


def _encode_semantic_id_map(entries: list[tuple[int, str]]) -> "np.ndarray":
    """Encode ``(semantic_id, raw_label)`` pairs into a SemanticIdMap byte buffer for decode tests.

    Layout mirrors the OVRTX render var: packed ``SemanticIdentifierMap`` entries, the UTF-8 label blob,
    then a trailing little-endian ``uint32`` entry count.
    """
    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    header_size = len(entries) * entry_dtype.itemsize
    entry_arr = np.zeros(len(entries), dtype=entry_dtype)
    label_blob = b""
    for i, (semantic_id, label) in enumerate(entries):
        label_bytes = label.encode("utf-8")
        entry_arr[i]["id"][0] = semantic_id
        entry_arr[i]["label_length"] = len(label_bytes)
        entry_arr[i]["label_offset"] = header_size + len(label_blob)
        label_blob += label_bytes
    buffer = entry_arr.tobytes() + label_blob + int(len(entries)).to_bytes(4, "little")
    return np.frombuffer(buffer, dtype=np.uint8).copy()


def test_parse_semantic_label_splits_type_and_label():
    """Raw ``"<type>: <label>;"`` strings parse into ``{type: label}`` dicts like Replicator idToLabels."""
    assert parse_semantic_label("class: cone;") == {"class": "cone"}
    assert parse_semantic_label("class: robot; type: arm;") == {"class": "robot", "type": "arm"}
    assert parse_semantic_label("") == {}


def test_decode_semantic_id_map_round_trips_entries():
    """Decoding an encoded SemanticIdMap recovers the semantic-ID-to-label mapping."""
    id_map = _encode_semantic_id_map([(2, "class: cone;"), (3, "class: cube;")])
    assert decode_semantic_id_map(id_map) == {2: {"class": "cone"}, 3: {"class": "cube"}}


def test_decode_semantic_id_map_handles_empty_buffer():
    """A too-small SemanticIdMap buffer decodes to an empty mapping instead of raising."""
    assert decode_semantic_id_map(np.zeros(0, dtype=np.uint8)) == {}


def test_decode_semantic_id_map_rejects_corrupt_entry_count():
    """A SemanticIdMap whose entry count exceeds the buffer raises instead of reading out of bounds."""
    id_map = _encode_semantic_id_map([(2, "class: cone;")])
    id_map[-4:] = np.frombuffer(int(999).to_bytes(4, "little"), dtype=np.uint8)
    with pytest.raises(ValueError, match="Corrupt SemanticIdMap"):
        decode_semantic_id_map(id_map)


def test_decode_semantic_id_map_rejects_label_spilling_into_count_field():
    """A label whose bytes run into the trailing 4-byte entry count is rejected (bound is data.size - 4)."""
    id_map = _encode_semantic_id_map([(2, "class: cone;")])
    # Inflate the single entry's label_length (u4 at byte offset 16 of the 24-byte entry) so the label
    # would extend into the trailing entry-count word.
    id_map[16:20] = np.frombuffer(int(len("class: cone;") + 4).to_bytes(4, "little"), dtype=np.uint8)
    with pytest.raises(ValueError, match="Corrupt SemanticIdMap"):
        decode_semantic_id_map(id_map)


def test_build_semantic_id_to_labels_non_colorize_keys_by_id():
    """Non-colorized idToLabels is keyed by decimal semantic ID and always includes the reserved entries."""
    id_to_labels = build_semantic_id_to_labels({2: {"class": "cone"}}, colorize=False, device="cpu")
    assert id_to_labels == {
        "0": {"class": "BACKGROUND"},
        "1": {"class": "UNLABELLED"},
        "2": {"class": "cone"},
    }


def test_build_semantic_id_to_labels_colorize_keys_by_color():
    """Colorized idToLabels is keyed by ``"(r, g, b, a)"``; reserved IDs use the fixed BACKGROUND/UNLABELLED colors."""
    if not torch.cuda.is_available():
        pytest.skip("colorized key computation runs a Warp kernel that requires a CUDA device")

    id_to_labels = build_semantic_id_to_labels({2: {"class": "cone"}}, colorize=True, device="cuda:0")
    # BACKGROUND -> transparent black, UNLABELLED -> opaque black (matches the colorization kernel).
    assert id_to_labels["(0, 0, 0, 0)"] == {"class": "BACKGROUND"}
    assert id_to_labels["(0, 0, 0, 255)"] == {"class": "UNLABELLED"}
    assert {"class": "cone"} in id_to_labels.values()
    assert len(id_to_labels) == 3
