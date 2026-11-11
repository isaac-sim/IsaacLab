# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OVRTX segmentation annotator utilities (map decode -> idToLabels / idToSemantics)."""

import importlib.metadata
import importlib.util

import pytest
import torch

_REQUIRED_MODULES = ("isaaclab_ov", "ovrtx")
_MISSING_MODULES = [module for module in _REQUIRED_MODULES if importlib.util.find_spec(module) is None]

pytestmark = [
    pytest.mark.skipif(
        bool(_MISSING_MODULES),
        reason=f"requires optional modules: {', '.join(_MISSING_MODULES)}",
    ),
]

if not _MISSING_MODULES:
    import numpy as np  # noqa: E402
    from isaaclab_ov.renderers import ovrtx_annotator_utils as annotator_utils
    from isaaclab_ov.renderers.ovrtx_annotator_utils import (  # noqa: E402
        StableId,
        StableIdLabelPair,
        _parse_semantic_label,
        build_instance_id_to_labels_and_semantics,
        build_semantic_id_to_labels,
        decode_semantic_id_map,
        decode_stable_id_map,
        decode_stable_id_semantic_id_map,
    )
    from isaaclab_ov.renderers.ovrtx_compat import OVRTX_USES_NAMED_SEGMENTATION_MAPS

else:
    OVRTX_USES_NAMED_SEGMENTATION_MAPS = False


def _encode_identifier_map(
    entries: "list[StableIdLabelPair]", *, named_tensors: bool = OVRTX_USES_NAMED_SEGMENTATION_MAPS
) -> "annotator_utils.IdentifierMap":
    """Encode identifier/label pairs into the requested OVRTX wire format.

    Named tensors contain four-word identifiers, UTF-8 label bytes, and label boundaries. The packed
    layout contains identifier records, label bytes, and a trailing little-endian entry count.
    """
    if named_tensors:
        encoded = [label.encode("utf-8") for _, label in entries]
        return {
            "id": np.array([identifier for identifier, _ in entries], dtype=np.uint32).reshape(-1, 4),
            "label_offsets": np.cumsum([0, *(len(label) for label in encoded)], dtype=np.uint64),
            "label_data": np.frombuffer(b"".join(encoded), dtype=np.uint8),
        }
    entry_dtype = np.dtype([("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")])
    header_size = len(entries) * entry_dtype.itemsize
    entry_arr = np.zeros(len(entries), dtype=entry_dtype)
    label_blob = b""
    for i, (id_words, label) in enumerate(entries):
        label_bytes = label.encode("utf-8")
        entry_arr[i]["id"] = id_words
        entry_arr[i]["label_length"] = len(label_bytes)
        entry_arr[i]["label_offset"] = header_size + len(label_blob)
        label_blob += label_bytes
    buffer = entry_arr.tobytes() + label_blob + int(len(entries)).to_bytes(4, "little")
    return np.frombuffer(buffer, dtype=np.uint8).copy()


def test_parse_semantic_label_splits_type_and_label():
    """Raw ``"<type>: <label>;"`` strings parse into ``{type: label}`` dicts like Replicator idToLabels."""
    assert _parse_semantic_label("class: cone;") == {"class": "cone"}
    assert _parse_semantic_label("class: robot; type: arm;") == {"class": "robot", "type": "arm"}
    assert _parse_semantic_label("") == {}


def test_decode_semantic_id_map_round_trips_entries():
    """Decoding an encoded SemanticIdMap recovers the semantic-ID-to-label mapping."""
    id_map = _encode_identifier_map([((2, 0, 0, 0), "class: cone;"), ((3, 0, 0, 0), "class: cube;")])
    assert decode_semantic_id_map(id_map) == {2: {"class": "cone"}, 3: {"class": "cube"}}


def test_decode_semantic_id_map_handles_empty_buffer():
    """A too-small SemanticIdMap buffer decodes to an empty mapping instead of raising."""
    assert decode_semantic_id_map(_encode_identifier_map([])) == {}
    assert annotator_utils._decode_identifier_map_packed(np.zeros(0, dtype=np.uint8), "SemanticIdMap") == []


def test_decode_semantic_id_map_rejects_corrupt_entry_count():
    """Packed entry counts cannot extend beyond the buffer."""
    id_map = _encode_identifier_map([((2, 0, 0, 0), "class: cone;")], named_tensors=False)
    id_map[-4:] = np.frombuffer(int(999).to_bytes(4, "little"), dtype=np.uint8)
    with pytest.raises(ValueError, match="Corrupt SemanticIdMap"):
        annotator_utils._decode_identifier_map_packed(id_map, "SemanticIdMap")


def test_decode_semantic_id_map_rejects_label_spilling_into_count_field():
    """Packed labels cannot overlap the trailing entry count."""
    id_map = _encode_identifier_map([((2, 0, 0, 0), "class: cone;")], named_tensors=False)
    id_map[16:20] = np.frombuffer(int(len("class: cone;") + 4).to_bytes(4, "little"), dtype=np.uint8)
    with pytest.raises(ValueError, match="Corrupt SemanticIdMap"):
        annotator_utils._decode_identifier_map_packed(id_map, "SemanticIdMap")


@pytest.mark.parametrize(
    "identifiers, offsets, error",
    [
        ([[2, 0, 0]], [0, 12], "counts do not match"),
        ([[2, 0, 0, 0]], [0], "counts do not match"),
        ([[2, 0, 0, 0]], [12, 0], "not monotonic"),
        ([[2, 0, 0, 0]], [0, 16], "exceed the label buffer"),
    ],
    ids=["identifier-shape", "offset-count", "descending-offsets", "label-bounds"],
)
def test_named_identifier_map_rejects_invalid_shapes_and_offsets(identifiers, offsets, error):
    """Named-map shapes and label boundaries are validated independently of packed-buffer guards."""
    id_map = {
        "id": np.array(identifiers, dtype=np.uint32),
        "label_offsets": np.array(offsets, dtype=np.uint64),
        "label_data": np.frombuffer(b"class: cone;", dtype=np.uint8),
    }
    with pytest.raises(ValueError, match=error):
        annotator_utils._decode_identifier_map_named(id_map, "SemanticIdMap")


@pytest.mark.parametrize("stable_shape, semantic_shape", [((4,), (1, 4)), ((1, 3), (1, 4)), ((1, 4), (2, 4))])
def test_named_stable_semantic_map_rejects_mismatched_identifier_shapes(stable_shape, semantic_shape):
    """Stable and semantic identifiers must both contain the same number of four-word IDs."""
    with pytest.raises(ValueError, match="must have shape"):
        annotator_utils._decode_stable_id_semantic_id_map_named(
            {
                "stable_id": np.zeros(stable_shape, dtype=np.uint32),
                "semantic_id": np.zeros(semantic_shape, dtype=np.uint32),
            }
        )


def test_build_semantic_id_to_labels_non_colorize_keys_by_id():
    """Non-colorized idToLabels is keyed by integer semantic ID and always includes the reserved entries."""
    id_to_labels = build_semantic_id_to_labels({2: {"class": "cone"}}, colorize=False, device="cpu")
    assert id_to_labels == {
        0: {"class": "BACKGROUND"},
        1: {"class": "UNLABELLED"},
        2: {"class": "cone"},
    }


def test_build_semantic_id_to_labels_colorize_keys_by_color():
    """Colorized idToLabels is keyed by ``(r, g, b, a)`` tuples; reserved IDs use the fixed reserved colors."""
    if not torch.cuda.is_available():
        pytest.skip("colorized key computation runs a Warp kernel that requires a CUDA device")

    id_to_labels = build_semantic_id_to_labels({2: {"class": "cone"}}, colorize=True, device="cuda:0")
    # BACKGROUND -> transparent black, UNLABELLED -> opaque black (matches the colorization kernel).
    assert id_to_labels[(0, 0, 0, 0)] == {"class": "BACKGROUND"}
    assert id_to_labels[(0, 0, 0, 255)] == {"class": "UNLABELLED"}
    assert {"class": "cone"} in id_to_labels.values()
    assert len(id_to_labels) == 3


def _encode_stable_id_semantic_id_map(
    entries: "list[tuple[StableId, int]]", *, named_tensors: bool = OVRTX_USES_NAMED_SEGMENTATION_MAPS
) -> "annotator_utils.IdentifierMap":
    """Encode stable/semantic ID pairs as named tensors or packed fixed-size records.

    Both formats use four words per identifier. Packed records contain ``{uint32 stableId[4];
    uint32 semanticId[4]}`` with no label blob or trailing count.
    """
    if named_tensors:
        return {
            "stable_id": np.array([identifier for identifier, _ in entries], dtype=np.uint32).reshape(-1, 4),
            "semantic_id": np.array([[identifier, 0, 0, 0] for _, identifier in entries], dtype=np.uint32).reshape(
                -1, 4
            ),
        }
    entry_dtype = np.dtype([("stable_id", "<u4", (4,)), ("semantic_id", "<u4", (4,))])
    entry_arr = np.zeros(len(entries), dtype=entry_dtype)
    for i, (stable_id_words, semantic_id) in enumerate(entries):
        entry_arr[i]["stable_id"] = stable_id_words
        entry_arr[i]["semantic_id"][0] = semantic_id
    return np.frombuffer(entry_arr.tobytes(), dtype=np.uint8).copy()


def test_decode_stable_id_map_round_trips_prim_paths():
    """Decoding an encoded StableIdMap recovers the four-word stable ID to prim-path mapping."""
    stable_id_map = _encode_identifier_map([((1, 0, 0, 0), "/World/Cone"), ((2, 3, 0, 0), "/World/Cube")])
    assert decode_stable_id_map(stable_id_map) == {(1, 0, 0, 0): "/World/Cone", (2, 3, 0, 0): "/World/Cube"}


def test_decode_stable_id_map_rejects_corrupt_entry_count():
    """A StableIdMap whose entry count exceeds the buffer raises with the map name in the message."""
    stable_id_map = _encode_identifier_map([((1, 0, 0, 0), "/World/Cone")], named_tensors=False)
    stable_id_map[-4:] = np.frombuffer(int(999).to_bytes(4, "little"), dtype=np.uint8)
    with pytest.raises(ValueError, match="Corrupt StableIdMap"):
        annotator_utils._decode_identifier_map_packed(stable_id_map, "StableIdMap")


def test_decode_stable_id_semantic_id_map_indexes_by_pixel_id():
    """Each StableIdSemanticIdMap entry decodes to (stable_id, semanticId[0]) at position pixel_id - 2."""
    buffer = _encode_stable_id_semantic_id_map([((1, 0, 0, 0), 2), ((5, 6, 0, 0), 3)])
    assert decode_stable_id_semantic_id_map(buffer) == [((1, 0, 0, 0), 2), ((5, 6, 0, 0), 3)]


def test_decode_stable_id_semantic_id_map_handles_empty_buffer():
    """A StableIdSemanticIdMap smaller than one entry decodes to an empty list instead of raising."""
    assert decode_stable_id_semantic_id_map(_encode_stable_id_semantic_id_map([])) == []
    assert annotator_utils._decode_stable_id_semantic_id_map_packed(np.zeros(0, dtype=np.uint8)) == []


def test_build_instance_id_to_labels_and_semantics_non_colorize_keys_by_id():
    """Non-colorized instance maps are keyed by integer pixel ID with the reserved BACKGROUND/UNLABELLED entries."""
    id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
        stable_id_semantic_id_map=[((1, 0, 0, 0), 2)],
        stable_id_to_path={(1, 0, 0, 0): "/World/Cone"},
        semantic_id_to_labels={2: {"class": "cone"}},
        colorize=False,
        device="cpu",
    )
    assert id_to_labels == {0: "BACKGROUND", 1: "UNLABELLED", 2: "/World/Cone"}
    assert id_to_semantics == {
        0: {"class": "BACKGROUND"},
        1: {"class": "UNLABELLED"},
        2: {"class": "cone"},
    }


def test_build_instance_id_to_labels_and_semantics_colorize_keys_by_color():
    """Colorized instance maps are keyed by ``(r, g, b, a)`` tuples; reserved IDs use the fixed reserved colors."""
    if not torch.cuda.is_available():
        pytest.skip("colorized key computation runs a Warp kernel that requires a CUDA device")

    id_to_labels, id_to_semantics = build_instance_id_to_labels_and_semantics(
        stable_id_semantic_id_map=[((1, 0, 0, 0), 2)],
        stable_id_to_path={(1, 0, 0, 0): "/World/Cone"},
        semantic_id_to_labels={2: {"class": "cone"}},
        colorize=True,
        device="cuda:0",
    )
    # BACKGROUND -> transparent black, UNLABELLED -> opaque black (matches the colorization kernel).
    assert id_to_labels[(0, 0, 0, 0)] == "BACKGROUND"
    assert id_to_labels[(0, 0, 0, 255)] == "UNLABELLED"
    assert "/World/Cone" in id_to_labels.values()
    assert id_to_semantics[(0, 0, 0, 0)] == {"class": "BACKGROUND"}
    assert id_to_semantics[(0, 0, 0, 255)] == {"class": "UNLABELLED"}
    assert {"class": "cone"} in id_to_semantics.values()
    assert len(id_to_labels) == 3 and len(id_to_semantics) == 3


@pytest.mark.parametrize("runtime_version", [None, "0.6.0", "0.7.0.dev1", "0.8.0"])
def test_segmentation_decoders_select_installed_wire_format(monkeypatch, runtime_version):
    """Import-time selection decodes real payloads, including prereleases and missing docs metadata."""
    from isaaclab_ov.renderers import ovrtx_compat

    def distribution_version(name):
        assert name == "ovrtx"
        if runtime_version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return runtime_version

    monkeypatch.setattr(importlib.metadata, "version", distribution_version)
    spec = importlib.util.spec_from_file_location("isaaclab_ov.renderers._compat_probe", ovrtx_compat.__file__)
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    monkeypatch.setattr(ovrtx_compat, "OVRTX_USES_NAMED_SEGMENTATION_MAPS", compat.OVRTX_USES_NAMED_SEGMENTATION_MAPS)
    spec = importlib.util.spec_from_file_location("isaaclab_ov.renderers._decoder_probe", annotator_utils.__file__)
    decoders = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoders)
    named = runtime_version in ("0.7.0.dev1", "0.8.0")
    assert decoders.decode_semantic_id_map(
        _encode_identifier_map([((2, 0, 0, 0), "class: cone;")], named_tensors=named)
    ) == {2: {"class": "cone"}}
    assert decoders.decode_stable_id_semantic_id_map(
        _encode_stable_id_semantic_id_map([((42, 0, 0, 0), 2)], named_tensors=named)
    ) == [((42, 0, 0, 0), 2)]
