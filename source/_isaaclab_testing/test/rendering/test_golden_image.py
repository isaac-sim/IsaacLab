# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer-independent golden-image and segmentation validation."""

import copy
from pathlib import Path

import pytest
import torch
from golden_image import compare_to_golden
from PIL import Image
from rendering_runner import validate_motion_vectors, validate_segmentation

from isaaclab.renderers.output_contract import RenderBufferKind

pytestmark = pytest.mark.kitless


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (24, 24), color)


def _compare(image: Image.Image, golden: Path, artifacts: Path):
    return compare_to_golden(
        image,
        golden,
        label="unit-test",
        artifact_dir=artifacts,
        max_diff_pct=0.0,
        min_ssim=1.0,
    )


def test_identical_image_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "golden.png"
    image = _image((20, 40, 80))
    image.save(golden)

    comparison = _compare(image, golden, tmp_path / "artifacts")

    assert comparison.passed
    assert comparison.diff_pct == 0.0
    assert comparison.ssim == pytest.approx(1.0)
    assert comparison.actual_path is None


def test_difference_fails_and_writes_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "golden.png"
    _image((255, 0, 0)).save(golden)

    comparison = _compare(_image((0, 0, 255)), golden, tmp_path / "artifacts")

    assert not comparison.passed
    assert comparison.diff_pct == 100.0
    assert comparison.actual_path is not None and comparison.actual_path.exists()
    assert comparison.golden_path is not None and comparison.golden_path.exists()


def test_pixel_threshold_uses_rgb_distance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "golden.png"
    _image((0, 0, 0)).save(golden)

    small_shift = compare_to_golden(
        _image((6, 6, 0)),
        golden,
        label="small-shift",
        artifact_dir=tmp_path / "artifacts",
        max_diff_pct=0.0,
        min_ssim=None,
    )
    large_delta = compare_to_golden(
        _image((6, 8, 1)),
        golden,
        label="large-delta",
        artifact_dir=tmp_path / "artifacts",
        max_diff_pct=0.0,
        min_ssim=None,
    )

    assert small_shift.passed and small_shift.diff_pct == 0.0
    assert not large_delta.passed and large_delta.diff_pct == 100.0


def test_missing_baseline_is_bootstrapped_but_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "missing.png"

    comparison = _compare(_image((1, 2, 3)), golden, tmp_path / "artifacts")

    assert golden.exists()
    assert not comparison.passed
    assert comparison.error is not None and "missing" in comparison.error.lower()


def test_update_mode_overwrites_and_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAACLAB_UPDATE_GOLDENS", "1")
    golden = tmp_path / "golden.png"
    _image((255, 0, 0)).save(golden)
    replacement = _image((0, 255, 0))

    comparison = _compare(replacement, golden, tmp_path / "artifacts")

    assert comparison.passed
    assert Image.open(golden).getpixel((0, 0)) == (0, 255, 0)


def test_semantic_segmentation_labels_and_visible_area() -> None:
    kind = RenderBufferKind.SEMANTIC_SEGMENTATION
    table, background = (153, 74, 33, 255), (0, 0, 0, 0)
    outputs = {kind: torch.tensor([background, table], dtype=torch.uint8).reshape(1, 1, 2, 4)}
    info = {kind: {"idToLabels": {table: {"class": "table"}}}}
    expected = {"/World/Table": "table"}

    validate_segmentation(outputs, info, expected, min_semantic_pixels={"table": 1})
    with pytest.raises(AssertionError, match="visible pixels"):
        validate_segmentation(outputs, info, expected, min_semantic_pixels={"table": 2})

    # Numeric IDs may change, but each class must retain its spatial mask.
    robot = (246, 184, 59, 255)
    palette = {"table": table, "robot": robot}
    expected["/World/Robot"] = "robot"
    reference = torch.tensor([background, table, robot], dtype=torch.uint8).reshape(1, 1, 3, 4)
    renumbered = torch.tensor([background, robot, table], dtype=torch.uint8).reshape(1, 1, 3, 4)
    info[kind]["idToLabels"] = {robot: {"class": "table"}, table: {"class": "robot"}}
    outputs[kind] = renumbered.clone()
    validate_segmentation(outputs, info, expected, semantic_colors=palette)
    assert torch.equal(outputs[kind], reference)

    outputs[kind] = renumbered
    info[kind]["idToLabels"] = {robot: {"class": "robot"}, table: {"class": "table"}}
    validate_segmentation(outputs, info, expected, semantic_colors=palette)
    assert not torch.equal(outputs[kind], reference)


def _instance_segmentation_data():
    background, unlabelled, robot = (0, 0, 0, 0), (0, 0, 0, 255), (12, 34, 56, 255)
    kind = RenderBufferKind.INSTANCE_SEGMENTATION
    outputs = {kind: torch.tensor([background, robot], dtype=torch.uint8).reshape(1, 1, 2, 4)}
    info = {
        kind: {
            "idToLabels": {background: "BACKGROUND", unlabelled: "UNLABELLED", robot: "/World/Robot"},
            "idToSemantics": {
                background: {"class": "BACKGROUND"},
                unlabelled: {"class": "UNLABELLED"},
                robot: {"class": "robot"},
            },
        }
    }
    return outputs, info, {"/World/Robot": "robot"}


def test_instance_segmentation_metadata_matches_pixels_and_scene() -> None:
    outputs, info, expected = _instance_segmentation_data()

    validate_segmentation(outputs, info, expected)

    invalid_info = copy.deepcopy(info)
    invalid_info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToSemantics"][(12, 34, 56, 255)] = {"class": "cube"}
    with pytest.raises(AssertionError, match="metadata mismatch"):
        validate_segmentation(outputs, invalid_info, expected)


def test_instance_segmentation_rejects_metadata_colors_absent_from_pixels() -> None:
    outputs, info, expected = _instance_segmentation_data()
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToLabels"][(99, 88, 77, 255)] = "/World/Extra"
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToSemantics"][(99, 88, 77, 255)] = {"class": "extra"}

    with pytest.raises(AssertionError, match="unrendered colors"):
        validate_segmentation(outputs, info, expected)


def test_partial_instance_metadata_accepts_scene_valid_static_mapping() -> None:
    outputs, info, expected = _instance_segmentation_data()
    occluded_color = (99, 88, 77, 255)
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToLabels"][occluded_color] = "/World/Occluded"
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToSemantics"][occluded_color] = {"class": "cube"}
    expected.update({"/World/Occluded": "cube", "/World/Clone": "robot"})

    validate_segmentation(outputs, info, expected, exact_instance_metadata=False)


def test_partial_instance_metadata_rejects_unknown_scene_instance() -> None:
    outputs, info, expected = _instance_segmentation_data()
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToLabels"][(12, 34, 56, 255)] = "/World/Unknown"

    with pytest.raises(AssertionError, match="scene-valid subset"):
        validate_segmentation(outputs, info, expected, exact_instance_metadata=False)


def test_motion_vectors_require_localized_motion() -> None:
    motion = torch.full((2, 16, 16, 4), 1.0e-5)
    motion[:, 4:10, 4:10, :2] = 1.0
    validate_motion_vectors(motion)
    # A single larger vector must not hide otherwise uniform whole-frame motion.
    motion[..., :2] = 1.0
    motion[:, 0, 0, :2] = 2.0
    with pytest.raises(AssertionError, match="not localized"):
        validate_motion_vectors(motion)
