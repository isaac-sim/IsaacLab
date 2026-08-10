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
from rendering_runner import _validate_segmentation

from isaaclab.renderers.output_contract import RenderBufferKind


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

    _validate_segmentation(outputs, info, expected)

    invalid_info = copy.deepcopy(info)
    invalid_info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToSemantics"][(12, 34, 56, 255)] = {"class": "cube"}
    with pytest.raises(AssertionError, match="metadata mismatch"):
        _validate_segmentation(outputs, invalid_info, expected)


def test_instance_segmentation_rejects_metadata_colors_absent_from_pixels() -> None:
    outputs, info, expected = _instance_segmentation_data()
    info[RenderBufferKind.INSTANCE_SEGMENTATION]["idToLabels"][(99, 88, 77, 255)] = "/World/Extra"

    with pytest.raises(AssertionError, match="rendered colors"):
        _validate_segmentation(outputs, info, expected)
