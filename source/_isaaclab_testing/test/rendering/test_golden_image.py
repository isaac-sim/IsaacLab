# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for renderer-independent golden-image and segmentation validation."""

import copy
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import rendering_runner
import torch
from golden_image import ImageComparison, camera_output_image, compare_to_golden, read_golden_range
from PIL import Image, PngImagePlugin
from rendering_cases import RenderCase
from rendering_runner import validate_motion_vectors, validate_segmentation
from rendering_scene_cfgs import RenderingSceneCfg, RenderingSceneSpec

from isaaclab.renderers.output_contract import RenderBufferKind


def _image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (24, 24), color)


def _compare(image: Image.Image, golden: Path, artifacts: Path) -> ImageComparison:
    return compare_to_golden(
        image,
        golden,
        label="unit-test",
        artifact_dir=artifacts,
        max_diff_pct=0.0,
        min_ssim=1.0,
    )


@pytest.mark.parametrize(
    ("data_type", "output", "expected_pixels", "expected_range"),
    [
        ("depth", torch.tensor([[[[2.0], [6.0]]]]), [(0, 0, 0), (255, 255, 255)], (2.0, 6.0)),
        (
            "rgb_hdr",
            torch.tensor([[[[float("inf"), -0.1, float("nan")], [0.1, 1.0, 3.0]]]]),
            [(8, 0, 8), (16, 90, 255)],
            (-0.1, 3.0),
        ),
    ],
)
def test_depth_hdr_pngs_use_the_golden_minmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data_type: str,
    output: torch.Tensor,
    expected_pixels: list[tuple[int, int, int]],
    expected_range: tuple[float, float],
) -> None:
    """Depth/HDR share PNG comparison and retain the reference normalization range."""
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "linear.png"
    assert read_golden_range(golden) is None
    kwargs = {
        "label": "linear",
        "artifact_dir": tmp_path / "artifacts",
        "max_diff_pct": 0.0,
        "min_ssim": None,
    }

    compare_to_golden(camera_output_image(output, data_type), golden, **kwargs)
    with Image.open(golden) as image:
        assert [image.getpixel((x, 0)) for x in range(image.width)] == expected_pixels
    value_range = read_golden_range(golden)
    assert value_range == pytest.approx(expected_range)
    comparison = compare_to_golden(camera_output_image(output, data_type, value_range), golden, **kwargs)
    assert comparison.passed and comparison.diff_pct == 0.0


def test_runner_reuses_golden_range_for_rescaled_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """A scaled depth capture must fail through the runner's reference-range selection."""
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    case = RenderCase("physx", "isaac_rtx", (RenderBufferKind.DEPTH,), scene="range_reuse")
    golden = tmp_path / case.scene / case.golden_filename(RenderBufferKind.DEPTH, "kit")
    golden.parent.mkdir()
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("range_min", "2.0")
    pnginfo.add_text("range_max", "6.0")
    image = Image.new("RGB", (2, 1))
    image.putdata([(0, 0, 0), (255, 255, 255)])
    image.save(golden, pnginfo=pnginfo)

    spec = RenderingSceneSpec(cfg=RenderingSceneCfg(num_envs=1, env_spacing=1.0), image_max_diff_pct=0.0)
    output = torch.tensor([[[[4.0], [12.0]]]])
    runtime = SimpleNamespace(camera_outputs=lambda: ({RenderBufferKind.DEPTH: output}, {}))
    monkeypatch.setattr(rendering_runner, "_GOLDEN_ROOT", tmp_path)
    monkeypatch.setattr(rendering_runner, "_ARTIFACT_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(rendering_runner, "make_rendering_scene_spec", lambda *_args: spec)
    monkeypatch.setattr(rendering_runner, "build_rendering_scene", lambda *_args, **_kwargs: nullcontext(runtime))

    with pytest.raises(AssertionError, match="Pixel difference"):
        rendering_runner.run_rendering_case(case, request)


@pytest.mark.parametrize(
    ("values", "expected_range", "expected_pixels"),
    [
        ([2.0, 6.0], ("2.0", "6.0"), (0, 255)),
        ([5.0, 5.0], ("0.0", "5.0"), (255, 255)),
        ([0.0, 0.0], ("0.0", "1.0"), (0, 0)),
    ],
)
def test_camera_output_range_is_independent_of_grid_padding(
    values: list[float], expected_range: tuple[str, str], expected_pixels: tuple[int, int]
) -> None:
    """Repeating views must not introduce zero into the measured range."""
    output = torch.tensor(values).reshape(1, 1, 2, 1)
    single = camera_output_image(output, "depth")
    batch = camera_output_image(output.repeat(4, 1, 1, 1), "depth")
    assert single.info == batch.info == dict(zip(("range_min", "range_max"), expected_range))
    assert tuple(single.getpixel((x, 0))[0] for x in range(2)) == expected_pixels


@pytest.mark.parametrize(
    "bounds", [("nan", "1"), ("0", "inf"), ("3", "2"), ("2", "2"), ("bad", "1"), (None, "6"), ("0", None)]
)
def test_invalid_golden_range_rejects_rescaled_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bounds: tuple[str | None, str | None]
) -> None:
    """Invalid bounds must fail even when independent normalization reproduces the golden pixels."""
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "invalid-range.png"
    pnginfo = PngImagePlugin.PngInfo()
    if bounds[0] is not None:
        pnginfo.add_text("range_min", bounds[0])
    if bounds[1] is not None:
        pnginfo.add_text("range_max", bounds[1])
    output = torch.tensor([[[[2.0], [6.0]]]])
    camera_output_image(output, "depth").save(golden, pnginfo=pnginfo)
    assert read_golden_range(golden) is None
    actual = camera_output_image(output * 2.0, "depth", read_golden_range(golden))
    comparison = compare_to_golden(
        actual,
        golden,
        label="invalid-range",
        artifact_dir=tmp_path,
        max_diff_pct=0.0,
        min_ssim=None,
    )
    assert not comparison.passed
    assert "Invalid normalization bounds" in comparison.error
    assert comparison.golden_path == golden


def test_corrupt_golden_returns_comparison_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PNG errors must retain the golden path in the comparison result."""
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS", raising=False)
    golden = tmp_path / "broken.png"
    golden.write_bytes(b"not a PNG")
    assert read_golden_range(golden) is None
    comparison = _compare(_image((20, 40, 80)), golden, tmp_path)
    assert not comparison.passed
    assert comparison.golden_path == golden
    assert str(golden) in comparison.error


@pytest.mark.parametrize(("value_range", "input_level"), [(None, 0), (None, 40), ((0.0, 0.5), 40)])
def test_rgb_uses_golden_range_or_unit_defaults(
    tmp_path: Path, value_range: tuple[float, float] | None, input_level: int
) -> None:
    """RGB keeps its uint8 conversion and honors the same normalization metadata as depth/HDR."""
    golden = tmp_path / "rgb.png"
    pnginfo = PngImagePlugin.PngInfo()
    expected_range = value_range or (0.0, 1.0)
    level = input_level
    if value_range is not None:
        pnginfo.add_text("range_min", repr(value_range[0]))
        pnginfo.add_text("range_max", repr(value_range[1]))
        level *= 2
    Image.new("RGB", (1, 1), (level, level, level)).save(golden, pnginfo=pnginfo)

    value_range = read_golden_range(golden)
    assert value_range == expected_range
    image = camera_output_image(torch.full((1, 1, 1, 3), input_level, dtype=torch.uint8), "rgb", value_range)
    comparison = compare_to_golden(
        image,
        golden,
        label="rgb",
        artifact_dir=tmp_path,
        max_diff_pct=0.0,
        min_ssim=None,
    )

    assert comparison.passed
    assert comparison.diff_pct == 0.0


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
    with Image.open(golden) as image:
        assert float(image.info["range_min"]) == 0.0
        assert float(image.info["range_max"]) == 1.0


def test_update_mode_overwrites_and_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ISAACLAB_UPDATE_GOLDENS", "1")
    golden = tmp_path / "golden.png"
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("range_min", "2.0")
    pnginfo.add_text("range_max", "6.0")
    _image((255, 0, 0)).save(golden, pnginfo=pnginfo)

    image = camera_output_image(torch.tensor([[[[0, 128, 0]]]]), "rgb", read_golden_range(golden))
    comparison = compare_to_golden(
        image,
        golden,
        label="update",
        artifact_dir=tmp_path / "artifacts",
        max_diff_pct=0.0,
    )

    assert comparison.passed
    assert Image.open(golden).getpixel((0, 0)) == (0, 128, 0)
    monkeypatch.delenv("ISAACLAB_UPDATE_GOLDENS")
    assert read_golden_range(golden) == (0.0, 1.0)


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
