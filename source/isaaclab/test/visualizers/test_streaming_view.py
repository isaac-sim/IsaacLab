# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for streaming view colorization and layout utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from isaaclab.envs.utils.camera_colorizer import (
    CameraFrameColorizer,
    sensor_key_for_gt_type,
    sensor_keys_for_gt_types,
)
from isaaclab.envs.utils.camera_view import _best_streaming_cols, compose_streaming_grid

# ---------------------------------------------------------------------------
# CameraFrameColorizer
# ---------------------------------------------------------------------------


def _fake_rgb(h: int = 48, w: int = 64) -> torch.Tensor:
    return torch.randint(0, 256, (h, w, 4), dtype=torch.uint8)


def _fake_depth(h: int = 48, w: int = 64) -> torch.Tensor:
    return torch.rand(h, w, 1)


def _fake_seg(h: int = 48, w: int = 64) -> torch.Tensor:
    t = torch.zeros(h, w, 4, dtype=torch.uint8)
    t[10:30, 10:30, 0] = 1  # class 1 patch
    t[30:45, 5:20, 0] = 2  # class 2 patch
    return t


def test_colorize_rgb_drops_alpha_channel():
    rgb_t = _fake_rgb()
    out = CameraFrameColorizer.colorize(rgb_t, "rgb")
    expected = rgb_t.numpy()[..., :3]
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, expected)


def test_colorize_depth_clamps_range():
    """Depth outside [depth_min, depth_max] maps to the color of the nearest bound."""

    def colorize(depth: float) -> np.ndarray:
        return CameraFrameColorizer.colorize(torch.full((48, 64, 1), depth), "depth", depth_min=1.0, depth_max=5.0)

    out_near = colorize(0.05)
    assert out_near.shape == (48, 64, 3)
    assert out_near.dtype == np.uint8
    np.testing.assert_array_equal(out_near, colorize(1.0))
    np.testing.assert_array_equal(colorize(9.0), colorize(5.0))


def test_colorize_depth_turbo_near_is_not_same_as_far():
    near = torch.zeros(4, 4, 1)
    far = torch.ones(4, 4, 1) * 10.0
    out_near = CameraFrameColorizer.colorize(near, "depth", depth_min=0.0, depth_max=10.0)
    out_far = CameraFrameColorizer.colorize(far, "depth", depth_min=0.0, depth_max=10.0)
    # Colors should differ — turbo is not monotone in any channel but the total RGB differs
    assert not np.array_equal(out_near, out_far), "Near and far depth should produce different colors"


def test_colorize_segmentation_background_is_dark_grey():
    seg = torch.zeros(10, 10, 4, dtype=torch.uint8)  # all background
    out = CameraFrameColorizer.colorize(seg, "segmentation")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, np.full((10, 10, 3), 40, dtype=np.uint8))


def test_colorize_unsupported_type_raises():
    with pytest.raises(ValueError, match="not supported"):
        CameraFrameColorizer.colorize(_fake_rgb(), "optical_flow")


# ---------------------------------------------------------------------------
# sensor_key_for_gt_type
# ---------------------------------------------------------------------------


def test_sensor_key_rgb():
    assert sensor_key_for_gt_type("rgb") == "rgb"


def test_sensor_key_depth_primary():
    assert sensor_key_for_gt_type("depth", frozenset({"rgb", "depth"})) == "depth"


def test_sensor_key_depth_fallback():
    assert sensor_key_for_gt_type("depth", frozenset({"rgb", "distance_to_image_plane"})) == "distance_to_image_plane"


def test_sensor_key_depth_missing_raises():
    with pytest.raises(KeyError):
        sensor_key_for_gt_type("depth", frozenset({"rgb"}))


def test_sensor_key_segmentation():
    assert sensor_key_for_gt_type("segmentation") == "semantic_segmentation"


def test_sensor_key_unsupported_raises():
    with pytest.raises(ValueError):
        sensor_key_for_gt_type("optical_flow")


def test_sensor_keys_for_gt_types_deduplication():
    keys = sensor_keys_for_gt_types(["rgb", "depth", "rgb"])
    assert keys == ["rgb", "depth"], "Duplicate gt types should not produce duplicate sensor keys"


# ---------------------------------------------------------------------------
# Layout algorithm: _best_streaming_cols and compose_streaming_grid
# ---------------------------------------------------------------------------


def test_layout_single_gt_square_grid():
    # 16 envs × 1 GT, 240h×320w frame → balanced-grid algorithm picks 4 cols (4×4, no ragged row)
    # over 5 cols (5 cols → 5+5+5+1 = ragged last row) because completeness is prioritised
    # over aspect-ratio optimisation.
    cols = _best_streaming_cols(16, 1, 240, 320)
    assert cols == 4


def test_layout_multi_gt_rows_envs():
    # 3 envs × 3 GTs, 240×320 frame → 2 cols is most square: 2*3*320=1920 × 2*240=480
    # vs 1 col: 960 × 720. log(1920/480)=1.39, log(960/720)=0.29 → 1 col wins
    cols = _best_streaming_cols(3, 3, 240, 320)
    assert cols == 1  # 1 env-col × 3 rows


def test_layout_6_envs_2_gt():
    # 6 envs × 2 GTs, 240×320: should pick 2 env-cols (1280×720) not 1 (640×1440)
    cols = _best_streaming_cols(6, 2, 240, 320)
    assert cols == 2


def test_compose_grid_single_gt_single_env():
    h, w = 48, 64
    frame = np.ones((h, w, 3), dtype=np.uint8) * 128
    composite = compose_streaming_grid([frame], n_envs=1, n_gt=1)
    assert composite.shape == (h, w, 3)
    np.testing.assert_array_equal(composite, frame)


def test_compose_grid_pixels_placed_correctly():
    h, w = 10, 10
    # 2 envs × 2 GTs, red / green / blue / yellow
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    frames = [np.full((h, w, 3), c, dtype=np.uint8) for c in colors]
    # _best_streaming_cols(2, 2, 10, 10) → 2 gives W=40,H=10 → 1 gives W=20,H=20 → 1 wins
    composite = compose_streaming_grid(frames, n_envs=2, n_gt=2)
    # 1 env-col × 2 rows, 2 GT cols → shape (20, 20, 3)
    assert composite.shape == (20, 20, 3)
    # env0_gt0 = red at top-left
    np.testing.assert_array_equal(composite[:h, :w], np.full((h, w, 3), colors[0], dtype=np.uint8))
    # env0_gt1 = green at top-right
    np.testing.assert_array_equal(composite[:h, w:], np.full((h, w, 3), colors[1], dtype=np.uint8))
    # env1_gt0 = blue at bottom-left
    np.testing.assert_array_equal(composite[h:, :w], np.full((h, w, 3), colors[2], dtype=np.uint8))


# ---------------------------------------------------------------------------
# High-impact streaming view feature tests
# ---------------------------------------------------------------------------


def test_camera_colorizer_segmentation_golden_ratio_hues():
    n_classes = 10
    h, w = 10, 10
    mean_colors = []
    for i in range(1, n_classes + 1):
        seg = torch.zeros(h, w, 4, dtype=torch.uint8)
        seg[:, :, 0] = i
        out = CameraFrameColorizer.colorize(seg, "segmentation")
        mean_colors.append(out.mean(axis=(0, 1)))

    for i in range(len(mean_colors)):
        for j in range(i + 1, len(mean_colors)):
            assert not np.allclose(mean_colors[i], mean_colors[j]), (
                f"Class IDs {i + 1} and {j + 1} produced the same mean color {mean_colors[i]}"
            )


def test_best_streaming_cols_never_exceeds_env_count():
    for n_envs in [1, 2, 3, 4, 6, 8, 16]:
        for n_gt in [1, 2, 3]:
            cols = _best_streaming_cols(n_envs, n_gt, 240, 320)
            assert cols <= n_envs, f"_best_streaming_cols returned {cols} > n_envs={n_envs} for n_gt={n_gt}"


def test_streaming_gt_types_validated_on_setup():
    import unittest.mock as mock

    from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg

    cfg = NewtonGLVisualizerCfg(streaming_view=True, streaming_gt_types=["rgb", "optical_flow"])
    viz = mock.MagicMock()
    viz.cfg = cfg
    viz._uses_streaming_view = lambda: bool(cfg.streaming_view)

    with pytest.raises(ValueError, match="optical_flow"):
        NewtonGLVisualizer._setup_streaming_view(viz, 4)


# ---------------------------------------------------------------------------
# Normals colorization tests
# ---------------------------------------------------------------------------


def test_colorize_normals_maps_xyz_to_rgb():
    """Normals map XYZ in [-1, 1] to RGB: up (0, 0, 1) is (128, 128, 255) and right (1, 0, 0) is (255, 128, 128)."""
    n = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])  # (1, 2, 4) XYZW
    out = CameraFrameColorizer.colorize(n, "normals")
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.uint8
    # 0 maps to 127.5, which may land on either 127 or 128.
    np.testing.assert_allclose(out, [[[127.5, 127.5, 255.0], [255.0, 127.5, 127.5]]], atol=1.0)


def test_normals_sensor_key():
    """Sensor key for 'normals' is 'normals'."""
    assert sensor_key_for_gt_type("normals") == "normals"


def test_best_streaming_cols_target_aspect_16_9():
    """target_aspect=16/9 should prefer a wider layout than the default square target."""
    cols_widescreen = _best_streaming_cols(6, 1, 240, 320, target_aspect=16 / 9)
    cols_square = _best_streaming_cols(6, 1, 240, 320)
    assert cols_widescreen > cols_square, (
        f"Expected more columns for 16:9 target ({cols_widescreen}) than square ({cols_square})"
    )


def test_compose_streaming_grid_invalid_target_aspect_fallback():
    """Non-positive or non-finite target_aspect falls back to 1.0 without raising."""
    import math

    h, w = 48, 64
    frames = [np.zeros((h, w, 3), dtype=np.uint8)] * 4
    shape_default = compose_streaming_grid(frames, 4, 1).shape
    assert compose_streaming_grid(frames, 4, 1, target_aspect=float("nan")).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=0.0).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=-1.0).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=math.inf).shape == shape_default
