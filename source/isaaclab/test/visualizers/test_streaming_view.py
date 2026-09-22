# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for streaming view colorization and layout utilities."""

from __future__ import annotations

import math
from unittest import mock

import numpy as np
import pytest
import torch

from isaaclab.envs.utils.camera_colorizer import (
    SUPPORTED_GT_TYPES,
    CameraFrameColorizer,
    sensor_key_for_gt_type,
    sensor_keys_for_gt_types,
)
from isaaclab.envs.utils.camera_view import _best_streaming_cols, compose_streaming_grid

pytestmark = pytest.mark.unit

_H, _W = 48, 64


def _fake_frame(gt_type: str) -> torch.Tensor:
    if gt_type == "rgb":
        return torch.randint(0, 256, (_H, _W, 4), dtype=torch.uint8)
    if gt_type == "depth":
        return torch.rand(_H, _W, 1)
    if gt_type == "segmentation":
        frame = torch.zeros(_H, _W, 4, dtype=torch.uint8)
        frame[10:30, 10:30, 0] = 1
        frame[30:45, 5:20, 0] = 2
        return frame
    normals = torch.zeros(_H, _W, 4)  # XYZW
    normals[..., 2] = 1.0
    return normals


#
# CameraFrameColorizer
#


@pytest.mark.parametrize("gt_type", ["rgb", "depth", "segmentation", "normals"])
def test_colorize_output_is_rgb_uint8(gt_type):
    out = CameraFrameColorizer.colorize(_fake_frame(gt_type), gt_type, depth_min=0.1, depth_max=10.0)
    assert out.shape == (_H, _W, 3)
    assert out.dtype == np.uint8


def test_colorize_rgb_drops_alpha_channel():
    frame = _fake_frame("rgb")
    np.testing.assert_array_equal(CameraFrameColorizer.colorize(frame, "rgb"), frame.numpy()[..., :3])


def test_colorize_depth_clamps_and_separates_near_from_far():
    # uniform input below depth_min maps to one color
    near = torch.full((_H, _W, 1), 0.05)
    out_near = CameraFrameColorizer.colorize(near, "depth", depth_min=1.0, depth_max=5.0)
    assert (out_near == out_near[0, 0]).all()
    # near and far depth get different colors
    far = torch.full((4, 4, 1), 10.0)
    out_near = CameraFrameColorizer.colorize(torch.zeros(4, 4, 1), "depth", depth_min=0.0, depth_max=10.0)
    out_far = CameraFrameColorizer.colorize(far, "depth", depth_min=0.0, depth_max=10.0)
    assert not np.array_equal(out_near, out_far)


def test_colorize_segmentation_colors():
    """Background is dark grey and distinct class ids get distinct (golden-ratio spaced) colors."""
    background = CameraFrameColorizer.colorize(torch.zeros(10, 10, 4, dtype=torch.uint8), "segmentation")
    np.testing.assert_array_equal(background, np.full((10, 10, 3), 40, dtype=np.uint8))

    colors = []
    for class_id in range(1, 11):
        seg = torch.full((2, 2, 4), class_id, dtype=torch.uint8)
        colors.append(CameraFrameColorizer.colorize(seg, "segmentation")[0, 0])
    assert len({tuple(color) for color in colors}) == len(colors)


@pytest.mark.parametrize(
    "normal, channel",
    [((0.0, 0.0, 1.0, 0.0), 2), ((1.0, 0.0, 0.0, 0.0), 0)],
    ids=["up_is_blue", "right_is_red"],
)
def test_colorize_normals_axis_maps_to_channel(normal, channel):
    out = CameraFrameColorizer.colorize(torch.tensor([[normal]]), "normals")
    assert out[0, 0, channel] > 200
    assert all(out[0, 0, other] < 150 for other in range(3) if other != channel)


def test_colorize_unsupported_type_raises():
    with pytest.raises(ValueError, match="not supported"):
        CameraFrameColorizer.colorize(_fake_frame("rgb"), "optical_flow")


#
# sensor keys
#


@pytest.mark.parametrize(
    "gt_type, available, expected",
    [
        ("rgb", None, "rgb"),
        ("depth", frozenset({"rgb", "depth"}), "depth"),
        ("depth", frozenset({"rgb", "distance_to_image_plane"}), "distance_to_image_plane"),
        ("segmentation", None, "semantic_segmentation"),
        ("normals", None, "normals"),
    ],
)
def test_sensor_key_for_gt_type(gt_type, available, expected):
    args = () if available is None else (available,)
    assert sensor_key_for_gt_type(gt_type, *args) == expected


@pytest.mark.parametrize(
    "gt_type, available, error",
    [("depth", frozenset({"rgb"}), KeyError), ("optical_flow", None, ValueError)],
)
def test_sensor_key_for_gt_type_errors(gt_type, available, error):
    args = () if available is None else (available,)
    with pytest.raises(error):
        sensor_key_for_gt_type(gt_type, *args)


def test_sensor_keys_for_gt_types_deduplicates():
    assert sensor_keys_for_gt_types(["rgb", "depth", "rgb"]) == ["rgb", "depth"]
    assert "normals" in SUPPORTED_GT_TYPES


#
# layout: _best_streaming_cols and compose_streaming_grid
#


@pytest.mark.parametrize(
    "n_envs, n_gt, expected_cols",
    [
        # 16 envs × 1 GT: 4×4 is complete, 5 cols would leave a ragged last row
        (16, 1, 4),
        # 3 envs × 3 GTs: 1 col (960×720) is closer to square than 2 cols (1920×480)
        (3, 3, 1),
        # 6 envs × 2 GTs: 2 env-cols (1280×720) beats 1 col (640×1440)
        (6, 2, 2),
    ],
)
def test_best_streaming_cols_prefers_complete_square_layouts(n_envs, n_gt, expected_cols):
    assert _best_streaming_cols(n_envs, n_gt, 240, 320) == expected_cols


def test_best_streaming_cols_never_exceeds_env_count():
    for n_envs in [1, 2, 3, 4, 6, 8, 16]:
        for n_gt in [1, 2, 3]:
            assert _best_streaming_cols(n_envs, n_gt, 240, 320) <= n_envs


def test_best_streaming_cols_widescreen_target_adds_columns():
    assert _best_streaming_cols(6, 1, 240, 320, target_aspect=16 / 9) > _best_streaming_cols(6, 1, 240, 320)


def test_compose_grid_places_frames_in_env_rows_and_gt_columns():
    h, w = 10, 10
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    frames = [np.full((h, w, 3), color, dtype=np.uint8) for color in colors]
    # 2 envs × 2 GTs → 1 env-col × 2 rows, 2 GT cols
    composite = compose_streaming_grid(frames, n_envs=2, n_gt=2)
    assert composite.shape == (2 * h, 2 * w, 3)
    for (row, col), color in zip([(0, 0), (0, 1), (1, 0), (1, 1)], colors):
        assert (composite[row * h : (row + 1) * h, col * w : (col + 1) * w] == color).all()

    single = compose_streaming_grid([frames[0]], n_envs=1, n_gt=1)
    np.testing.assert_array_equal(single, frames[0])


@pytest.mark.parametrize("target_aspect", [float("nan"), 0.0, -1.0, math.inf])
def test_compose_grid_invalid_target_aspect_falls_back(target_aspect):
    frames = [np.zeros((_H, _W, 3), dtype=np.uint8)] * 4
    assert compose_streaming_grid(frames, 4, 1, target_aspect=target_aspect).shape == (
        compose_streaming_grid(frames, 4, 1).shape
    )


def test_streaming_gt_types_validated_on_setup():
    from isaaclab_visualizers.newton import NewtonGLVisualizer, NewtonGLVisualizerCfg

    cfg = NewtonGLVisualizerCfg(streaming_view=True, streaming_gt_types=["rgb", "optical_flow"])
    viz = mock.MagicMock()
    viz.cfg = cfg
    viz._uses_streaming_view = lambda: bool(cfg.streaming_view)

    with pytest.raises(ValueError, match="optical_flow"):
        NewtonGLVisualizer._setup_streaming_view(viz, 4)
