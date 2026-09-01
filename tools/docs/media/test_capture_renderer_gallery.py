# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse

import numpy as np
import pytest
import torch

from tools.docs.media import capture_renderer_gallery
from tools.docs.media.capture_renderer_gallery import motion_vectors_to_image, thumbnail_frame_index


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (torch.tensor([[float("nan"), 1.5, 4.0, float("inf")]]), (1.5, 4.0)),
        (torch.tensor([[4.0, 20.0]]), (2.0, 13.0)),
    ],
)
def test_depth_display_bounds_use_finite_frame_extents_with_gallery_limits(depth, expected):
    assert capture_renderer_gallery.depth_display_bounds(depth) == expected


def test_depth_display_bounds_fall_back_when_frame_has_no_finite_samples():
    depth = torch.tensor([[float("nan"), float("inf")]])

    assert capture_renderer_gallery.depth_display_bounds(depth) == (2.0, 13.0)


def test_gallery_capture_requires_explicit_scene_path():
    parser = argparse.ArgumentParser()
    capture_renderer_gallery.add_gallery_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--renderer-backend", "newton"])


def test_thumbnail_uses_sixth_captured_frame():
    assert thumbnail_frame_index(6) == 5


@pytest.mark.parametrize("frame_count", [0, 5])
def test_thumbnail_requires_six_frames(frame_count):
    with pytest.raises(ValueError, match="six"):
        thumbnail_frame_index(frame_count)


def test_motion_vector_image_includes_direction_arrows():
    vectors = torch.zeros((64, 64, 4), dtype=torch.float32)
    vectors[8:56, 8:56, 0] = 1.0
    vectors[8:56, 8:56, 1] = 0.5

    image = motion_vectors_to_image(vectors)

    pixels = np.asarray(image)
    assert pixels.shape == (64, 64, 3)
    assert np.any(np.all(pixels == 255, axis=-1))


def test_motion_vector_arrows_convert_v_axis_to_screen_y():
    vectors = torch.zeros((64, 64, 4), dtype=torch.float32)
    vectors[32, 32, 1] = -1.0

    pixels = np.asarray(motion_vectors_to_image(vectors))
    white_pixels = np.all(pixels == 255, axis=-1)

    assert np.any(white_pixels[33:, 28:37])
    assert not np.any(white_pixels[:32])
