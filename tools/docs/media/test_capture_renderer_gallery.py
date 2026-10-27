# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from tools.docs.media import capture_renderer_gallery
from tools.docs.media.capture_renderer_gallery import motion_vectors_to_image, thumbnail_frame_index


@pytest.mark.parametrize(
    ("depth", "expected"),
    [
        (torch.tensor([[float("nan"), 1.5, 4.0, float("inf")]]), (1.5, 4.0)),
        (torch.tensor([[4.0, 20.0]]), (4.0, 20.0)),
    ],
)
def test_depth_display_bounds_use_finite_frame_extents(depth, expected):
    assert capture_renderer_gallery.depth_display_bounds(depth) == expected


def test_depth_display_bounds_reject_frame_with_no_finite_samples():
    depth = torch.tensor([[float("nan"), float("inf")]])

    with pytest.raises(ValueError, match="finite"):
        capture_renderer_gallery.depth_display_bounds(depth)


def test_gallery_capture_defaults_to_published_scene():
    parser = argparse.ArgumentParser()
    capture_renderer_gallery.add_gallery_arguments(parser)

    args = parser.parse_args(["--renderer-backend", "newton"])

    assert args.scene == (
        "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/Isaac/IsaacLab/Docs/"
        "Renderers/renderer_gallery_scene.usda"
    )


def test_gallery_capture_writes_camera_channels(tmp_path, monkeypatch):
    """Exercise the capture entry point and image conversion without starting a renderer."""
    import isaaclab.sim as sim_utils
    from isaaclab import cloner, utils

    parser = argparse.ArgumentParser()
    capture_renderer_gallery.add_gallery_arguments(parser)
    args = parser.parse_args(["--renderer-backend", "newton", "--frames", "6", "--output-dir", str(tmp_path)])
    args.device = "cpu"
    outputs = {
        "rgb": torch.full((1, 2, 2, 3), 128, dtype=torch.uint8),
        "albedo": torch.full((1, 2, 2, 3), 0.5),
        "depth": torch.tensor([[[[1.0], [2.0]], [[3.0], [4.0]]]]),
        "normals": torch.tensor([[[[1.0, 0.0, -1.0]] * 2] * 2]),
        "semantic_segmentation": torch.zeros((1, 2, 2, 1), dtype=torch.int32),
        "instance_segmentation": torch.full((1, 2, 2, 3), 64, dtype=torch.uint8),
    }
    camera = MagicMock(
        data=SimpleNamespace(output={key: SimpleNamespace(torch=value) for key, value in outputs.items()})
    )
    monkeypatch.setattr(sim_utils, "create_new_stage", MagicMock())
    monkeypatch.setattr(sim_utils, "SimulationContext", MagicMock())
    monkeypatch.setattr(cloner, "clone_plan_from_env_0", MagicMock())
    monkeypatch.setattr(cloner, "replicate", MagicMock())
    monkeypatch.setattr(utils, "instantiate", MagicMock())
    monkeypatch.setattr(capture_renderer_gallery, "_create_camera_and_reset", lambda *_: camera)

    capture_renderer_gallery._capture(args)

    for channel, expected in (
        ("rgb", (128, 128, 128)),
        ("albedo", (127, 127, 127)),
        ("normals", (255, 127, 0)),
        ("semantic_segmentation", (40, 40, 40)),
        ("instance_segmentation", (64, 64, 64)),
    ):
        with Image.open(tmp_path / capture_renderer_gallery.gallery_asset_name("newton", channel)) as image:
            np.testing.assert_allclose(np.asarray(image)[0, 0], expected, atol=1 if channel == "rgb" else 0)
    with Image.open(tmp_path / capture_renderer_gallery.gallery_asset_name("newton", "depth")) as image:
        pixels = np.asarray(image)
        assert pixels.shape == (2, 2, 3)
        assert np.unique(pixels.reshape(-1, 3), axis=0).shape[0] == 4


def test_gallery_scene_preserves_nucleus_uri():
    scene_uri = "omniverse://example.com/path/to/scene.usda"

    assert capture_renderer_gallery.resolve_gallery_scene(scene_uri) == scene_uri


def test_gallery_scene_resolves_existing_local_path(tmp_path, monkeypatch):
    scene_path = tmp_path / "scene.usda"
    scene_path.touch()
    monkeypatch.chdir(tmp_path)

    assert capture_renderer_gallery.resolve_gallery_scene("scene.usda") == str(scene_path)


def test_gallery_scene_rejects_missing_local_path(tmp_path):
    scene_path = tmp_path / "missing.usda"

    with pytest.raises(ValueError, match="Scene does not exist"):
        capture_renderer_gallery.resolve_gallery_scene(str(scene_path))


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
