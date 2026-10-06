# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming display channel, colorization, and pixel layout contracts."""

import math
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp

from isaaclab.envs.utils.camera_colorizer import CameraFrameColorizer, sensor_key_for_gt_type, sensor_keys_for_gt_types
from isaaclab.envs.utils.camera_view import compose_streaming_grid
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.image_composition import compose_image, prepare_image_composition


def test_colorize_rgb_drops_alpha_channel():
    pixels = torch.randint(0, 256, (4, 6, 4), dtype=torch.uint8)
    out = CameraFrameColorizer.colorize(pixels, "rgb")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, pixels.numpy()[..., :3])
    with pytest.raises(ValueError, match="not supported"):
        CameraFrameColorizer.colorize(pixels, "optical_flow")


def test_colorize_depth_clamps_range():
    depth = torch.tensor([[[0.05], [1.0], [5.0], [10.0]]])
    out = CameraFrameColorizer.colorize(depth, "depth", depth_min=1.0, depth_max=5.0)
    assert out.shape == (1, 4, 3) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], out[0, 1])
    np.testing.assert_array_equal(out[0, 2], out[0, 3])
    assert not np.array_equal(out[0, 1], out[0, 2])


def test_colorize_segmentation_background_and_classes():
    seg = torch.zeros(1, 11, 4, dtype=torch.uint8)
    seg[0, :, 0] = torch.arange(11)
    out = CameraFrameColorizer.colorize(seg, "segmentation")
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[0, 0], [40, 40, 40])
    assert len(np.unique(out[0, 1:], axis=0)) == 10


@pytest.mark.parametrize(
    "gt,available,expected",
    [
        ("rgb", None, "rgb"),
        ("rgb", {"rgba"}, "rgba"),
        ("rgb", {"rgb", "rgba"}, "rgb"),
        ("depth", {"depth", "distance_to_image_plane"}, "depth"),
        ("depth", {"rgb", "distance_to_image_plane"}, "distance_to_image_plane"),
        ("segmentation", None, "semantic_segmentation"),
        ("normals", None, "normals"),
    ],
)
def test_sensor_key_for_display_channel(gt, available, expected):
    available = frozenset(available) if available is not None else None
    assert sensor_key_for_gt_type(gt, available) == expected
    assert sensor_key_for_gt_type(gt, available, required=False) == expected


def test_sensor_key_missing_or_unknown():
    with pytest.raises(KeyError):
        sensor_key_for_gt_type("depth", frozenset({"rgb"}))
    assert sensor_key_for_gt_type("depth", frozenset({"rgb"}), required=False) is None
    with pytest.raises(ValueError):
        sensor_key_for_gt_type("optical_flow")


def test_sensor_keys_for_gt_types_deduplication():
    assert sensor_keys_for_gt_types(["rgb", "depth", "rgb"]) == ["rgb", "depth"]


@pytest.mark.parametrize(
    "envs,channels,aspect,columns",
    [
        (1, 1, 1.0, 1),
        (2, 2, 1.0, 1),
        (16, 1, 1.0, 4),
        (3, 3, 1.0, 1),
        (6, 2, 1.0, 2),
        (6, 1, 1.0, 2),
        (6, 1, 16 / 9, 3),
    ],
)
@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_compose_grid_pixels_placed_correctly(envs, channels, aspect, columns, device):
    height, width = 6, 8
    frames = [np.full((height, width, 3), i + 1, dtype=np.uint8) for i in range(envs * channels)]
    composite = compose_streaming_grid(frames, envs, channels, target_aspect=aspect)
    assert composite.shape == (math.ceil(envs / columns) * height, columns * channels * width, 3)
    # Read rows of complete tiles, independently of the compositing loop's environment/channel indexing.
    tiles = composite.reshape(-1, height, columns * channels, width, 3).transpose(0, 2, 1, 3, 4)
    np.testing.assert_array_equal(tiles.reshape(-1, height, width, 3)[: len(frames)], frames)
    assert not tiles.reshape(-1, height, width, 3)[len(frames) :].any()

    # Use non-consecutive source rows to catch a composer that ignores the selection.
    selected = list(range(envs * 2 - 1, 0, -2))
    sources = []
    for channel in range(channels):
        batch = np.full((envs * 2, height, width, 4), 255, dtype=np.uint8)
        for row, env in enumerate(selected):
            batch[env, ..., :3] = frames[row * channels + channel]
        sources.append(wp.array(batch, device=device))
    params = prepare_image_composition(tuple(sources), ("rgb",) * channels, selected, target_aspect=aspect)
    output = wp.empty(params.output_shape, dtype=wp.uint8, device=device)
    compose_image(output, tuple(sources), params)
    np.testing.assert_array_equal(output.numpy()[..., :3], composite)
    assert np.all(output.numpy()[..., 3] == 255)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_device_colorization_matches_recording_and_reuses_storage(device, monkeypatch):
    """Mixed sensor outputs compose on their device, including strides, invalid depth, and packed/raw IDs."""
    wp.init()
    rng = np.random.default_rng(3)
    shape = (2, 16, 16)
    rgb = torch.from_numpy(rng.integers(0, 256, (*shape, 4), dtype=np.uint8)).to(device)[..., :3]
    depth = rng.uniform(-1, 15, (*shape, 1)).astype(np.float32)
    depth[0, 0, :3, 0] = (np.nan, np.inf, -np.inf)
    depth[1, ..., 0] = (0.1 + np.arange(256).reshape(16, 16) / 256 * 9.9).astype(np.float32)
    normals = rng.uniform(-1.5, 1.5, (*shape, 3)).astype(np.float32)
    ids = rng.integers(0, 2**24, (*shape, 1), dtype=np.int32)
    ids[0, 0, :5, 0] = (0, 1, 2**24 - 1, -1, 2**31 - 1)
    packed = np.concatenate([ids % 256, (ids // 256) % 256, ids // 65536], axis=-1).astype(np.uint8)
    host = (rgb.cpu().numpy(), depth, normals, ids, packed)
    sources = (wp.from_torch(rgb), *(wp.array(array, device=device) for array in host[1:]))
    # Reject an incompatible layout during preparation, before a kernel can read past its channels.
    with pytest.raises(ValueError, match="channels"):
        prepare_image_composition((sources[1],), ("rgb",), [0])
    channels = ("rgb", "depth", "normals", "segmentation", "segmentation")
    params = prepare_image_composition(sources, channels, [1, 0])
    output = wp.empty(params.output_shape, dtype=wp.uint8, device=device)
    pointer = output.ptr
    for _ in range(2):
        with monkeypatch.context() as execution:
            from isaaclab.sim import SimulationContext

            forbidden = Mock(side_effect=AssertionError("Composition must only read arrays and write its output"))
            execution.setattr(SimulationContext, "instance", forbidden)
            execution.setattr(wp.array, "numpy", forbidden)
            execution.setattr(torch.Tensor, "cpu", forbidden)
            execution.setattr(wp, "empty", forbidden)
            compose_image(output, sources, params)
        frames = [CameraFrameColorizer.colorize(array[env], gt) for env in (1, 0) for array, gt in zip(host, channels)]
        expected = compose_streaming_grid(frames, 2, len(channels))
        np.testing.assert_array_equal(output.numpy()[..., :3], expected)
        assert output.ptr == pointer
        assert np.all(output.numpy()[..., 3] == 255)
        depth.fill(2.0)
        sources[1].assign(depth)


def test_colorize_normals_maps_xyz_to_rgb():
    """Normals map XYZ in [-1, 1] to RGB: up (0, 0, 1) is (128, 128, 255) and right (1, 0, 0) is (255, 128, 128)."""
    n = torch.tensor([[[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])  # (1, 2, 4) XYZW
    out = CameraFrameColorizer.colorize(n, "normals")
    assert out.shape == (1, 2, 3)
    assert out.dtype == np.uint8
    # 0 maps to 127.5, which may land on either 127 or 128.
    np.testing.assert_allclose(out, [[[127.5, 127.5, 255.0], [255.0, 127.5, 127.5]]], atol=1.0)


def test_compose_streaming_grid_invalid_target_aspect_fallback():
    """Non-positive or non-finite target_aspect falls back to 1.0 without raising."""
    h, w = 48, 64
    frames = [np.zeros((h, w, 3), dtype=np.uint8)] * 4
    shape_default = compose_streaming_grid(frames, 4, 1).shape
    assert compose_streaming_grid(frames, 4, 1, target_aspect=float("nan")).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=0.0).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=-1.0).shape == shape_default
    assert compose_streaming_grid(frames, 4, 1, target_aspect=math.inf).shape == shape_default
