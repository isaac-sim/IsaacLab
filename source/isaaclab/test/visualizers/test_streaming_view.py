# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Streaming display channel, colorization, and pixel layout contracts."""

import math
from colorsys import hsv_to_rgb
from importlib.util import find_spec
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import warp as wp
from matplotlib import colormaps

from isaaclab.envs.utils import camera_view
from isaaclab.envs.utils.camera_view import image_grid_columns, sensor_key_for_gt_type
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.images import compose_image


def _reference_colorize(data, channel):
    """NumPy/stdlib reference independent of the device composition kernels."""
    if channel == "rgb":
        return data[..., :3]
    if channel == "depth":
        normalized = np.clip((data[..., 0] - 0.1) / 9.9, 0.0, 1.0)
        return (colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    if channel == "normals":
        return np.clip((data[..., :3] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    ids = data[..., 0].astype(np.int32)
    if data.shape[-1] >= 3:
        ids = (data[..., :3].astype(np.int32) * (1, 256, 65536)).sum(axis=-1)
    return np.array(
        [
            (40, 40, 40)
            if identifier == 0
            else np.array(hsv_to_rgb((int(identifier) * 0.6180339887) % 1, 0.75, 0.9)) * 255
            for identifier in ids.flat
        ],
        dtype=np.uint8,
    ).reshape((*ids.shape, 3))


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
    actual_columns = image_grid_columns(envs, channels, height, width, aspect)
    shape = (math.ceil(envs / actual_columns) * height, actual_columns * channels * width, 4)
    assert shape == (math.ceil(envs / columns) * height, columns * channels * width, 4)

    # Use non-consecutive source rows to catch a composer that ignores the selection.
    selected = list(range(envs * 2 - 1, 0, -2))
    sources = []
    for channel in range(channels):
        batch = np.full((envs * 2, height, width, 4), 255, dtype=np.uint8)
        for row, env in enumerate(selected):
            batch[env, ..., :3] = frames[row * channels + channel]
        sources.append(wp.array(batch, device=device))
    env_ids = wp.array(selected, dtype=wp.int32, device=device)
    depth_colors = wp.empty((0, 3), dtype=wp.uint8, device=device)
    output = wp.empty(shape, dtype=wp.uint8, device=device)
    compose_image(output, tuple(sources), env_ids, ("rgb",) * channels, depth_colors)
    # Read complete tiles independently of the kernel's destination-index calculation.
    pixels = output.numpy()
    tiles = pixels[..., :3].reshape(-1, height, columns * channels, width, 3).transpose(0, 2, 1, 3, 4)
    tiles = tiles.reshape(-1, height, width, 3)
    np.testing.assert_array_equal(tiles[: len(frames)], frames)
    assert not tiles[len(frames) :].any()
    assert np.all(pixels[..., 3] == 255)


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_device_colorization_matches_reference_and_reuses_storage(device, monkeypatch):
    """Mixed sensor outputs compose on their device, including strides, invalid depth, and packed/raw IDs."""
    wp.init()
    rng = np.random.default_rng(3)
    shape = (2, 16, 16)
    rgb = torch.from_numpy(rng.integers(0, 256, (*shape, 8), dtype=np.uint8)).to(device)[..., ::2]
    depth = rng.uniform(-1, 15, (*shape, 1)).astype(np.float32)
    depth[0, 0, :3, 0] = (np.nan, np.inf, -np.inf)
    depth[1, ..., 0] = (0.1 + np.arange(256).reshape(16, 16) / 256 * 9.9).astype(np.float32)
    normals = rng.uniform(-1.5, 1.5, (*shape, 3)).astype(np.float32)
    ids = rng.integers(0, 2**24, (*shape, 1), dtype=np.int32)
    ids[0, 0, :5, 0] = (0, 1, 2**24 - 1, -1, 2**31 - 1)
    packed = np.concatenate([ids % 256, (ids // 256) % 256, ids // 65536], axis=-1).astype(np.uint8)
    host = (rgb.cpu().numpy(), depth, normals, ids, packed)
    sources = (wp.from_torch(rgb), *(wp.array(array, device=device) for array in host[1:]))
    channels = ("rgb", "depth", "normals", "segmentation", "segmentation")
    env_ids = wp.array([1, 0], dtype=wp.int32, device=device)
    colors = (colormaps["turbo"](np.arange(256) / 255.0)[..., :3] * 255).astype(np.uint8)
    depth_colors = wp.array(colors, device=device)
    output = wp.empty((shape[0] * shape[1], len(channels) * shape[2], 4), dtype=wp.uint8, device=device)
    pointer = output.ptr
    for _ in range(2):
        launch = Mock(wraps=wp.launch)
        with monkeypatch.context() as execution:
            from isaaclab.sim import SimulationContext

            forbidden = Mock(side_effect=AssertionError("Composition must only read arrays and write its output"))
            execution.setattr(SimulationContext, "instance", forbidden)
            execution.setattr(wp.array, "numpy", forbidden)
            execution.setattr(torch.Tensor, "cpu", forbidden)
            execution.setattr(wp, "empty", forbidden)
            execution.setattr(wp, "launch", launch)
            compose_image(output, sources, env_ids, channels, depth_colors)
        # Different display channels must not share a runtime-switched kernel.
        assert len({call.args[0] for call in launch.call_args_list}) == len(set(channels))
        expected = np.concatenate([
            np.concatenate([_reference_colorize(array[env], gt) for array, gt in zip(host, channels)], axis=1)
            for env in (1, 0)
        ])  # fmt: skip
        np.testing.assert_array_equal(output.numpy()[..., :3], expected)
        assert output.ptr == pointer
        assert np.all(output.numpy()[..., 3] == 255)
        depth.fill(2.0)
        sources[1].assign(depth)


def test_image_grid_columns_invalid_target_aspect():
    """Invalid legacy display aspects retain the default grid shape."""
    for aspect in (float("nan"), 0.0, -1.0, math.inf):
        assert image_grid_columns(4, 1, 48, 64, aspect) == 2


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_window_and_recorder_share_fixed_device_image(device, monkeypatch):
    """One selected frame serves both consumers; resizing presentation preserves captured storage."""
    from copy import deepcopy
    from types import SimpleNamespace
    from unittest.mock import PropertyMock

    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils.warp import ProxyArray
    from isaaclab.visualizers import GLWindowCfg, ImageViewCfg

    assert find_spec("isaaclab.envs.utils.camera_colorizer") is None
    assert not {
        "camera_gt_batch",
        "camera_rgb_batch",
        "compose_streaming_grid",
        "compose_rgb_grid_tensor",
    }.intersection(vars(camera_view))
    rgb = wp.array(np.arange(3, dtype=np.uint8)[:, None, None, None] * np.ones((3, 2, 2, 4), np.uint8), device=device)
    depth = wp.full((3, 2, 2, 1), 2.0, dtype=wp.float32, device=device)
    data = SimpleNamespace(output={"rgba": ProxyArray(rgb), "distance_to_image_plane": ProxyArray(depth)})
    camera = Mock(cfg=SimpleNamespace(data_types=list(data.output)))
    acquired = PropertyMock(return_value=data)
    type(camera).data = acquired
    sim = object.__new__(SimulationContext)
    sim._scene_data_provider = SimpleNamespace(get_camera_sensors=Mock(return_value={"front": camera}))
    sim._image_views, sim._physics_step_count = {}, 0
    declaration = ImageViewCfg(source="front", envs=(2, 0), channels=("rgb", "depth"), size=(16, 20))
    window = deepcopy(GLWindowCfg(view=declaration))
    recorder_cfg = deepcopy(VideoRecorderCfg(view=declaration))
    monkeypatch.setattr("isaaclab.envs.utils.video_recorder.ImageSequenceClip", Mock())
    recorder = VideoRecorder(recorder_cfg, SimpleNamespace(sim=sim))
    view = sim.get_image_view(window.view)
    image = view.read(0)
    pixels = recorder._get_frame()
    assert sim.get_image_view(recorder_cfg.view) is view
    assert acquired.call_count == 1
    assert image.device == wp.get_device(device)
    colors = (colormaps["turbo"]((2.0 - 0.1) / 9.9)[:3] * np.array(255)).astype(np.uint8)
    expected = np.zeros((20, 16, 3), np.uint8)
    expected[2:10, :8] = 2
    expected[2:18, 8:] = colors
    np.testing.assert_array_equal(pixels, expected)

    # The window is only a consumer. Changing its dimensions leaves the sensor and view allocations intact.
    pointer, source_pointer = image.ptr, rgb.ptr
    window.size = (1920, 1080)
    with monkeypatch.context() as execution:
        forbidden = Mock(side_effect=AssertionError("GPU frame execution must not download or allocate images"))
        execution.setattr(wp.array, "numpy", forbidden)
        execution.setattr(wp, "empty", forbidden)
        execution.setattr(SimulationContext, "instance", forbidden)
        with wp.ScopedCapture(device=device) as capture:
            view.read(1)
    rgb.fill_(7)
    wp.capture_launch(capture.graph)
    assert image.ptr == pointer and rgb.ptr == source_pointer
    replayed = image.numpy()[..., :3]
    assert np.all(replayed[2:18, :8] == 7)
    np.testing.assert_array_equal(replayed[2:18, 8:], expected[2:18, 8:])
    assert not replayed[:2].any() and not replayed[-2:].any()
    camera.update.assert_not_called()
    camera.close.assert_not_called()
    declaration.envs = ()
    assert view.read(2) is None
    assert recorder._get_frame() is None
    view.close()
    camera.close.assert_not_called()
