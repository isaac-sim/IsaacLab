# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for :mod:`isaaclab.utils.images`."""

import pytest
import torch

from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.images import (
    is_depth_like,
    is_normals_like,
    is_rgb_like,
    make_camera_output_grid,
    normalize_camera_image,
    normalize_camera_output_for_display,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=test_devices(DeviceScope.CPU_AND_DEFAULT_CUDA))
def device(request):
    return request.param


@pytest.mark.parametrize(
    ("predicate", "matches", "rejects"),
    [
        (
            is_rgb_like,
            ["rgb", "rgba", "albedo", "simple_shading_constant_diffuse", "simple_shading_diffuse_mdl"],
            ["depth", "distance_to_camera", "normals", "semantic_segmentation"],
        ),
        (
            is_depth_like,
            ["depth", "depth_linear", "distance_to_camera", "distance_to_plane"],
            ["rgb", "albedo", "normals"],
        ),
        (is_normals_like, ["normals", "normals_object_frame"], ["rgb", "albedo", "depth", "distance_to_camera"]),
    ],
)
def test_image_type_predicates(predicate, matches, rejects):
    for data_type in matches:
        assert predicate(data_type), data_type
    for data_type in rejects:
        assert not predicate(data_type), data_type


@pytest.mark.parametrize(
    ("data_type", "dtype", "channel_dim", "strided"),
    [
        ("rgb", torch.uint8, -1, False),
        ("albedo", torch.uint8, -1, False),
        ("simple_shading_diffuse_mdl", torch.uint8, -1, False),
        ("semantic_segmentation", torch.uint8, -1, False),
        ("rgb", torch.float32, -1, False),
        ("rgb", torch.uint8, -1, True),
        ("rgb", torch.uint8, 1, False),
        ("rgb", torch.float32, 1, False),
    ],
)
def test_normalize_rgb_image(device, data_type, dtype, channel_dim, strided):
    channels = 4 if data_type == "semantic_segmentation" else 3
    shape = (2, channels, 8, 8) if channel_dim == 1 else (2, 8, 8, channels * (2 if strided else 1))
    src = torch.randint(0, 255, shape, dtype=torch.uint8, device=device).to(dtype)
    if strided:
        src = src[..., ::2]
        assert not src.is_contiguous()

    out = normalize_camera_image(src, data_type, channel_dim=channel_dim)

    spatial_dims = (2, 3) if channel_dim == 1 else (1, 2)
    expected = src.float() / 255.0
    expected -= expected.mean(dim=spatial_dims, keepdim=True)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)
    assert out.dtype == torch.float32


def test_normalize_rgb_reuses_output(device):
    src = torch.randint(0, 255, (2, 8, 8, 6), dtype=torch.uint8, device=device)
    out = torch.empty(src.shape, dtype=torch.float32, device=device)
    ptr = out.data_ptr()

    assert normalize_camera_image(src, "rgb", out=out) is out
    assert out.data_ptr() == ptr
    expected = src.float() / 255.0
    expected -= expected.mean(dim=(1, 2), keepdim=True)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_normalize_segmentation_preserves_labels(device):
    src = torch.randint(0, 4, (2, 8, 8, 1), dtype=torch.int32, device=device)
    out = normalize_camera_image(src, "semantic_segmentation")

    torch.testing.assert_close(out, src.float())


@pytest.mark.parametrize("data_type", ["depth", "distance_to_camera", "distance_to_plane"])
def test_normalize_depth_replaces_inf_in_place(device, data_type):
    src = torch.tensor([[1.0, float("inf"), 3.0], [float("inf"), 2.0, 4.0]], device=device)
    out = normalize_camera_image(src, data_type)

    assert out is src
    expected = torch.tensor([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]], device=device)
    torch.testing.assert_close(out, expected)


def test_normalize_normals(device):
    src = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0], device=device)
    out = normalize_camera_image(src, "normals")

    expected = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], device=device)
    torch.testing.assert_close(out, expected)


@pytest.mark.parametrize("data_type", ["instance_segmentation", "motion_vectors"])
def test_normalize_unknown_type_passthrough(device, data_type):
    src = torch.ones((2, 4, 4, 3), device=device)
    assert normalize_camera_image(src, data_type) is src


@pytest.mark.parametrize(
    ("data_type", "values", "expected"),
    [
        ("rgb", [0.0, 127.0, 255.0], [0.0, 127.0 / 255.0, 1.0]),
        ("albedo", [255.0, 128.0, 64.0, 9.0], [1.0, 128.0 / 255.0, 64.0 / 255.0]),
        ("motion_vectors", [4.0, -2.0], [1.0, 0.0, 0.0]),
        ("motion_vectors", [0.0, 4.0], [0.5, 1.0, 0.0]),
    ],
)
def test_normalize_display_colors(device, data_type, values, expected):
    src = torch.tensor([[[values]]], device=device)
    out = normalize_camera_output_for_display(src, data_type)

    torch.testing.assert_close(out, torch.tensor([[[expected]]], device=device))


@pytest.mark.parametrize("data_type", ["depth", "distance_to_camera", "distance_to_image_plane"])
@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([0.0, 2.0, 4.0], [0.0, 0.5, 1.0]),
        ([0.0, 2.0, float("inf"), 4.0, float("nan")], [0.0, 0.5, 0.0, 1.0, 0.0]),
        ([float("inf"), float("nan")], [0.0, 0.0]),
    ],
)
def test_normalize_display_depth(device, data_type, values, expected):
    src = torch.tensor(values, device=device).reshape(1, 1, -1, 1)
    out = normalize_camera_output_for_display(src, data_type)

    torch.testing.assert_close(out, torch.tensor(expected, device=device).reshape_as(src))


def test_single_image_grid_is_channel_first(device):
    images = torch.ones((1, 2, 3, 3), device=device)
    grid = make_camera_output_grid(images)
    assert grid.shape == (3, 2, 3)
