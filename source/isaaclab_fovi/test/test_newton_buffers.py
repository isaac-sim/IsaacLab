# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Newton foveated camera's render layout and publication, on stand-in renderer buffers."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors import foveated_camera as base_camera
from isaaclab_fovi.sensors import foveated_newton_camera as newton_camera
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera

from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless


@pytest.mark.parametrize("pooled", [False, True])
def test_native_grid_render_layout_preserves_rays_and_upright_outputs(monkeypatch, pooled) -> None:
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32) if pooled else None
    ray_coords = highres if pooled else coords
    pooler = torch.nn.AdaptiveAvgPool1d(len(coords)) if pooled else None
    monkeypatch.setattr(newton_camera, "build_lowres_coords", lambda *args, **kwargs: (coords, 16))
    monkeypatch.setattr(newton_camera, "build_highres_coords", lambda *args, **kwargs: highres)
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (pooler, 4))
    monkeypatch.setattr(FoveatedNewtonCamera, "_initialize_gaze", lambda self: None)

    def initialize_renderer(camera):
        # Newton launches image tiles; a vector-shaped width of one is not valid.
        assert camera.cfg.height == camera.cfg.width == ray_coords.resolution
        rgba = torch.zeros((2, camera.cfg.height, camera.cfg.width, 4), dtype=torch.uint8)
        camera._data = SimpleNamespace(
            output={
                "rgba": ProxyArray(wp.from_torch(rgba)),
                "rgb": ProxyArray(wp.from_torch(rgba[..., :3])),
            }
        )
        camera._render_data = SimpleNamespace(camera_rays=None)
        camera._renderer = SimpleNamespace(
            cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
            cleanup=lambda render_data: None,
        )
        camera.renderer_rgba = rgba

    def render(camera, env_mask):
        assert camera._data.output["rgba"].torch.shape == camera.renderer_rgba.shape
        values = torch.arange(camera.renderer_rgba.numel()).reshape(camera.renderer_rgba.shape) + camera.frame_offset
        camera.renderer_rgba.copy_(values.to(torch.uint8))

    monkeypatch.setattr(newton_camera.Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(newton_camera.Camera, "_update_buffers_impl", render)
    camera = object.__new__(FoveatedNewtonCamera)
    camera._initialize_handle = None
    camera._invalidate_initialize_handle = None
    camera._prim_deletion_handle = None
    camera._view = None
    camera._renderer = None
    camera._device = "cpu"
    camera.cfg = SimpleNamespace(
        foveated_style="warped_cartesian_as_grid",
        mask_to_sensor_fov=False,
        cmf_a_deg=0.591,
        pool_enabled=pooled,
        pool_mode="avg",
        pool_int_mode="avg",
        pool_sample_cortex=False,
        image_rotation_deg=0,
        update_period=0.0,
        data_types=["rgba", "rgb"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()

    rays = wp.to_torch(camera._foveated_rays)
    assert rays.shape == (ray_coords.resolution, ray_coords.resolution, 1, 2, 3)
    torch.testing.assert_close(rays.reshape(-1, 2, 3)[:, 1], camera._ray_directions)
    public_rgba = camera._data.output["rgba"]
    camera.frame_offset = 0
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))
    expected = camera.renderer_rgba.reshape(2, len(ray_coords), 1, 4).clone()
    expected[:, ~ray_coords.valid_mask] = 0
    if pooled:
        # AdaptiveAvgPool1d maps the 4 * N_lo canonical samples to N_lo by averaging each run of four.
        expected = expected.reshape(2, len(coords), 4, 1, 4).float().mean(dim=2).to(torch.uint8)
    expected = coords.as_grid(expected.squeeze(2), sample_dim=1)
    torch.testing.assert_close(camera._data.output["rgba"].torch, expected, rtol=0, atol=0)
    # The published buffers are stable across frames, and rgb shares rgba's memory.
    assert camera._data.output["rgba"] is public_rgba
    torch.testing.assert_close(camera._data.output["rgb"].torch, expected[..., :3], rtol=0, atol=0)
    assert camera._data.output["rgb"].warp.ptr == public_rgba.warp.ptr

    # Newton redraws every row, but only the updated environment's publication changes.
    camera.frame_offset = 1
    camera._update_buffers_impl(wp.array([False, True], dtype=wp.bool, device="cpu"))
    published = camera._data.output["rgba"].torch
    torch.testing.assert_close(published[0], expected[0], rtol=0, atol=0)
    assert not torch.equal(published[1], expected[1])


def test_pooled_segmentation_publishes_modal_labels_at_full_precision(monkeypatch) -> None:
    """A pooled camera's segmentation output holds each sample's majority label exactly, IDs above 2**24 included."""
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32)
    runs = torch.arange(len(coords))
    # Each pooled sample's neighbours are a run of four rendered samples, nearest first.
    neighbors = torch.stack([4 * runs + j for j in range(4)])
    average = torch.nn.AdaptiveAvgPool1d(len(coords))

    monkeypatch.setattr(newton_camera, "build_lowres_coords", lambda *args, **kwargs: (coords, 16))
    monkeypatch.setattr(newton_camera, "build_highres_coords", lambda *args, **kwargs: highres)
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (average, 4))
    # The camera reads the modal neighbourhoods from its standard pooler.
    monkeypatch.setattr(base_camera, "segmentation_neighbor_indices", lambda pooler: neighbors)
    monkeypatch.setattr(FoveatedNewtonCamera, "_initialize_gaze", lambda self: None)

    def initialize_renderer(camera):
        size = (2, camera.cfg.height, camera.cfg.width)
        camera.renderer_rgba = torch.zeros((*size, 4), dtype=torch.uint8)
        camera.renderer_labels = torch.zeros((*size, 1), dtype=torch.int32)
        camera._data = SimpleNamespace(
            output={
                "rgba": ProxyArray(wp.from_torch(camera.renderer_rgba)),
                "semantic_segmentation": ProxyArray(wp.from_torch(camera.renderer_labels)),
            }
        )
        camera._render_data = SimpleNamespace(camera_rays=None)
        camera._renderer = SimpleNamespace(cleanup=lambda render_data: None)

    # In each run, three samples share a label above 2**24 and one differs; float32 would merge neighbouring labels.
    majority = 2**24 + 1 + 2 * runs
    labels = majority[:, None].repeat(1, 4)
    labels[:, 2] = 2**30 + runs
    labels = torch.stack((labels.reshape(-1), (labels + 7).reshape(-1)))

    def render(camera, env_mask):
        camera.renderer_labels.copy_(labels.to(torch.int32).reshape(camera.renderer_labels.shape))

    monkeypatch.setattr(newton_camera.Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(newton_camera.Camera, "_update_buffers_impl", render)
    camera = object.__new__(FoveatedNewtonCamera)
    camera._initialize_handle = None
    camera._invalidate_initialize_handle = None
    camera._prim_deletion_handle = None
    camera._view = None
    camera._renderer = None
    camera._device = "cpu"
    camera.cfg = SimpleNamespace(
        foveated_style="warped_cartesian_as_grid",
        mask_to_sensor_fov=False,
        update_period=0.0,
        pool_enabled=True,
        pool_mode="avg",
        pool_int_mode="mode",
        pool_sample_cortex=False,
        image_rotation_deg=0,
        data_types=["rgba", "semantic_segmentation"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))

    # The modal label of each run after the footprint mask zeroes samples outside the retina, ties to the nearest.
    rendered = labels.clone()
    rendered[:, ~highres.valid_mask] = 0
    expected = torch.zeros(2, len(coords), dtype=torch.int64)
    for env in range(2):
        for sample in range(len(coords)):
            run = rendered[env, 4 * sample : 4 * sample + 4].tolist()
            expected[env, sample] = max(run, key=lambda label: (run.count(label), -run.index(label)))
    published = camera._data.output["semantic_segmentation"].torch
    assert published.dtype == torch.int32
    torch.testing.assert_close(published[..., 0].long(), coords.as_grid(expected, sample_dim=1), rtol=0, atol=0)
    assert camera.image_shape == (16, 16) and camera._data.image_shape == (16, 16)


@pytest.mark.parametrize("size, valid", [(83, False), (100, False), (128, True)])
def test_native_grid_fractional_pool_resolution_validates_newton_tiles(size, valid) -> None:
    camera = SimpleNamespace(cfg=SimpleNamespace(width=size, height=size))
    if valid:
        FoveatedNewtonCamera._validate_native_render_layout(camera, 16, 8)
    else:
        with pytest.raises(ValueError, match=f"{size}x{size}.*16x8"):
            FoveatedNewtonCamera._validate_native_render_layout(camera, 16, 8)
