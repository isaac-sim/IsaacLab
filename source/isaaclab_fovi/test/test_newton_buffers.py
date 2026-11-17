# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Newton anti-aliasing consumes IsaacLab's public renderer-buffer views."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors import foveated_newton_camera as newton_camera
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera

from isaaclab.renderers.output_contract import RenderBufferKind, RenderBufferSpec
from isaaclab.sensors.camera.camera_data import CameraData
from isaaclab.utils.warp import ProxyArray


def test_pooled_allocation_accepts_proxy_arrays_and_preserves_render_buffers() -> None:
    rgba = torch.zeros((1, 8, 1, 4), dtype=torch.uint8)
    rgba_proxy = ProxyArray(wp.from_torch(rgba))
    data = CameraData.allocate(
        ["rgba"],
        8,
        1,
        1,
        "cpu",
        {RenderBufferKind.RGBA: RenderBufferSpec(channels=4, dtype=wp.uint8)},
    )
    data.output["rgba"] = rgba_proxy
    camera = SimpleNamespace()
    camera._data = data
    camera._pooler = torch.nn.Identity()
    camera._pooler_int = None
    camera._num_valid_samples = 4
    camera._num_ray_samples = 8
    camera._pool_k = 2
    camera.cfg = SimpleNamespace(data_types=["rgba"], pool_int_mode="max", pool_mode="avg", pool_sample_cortex=False)

    FoveatedNewtonCamera._allocate_pooled_buffers(camera)

    assert camera._pooled_output["rgba"].shape == (1, 4, 1, 4)
    assert camera._pooled_output["rgba"].dtype == torch.uint8
    assert camera._highres_output["rgba"] is rgba_proxy
    assert camera._highres_output is not data.output


@pytest.mark.parametrize("pooled", [False, True])
def test_native_grid_render_layout_preserves_rays_and_upright_outputs(monkeypatch, pooled) -> None:
    coords = SamplingCoords(1.0, 0.036021, 16, style="warped_cartesian_as_grid", device="cpu")
    highres = coords.clone(resolution=32) if pooled else None
    ray_coords = highres if pooled else coords
    pooler = torch.nn.AdaptiveAvgPool1d(len(coords)) if pooled else None
    monkeypatch.setattr(newton_camera, "build_lowres_coords", lambda *args, **kwargs: (coords, 16))
    monkeypatch.setattr(newton_camera, "build_highres_coords", lambda *args, **kwargs: highres)
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (pooler, 4))
    monkeypatch.setattr(FoveatedNewtonCamera, "_initialize_calibrated_gaze", lambda self: None)

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
        values = torch.arange(camera.renderer_rgba.numel()).reshape(camera.renderer_rgba.shape)
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
    camera._spawn_horizontal_aperture = 1.0
    camera._spawn_focal_length = 1.0
    camera.cfg = SimpleNamespace(
        foveated_style="warped_cartesian_as_grid",
        mask_to_sensor_fov=False,
        cmf_a_deg=0.591,
        pool_enabled=pooled,
        pool_mode="avg",
        pool_int_mode="avg",
        pool_sample_cortex=False,
        field_geometry="planar",
        image_rotation_deg=0,
        data_types=["rgba", "rgb"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()

    rays = wp.to_torch(camera._foveated_rays)
    assert rays.shape == (1, ray_coords.resolution, ray_coords.resolution, 2, 3)
    torch.testing.assert_close(rays.reshape(-1, 2, 3)[:, 1], camera._ray_directions)
    public_rgba = camera._data.output["rgba"]
    camera._update_buffers_impl(None)
    expected = camera.renderer_rgba.reshape(2, len(ray_coords), 1, 4).clone()
    expected[:, ~ray_coords.valid_mask] = 0
    if pooled:
        expected = newton_camera.pool_hwc_buffer(pooler, expected.float()).to(torch.uint8)
    expected = coords.as_grid(expected.squeeze(2), sample_dim=1)
    torch.testing.assert_close(camera._data.output["rgba"], expected, rtol=0, atol=0)
    assert camera._data.output["rgba"] is public_rgba
    assert camera._data.output["rgb"].untyped_storage().data_ptr() == public_rgba.untyped_storage().data_ptr()


@pytest.mark.parametrize("render_order", ["tiled", "pixel_priority", "view_priority"])
def test_native_grid_fractional_pool_resolution_validates_newton_tiles(render_order) -> None:
    camera = SimpleNamespace(
        cfg=SimpleNamespace(width=83, height=83),
        _renderer=SimpleNamespace(
            cfg=SimpleNamespace(render_order=render_order, tile_rendering_width=16, tile_rendering_height=8)
        ),
    )
    if render_order == "tiled":
        with pytest.raises(ValueError, match="83x83.*16x8"):
            FoveatedNewtonCamera._validate_native_render_layout(camera)
    else:
        FoveatedNewtonCamera._validate_native_render_layout(camera)
