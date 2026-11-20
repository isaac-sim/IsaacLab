# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Newton foveated camera's render layout and publication, on stand-in renderer buffers."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import warp as wp
from fovi.sensing.coords import SamplingCoords
from fovi.sensing.projection import CameraModel
from isaaclab_fovi.sensors import foveated_camera as base_camera
from isaaclab_fovi.sensors import foveated_newton_camera as newton_camera
from isaaclab_fovi.sensors.foveated_newton_camera import FoveatedNewtonCamera, explicit_rays
from isaaclab_fovi.sensors.native_grid import native_grid_directions

from isaaclab.sensors.camera.camera import Camera
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless


def new_camera() -> FoveatedNewtonCamera:
    """A Newton camera without its constructor: the state its constructor sets and the handles its destructor reads."""
    camera = object.__new__(FoveatedNewtonCamera)
    camera._gaze_angles = camera._gaze_rotation = camera._visibility = camera._grid_publish = None
    camera._published_shape = camera._segmentation_neighbors = None
    camera._initialize_handle = None
    camera._invalidate_initialize_handle = None
    camera._prim_deletion_handle = None
    camera._view = None
    camera._renderer = None
    camera._device = "cpu"
    return camera


@pytest.mark.parametrize("pooled", [False, True])
def test_native_grid_render_layout_preserves_rays_and_upright_outputs(monkeypatch, pooled) -> None:
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32) if pooled else None
    ray_coords = highres if pooled else coords
    pooler = torch.nn.AdaptiveAvgPool1d(len(coords)) if pooled else None
    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, highres))
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (pooler, 4))

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
        assert camera._data.output["rgba"] is public_rgba
        values = torch.arange(camera.renderer_rgba.numel()).reshape(camera.renderer_rgba.shape) + camera.frame_offset
        camera.renderer_rgba.copy_(values.to(torch.uint8))

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", render)
    camera = new_camera()
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style="warped_cartesian_as_grid",
        mask_to_sensor_fov=False,
        cmf_a_deg=0.591,
        pool_enabled=pooled,
        pool_mode="avg",
        pool_sample_cortex=False,
        image_rotation_deg=0,
        update_period=0.0,
        data_types=["rgba", "rgb"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()

    rays = wp.to_torch(camera._render_data.camera_rays)
    assert rays.shape == (ray_coords.resolution, ray_coords.resolution, 1, 2, 3)
    expected_rays = native_grid_directions(ray_coords, image_rotation_deg=0).float()
    torch.testing.assert_close(rays.reshape(-1, 2, 3)[:, 1], expected_rays)
    public_rgba = camera._data.output["rgba"]
    camera.frame_offset = 0
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))
    expected = camera.renderer_rgba.reshape(2, len(ray_coords), 1, 4).clone()
    expected[:, ~ray_coords.valid_mask] = 0
    if pooled:
        # AdaptiveAvgPool1d maps the 4 * N_lo canonical samples to N_lo by averaging each run of four, which the
        # camera rounds to the nearest uint8 value.
        expected = expected.reshape(2, len(coords), 4, 1, 4).float().mean(dim=2).round().to(torch.uint8)
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


def test_sample_set_publication_keeps_environments_that_were_not_updated(monkeypatch) -> None:
    """A sample-set camera publishes its own buffers: Newton redraws every row, but only updated rows change."""
    coords = SamplingCoords(60.0, 0.591, 8, style="isotropic", field_geometry="spherical", device="cpu")
    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, None))
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (None, None))

    def initialize_renderer(camera):
        depth = torch.zeros((2, camera.cfg.height, camera.cfg.width, 1))
        camera._data = SimpleNamespace(output={"distance_to_camera": ProxyArray(wp.from_torch(depth))})
        camera._render_data = SimpleNamespace(camera_rays=None)
        camera.renderer_depth = depth

    def render(camera, env_mask):
        camera.renderer_depth.fill_(camera.frame_value)

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", render)
    camera = new_camera()
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style="isotropic",
        mask_to_sensor_fov=False,
        pool_enabled=False,
        image_rotation_deg=0,
        update_period=0.0,
        data_types=["distance_to_camera"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()
    published = camera._data.output["distance_to_camera"]
    assert published.torch.shape == (2, len(coords), 1, 1)
    assert published.warp.ptr != camera._flat_render_output["distance_to_camera"].warp.ptr

    camera.frame_value = 1.0
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))
    camera.frame_value = 2.0
    camera._update_buffers_impl(wp.array([False, True], dtype=wp.bool, device="cpu"))
    assert camera._data.output["distance_to_camera"] is published
    assert bool((published.torch[0] == 1.0).all()) and bool((published.torch[1] == 2.0).all())


def test_sample_set_sensor_mask_follows_the_published_frame(monkeypatch) -> None:
    """An environment that was not updated keeps its published samples and the sensor visibility that masked them."""
    coords = SamplingCoords(60.0, 0.591, 8, style="isotropic", field_geometry="spherical", device="cpu")
    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, None))
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (None, None))

    def initialize_renderer(camera):
        camera.renderer_depth = torch.ones((2, camera.cfg.height, camera.cfg.width, 1))
        camera._data = SimpleNamespace(output={"distance_to_camera": ProxyArray(wp.from_torch(camera.renderer_depth))})
        camera._render_data = SimpleNamespace(camera_rays=None)

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", lambda camera, env_mask: None)
    camera = new_camera()
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style="isotropic",
        mask_to_sensor_fov=True,
        pool_enabled=False,
        image_rotation_deg=0,
        update_period=0.0,
        data_types=["distance_to_camera"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()
    camera._view = SimpleNamespace(count=2, close=lambda: None)
    camera._data.quat_w_ros = ProxyArray(wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2), dtype=wp.quatf))

    # The second camera looks away from the physical camera's image, so it sees nothing.
    camera.set_gaze_angles(torch.tensor([[0.0, 0.0], [0.0, 180.0]]))
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))
    seen = camera.sensor_fov_valid_fraction.clone()
    assert seen[0] > 0.5 and seen[1] == 0
    # Both look ahead now, but only the first camera's frame is published.
    camera.set_gaze_angles(torch.zeros(2, 2))
    camera._update_buffers_impl(wp.array([True, False], dtype=wp.bool, device="cpu"))
    assert not camera._data.output["distance_to_camera"].torch[1].any()
    torch.testing.assert_close(camera.sensor_fov_valid_fraction, seen)


def test_pooled_segmentation_publishes_modal_labels_at_full_precision(monkeypatch) -> None:
    """A pooled camera's segmentation output holds each sample's majority label exactly, IDs above 2**24 included."""
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32)
    runs = torch.arange(len(coords))
    # Each pooled sample's neighbours are a run of four rendered samples, nearest first.
    neighbors = torch.stack([4 * runs + j for j in range(4)])
    average = torch.nn.AdaptiveAvgPool1d(len(coords))

    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, highres))
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (average, 4))
    # The camera reads the modal neighbourhoods from its standard pooler.
    monkeypatch.setattr(base_camera, "segmentation_neighbor_indices", lambda pooler: neighbors)

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

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", render)
    camera = new_camera()
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style="warped_cartesian_as_grid",
        mask_to_sensor_fov=False,
        update_period=0.0,
        pool_enabled=True,
        pool_mode="avg",
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


def test_native_grid_render_size_must_fill_newton_tiles() -> None:
    """A fractional pool_res_mult can give a grid that Newton's tiles do not divide; the camera rejects it."""
    FoveatedNewtonCamera._validate_native_render_layout(
        SimpleNamespace(cfg=SimpleNamespace(width=128, height=128)), 16, 8
    )
    with pytest.raises(ValueError, match="83x83.*16x8"):
        FoveatedNewtonCamera._validate_native_render_layout(
            SimpleNamespace(cfg=SimpleNamespace(width=83, height=83)), 16, 8
        )


def test_explicit_rays_keep_sample_order_and_pad_whole_tiles() -> None:
    """Newton receives one ray per sample from the optical centre, in sample order, padded to whole tiles."""
    wp.init()
    directions = torch.nn.functional.normalize(torch.randn(5, 3, generator=torch.Generator().manual_seed(0)), dim=-1)
    rays = explicit_rays(directions, 2, 4)
    assert rays.shape == (2, 4, 1, 2) and rays.dtype == wp.vec3f
    rays = wp.to_torch(rays).reshape(8, 2, 3)
    torch.testing.assert_close(rays[:5, 1], directions)
    torch.testing.assert_close(rays[5:, 1], directions[-1].expand(3, -1))
    assert not rays[:, 0].any()
    with pytest.raises(ValueError, match="shape"):
        explicit_rays(torch.zeros(3, 2), 3, 1)
    with pytest.raises(ValueError, match="cannot hold"):
        explicit_rays(torch.zeros(5, 3), 2, 2)


def test_pooled_rgb_is_the_view_of_pooled_rgba(monkeypatch) -> None:
    """The renderer publishes rgb as a view of rgba even when only rgba is requested, and so do pooled sample sets."""
    style = "isotropic"
    coords = SamplingCoords(60.0, 0.591, 16, style=style, field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32)
    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, highres))
    monkeypatch.setattr(
        newton_camera, "build_pooler", lambda *args, **kwargs: (torch.nn.AdaptiveAvgPool1d(len(coords)), 4)
    )

    def initialize_renderer(camera):
        camera.renderer_rgba = torch.zeros((2, camera.cfg.height, camera.cfg.width, 4), dtype=torch.uint8)
        rgba = ProxyArray(wp.from_torch(camera.renderer_rgba))
        camera._data = SimpleNamespace(
            output={"rgba": rgba, "rgb": ProxyArray(wp.from_torch(camera.renderer_rgba[..., :3]))}
        )
        camera._render_data = SimpleNamespace(camera_rays=None)
        camera._renderer = SimpleNamespace(cleanup=lambda render_data: None)

    def render(camera, env_mask):
        camera.renderer_rgba.copy_(torch.tensor([10, 20, 30, 255], dtype=torch.uint8))

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", render)
    camera = new_camera()
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style=style,
        mask_to_sensor_fov=False,
        update_period=0.0,
        pool_enabled=True,
        pool_mode="avg",
        pool_sample_cortex=False,
        image_rotation_deg=0,
        data_types=["rgba"],
        renderer_cfg=SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8),
    )
    camera._initialize_impl()
    camera._update_buffers_impl(wp.ones(2, dtype=wp.bool, device="cpu"))

    rgb, rgba = camera._data.output["rgb"].torch, camera._data.output["rgba"].torch
    assert rgb.any()
    torch.testing.assert_close(rgb, rgba[..., :3], rtol=0, atol=0)
    # rgb is a view of rgba, not a copy: a write to rgba shows through it.
    assert rgb.data_ptr() == rgba.data_ptr()
    rgba[..., 0] = 7
    assert bool((rgb[..., 0] == 7).all())


def test_runtime_intrinsics_leave_the_retinas_rays_alone(monkeypatch) -> None:
    """The retina sets a foveated camera's rays, so runtime intrinsics, such as a viewer's zoom sets, leave the camera
    as configured with a warning to the caller, rather than reaching the renderer, which would replace the rays with a
    pinhole's."""

    def replace_rays(*args, **kwargs):
        raise AssertionError("the stock setter replaces the renderer's rays")

    monkeypatch.setattr(Camera, "set_intrinsic_matrices", replace_rays)
    camera = new_camera()
    with pytest.warns(UserWarning, match="leaves a foveated camera as configured") as caught:
        camera.set_intrinsic_matrices(torch.eye(3).repeat(2, 1, 1))
    # The warning points at the caller's line.
    assert caught[0].filename == __file__
