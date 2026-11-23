# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Newton foveated camera's render layout and publication, on stand-in renderer buffers."""

from collections.abc import Callable
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
from isaaclab_fovi.sensors.pooling import pool_segmentation_buffer

from isaaclab.sensors.camera.camera import Camera
from isaaclab.test.utils import DeviceScope, test_devices
from isaaclab.utils.warp import ProxyArray

pytestmark = pytest.mark.kitless

DEVICE = next(iter(test_devices(DeviceScope.DEFAULT_CUDA)), "cpu")
TILES = SimpleNamespace(render_order="tiled", tile_rendering_width=8, tile_rendering_height=8)


def stub_camera(
    monkeypatch,
    coords: SamplingCoords,
    outputs: dict[str, tuple[int, torch.dtype]],
    render: Callable[[FoveatedNewtonCamera, wp.array], None] | None = None,
    *,
    highres: SamplingCoords | None = None,
    pooler: torch.nn.Module | None = None,
    **cfg,
) -> FoveatedNewtonCamera:
    """An initialized Newton camera whose renderer writes ``camera.renderer[name]``, shape (2, H, W, C), for each
    output (channels, dtype); rgb is a view of rgba when both are requested, as the renderer publishes it.

    ``render(camera, env_mask)`` fills the renderer buffers each update.
    """
    monkeypatch.setattr(base_camera, "build_retinal_coords", lambda *args, **kwargs: (coords, highres))
    monkeypatch.setattr(newton_camera, "build_pooler", lambda *args, **kwargs: (pooler, 4))

    def initialize_renderer(camera):
        camera.renderer = {}
        for name, (channels, dtype) in outputs.items():
            if name == "rgb" and "rgba" in outputs:
                continue
            camera.renderer[name] = torch.zeros((2, camera.cfg.height, camera.cfg.width, channels), dtype=dtype)
        if "rgb" in outputs and "rgba" in outputs:
            camera.renderer["rgb"] = camera.renderer["rgba"][..., :3]
        camera._data = SimpleNamespace(
            output={name: ProxyArray(wp.from_torch(t)) for name, t in camera.renderer.items()}
        )
        camera._render_data = SimpleNamespace(camera_rays=None)
        camera._renderer = SimpleNamespace(cfg=TILES, cleanup=lambda render_data: None)

    monkeypatch.setattr(Camera, "_initialize_impl", initialize_renderer)
    monkeypatch.setattr(Camera, "_update_buffers_impl", render or (lambda camera, env_mask: None))
    # The state the constructor sets and the handles the destructor reads.
    camera = object.__new__(FoveatedNewtonCamera)
    camera._gaze_angles = camera._gaze_rotation = camera._visibility = camera._grid_publish = None
    camera._published_shape = camera._segmentation_neighbors = None
    camera._initialize_handle = camera._invalidate_initialize_handle = camera._prim_deletion_handle = None
    camera._view = camera._renderer = None
    camera._device = "cpu"
    camera.cfg = SimpleNamespace(
        camera_model=asdict(CameraModel("pinhole", (480, 640), (300.0, 300.0, 319.5, 239.5))),
        foveated_style=coords.style,
        mask_to_sensor_fov=False,
        cmf_a_deg=0.591,
        pool_enabled=highres is not None,
        pool_mode="avg",
        pool_sample_cortex=False,
        image_rotation_deg=0,
        update_period=0.0,
        data_types=list(outputs),
        renderer_cfg=TILES,
    )
    for name, value in cfg.items():
        setattr(camera.cfg, name, value)
    camera._initialize_impl()
    return camera


def updated(*rows: bool) -> wp.array:
    return wp.array(list(rows), dtype=wp.bool, device="cpu")


def test_native_grid_render_layout_preserves_rays_and_upright_outputs(monkeypatch) -> None:
    """A pooled native grid renders its photoreceptor rays in order, and publishes the pooled retina upright."""
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32)

    def render(camera, env_mask):
        assert camera._data.output["rgba"] is public_rgba
        rgba = camera.renderer["rgba"]
        rgba.copy_((torch.arange(rgba.numel()).reshape(rgba.shape) + camera.frame_offset).to(torch.uint8))

    camera = stub_camera(
        monkeypatch,
        coords,
        {"rgba": (4, torch.uint8), "rgb": (3, torch.uint8)},
        render,
        highres=highres,
        pooler=torch.nn.AdaptiveAvgPool1d(len(coords)),
    )
    # Newton launches image tiles; a vector-shaped width of one is not valid.
    assert camera.renderer["rgba"].shape[1:3] == (highres.resolution, highres.resolution)
    rays = wp.to_torch(camera._render_data.camera_rays)
    assert rays.shape == (highres.resolution, highres.resolution, 1, 2, 3)
    torch.testing.assert_close(
        rays.reshape(-1, 2, 3)[:, 1], native_grid_directions(highres, image_rotation_deg=0).float()
    )
    public_rgba = camera._data.output["rgba"]
    camera.frame_offset = 0
    camera._update_buffers_impl(updated(True, True))
    expected = camera.renderer["rgba"].reshape(2, len(highres), 1, 4).clone()
    expected[:, ~highres.valid_mask] = 0
    # AdaptiveAvgPool1d maps the 4 * N_lo canonical samples to N_lo by averaging each run of four, which the camera
    # rounds to the nearest uint8 value.
    expected = expected.reshape(2, len(coords), 4, 1, 4).float().mean(dim=2).round().to(torch.uint8)
    expected = coords.as_grid(expected.squeeze(2), sample_dim=1)
    torch.testing.assert_close(camera._data.output["rgba"].torch, expected, rtol=0, atol=0)
    # The published buffers are stable across frames, and rgb shares rgba's memory.
    assert camera._data.output["rgba"] is public_rgba
    torch.testing.assert_close(camera._data.output["rgb"].torch, expected[..., :3], rtol=0, atol=0)
    assert camera._data.output["rgb"].warp.ptr == public_rgba.warp.ptr

    # Newton redraws every row, but only the updated environment's publication changes.
    camera.frame_offset = 1
    camera._update_buffers_impl(updated(False, True))
    published = camera._data.output["rgba"].torch
    torch.testing.assert_close(published[0], expected[0], rtol=0, atol=0)
    assert not torch.equal(published[1], expected[1])


def test_sample_set_publication_and_sensor_mask_follow_the_published_frame(monkeypatch) -> None:
    """A sample-set camera publishes its own buffers, and its sensor mask reuses one visibility buffer: Newton redraws
    every row, but an environment that was not updated keeps its published samples and the visibility that masked
    them."""
    coords = SamplingCoords(60.0, 0.591, 8, style="isotropic", field_geometry="spherical", device="cpu")

    def render(camera, env_mask):
        camera.renderer["distance_to_camera"].fill_(camera.frame_value)

    camera = stub_camera(
        monkeypatch, coords, {"distance_to_camera": (1, torch.float32)}, render, mask_to_sensor_fov=True
    )
    camera._view = SimpleNamespace(count=2, close=lambda: None)
    camera._data.quat_w_ros = ProxyArray(wp.from_torch(torch.tensor([[0.0, 0.0, 0.0, 1.0]] * 2), dtype=wp.quatf))
    published = camera._data.output["distance_to_camera"]
    assert published.torch.shape == (2, len(coords), 1, 1)
    assert published.warp.ptr != camera._flat_render_output["distance_to_camera"].warp.ptr

    # The second camera looks away from the physical camera's image, so it sees nothing.
    camera.set_gaze_angles(torch.tensor([[0.0, 0.0], [0.0, 180.0]]))
    camera.frame_value = 1.0
    camera._update_buffers_impl(updated(True, True))
    visibility = camera._visibility["valid"].torch
    seen = camera.sensor_fov_valid_fraction.clone()
    assert seen[0] > 0.5 and seen[1] == 0
    sample = published.torch[..., 0, 0]
    assert bool((sample[0] == visibility[0].float()).all()) and not sample[1].any()

    # Both look ahead now, but only the first camera's frame is published: the second keeps its samples and visibility.
    camera.set_gaze_angles(torch.zeros(2, 2))
    camera.frame_value = 2.0
    camera._update_buffers_impl(updated(True, False))
    assert camera._data.output["distance_to_camera"] is published
    assert camera._visibility["valid"].torch is visibility
    assert not published.torch[1].any()
    torch.testing.assert_close(camera.sensor_fov_valid_fraction, seen)
    first = published.torch[0].clone()

    # Now only the second camera's frame is published, and the first keeps its samples.
    camera.frame_value = 3.0
    camera._update_buffers_impl(updated(False, True))
    torch.testing.assert_close(published.torch[0], first)
    assert bool((published.torch[1][camera._visibility["valid"].torch[1]] == 3.0).all())
    assert camera.sensor_fov_valid_fraction[1] > 0.5


def test_modal_pooling_preserves_ids_above_float_precision(monkeypatch) -> None:
    """Segmentation keeps each pooled sample's majority label exactly, IDs above float precision included: on the
    camera's published output, and for wider IDs and multi-channel labels in the pooling itself."""
    coords = SamplingCoords(60.0, 0.591, 16, style="warped_cartesian_as_grid", field_geometry="spherical", device="cpu")
    highres = coords.clone(resolution=32)
    runs = torch.arange(len(coords))
    # Each pooled sample's neighbours are a run of four rendered samples, nearest first.
    monkeypatch.setattr(
        base_camera, "segmentation_neighbor_indices", lambda pooler: torch.stack([4 * runs + j for j in range(4)])
    )
    # In each run, three samples share a label above 2**24 and one differs; float32 would merge neighbouring labels.
    majority = 2**24 + 1 + 2 * runs
    labels = majority[:, None].repeat(1, 4)
    labels[:, 2] = 2**30 + runs
    labels = torch.stack((labels.reshape(-1), (labels + 7).reshape(-1)))

    def render(camera, env_mask):
        target = camera.renderer["semantic_segmentation"]
        target.copy_(labels.to(torch.int32).reshape(target.shape))

    camera = stub_camera(
        monkeypatch,
        coords,
        {"rgba": (4, torch.uint8), "semantic_segmentation": (1, torch.int32)},
        render,
        highres=highres,
        pooler=torch.nn.AdaptiveAvgPool1d(len(coords)),
    )
    camera._update_buffers_impl(updated(True, True))
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

    # Three outputs over five inputs; 5 marks an excluded neighbour.
    table = torch.tensor([[0, 3, 5], [1, 4, 5], [2, 5, 5]], device=DEVICE)
    # 2**24 + 1 and 2**24 + 3 round to other values in float32; int64 IDs must come back exactly.
    big = 2**40
    ids = torch.tensor([2**24 + 1, 2**24 + 1, 2**24 + 3, big + 7, big + 7], dtype=torch.int64)
    pooled = pool_segmentation_buffer(table, torch.stack([ids, ids.flip(0)])[:, :, None, None].to(DEVICE))
    assert pooled.dtype == torch.int64
    assert torch.equal(pooled[..., 0, 0].cpu(), torch.tensor([[2**24 + 1, big + 7, 0], [big + 7, 2**24 + 1, 0]]))
    # Two int64 channels exceed the 8-byte packed key; vectors differing only in the second channel stay distinct.
    vectors = torch.tensor([[1, 2**40], [1, 2**40 + 1], [1, 2**40 + 1], [5, 6], [7, 8]], dtype=torch.int64)
    pooled = pool_segmentation_buffer(table, vectors[None, :, None, :].to(DEVICE))
    assert torch.equal(pooled[0, :, 0].cpu(), torch.tensor([[1, 2**40 + 1], [5, 6], [0, 0]]))


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
    coords = SamplingCoords(60.0, 0.591, 16, style="isotropic", field_geometry="spherical", device="cpu")

    def render(camera, env_mask):
        camera.renderer["rgba"].copy_(torch.tensor([10, 20, 30, 255], dtype=torch.uint8))

    camera = stub_camera(
        monkeypatch,
        coords,
        {"rgba": (4, torch.uint8), "rgb": (3, torch.uint8)},
        render,
        highres=coords.clone(resolution=32),
        pooler=torch.nn.AdaptiveAvgPool1d(len(coords)),
        data_types=["rgba"],
    )
    camera._update_buffers_impl(updated(True, True))
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
    camera = object.__new__(FoveatedNewtonCamera)
    camera._initialize_handle = camera._invalidate_initialize_handle = camera._prim_deletion_handle = None
    camera._view = camera._renderer = None
    with pytest.warns(UserWarning, match="leaves a foveated camera as configured") as caught:
        camera.set_intrinsic_matrices(torch.eye(3).repeat(2, 1, 1))
    # The warning points at the caller's line.
    assert caught[0].filename == __file__
