# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The native grid's rays, lens textures and published layout agree with independent geometry."""

import math

import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors.calibrated_geometry import retinal_directions
from isaaclab_fovi.sensors.lens_lut import build_native_grid_textures
from isaaclab_fovi.sensors.native_grid import (
    allocate_grid_outputs,
    bounded_angular_radius,
    copy_grid_outputs,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
    native_grid_exit_uv,
    unbounded_angular_radius,
)

from isaaclab.test.utils import DeviceScope, test_devices

pytestmark = pytest.mark.kitless


def make_coords(resolution: int = 64, coverage: str = "circular", field: str = "spherical") -> SamplingCoords:
    """A wide grid; ``"square_shells"`` is the square footprint with square (infinity-norm) shells."""
    fov = 80.0 if coverage == "wang" else 170.0
    square_shells = coverage == "square_shells"
    return SamplingCoords(
        fov=fov,
        cmf_a=0.036021 * fov,
        res=resolution,
        style="warped_cartesian_as_grid",
        fov_type="square" if square_shells else coverage,
        radius_norm=math.inf if square_shells else 2.0,
        field_geometry=field,
        device="cpu",
    )


def texel_centers(resolution: int) -> torch.Tensor:
    """Native texture UV of each canonical sample: columns follow native x, rows native y."""
    axis = (torch.arange(resolution) + 0.5) / resolution
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((col, row), -1).reshape(-1, 2)


def eccentricity_rays(visual: torch.Tensor, fov_deg: float, rotation_deg: float = 0.0) -> torch.Tensor:
    """OpenGL rays for visual-chart points: angle off the axis is the visual radius times the half field of view."""
    visual = visual.to(torch.float64)
    theta = torch.linalg.vector_norm(visual, dim=-1) * math.radians(fov_deg) / 2
    phi = torch.atan2(visual[..., 1], visual[..., 0]) + math.radians(rotation_deg)
    return torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), -theta.cos()), -1)


@pytest.mark.parametrize(
    ("coverage", "resolution", "rotation"),
    # Circular padding outside the field, and odd square-shell grids with a texel on the axis, under a roll.
    [("circular", 64, 0.0), ("square_shells", 15, 37.0)],
)
def test_every_texel_roundtrips(coverage: str, resolution: int, rotation: float) -> None:
    coords = make_coords(resolution, coverage)
    rays = native_grid_directions(coords, image_rotation_deg=rotation)
    uv, valid = native_grid_exit_uv(rays, coords, image_rotation_deg=rotation)
    assert valid.all()
    torch.testing.assert_close(uv, texel_centers(resolution), atol=2e-6, rtol=0)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(len(coords)), atol=1e-6, rtol=0)


@pytest.mark.parametrize(("coverage", "rotation"), [("wang", 180.0), ("square_shells", 37.0)])
def test_inverse_maps_independent_rays_to_their_texels(coverage: str, rotation: float) -> None:
    coords = make_coords(64, coverage)
    half_fov = math.radians(coords.fov) / 2
    # Rays through fovi's own sample positions inside the field land on those samples' texels.
    inside = torch.linalg.vector_norm(coords.cartesian, dim=-1) <= (math.sqrt(2) if coverage == "square_shells" else 1)
    rays = eccentricity_rays(coords.cartesian[inside], coords.fov, rotation)
    uv, valid = native_grid_exit_uv(rays, coords, image_rotation_deg=rotation)
    assert inside.sum() > len(coords) // 2
    assert valid.all()
    torch.testing.assert_close(uv, texel_centers(64)[inside], atol=2e-6, rtol=0)
    # The optical axis is the texture centre and the nominal rim along each rolled native axis is its edge.
    roll = math.radians(rotation)
    rim = [
        (math.sin(half_fov) * math.cos(angle), math.sin(half_fov) * math.sin(angle), -math.cos(half_fov))
        for angle in (roll, roll + math.pi / 2)
    ]
    hand_rays = torch.tensor([(0.0, 0.0, -1.0), *rim], dtype=torch.float64)
    uv, _ = native_grid_exit_uv(hand_rays, coords, image_rotation_deg=rotation)
    torch.testing.assert_close(uv, torch.tensor([[0.5, 0.5], [0.5, 1.0], [1.0, 0.5]]), atol=1e-6, rtol=0)


def test_valid_samples_match_the_software_sampler_rays() -> None:
    """The renderer's rays are the ones fovi's software sampler uses on a real camera's images."""
    coords = make_coords()
    actual = native_grid_directions(coords)
    expected = retinal_directions(coords.cartesian[coords.valid_mask], coords.fov)
    torch.testing.assert_close(actual[coords.valid_mask], expected, atol=3e-7, rtol=0)
    with pytest.raises(ValueError, match="spherical"):
        native_grid_directions(make_coords(16, field="planar"))


def test_spherical_padding_matches_rim_slope_and_does_not_wrap() -> None:
    radius = torch.tensor([0.0, 0.99999, 1.0, 1.00001, 2.0, 10.0], dtype=torch.float64)
    half_fov = math.radians(170) / 2
    angles = bounded_angular_radius(radius, half_fov)
    assert (angles[1:] > angles[:-1]).all()
    assert (angles < math.pi).all()
    left = (angles[2] - angles[1]) / (radius[2] - radius[1])
    right = (angles[3] - angles[2]) / (radius[3] - radius[2])
    torch.testing.assert_close(left, right, atol=2e-5, rtol=0)
    torch.testing.assert_close(unbounded_angular_radius(angles, half_fov), radius, atol=1e-10, rtol=0)


@pytest.mark.parametrize(("coverage", "rotation"), [("circular", 0.0), ("square_shells", 37.0)])
def test_textures_match_independent_geometry(coverage: str, rotation: float) -> None:
    coords = make_coords(64, coverage)
    size = 256
    direction, inverse = build_native_grid_textures(coords, image_rotation_deg=rotation, exit_size=size)
    assert direction.shape == (64, 64, 3) and inverse.shape == (size, size, 2)
    inside = coords.valid_mask & (torch.linalg.vector_norm(coords.cartesian, dim=-1) <= math.sqrt(2))
    expected = eccentricity_rays(coords.cartesian[inside], coords.fov, rotation).float()
    torch.testing.assert_close(direction.reshape(-1, 3)[inside], expected, atol=1e-6, rtol=0)
    # Forward-hemisphere octahedral texel centres decode to (x, y, -(1 - |x| - |y|)) before normalization.
    axis = (torch.arange(size, dtype=torch.float64) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    forward = x.abs() + y.abs() < 1
    texel_rays = torch.nn.functional.normalize(torch.stack((x, y, x.abs() + y.abs() - 1), -1), dim=-1)
    # Each stored UV, mapped back through fovi's native-to-visual warp, must look along its texel's ray.
    stored = forward & (inverse[..., 0] >= 0)
    native = (inverse[stored].to(torch.float64) * 2 - 1).flip(-1)
    visual = coords.native_to_visual(native)
    # Circular grids pad outside the nominal field on another curve; square shells reach their corners linearly.
    linear = torch.ones(len(visual), dtype=torch.bool)
    if coverage == "circular":
        linear = torch.linalg.vector_norm(visual, dim=-1) <= 1
    assert linear.sum() > 1000
    torch.testing.assert_close(
        eccentricity_rays(visual[linear], coords.fov, rotation), texel_rays[stored][linear], atol=1e-5, rtol=0
    )
    # The antipode lies outside every grid.
    assert (inverse[0, 0] == -1).all() and (inverse[-1, -1] == -1).all()
    assert torch.isfinite(inverse).all()


def test_upright_grid_masks_and_channel_aliases() -> None:
    coords = make_coords(16)
    # Coordinate channels make rotation/transposition failures observable.
    values = torch.cat((coords.cartesian, torch.ones(len(coords), 2)), -1)[None, :, None]
    source = {"rgba": values.clone()}
    source["rgb"] = source["rgba"][..., :3]
    mask_grid_samples(source, coords)
    target = allocate_grid_outputs(source, coords)
    copy_grid_outputs(source, target, grid_sample_indices(coords))
    expected = coords.as_grid(source["rgba"][:, :, 0], sample_dim=1)
    torch.testing.assert_close(target["rgba"], expected)
    assert target["rgb"].untyped_storage().data_ptr() == target["rgba"].untyped_storage().data_ptr()
    assert target["rgba"][0, :, :, 0][8, 4] < target["rgba"][0, :, :, 0][8, 11]
    assert target["rgba"][0, :, :, 1][4, 8] > target["rgba"][0, :, :, 1][11, 8]
    assert (target["rgba"][0][~coords.as_grid(coords.valid_mask)] == 0).all()


@pytest.mark.parametrize("device", test_devices(DeviceScope.DEFAULT_CUDA))
def test_published_grid_copy_is_graph_capturable(device: str) -> None:
    coords = make_coords(16).to(device=device)
    source = {"value": torch.arange(3 * len(coords) * 4, device=device).to(torch.uint8).reshape(3, len(coords), 1, 4)}
    target = allocate_grid_outputs(source, coords)
    indices = grid_sample_indices(coords)
    copy_grid_outputs(source, target, indices)
    torch.testing.assert_close(target["value"], coords.as_grid(source["value"][:, :, 0], sample_dim=1), atol=0, rtol=0)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_grid_outputs(source, target, indices)
    source["value"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(target["value"]) == 0
