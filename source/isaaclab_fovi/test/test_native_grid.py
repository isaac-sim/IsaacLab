# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native sensor texture geometry agrees with Fovi's ordered samples."""

import math

import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors.generalized_projection import build_native_grid_textures
from isaaclab_fovi.sensors.native_grid import (
    allocate_grid_outputs,
    bounded_angular_radius,
    canonical_to_grid,
    copy_grid_outputs,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
    native_grid_exit_uv,
    unbounded_angular_radius,
)


def make_coords(field: str, resolution: int = 64, coverage: str = "circular") -> SamplingCoords:
    fov = (80.0 if coverage == "wang" else 170.0) if field == "spherical" else 1.0
    return SamplingCoords(
        fov=fov,
        cmf_a=0.036021 * fov,
        res=resolution,
        style="warped_cartesian_as_grid",
        fov_type=coverage,
        field_geometry=field,
        device="cpu",
    )


@pytest.mark.parametrize("field", ["planar", "spherical"])
@pytest.mark.parametrize("resolution", [64, 128])
@pytest.mark.parametrize("rotation", [0.0, 180.0, 37.0])
@pytest.mark.parametrize("coverage", ["circular", "square", "wang"])
def test_analytic_inverse_returns_every_native_texel(field, resolution, rotation, coverage):
    coords = make_coords(field, resolution, coverage)
    rays = native_grid_directions(coords, tangent_scale=0.7, image_rotation_deg=rotation)
    uv, valid = native_grid_exit_uv(rays, coords, tangent_scale=0.7, image_rotation_deg=rotation)
    axis = (torch.arange(resolution) + 0.5) / resolution
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    expected = torch.stack((col, row), -1).reshape(-1, 2)
    assert valid.all()
    torch.testing.assert_close(uv, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(len(coords)), atol=1e-6, rtol=0)


@pytest.mark.parametrize("coverage", ["circular", "square", "wang"])
def test_spherical_valid_samples_match_calibrated_software_rays(coverage):
    from isaaclab_fovi.sensors.calibrated_geometry import retinal_directions

    coords = make_coords("spherical", coverage=coverage)
    actual = native_grid_directions(coords, tangent_scale=0.7)
    expected = retinal_directions(coords.cartesian[coords.valid_mask], coords.fov)
    torch.testing.assert_close(actual[coords.valid_mask], expected, atol=3e-7, rtol=0)


def test_planar_equidistant_preserves_lens_projection():
    from isaaclab_fovi.sensors.calibrated_geometry import retinal_directions

    coords = make_coords("planar")
    rays = native_grid_directions(
        coords,
        tangent_scale=0.7,
        projection="fisheye_equidistant",
        camera_fov_rad=math.radians(150),
    )
    expected = retinal_directions(coords.cartesian[coords.valid_mask], 150)
    torch.testing.assert_close(rays[coords.valid_mask], expected, atol=3e-7, rtol=0)
    uv, valid = native_grid_exit_uv(
        rays,
        coords,
        tangent_scale=0.7,
        projection="fisheye_equidistant",
        camera_fov_rad=math.radians(150),
    )
    torch.testing.assert_close(uv, (coords.cortical.flip(-1) + 1) / 2, atol=2e-6, rtol=0)
    assert valid.all()


def test_spherical_padding_matches_rim_slope_and_does_not_wrap():
    radius = torch.tensor([0.0, 0.99999, 1.0, 1.00001, 2.0, 10.0], dtype=torch.float64)
    half_fov = math.radians(170) / 2
    angles = bounded_angular_radius(radius, half_fov)
    assert (angles[1:] > angles[:-1]).all()
    assert (angles < math.pi).all()
    left = (angles[2] - angles[1]) / (radius[2] - radius[1])
    right = (angles[3] - angles[2]) / (radius[3] - radius[2])
    torch.testing.assert_close(left, right, atol=2e-5, rtol=0)
    torch.testing.assert_close(unbounded_angular_radius(angles, half_fov), radius, atol=1e-10, rtol=0)


@pytest.mark.parametrize("field", ["planar", "spherical"])
def test_direction_texture_is_native_rays_without_assignment(field):
    coords = make_coords(field, 64)
    direction, inverse = build_native_grid_textures(coords, tangent_scale=0.7, image_rotation_deg=0, exit_size=256)
    expected = native_grid_directions(coords, tangent_scale=0.7)
    torch.testing.assert_close(direction.reshape(-1, 3), expected, atol=0, rtol=0)
    assert direction.shape == (64, 64, 3)
    assert inverse.shape == (256, 256, 2)
    assert torch.isfinite(inverse).all()
    assert (inverse == -1).any()


def test_upright_grid_masks_and_channel_aliases():
    coords = make_coords("planar", 16)
    # Coordinate channels make rotation/transposition failures observable.
    values = torch.cat((coords.cartesian, torch.ones(len(coords), 2)), -1)[None, :, None]
    source = {"rgba": values.clone(), "rgb": values[..., :3]}
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
    torch.testing.assert_close(canonical_to_grid(source["rgba"], coords), expected)


@pytest.mark.parametrize("dtype,channels", [(torch.uint8, 4), (torch.float32, 1), (torch.int32, 1)])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_output_permutation_preserves_dtype_and_captures(dtype, channels, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    coords = make_coords("planar", 16).to(device=device)
    source = {
        "value": torch.arange(3 * len(coords) * channels, device=device).to(dtype).reshape(3, len(coords), 1, channels)
    }
    target = allocate_grid_outputs(source, coords)
    indices = grid_sample_indices(coords)
    copy_grid_outputs(source, target, indices)
    expected = canonical_to_grid(source["value"], coords)
    torch.testing.assert_close(target["value"], expected, atol=0, rtol=0)
    if device == "cuda":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            copy_grid_outputs(source, target, indices)
        source["value"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.count_nonzero(target["value"]) == 0
