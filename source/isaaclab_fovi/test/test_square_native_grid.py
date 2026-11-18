# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Square-shell sensing uses every renderer texel with an analytic inverse."""

import math

import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors.lens_lut import build_native_grid_textures
from isaaclab_fovi.sensors.native_grid import (
    allocate_grid_outputs,
    copy_grid_outputs,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
    native_grid_exit_uv,
)

pytestmark = pytest.mark.kitless


def make_coords(res: int = 16) -> SamplingCoords:
    fov = 170.0
    return SamplingCoords(
        fov,
        0.036021 * fov,
        res,
        style="warped_cartesian_as_grid",
        fov_type="square",
        radius_norm=math.inf,
        field_geometry="spherical",
    )


@pytest.mark.parametrize(
    ("res", "rotation"),
    [
        # A single texel sits on the optical axis, where the sensor roll has no effect.
        (1, 0.0),
        # Odd grids put a texel on the optical axis; even grids straddle it.
        (15, 37.0),
        (64, 180.0),
    ],
)
def test_every_texel_roundtrips(res: int, rotation: float) -> None:
    coords = make_coords(res)
    rays = native_grid_directions(coords, image_rotation_deg=rotation)
    uv, visible = native_grid_exit_uv(rays, coords, image_rotation_deg=rotation)
    axis = (torch.arange(res) + 0.5) / res
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    expected = torch.stack((col, row), -1).reshape(-1, 2)
    assert visible.all() and coords.valid_mask.all()
    torch.testing.assert_close(uv, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(len(coords)), atol=1e-6, rtol=0)


def texel_centers(res: int) -> torch.Tensor:
    """Native texture UV of each canonical sample, in row-major sample order."""
    axis = (torch.arange(res) + 0.5) / res
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack((col, row), -1).reshape(-1, 2)


def eccentricity_rays(visual: torch.Tensor, fov_deg: float, rotation_deg: float = 0.0) -> torch.Tensor:
    """OpenGL rays for visual-chart points, at Fovi's spherical eccentricity: Euclidean visual radius times the half
    FoV. Square-shell corners reach sqrt(2) half FoVs, behind the optical hemisphere for wide fields."""
    visual = visual.to(torch.float64)
    theta = torch.linalg.vector_norm(visual, dim=-1) * math.radians(fov_deg) / 2
    phi = torch.atan2(visual[..., 1], visual[..., 0]) + math.radians(rotation_deg)
    return torch.stack((theta.sin() * phi.cos(), theta.sin() * phi.sin(), -theta.cos()), -1)


def ray_visual(rays: torch.Tensor, fov_deg: float, rotation_deg: float = 0.0) -> torch.Tensor:
    """Visual-chart point of each ray: its eccentricity over the half FoV along its unrolled azimuth."""
    rays = rays.to(torch.float64)
    theta = torch.atan2(torch.linalg.vector_norm(rays[..., :2], dim=-1), -rays[..., 2])
    phi = torch.atan2(rays[..., 1], rays[..., 0]) - math.radians(rotation_deg)
    radius = theta / (math.radians(fov_deg) / 2)
    return torch.stack((radius * phi.cos(), radius * phi.sin()), -1)


def octahedral_texel_rays(size: int) -> torch.Tensor:
    """OpenGL rays at octahedral texel centres; the outer triangles fold onto the back hemisphere."""
    axis = (torch.arange(size, dtype=torch.float64) + 0.5) / size * 2 - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    forward = 1 - x.abs() - y.abs()
    back = forward < 0
    fx = torch.where(back, (1 - y.abs()) * x.sign(), x)
    fy = torch.where(back, (1 - x.abs()) * y.sign(), y)
    return torch.nn.functional.normalize(torch.stack((fx, fy, -forward), -1), dim=-1)


@pytest.mark.parametrize("rotation", [0.0, 37.0])
def test_inverse_maps_independent_rays_to_their_texels(rotation: float) -> None:
    coords = make_coords(64)
    half_fov = math.radians(coords.fov) / 2
    # Square shells keep every sample, so every one of Fovi's sample positions must land on its own texel.
    rays = eccentricity_rays(coords.cartesian, coords.fov, rotation)
    assert (rays[:, 2] > 0).any()
    uv, visible = native_grid_exit_uv(rays, coords, image_rotation_deg=rotation)
    assert visible.all()
    torch.testing.assert_close(uv, texel_centers(64), atol=2e-6, rtol=0)
    torch.testing.assert_close(
        native_grid_directions(coords, image_rotation_deg=rotation), rays.float(), atol=1e-6, rtol=0
    )
    # The optical axis is the texture centre, the nominal rim along each rolled native axis is an edge midpoint, and
    # the square's corners lie sqrt(2) half FoVs out along the rolled diagonals.
    roll = math.radians(rotation)

    def ray(angle: float, azimuth: float) -> tuple[float, float, float]:
        return (
            math.sin(angle) * math.cos(roll + azimuth),
            math.sin(angle) * math.sin(roll + azimuth),
            -math.cos(angle),
        )

    corner = math.sqrt(2) * half_fov
    hand_rays = torch.tensor(
        [
            (0.0, 0.0, -1.0),
            ray(half_fov, 0.0),
            ray(half_fov, math.pi / 2),
            ray(half_fov, math.pi),
            ray(corner, math.pi / 4),
            ray(corner, -math.pi / 4),
            ray(corner, -3 * math.pi / 4),
        ],
        dtype=torch.float64,
    )
    uv, visible = native_grid_exit_uv(hand_rays, coords, image_rotation_deg=rotation)
    assert visible.all()
    expected = torch.tensor([[0.5, 0.5], [0.5, 1.0], [1.0, 0.5], [0.5, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    torch.testing.assert_close(uv, expected, atol=1e-6, rtol=0)


@pytest.mark.parametrize("rotation", [0.0, 37.0])
def test_textures_match_independent_geometry(rotation: float) -> None:
    coords = make_coords(64)
    size = 256
    direction, inverse = build_native_grid_textures(coords, image_rotation_deg=rotation, exit_size=size)
    assert direction.shape == (64, 64, 3) and inverse.shape == (size, size, 2)
    expected = eccentricity_rays(coords.cartesian, coords.fov, rotation).float()
    torch.testing.assert_close(direction.reshape(-1, 3), expected, atol=1e-6, rtol=0)
    # A texel's ray is on the sensor exactly when its visual point lies in the unit square.
    texel_rays = octahedral_texel_rays(size)
    extent = ray_visual(texel_rays, coords.fov, rotation).abs().amax(-1)
    inside, outside = extent < 1 - 1e-6, extent > 1 + 1e-6
    assert (texel_rays[inside][:, 2] > 0).any() and outside.any()
    assert (inverse[inside] >= 0).all() and (inverse[outside] == -1).all()
    # Each stored UV, mapped back through Fovi's native-to-visual warp, must look along its texel's ray.
    native = (inverse[inside].to(torch.float64) * 2 - 1).flip(-1)
    torch.testing.assert_close(
        eccentricity_rays(coords.native_to_visual(native), coords.fov, rotation),
        texel_rays[inside],
        atol=1e-5,
        rtol=0,
    )


def test_textures_and_public_modalities() -> None:
    coords = make_coords()
    directions, inverse = build_native_grid_textures(coords, image_rotation_deg=0, exit_size=64)
    torch.testing.assert_close(directions.reshape(-1, 3), native_grid_directions(coords))
    assert torch.isfinite(inverse).all()
    assert (inverse == -1).any()
    assert grid_output_shape(coords) == (16, 16)
    source = {
        "rgba": torch.ones(2, len(coords), 1, 4, dtype=torch.uint8),
        "depth": torch.arange(len(coords), dtype=torch.float32)[None, :, None, None],
        "segmentation": torch.arange(len(coords), dtype=torch.int32)[None, :, None, None],
    }
    source["rgb"] = source["rgba"][..., :3]
    mask_grid_samples(source, coords)
    assert source["rgba"].all()
    target = allocate_grid_outputs(source, coords)
    pointers = {name: tensor.data_ptr() for name, tensor in target.items()}
    copy_grid_outputs(source, target, grid_sample_indices(coords))
    for name in source:
        torch.testing.assert_close(target[name], coords.as_grid(source[name][:, :, 0], sample_dim=1))
        assert target[name].data_ptr() == pointers[name]
    assert target["rgb"].untyped_storage().data_ptr() == target["rgba"].untyped_storage().data_ptr()


def test_camera_config_rejects_wang_with_square_shells() -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg

    kwargs = dict(prim_path="/World/Camera", foveated_style="warped_cartesian_as_grid", radius_norm=math.inf)
    with pytest.raises(ValueError, match="wang"):
        FoveatedCameraCfg(**kwargs, fov_type="wang")
    cfg = FoveatedCameraCfg(**kwargs, fov_type="square")
    assert cfg.fov_type == "square"


@pytest.mark.parametrize("coverage", ["square", "circular"])
def test_camera_pooling_preserves_square_shells(coverage: str) -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg
    from isaaclab_fovi.sensors.pooling import build_highres_coords, build_lowres_coords

    from isaaclab.sim import PinholeCameraCfg

    cfg = FoveatedCameraCfg(
        prim_path="/World/Camera",
        spawn=PinholeCameraCfg(focal_length=1.0, horizontal_aperture=2.0),
        cmf_a_deg=9.0,
        foveated_style="warped_cartesian_as_grid",
        fov_type=coverage,
        radius_norm=math.inf,
        foveated_res=16,
        pool_enabled=True,
        pool_res_mult=2.0,
    )
    lowres, matched = build_lowres_coords(cfg, "cpu")
    highres = build_highres_coords(cfg, matched, "cpu")
    assert lowres.radius_norm == highres.radius_norm == math.inf
    for coords in (lowres, highres):
        assert coords.fov_type == coverage
        boundary = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        torch.testing.assert_close(coords.native_to_visual(boundary), boundary)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(radius_norm=1.0, foveated_style="warped_cartesian_as_grid"),
        dict(radius_norm=math.inf, foveated_style="isotropic"),
    ],
)
def test_camera_rejects_unsupported_radius_norm(kwargs: dict[str, float | str]) -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg

    with pytest.raises(ValueError, match="radius_norm"):
        FoveatedCameraCfg(prim_path="/World/Camera", **kwargs)
