# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Square-shell sensing uses every renderer texel with an analytic inverse."""

import math

import pytest
import torch
from fovi.sensing.coords import SamplingCoords
from isaaclab_fovi.sensors.generalized_projection import build_native_grid_textures
from isaaclab_fovi.sensors.native_grid import (
    allocate_grid_outputs,
    copy_grid_outputs,
    grid_output_shape,
    grid_sample_indices,
    mask_grid_samples,
    native_grid_directions,
    native_grid_exit_uv,
)


def make_coords(field: str, res: int = 16) -> SamplingCoords:
    fov = 170.0 if field == "spherical" else 1.0
    return SamplingCoords(
        fov,
        0.036021 * fov,
        res,
        style="warped_cartesian_as_grid",
        fov_type="square",
        radius_norm=math.inf,
        field_geometry=field,
    )


@pytest.mark.parametrize("field", ["planar", "spherical"])
@pytest.mark.parametrize("projection", ["pinhole", "fisheye_equidistant"])
@pytest.mark.parametrize("rotation", [0.0, 37.0, 180.0])
@pytest.mark.parametrize("res", [1, 15, 64])
def test_every_texel_roundtrips(field: str, projection: str, rotation: float, res: int) -> None:
    coords = make_coords(field, res)
    kwargs = dict(
        tangent_scale=0.7,
        image_rotation_deg=rotation,
        projection=projection,
        camera_fov_rad=math.radians(150),
    )
    rays = native_grid_directions(coords, **kwargs)
    uv, visible = native_grid_exit_uv(rays, coords, **kwargs)
    axis = (torch.arange(res) + 0.5) / res
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    expected = torch.stack((col, row), -1).reshape(-1, 2)
    assert visible.all() and coords.valid_mask.all()
    torch.testing.assert_close(uv, expected, atol=2e-6, rtol=0)
    torch.testing.assert_close(rays.norm(dim=-1), torch.ones(len(coords)), atol=1e-6, rtol=0)


@pytest.mark.parametrize("field", ["planar", "spherical"])
def test_textures_and_public_modalities(field: str) -> None:
    coords = make_coords(field)
    directions, inverse = build_native_grid_textures(coords, tangent_scale=0.7, image_rotation_deg=0, exit_size=64)
    torch.testing.assert_close(directions.reshape(-1, 3), native_grid_directions(coords, tangent_scale=0.7))
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


def test_camera_config_rejects_wang_and_requires_native_layout() -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg

    kwargs = dict(prim_path="/World/Camera", foveated_style="warped_cartesian_as_grid", radius_norm=math.inf)
    with pytest.raises(ValueError, match="native_grid"):
        FoveatedCameraCfg(**kwargs, fov_type="square")
    with pytest.raises(ValueError, match="wang"):
        FoveatedCameraCfg(**kwargs, ndc_layout="native_grid", fov_type="wang")
    cfg = FoveatedCameraCfg(**kwargs, ndc_layout="native_grid", fov_type="square")
    assert cfg.ndc_layout == "native_grid"


@pytest.mark.parametrize("coverage", ["square", "circular"])
def test_camera_pooling_preserves_square_shells(coverage: str) -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg
    from isaaclab_fovi.sensors.pooling import build_highres_coords, build_lowres_coords

    cfg = FoveatedCameraCfg(
        prim_path="/World/Camera",
        foveated_style="warped_cartesian_as_grid",
        fov_type=coverage,
        radius_norm=math.inf,
        ndc_layout="native_grid",
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
        dict(radius_norm=1.0, foveated_style="warped_cartesian_as_grid", ndc_layout="native_grid"),
        dict(radius_norm=math.inf, foveated_style="isotropic"),
    ],
)
def test_camera_rejects_unsupported_radius_norm(kwargs: dict[str, float | str]) -> None:
    from isaaclab_fovi.sensors.foveated_camera_cfg import FoveatedCameraCfg

    with pytest.raises(ValueError, match="radius_norm"):
        FoveatedCameraCfg(prim_path="/World/Camera", **kwargs)
