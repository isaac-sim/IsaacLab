# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from PIL import Image

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.from_files import from_files_cfg

ASSET_DIR = Path(from_files_cfg.__file__).parent / "data" / "default_ground_plane"
USD_PATH = ASSET_DIR / "default_ground_plane.usda"
NVIDIA_GREEN = (118, 185, 0)


def test_default_ground_plane_usd_contract():
    """Validate the ground plane's spawner-compatible structure and metric UV mapping."""
    stage = Usd.Stage.Open(str(USD_PATH))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/World"

    for prim_path in (
        "/World/Environment/Geometry",
        "/World/GroundPlane/CollisionPlane",
        "/World/Looks/theGrid/Shader",
        "/World/SphereLight",
    ):
        assert stage.GetPrimAtPath(prim_path).IsValid()

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Environment/Geometry"))
    shader = UsdShade.Shader(stage.GetPrimAtPath("/World/Looks/theGrid/Shader"))
    # ``primvars:st`` is the only texture mapping: OmniPBR reads UV set 0 rather than projecting.
    assert shader.GetInput("project_uvw").Get() is False
    assert not shader.GetInput("world_or_object")
    assert tuple(shader.GetInput("texture_scale").Get()) == pytest.approx((1.0, 1.0))
    assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()] == [
        (-10.0, -10.0),
        (10.0, -10.0),
        (10.0, 10.0),
        (-10.0, 10.0),
    ]

    for input_name, filename in (
        ("diffuse_texture", "albedo.png"),
        ("emissive_color_texture", "emissive_color.png"),
        ("emissive_mask_texture", "emissive_mask.png"),
    ):
        assert shader.GetInput(input_name).Get().path == f"./{filename}"
        assert (ASSET_DIR / filename).is_file()


def test_default_ground_plane_texture_contract():
    """Validate the selected colors, metric line widths, and seamless landmark edge."""
    for filename in ("albedo.png", "emissive_color.png", "emissive_mask.png"):
        with Image.open(ASSET_DIR / filename) as image:
            assert image.size == (1000, 1000)

    with Image.open(ASSET_DIR / "albedo.png") as image:
        sample_y = 30

        # A 10 mm primary line occupies two texels at 200 texels per meter.
        assert image.getpixel((198, sample_y)) != NVIDIA_GREEN
        assert [image.getpixel((x, sample_y)) for x in range(199, 201)] == [NVIDIA_GREEN] * 2
        assert image.getpixel((201, sample_y)) != NVIDIA_GREEN

        # The former 0.1 m grey subdivisions are absent from the texture.
        subdivision_color = (217, 220, 213)
        assert subdivision_color not in set(image.get_flattened_data())

        # The 10 mm landmark is also two texels wide across the repeating texture seam.
        assert image.getpixel((998, sample_y)) != NVIDIA_GREEN
        assert [image.getpixel((x, sample_y)) for x in (999, 0)] == [NVIDIA_GREEN] * 2
        assert image.getpixel((1, sample_y)) != NVIDIA_GREEN


def test_ground_plane_defaults_to_bundled_appearance():
    """Validate that the default config selects the bundled asset without tinting it."""
    cfg = from_files_cfg.GroundPlaneCfg()

    assert Path(cfg.usd_path) == USD_PATH
    assert cfg.color is None
