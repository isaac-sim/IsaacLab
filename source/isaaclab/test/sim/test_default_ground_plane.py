# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from pathlib import Path

import pytest
from PIL import Image

from pxr import Usd, UsdGeom, UsdShade

from isaaclab.sim.spawners.from_files import from_files_cfg
from isaaclab.utils.assets import retrieve_file_path

USD_PATH = from_files_cfg._DEFAULT_GROUND_PLANE_USD
NVIDIA_GREEN = (118, 185, 0)
OFF_WHITE = (221, 218, 210)
DARK_GREY = (90, 93, 97)
ROUGH = 255
SMOOTH = 13


@pytest.fixture(scope="module")
def default_ground_plane_asset() -> tuple[Path, Path]:
    """Resolve the bundled ground-plane USD and its texture directory."""
    local_usd_path = Path(retrieve_file_path(USD_PATH))
    return local_usd_path, local_usd_path.parent / "Materials" / "Textures"


def test_default_ground_plane_usd_contract(default_ground_plane_asset: tuple[Path, Path]):
    """Validate the ground plane's spawner-compatible structure and metric UV mapping."""
    local_usd_path, texture_dir = default_ground_plane_asset
    stage = Usd.Stage.Open(str(local_usd_path))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/World"
    assert stage.GetRootLayer().customLayerData["textureTileSizeMeters"] == 2.0

    for prim_path in (
        "/World/Environment/Geometry",
        "/World/Environment/GlossyChecks",
        "/World/GroundPlane/CollisionPlane",
        "/World/Looks/theGrid/Shader",
        "/World/Looks/GlossyChecks/Shader",
    ):
        assert stage.GetPrimAtPath(prim_path).IsValid()
    assert not stage.GetPrimAtPath("/World/SphereLight").IsValid()
    assert UsdGeom.Imageable(stage.GetPrimAtPath("/World/GroundPlane/CollisionPlane")).GetVisibilityAttr().Get() == (
        UsdGeom.Tokens.invisible
    )

    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Environment/Geometry"))
    assert len(mesh.GetPointsAttr().Get()) == 4
    assert list(mesh.GetFaceVertexCountsAttr().Get()) == [4]
    assert [tuple(uv) for uv in UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get()] == [
        (-25.0, -25.0),
        (25.0, -25.0),
        (25.0, 25.0),
        (-25.0, 25.0),
    ]

    base_shader = UsdShade.Shader.Get(stage, "/World/Looks/theGrid/Shader")
    glossy_shader = UsdShade.Shader.Get(stage, "/World/Looks/GlossyChecks/Shader")
    assert base_shader.GetInput("project_uvw").Get() is False
    assert tuple(base_shader.GetInput("texture_scale").Get()) == pytest.approx((1.0, 1.0))
    assert base_shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(1.0)
    assert base_shader.GetInput("reflection_roughness_texture_influence").Get() == pytest.approx(1.0)
    assert glossy_shader.GetInput("reflection_roughness_constant").Get() == pytest.approx(0.05)
    for input_name, filename in (
        ("diffuse_texture", "default_ground_plane_albedo.png"),
        ("reflectionroughness_texture", "default_ground_plane_roughness.png"),
    ):
        assert base_shader.GetInput(input_name).Get().path == f"./Materials/Textures/{filename}"
        assert (texture_dir / filename).is_file()


def test_default_ground_plane_texture_contract(default_ground_plane_asset: tuple[Path, Path]):
    """Validate the 1 m checker cells and 2 m NVIDIA-green landmarks."""
    _, texture_dir = default_ground_plane_asset
    with Image.open(texture_dir / "default_ground_plane_albedo.png") as image:
        assert image.size == (512, 512)
        assert set(image.get_flattened_data()) == {OFF_WHITE, DARK_GREY, NVIDIA_GREEN}

        pixels_per_meter = image.width / 2.0
        for row in range(2):
            for column in range(2):
                center = (round((column + 0.5) * pixels_per_meter), round((row + 0.5) * pixels_per_meter))
                expected_color = OFF_WHITE if (row + column) % 2 == 0 else DARK_GREY
                assert image.getpixel(center) == expected_color

        # Quarter diamonds join across repeated texture edges into one landmark every 2 m.
        for corner in ((0, 0), (image.width - 1, 0), (0, image.height - 1), (image.width - 1,) * 2):
            assert image.getpixel(corner) == NVIDIA_GREEN

    with Image.open(texture_dir / "default_ground_plane_roughness.png") as image:
        assert image.size == (512, 512)
        assert set(image.get_flattened_data()) == {SMOOTH, ROUGH}
        assert image.getpixel((128, 128)) == ROUGH
        assert image.getpixel((384, 128)) == SMOOTH
        assert image.getpixel((0, 0)) == ROUGH


def test_ground_plane_defaults_to_bundled_appearance():
    """Validate that the default config selects the bundled asset without tinting it."""
    cfg = from_files_cfg.GroundPlaneCfg()

    assert cfg.usd_path == USD_PATH
    assert cfg.color is None
