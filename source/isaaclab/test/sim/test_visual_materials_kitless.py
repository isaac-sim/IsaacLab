# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Pure-USD visual material authoring and binding without launching Kit."""

from pathlib import Path

import pytest

from pxr import Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.utils.version import has_kit

pytestmark = pytest.mark.unit


def _make_cube_and_bind(stage, material_path: str) -> None:
    cube = UsdGeom.Cube.Define(stage, "/World/Cube").GetPrim()
    sim_utils.bind_visual_material(cube.GetPath(), material_path)
    bound_material, _ = UsdShade.MaterialBindingAPI(cube).ComputeBoundMaterial()
    assert bound_material.GetPath() == material_path


def test_preview_surface_authors_and_binds_without_kit() -> None:
    assert not has_kit()
    stage = sim_utils.create_new_stage()
    cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.6, 0.9))

    shader_prim = cfg.func("/World/Looks/Preview", cfg)

    shader = UsdShade.Shader(shader_prim)
    assert shader.GetIdAttr().Get() == "UsdPreviewSurface"
    assert shader.GetInput("diffuseColor").Get() == cfg.diffuse_color
    assert stage.GetPrimAtPath("/World/Looks/Preview").GetAttribute("outputs:surface").GetConnections() == [
        Sdf.Path("/World/Looks/Preview/Shader.outputs:surface")
    ]
    _make_cube_and_bind(stage, "/World/Looks/Preview")


def test_builtin_mdl_authors_and_binds_without_kit() -> None:
    assert not has_kit()
    stage = sim_utils.create_new_stage()
    cfg = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl")

    shader_prim = cfg.func("/World/Looks/OmniPBR", cfg)

    shader = UsdShade.Shader(shader_prim)
    source_asset = shader.GetSourceAsset("mdl")
    assert Path(source_asset.path).is_file()
    assert Path(source_asset.path).name == "OmniPBR.mdl"
    assert shader.GetSourceAssetSubIdentifier("mdl") == "OmniPBR"
    assert stage.GetPrimAtPath("/World/Looks/OmniPBR").GetAttribute("outputs:mdl:surface").GetConnections() == [
        Sdf.Path("/World/Looks/OmniPBR/Shader.outputs:out")
    ]
    _make_cube_and_bind(stage, "/World/Looks/OmniPBR")


def test_builtin_mdl_resolves_archive_kit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mdl_path = tmp_path / "mdl/core/Base/OmniPBR.mdl"
    mdl_path.parent.mkdir(parents=True)
    mdl_path.touch()
    monkeypatch.setenv("CARB_APP_PATH", str(tmp_path))
    stage = sim_utils.create_new_stage()
    cfg = sim_utils.MdlFileCfg(mdl_path="OmniPBR.mdl")

    shader_prim = cfg.func("/World/Looks/OmniPBR", cfg)

    assert Path(UsdShade.Shader(shader_prim).GetSourceAsset("mdl").path) == mdl_path
    assert stage.GetPrimAtPath("/World/Looks/OmniPBR").IsValid()
