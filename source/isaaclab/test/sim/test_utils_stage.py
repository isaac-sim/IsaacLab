# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for stage utilities."""

import pytest

from pxr import Sdf, Usd, UsdLux, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.stage import _is_prim_deletable, resolve_paths

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]


def test_create_new_stage_becomes_current():
    stage1 = sim_utils.create_new_stage()
    stage1.DefinePrim("/Stage1Marker", "Xform")
    assert isinstance(stage1, Usd.Stage)
    assert sim_utils.get_current_stage() == stage1
    assert isinstance(sim_utils.get_current_stage_id(), int)
    assert sim_utils.get_current_stage_id() >= 0
    assert sim_utils.is_current_stage_in_memory() is True

    # a new stage replaces the current one
    stage2 = sim_utils.create_new_stage()
    stage2.DefinePrim("/Stage2Marker", "Xform")
    assert stage2 != stage1
    assert sim_utils.get_current_stage() == stage2
    assert not sim_utils.get_current_stage().GetPrimAtPath("/Stage1Marker").IsValid()

    # use_stage switches temporarily and restores afterwards
    with sim_utils.use_stage(stage1):
        assert sim_utils.get_current_stage().GetPrimAtPath("/Stage1Marker").IsValid()
    assert sim_utils.get_current_stage().GetPrimAtPath("/Stage2Marker").IsValid()
    with pytest.raises(TypeError):
        with sim_utils.use_stage("not a stage"):  # type: ignore
            pass


@pytest.mark.parametrize("reload_in_place", [False, True])
def test_save_and_open_stage(tmp_path, reload_in_place):
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/TestCube", "Cube")
    save_path = str(tmp_path / "test_stage.usd")

    assert sim_utils.save_stage(save_path, save_and_reload_in_place=reload_in_place) is True

    current = sim_utils.get_current_stage()
    assert (current != stage) is reload_in_place
    assert current.GetPrimAtPath("/World/TestCube").GetTypeName() == "Cube"

    # open a different stage, then re-open the saved one: it becomes current
    sim_utils.create_new_stage()
    opened = sim_utils.open_stage(save_path)
    assert sim_utils.get_current_stage() == opened
    assert opened.GetPrimAtPath("/World/TestCube").GetTypeName() == "Cube"


def test_open_and_save_stage_reject_unsupported_files(tmp_path):
    sim_utils.create_new_stage()
    with pytest.raises(ValueError, match="not supported"):
        sim_utils.open_stage("/invalid/path/to/stage.invalid")
    with pytest.raises(ValueError, match="not supported"):
        sim_utils.save_stage(str(tmp_path / "test.invalid"))


@pytest.mark.parametrize(
    "ref",
    [
        "http://example.com/file.usd",
        "https://example.com/assets/textures/sky.hdr",
        "omniverse://path/to/file.usd",
        "s3://bucket-name/path/to/asset.usd",
    ],
)
def test_save_stage_preserves_uri_asset_paths(tmp_path, ref):
    stage = sim_utils.create_new_stage()
    UsdLux.DomeLight.Define(stage, "/World/skyLight").CreateTextureFileAttr().Set(ref)
    save_path = tmp_path / "test_stage.usda"

    assert sim_utils.save_stage(str(save_path), save_and_reload_in_place=False) is True

    saved = save_path.read_text(encoding="utf-8")
    assert f"@{ref}@" in saved
    assert "../" + ref.split("://")[0] not in saved


def test_close_stage():
    sim_utils.create_new_stage()
    assert sim_utils.close_stage() is True
    assert sim_utils.get_current_stage() is None


def test_clear_stage_respects_deletable_predicate():
    stage = sim_utils.create_new_stage()
    stage.DefinePrim("/World", "Xform")
    stage.DefinePrim("/World/Cube", "Cube")
    stage.DefinePrim("/World/Sphere", "Sphere")
    stage.DefinePrim("/Render/Product", "Xform")
    assert _is_prim_deletable(stage.GetPrimAtPath("/World/Cube")) is True
    assert _is_prim_deletable(stage.GetPseudoRoot()) is False

    sim_utils.clear_stage(lambda prim: prim.GetTypeName() == "Cube")
    assert not stage.GetPrimAtPath("/World/Cube").IsValid()
    assert stage.GetPrimAtPath("/World/Sphere").IsValid()

    # the default predicate deletes everything except protected namespaces
    sim_utils.clear_stage()
    assert not stage.GetPrimAtPath("/World").IsValid()
    assert stage.GetPrimAtPath("/Render/Product").IsValid()


def test_resolve_paths_reanchors_relative_and_keeps_search_path_assets(tmp_path):
    """Re-anchoring ``OmniPBR.mdl`` would make it relative to the working directory and resolve to nothing."""
    mesh_path = tmp_path / "meshes" / "mesh.usda"
    mesh_path.parent.mkdir()
    mesh_stage = Usd.Stage.CreateNew(str(mesh_path))
    mesh_stage.GetRootLayer().Save()

    source_path = tmp_path / "asset.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_stage.DefinePrim("/World", "Xform").GetReferences().AddReference("./meshes/mesh.usda")
    shader = UsdShade.Shader.Define(source_stage, "/World/Looks/red/red")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    source_stage.GetRootLayer().Save()

    # copy the layer one directory deeper, as export_prim_to_file does for instanceable meshes
    dest_path = tmp_path / "Props" / "instanceable_meshes.usda"
    dest_path.parent.mkdir()
    dest_layer = Sdf.Layer.CreateNew(str(dest_path))
    dest_layer.TransferContent(source_stage.GetRootLayer())

    resolve_paths(str(source_path), str(dest_path))

    contents = dest_layer.ExportToString()
    assert "@OmniPBR.mdl@" in contents
    assert "@../meshes/mesh.usda@" in contents
