# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest

pytestmark = pytest.mark.ov
from isaaclab_ov.stage import export_stage_to_string
from isaaclab_teleop.xr_visual_exclusions import exclude_visual_meshes

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def _scene(root: str = "/World/envs/env_0") -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    env = UsdGeom.Xform.Define(stage, root).GetPrim()
    env.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    UsdGeom.Mesh.Define(stage, "/Asset/head/mesh")
    UsdGeom.Mesh.Define(stage, "/Asset/body/mesh")
    visual = UsdGeom.Xform.Define(stage, f"{root}/Robot/visuals").GetPrim()
    visual.GetReferences().AddInternalReference("/Asset")
    visual.SetInstanceable(True)
    collision = UsdGeom.Mesh.Define(stage, f"{root}/Robot/collisions/head").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision)
    UsdGeom.Camera.Define(stage, f"{root}/Robot/Camera")
    return stage


@pytest.mark.parametrize(
    "settings_path,root", [(None, "/World/envs/env_0"), ("/Render/AuthoredSettings", "/World/custom/scene_0")]
)
def test_exclusion_export_hides_only_matched_meshes_and_restores_source(settings_path, root) -> None:
    stage = _scene(root)
    if settings_path is not None:
        stage.DefinePrim(settings_path, "RenderSettings")
        stage.SetMetadata("renderSettingsPrimPath", settings_path)
    original = stage.GetRootLayer().ExportToString()
    session = stage.GetSessionLayer().ExportToString()
    head = f"{root}/Robot/visuals/head/mesh"
    kept = (f"{root}/Robot/visuals/body/mesh", f"{root}/Robot/collisions/head")
    with exclude_visual_meshes(stage, ["{ENV_REGEX_NS}/Robot/visuals/head/mesh"], environment_root_path=root):
        assert UsdGeom.Imageable(stage.GetPrimAtPath(head)).ComputeVisibility() == UsdGeom.Tokens.invisible
        for path in kept:
            assert UsdGeom.Imageable(stage.GetPrimAtPath(path)).ComputeVisibility() == UsdGeom.Tokens.inherited
        assert stage.GetPrimAtPath(f"{root}/Robot/Camera").IsA(UsdGeom.Camera)
        layer = Sdf.Layer.CreateAnonymous()
        layer.ImportFromString(export_stage_to_string(stage, (root,), source_paths=(root,)))
        exported = Usd.Stage.Open(layer)
        assert UsdGeom.Imageable(exported.GetPrimAtPath(head)).ComputeVisibility() == UsdGeom.Tokens.invisible
        for path in kept:
            assert UsdGeom.Imageable(exported.GetPrimAtPath(path)).ComputeVisibility() == UsdGeom.Tokens.inherited
        # Unpartitioned eye cameras see every partition, including viewer-partition debug markers.
        settings = exported.GetPrimAtPath(exported.GetMetadata("renderSettingsPrimPath"))
        assert settings.GetAttribute("omni:rtx:scenePartitioning:showAllPartitionsByDefault").Get() is True
    assert stage.GetRootLayer().ExportToString() == original
    assert stage.GetSessionLayer().ExportToString() == session
    assert stage.GetPrimAtPath(head).IsInstanceProxy()
    assert UsdGeom.Imageable(stage.GetPrimAtPath(head)).ComputeVisibility() == UsdGeom.Tokens.inherited


@pytest.mark.parametrize("pattern", ["/missing", "/World/envs/env_0/Robot/collisions/head"])
def test_reject_missing_or_collision_mesh_before_edit(pattern):
    stage = _scene()
    original = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):
        with exclude_visual_meshes(stage, [pattern]):
            pass
    assert stage.GetSessionLayer().ExportToString() == original


def test_export_failure_removes_temporary_layer():
    stage = _scene()
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(RuntimeError):
        with exclude_visual_meshes(stage, ["/World/envs/env_0/Robot/visuals/head/mesh"]):
            raise RuntimeError("export failed")
    assert stage.GetSessionLayer().ExportToString() == before


@pytest.mark.parametrize("conflict", ["instance", "visibility"])
def test_stronger_session_opinion_fails_closed_and_is_preserved(conflict):
    stage = _scene()
    head = "/World/envs/env_0/Robot/visuals/head/mesh"
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        visual = stage.GetPrimAtPath("/World/envs/env_0/Robot/visuals")
        if conflict == "instance":
            visual.SetInstanceable(True)
        else:
            visual.SetInstanceable(False)
            UsdGeom.Imageable(stage.GetPrimAtPath(head)).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(RuntimeError, match="Stronger source opinion"):
        with exclude_visual_meshes(stage, [head]):
            pytest.fail("A stronger source opinion must not silently defeat the exclusion")
    assert stage.GetSessionLayer().ExportToString() == before
