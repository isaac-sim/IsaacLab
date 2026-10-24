# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pytest
from isaaclab_ov.stage import export_stage_to_string
from isaaclab_teleop.xr_scene_partition import partition_visual_meshes

from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def _scene():
    stage = Usd.Stage.CreateInMemory()
    env = UsdGeom.Xform.Define(stage, "/World/envs/env_0").GetPrim()
    env.CreateAttribute("primvars:omni:scenePartition", Sdf.ValueTypeNames.Token).Set("env_0")
    UsdGeom.Mesh.Define(stage, "/Asset/head/mesh")
    UsdGeom.Mesh.Define(stage, "/Asset/body/mesh")
    visual = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Robot/visuals").GetPrim()
    visual.GetReferences().AddInternalReference("/Asset")
    visual.SetInstanceable(True)
    collision = UsdGeom.Mesh.Define(stage, "/World/envs/env_0/Robot/collisions/head").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collision)
    UsdGeom.Camera.Define(stage, "/World/envs/env_0/Robot/Camera")
    return stage


@pytest.mark.parametrize("settings_path", [None, "/Render/AuthoredSettings"])
def test_partition_export_preserves_source_visibility_physics_and_camera(settings_path):
    stage = _scene()
    if settings_path is not None:
        stage.DefinePrim(settings_path, "RenderSettings")
        stage.SetMetadata("renderSettingsPrimPath", settings_path)
    original = stage.GetRootLayer().ExportToString()
    session = stage.GetSessionLayer().ExportToString()
    head = "/World/envs/env_0/Robot/visuals/head/mesh"
    with partition_visual_meshes(stage, [head]):
        prim = stage.GetPrimAtPath(head)
        assert UsdGeom.PrimvarsAPI(prim).FindPrimvarWithInheritance("omni:scenePartition").Get() == "xr_excluded"
        assert UsdGeom.Imageable(prim).ComputeVisibility() == "inherited"
        for path in ("/World/envs/env_0/Robot/visuals/body/mesh", "/World/envs/env_0/Robot/collisions/head"):
            assert (
                UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).FindPrimvarWithInheritance("omni:scenePartition").Get()
                == "env_0"
            )
        assert stage.GetPrimAtPath("/World/envs/env_0/Robot/Camera").IsA(UsdGeom.Camera)
        layer = Sdf.Layer.CreateAnonymous()
        layer.ImportFromString(export_stage_to_string(stage, 1, source_paths=("/World/envs/env_0",)))
        exported = Usd.Stage.Open(layer)
        settings = exported.GetPrimAtPath(exported.GetMetadata("renderSettingsPrimPath"))
        assert settings.GetAttribute("omni:rtx:scenePartitioning:showAllPartitionsByDefault").Get() is False
        assert (
            UsdGeom.PrimvarsAPI(exported.GetPrimAtPath(head)).FindPrimvarWithInheritance("omni:scenePartition").Get()
            == "xr_excluded"
        )
    assert stage.GetRootLayer().ExportToString() == original
    assert stage.GetSessionLayer().ExportToString() == session
    assert stage.GetPrimAtPath(head).IsInstanceProxy()


@pytest.mark.parametrize("pattern", ["/missing", "/World/envs/env_0/Robot/collisions/head"])
def test_reject_missing_or_collision_mesh_before_edit(pattern):
    stage = _scene()
    original = stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):
        with partition_visual_meshes(stage, [pattern]):
            pass
    assert stage.GetSessionLayer().ExportToString() == original


def test_export_failure_removes_temporary_layer():
    stage = _scene()
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(RuntimeError):
        with partition_visual_meshes(stage, ["/World/envs/env_0/Robot/visuals/head/mesh"]):
            raise RuntimeError("export failed")
    assert stage.GetSessionLayer().ExportToString() == before


@pytest.mark.parametrize("conflict", ["instance", "partition", "show_all"])
def test_stronger_session_opinion_fails_closed_and_is_preserved(conflict):
    stage = _scene()
    head = "/World/envs/env_0/Robot/visuals/head/mesh"
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        visual = stage.GetPrimAtPath("/World/envs/env_0/Robot/visuals")
        if conflict == "instance":
            visual.SetInstanceable(True)
        elif conflict == "partition":
            visual.SetInstanceable(False)
            UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(head)).CreatePrimvar(
                "omni:scenePartition", Sdf.ValueTypeNames.Token, UsdGeom.Tokens.constant
            ).Set("env_0")
        else:
            settings = stage.DefinePrim("/Render/IsaacLabSettings", "RenderSettings")
            settings.CreateAttribute(
                "omni:rtx:scenePartitioning:showAllPartitionsByDefault", Sdf.ValueTypeNames.Bool
            ).Set(True)
    before = stage.GetSessionLayer().ExportToString()
    with pytest.raises(RuntimeError, match="Stronger source opinion"):
        with partition_visual_meshes(stage, [head]):
            pytest.fail("A stronger source opinion must not silently defeat the exclusion")
    assert stage.GetSessionLayer().ExportToString() == before
