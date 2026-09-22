# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for USD replication on an in-memory stage (no simulator runtime)."""

import ast
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom

from isaaclab.cloner import ClonePlan, UsdReplicateContext, _fabric_notices, grid_transforms, usd_replicate

_ENV_IDS = np.asarray([0, 1], dtype=np.int64)


def _stage_with(*paths: str) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for path in paths:
        for prefix in Sdf.Path(path).GetPrefixes():
            stage.DefinePrim(prefix, "Xform")
    return stage


def _translation(stage: Usd.Stage, path: str) -> tuple[float, float, float]:
    return tuple(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(0).ExtractTranslation())


def test_asset_clone_uses_native_relationship_path_mapping() -> None:
    stage = Usd.Stage.CreateInMemory()
    layer = stage.GetRootLayer()
    source = Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/source")
    target = Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/target")
    Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/paint")
    Sdf.CreatePrimInLayer(layer, "/World/global")

    binding = Sdf.RelationshipSpec(source, "material:binding", custom=False)
    binding.targetPathList.explicitItems = [Sdf.Path("/World/envs/env_0/Robot/paint")]
    unrelated = Sdf.RelationshipSpec(source, "control:target", custom=False)
    unrelated.targetPathList.prependedItems = [Sdf.Path("/World/global")]
    output = Sdf.AttributeSpec(source, "output", Sdf.ValueTypeNames.Float)
    input_attribute = Sdf.AttributeSpec(target, "input", Sdf.ValueTypeNames.Float)
    input_attribute.connectionPathList.explicitItems = [output.path]
    joint = Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/joint")
    body = Sdf.RelationshipSpec(joint, "physics:body1", custom=False)
    body.targetPathList.explicitItems = [Sdf.Path("/World/envs/env_0/Robot/target")]

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Robot"],
        destinations=["/World/envs/env_{}/Robot"],
        env_ids=np.asarray([0, 2], dtype=np.int64),
    )

    cloned_binding = layer.GetRelationshipAtPath("/World/envs/env_2/Robot/source.material:binding")
    assert cloned_binding.targetPathList.explicitItems == [Sdf.Path("/World/envs/env_2/Robot/paint")]
    relationship = stage.GetRelationshipAtPath("/World/envs/env_2/Robot/source.material:binding")
    assert relationship.GetTargets() == [Sdf.Path("/World/envs/env_2/Robot/paint")]
    cloned_unrelated = layer.GetRelationshipAtPath("/World/envs/env_2/Robot/source.control:target")
    assert cloned_unrelated.targetPathList.prependedItems == [Sdf.Path("/World/global")]
    cloned_input = layer.GetAttributeAtPath("/World/envs/env_2/Robot/target.input")
    assert tuple(cloned_input.connectionPathList.explicitItems) == (Sdf.Path("/World/envs/env_2/Robot/source.output"),)
    cloned_body = layer.GetRelationshipAtPath("/World/envs/env_2/Robot/joint.physics:body1")
    assert cloned_body.targetPathList.explicitItems == [Sdf.Path("/World/envs/env_2/Robot/target")]


def test_usd_replicate_keeps_native_copy_spec_path_semantics() -> None:
    module = Path(__file__).parents[2] / "isaaclab" / "cloner" / "usd.py"
    calls = [
        node
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "CopySpec"
    ]
    assert calls and all(len(call.args) == 4 and not call.keywords for call in calls)


@pytest.mark.parametrize("predefined_ancestor", [False, True])
def test_usd_replicate_defines_nested_destination_ancestors(predefined_ancestor):
    """Copied prims under a nested scope compose as defined prims; existing ancestors are left untouched."""
    stage = _stage_with("/World/envs/env_0/Groceries/Object")
    stage.DefinePrim("/World/envs/env_1/Groceries" if predefined_ancestor else "/World/envs/env_1", "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=_ENV_IDS,
    )

    scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    assert scope.IsDefined(), "intermediate ancestor must compose as a defined prim"
    assert scope.GetTypeName() == ("Xform" if predefined_ancestor else "")
    assert stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object").IsDefined()


def test_usd_replicate_with_mask_and_depth_order():
    """Rows reach only their masked envs, and a child listed before its parent is still authored below it."""
    stage = _stage_with("/World/template/Parent/Child", "/World/template/B", "/World/envs/env_2")
    mask = np.asarray([[True, False, True], [True, False, True], [False, True, False]])

    usd_replicate(
        stage,
        sources=["/World/template/Parent/Child", "/World/template/Parent", "/World/template/B"],
        destinations=["/World/envs/env_{}/Parent/Child", "/World/envs/env_{}/Parent", "/World/envs/env_{}/B"],
        env_ids=np.arange(3, dtype=np.int64),
        mask=mask,
    )

    for env in range(3):
        assert stage.GetPrimAtPath(f"/World/envs/env_{env}/Parent/Child").IsValid() == mask[0, env]
        assert stage.GetPrimAtPath(f"/World/envs/env_{env}/B").IsValid() == mask[2, env]


def test_usd_replicate_authors_positions_on_env_roots_only():
    """Grid positions land on env roots as translate ops; nested assets keep their local offset."""
    camera_offset = (0.57, -0.8, 0.5)
    positions, _ = grid_transforms(2, 3.0)
    stage = _stage_with("/World/envs/env_0")
    camera = UsdGeom.Camera.Define(stage, "/World/envs/env_0/Camera")
    camera.AddTranslateOp().Set(camera_offset)

    usd_replicate(stage, ["/World/envs/env_0"], ["/World/envs/env_{}"], _ENV_IDS, positions=positions)
    usd_replicate(stage, ["/World/envs/env_0/Camera"], ["/World/envs/env_{}/Camera"], _ENV_IDS, positions=positions)

    for env in range(2):
        env_prim = stage.GetPrimAtPath(f"/World/envs/env_{env}")
        assert tuple(env_prim.GetAttribute("xformOp:translate").Get()) == pytest.approx(positions[env].tolist())
        camera_prim = stage.GetPrimAtPath(f"/World/envs/env_{env}/Camera")
        assert tuple(camera_prim.GetAttribute("xformOp:translate").Get()) == pytest.approx(camera_offset)


def test_usd_replicate_context_consumes_plan():
    """UsdReplicateContext applies its routed rows, env ids, and positions from the shared plan."""
    stage = _stage_with("/World/template/A", "/World/envs")
    plan = ClonePlan(
        sources=("/World/template/A",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.asarray([[False, True]], dtype=np.bool_),
        env_ids=np.asarray([10, 20], dtype=np.int64),
        positions=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        context_rows={UsdReplicateContext: (0,)},
    )
    UsdReplicateContext(stage).replicate(plan)

    assert not stage.GetPrimAtPath("/World/envs/env_10").IsValid()
    assert _translation(stage, "/World/envs/env_20") == (4.0, 5.0, 6.0)


def test_usd_replicate_self_copy_skips_copy_spec():
    """usd_replicate must not call Sdf.CopySpec when source and destination paths are identical."""
    stage = _stage_with("/World/envs/env_0/Robot/base_link", "/World/envs/env_1")
    copy_calls: list[tuple[str, str]] = []
    real_copy_spec = Sdf.CopySpec

    def capturing_copy_spec(src_layer, src_path, dst_layer, dst_path, *args):
        copy_calls.append((str(src_path), str(dst_path)))
        return real_copy_spec(src_layer, src_path, dst_layer, dst_path, *args)

    with patch.object(Sdf, "CopySpec", capturing_copy_spec):
        usd_replicate(stage, ["/World/envs/env_0"], ["/World/envs/env_{}"], _ENV_IDS, mask=np.ones((1, 2), dtype=bool))

    assert copy_calls == [("/World/envs/env_0", "/World/envs/env_1")]


def test_disabled_fabric_change_notifies_noops_when_usdrt_unavailable(monkeypatch):
    """Fabric notice suspension no-ops when Carbonite bindings exist but ``usdrt`` does not."""
    import builtins

    class _FakeBindings:
        def validate_with(self, fabric_id: int) -> bool:
            raise AssertionError("missing usdrt should prevent fabric-id lookup")

    monkeypatch.setattr(_fabric_notices, "get_bindings", lambda: _FakeBindings())
    real_import = builtins.__import__

    def _import_without_usdrt(name, *args, **kwargs):
        if name == "usdrt":
            raise ModuleNotFoundError("No module named 'usdrt'", name="usdrt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_without_usdrt)

    with _fabric_notices.disabled_fabric_change_notifies(Usd.Stage.CreateInMemory()):
        pass
