# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD cloning preserves hierarchy, authored overrides, and relationship targets."""

from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Sdf, Usd, UsdGeom

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import UsdReplicateContext, make_clone_plan, usd_replicate
from isaaclab.sim import SpawnerCfg


def test_usd_replicate_preserves_ancestors_and_relationships():
    """Noncontiguous copies define missing ancestors and rebase only targets inside the copied subtree."""
    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/Groceries/Object"
    for prefix in Sdf.Path(root).GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    stage.DefinePrim("/World/envs/env_2", "Xform")
    for prefix in Sdf.Path("/World/envs/env_7/Groceries").GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    layer = stage.GetRootLayer()
    source = Sdf.CreatePrimInLayer(layer, root + "/source")
    target = Sdf.CreatePrimInLayer(layer, root + "/target")
    Sdf.CreatePrimInLayer(layer, root + "/paint")
    Sdf.CreatePrimInLayer(layer, "/World/global")
    binding = Sdf.RelationshipSpec(source, "material:binding", custom=False)
    binding.targetPathList.explicitItems = [Sdf.Path(root + "/paint")]
    unrelated = Sdf.RelationshipSpec(source, "control:target", custom=False)
    unrelated.targetPathList.prependedItems = [Sdf.Path("/World/global")]
    output = Sdf.AttributeSpec(source, "output", Sdf.ValueTypeNames.Float)
    input_attribute = Sdf.AttributeSpec(target, "input", Sdf.ValueTypeNames.Float)
    input_attribute.connectionPathList.explicitItems = [output.path]
    joint = Sdf.CreatePrimInLayer(layer, root + "/joint")
    body = Sdf.RelationshipSpec(joint, "physics:body1", custom=False)
    body.targetPathList.explicitItems = [Sdf.Path(root + "/target")]

    template = "/World/envs/env_{}/Groceries/Object"
    usd_replicate(stage, [root], [template], np.asarray([0, 2, 7], dtype=np.int64))

    for world in (0, 2, 7):
        destination = template.format(world)
        assert stage.GetPrimAtPath(destination).IsDefined()
        assert stage.GetPrimAtPath(destination).GetParent().IsDefined()
        binding_path = destination + "/source.material:binding"
        assert layer.GetRelationshipAtPath(binding_path).targetPathList.explicitItems == [destination + "/paint"]
        assert stage.GetRelationshipAtPath(binding_path).GetTargets() == [destination + "/paint"]
        unrelated = layer.GetRelationshipAtPath(destination + "/source.control:target")
        assert unrelated.targetPathList.prependedItems == ["/World/global"]
        connection = layer.GetAttributeAtPath(destination + "/target.input").connectionPathList
        assert connection.explicitItems == [destination + "/source.output"]
        body = layer.GetRelationshipAtPath(destination + "/joint.physics:body1")
        assert body.targetPathList.explicitItems == [destination + "/target"]
    assert stage.GetPrimAtPath("/World/envs/env_7/Groceries").GetTypeName() == "Xform"


@pytest.mark.parametrize("independent_child, routed_assets", [(False, (0, 1)), (True, (0, 1)), (False, (0,))])
def test_context_clones_nested_declarations_parent_first(independent_child, routed_assets):
    """Selected children, overrides, and existing world transforms survive parent-first cloning."""
    stage = Usd.Stage.CreateInMemory()
    for world in range(2):
        root = UsdGeom.Xform.Define(stage, f"/World/envs/env_{world}")
        if world == 0:
            root.AddTranslateOp().Set((10, 20, 30))
        root.AddRotateZOp().Set(90)
        root.AddScaleOp().Set((2, 3, 4))
        root.SetResetXformStack(True)
    stage.DefinePrim("/Sources/Robot/Camera", "Camera").CreateAttribute("marker", Sdf.ValueTypeNames.Int).Set(1)
    stage.DefinePrim("/Sources/Robot/Body", "Xform")
    stage.DefinePrim("/Sources/Camera", "Camera").CreateAttribute("marker", Sdf.ValueTypeNames.Int).Set(2)
    child_source = "/Sources/Camera" if independent_child else "/Sources/Robot/Camera"
    cfgs = (
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot/Camera", spawn=SpawnerCfg(spawn_path=child_source)),
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path="/Sources/Robot")),
    )
    positions = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    plan = make_clone_plan(cfgs, ((0, 1), (0,)), 2, positions=positions)
    UsdReplicateContext(SimpleNamespace(stage=stage)).replicate(plan, routed_assets)
    for world in range(2):
        assert bool(stage.GetPrimAtPath(f"/World/envs/env_{world}/Robot/Body")) == (world == 0 and 1 in routed_assets)
        camera = stage.GetPrimAtPath(f"/World/envs/env_{world}/Robot/Camera")
        assert camera.GetAttribute("marker").Get() == (2 if independent_child else 1)
        transform = UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(0)
        np.testing.assert_allclose(transform.Transform((1, 0, 0)), positions[world] + [0, 2, 0], atol=1e-6)
        assert UsdGeom.Xformable(stage.GetPrimAtPath(f"/World/envs/env_{world}")).GetResetXformStack()
