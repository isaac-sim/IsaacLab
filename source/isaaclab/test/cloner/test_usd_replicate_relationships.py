# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relationship-path contracts for heterogeneous USD replication."""

import ast
from pathlib import Path

import numpy as np

from pxr import Sdf, Usd

from isaaclab.cloner import usd_replicate


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
