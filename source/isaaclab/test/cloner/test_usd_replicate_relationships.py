# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relationship-path contracts for heterogeneous USD replication."""

import torch

from pxr import Sdf, Usd

from isaaclab.cloner import usd_replicate


def test_asset_clone_rebases_sibling_material_binding_only() -> None:
    stage = Usd.Stage.CreateInMemory()
    layer = stage.GetRootLayer()
    source = Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/source")
    target = Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Robot/target")
    Sdf.CreatePrimInLayer(layer, "/World/envs/env_0/Materials/paint")

    binding = Sdf.RelationshipSpec(source, "material:binding", custom=False)
    binding.targetPathList.prependedItems = [Sdf.Path("/World/envs/env_0/Materials/paint")]
    unrelated = Sdf.RelationshipSpec(source, "control:target", custom=False)
    unrelated.targetPathList.prependedItems = [Sdf.Path("/World/envs/env_0/Materials/paint")]
    output = Sdf.AttributeSpec(source, "output", Sdf.ValueTypeNames.Float)
    input_attribute = Sdf.AttributeSpec(target, "input", Sdf.ValueTypeNames.Float)
    input_attribute.connectionPathList.explicitItems = [output.path]

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Robot"],
        destinations=["/World/envs/env_{}/Robot"],
        env_ids=torch.tensor([0, 2]),
    )

    cloned_binding = layer.GetRelationshipAtPath("/World/envs/env_2/Robot/source.material:binding")
    assert cloned_binding.targetPathList.prependedItems == [Sdf.Path("/World/envs/env_2/Materials/paint")]
    relationship = stage.GetRelationshipAtPath("/World/envs/env_2/Robot/source.material:binding")
    assert relationship.GetTargets() == [Sdf.Path("/World/envs/env_2/Materials/paint")]
    cloned_unrelated = layer.GetRelationshipAtPath("/World/envs/env_2/Robot/source.control:target")
    assert cloned_unrelated.targetPathList.prependedItems == [Sdf.Path("/World/envs/env_0/Materials/paint")]
    cloned_input = layer.GetAttributeAtPath("/World/envs/env_2/Robot/target.input")
    assert tuple(cloned_input.connectionPathList.explicitItems) == (Sdf.Path("/World/envs/env_2/Robot/source.output"),)
