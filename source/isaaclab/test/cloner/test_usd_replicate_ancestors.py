# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ancestor authoring in :func:`~isaaclab.cloner.usd_replicate`."""

from types import SimpleNamespace

import numpy as np
import pytest

from pxr import Sdf, Usd

from isaaclab.assets import AssetBaseCfg
from isaaclab.cloner import UsdReplicateContext, make_clone_plan, usd_replicate
from isaaclab.sim import SpawnerCfg


def _make_stage_with_source(source_path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for prefix in Sdf.Path(source_path).GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    return stage


def test_usd_replicate_defines_nested_destination_ancestors():
    """Copied prims under a nested scope compose as defined prims, keeping ancestors a target env already defines."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    stage.DefinePrim("/World/envs/env_1", "Xform")
    for prefix in Sdf.Path("/World/envs/env_2/Groceries").GetPrefixes():
        stage.DefinePrim(prefix, "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=np.asarray([0, 1, 2], dtype=np.int64),
    )

    copied_scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    copied_prim = stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
    assert copied_scope.IsDefined(), "intermediate ancestor must compose as a defined prim"
    assert copied_prim.IsDefined(), "copied prim must compose as a defined prim"

    # An ancestor already defined in the target env is left untouched.
    existing_scope = stage.GetPrimAtPath("/World/envs/env_2/Groceries")
    assert existing_scope.IsDefined()
    assert existing_scope.GetTypeName() == "Xform"
    assert stage.GetPrimAtPath("/World/envs/env_2/Groceries/Object").IsDefined()


@pytest.mark.parametrize("independent_child", [False, True])
def test_context_clones_nested_declarations_parent_first(independent_child):
    """An attached child is copied once; an independently authored child overrides its parent."""
    stage = _make_stage_with_source("/Sources/Robot/Camera")
    stage.GetPrimAtPath("/Sources/Robot/Camera").CreateAttribute("marker", Sdf.ValueTypeNames.Int).Set(1)
    stage.DefinePrim("/Sources/Camera", "Camera").CreateAttribute("marker", Sdf.ValueTypeNames.Int).Set(2)
    child_source = "/Sources/Camera" if independent_child else "/Sources/Robot/Camera"
    cfgs = (
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot/Camera", spawn=SpawnerCfg(spawn_path=child_source)),
        AssetBaseCfg(prim_path="/World/envs/env_[^/]+/Robot", spawn=SpawnerCfg(spawn_path="/Sources/Robot")),
    )
    plan = make_clone_plan(cfgs, ((0, 1), (0,)), 2)
    UsdReplicateContext(SimpleNamespace(stage=stage)).replicate(plan, (0, 1))
    for world in range(2):
        camera = stage.GetPrimAtPath(f"/World/envs/env_{world}/Robot/Camera")
        assert camera.GetAttribute("marker").Get() == (2 if independent_child else 1)
