# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ancestor authoring in :func:`~isaaclab.cloner.usd_replicate`."""

import torch

from pxr import Sdf, Usd

from isaaclab.cloner import usd_replicate


def _make_stage_with_source(source_path: str) -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    for prefix in Sdf.Path(source_path).GetPrefixes():
        stage.DefinePrim(prefix, "Xform")
    return stage


def test_usd_replicate_defines_nested_destination_ancestors():
    """Copied prims under a nested scope compose as defined prims in target envs."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    stage.DefinePrim("/World/envs/env_1", "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=torch.tensor([0, 1]),
    )

    copied_scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    copied_prim = stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object")
    assert copied_scope.IsDefined(), "intermediate ancestor must compose as a defined prim"
    assert copied_prim.IsDefined(), "copied prim must compose as a defined prim"


def test_usd_replicate_keeps_existing_ancestor_specs():
    """Ancestors already defined in the target env are left untouched."""
    stage = _make_stage_with_source("/World/envs/env_0/Groceries/Object")
    for prefix in Sdf.Path("/World/envs/env_1/Groceries").GetPrefixes():
        stage.DefinePrim(prefix, "Xform")

    usd_replicate(
        stage,
        sources=["/World/envs/env_0/Groceries/Object"],
        destinations=["/World/envs/env_{}/Groceries/Object"],
        env_ids=torch.tensor([0, 1]),
    )

    scope = stage.GetPrimAtPath("/World/envs/env_1/Groceries")
    assert scope.IsDefined()
    assert scope.GetTypeName() == "Xform"
    assert stage.GetPrimAtPath("/World/envs/env_1/Groceries/Object").IsDefined()
