# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ancestor authoring in :func:`~isaaclab.cloner.usd_replicate`."""

import numpy as np

from pxr import Sdf, Usd

from isaaclab.cloner import usd_replicate


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
