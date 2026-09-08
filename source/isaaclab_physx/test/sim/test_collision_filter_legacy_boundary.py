# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression for the legacy collision-authoring boundary around compiled PhysX policy."""

import numpy as np
import pytest
from isaaclab_physx.physics.collision_filter import apply_collision_filter

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import ClonePlan, filter_collisions
from isaaclab.physics import CollisionFilterCfg, CollisionGroupCfg, PhysicsManager


def test_late_legacy_groups_cannot_overlap_compiled_profiles(monkeypatch) -> None:
    """The assembly barrier protects exclusive generated PhysX memberships."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env_id in (0, 1):
        for asset in ("Robot", "Support"):
            path = f"/World/envs/env_{env_id}/{asset}/shape"
            UsdGeom.Cube.Define(stage, path)
            UsdPhysics.CollisionAPI.Apply(stage.GetPrimAtPath(path))
    plan = ClonePlan(
        sources=("/World/envs/env_0",),
        destinations=("/World/envs/env_{}",),
        clone_mask=np.ones((1, 2), dtype=np.bool_),
        env_ids=np.asarray([0, 1]),
    )
    cfg = CollisionFilterCfg(
        groups={
            "robot": CollisionGroupCfg(
                prim_path_exprs=(r"{ENV_REGEX_NS}/Robot/shape",),
                filtered_groups=("support",),
            ),
            "support": CollisionGroupCfg(prim_path_exprs=(r"{ENV_REGEX_NS}/Support/shape",)),
        }
    )
    apply_collision_filter(stage, "/physicsScene", plan, cfg, isolate_environments=True, replicate_physics=True)
    root_before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", True)

    with pytest.raises(RuntimeError, match=r"cannot run after PhysicsManager\.apply_collision_filter"):
        filter_collisions(
            stage,
            "/physicsScene",
            "/World/lateLegacyCollisionGroups",
            ["/World/envs/env_0", "/World/envs/env_1"],
        )

    assert stage.GetRootLayer().ExportToString() == root_before
    assert not stage.GetPrimAtPath("/World/lateLegacyCollisionGroups").IsValid()
