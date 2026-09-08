# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Compatibility tests for legacy PhysX collision-group authoring."""

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.cloner import filter_collisions
from isaaclab.physics import PhysicsManager
from isaaclab.scene import InteractiveScene


def _stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.Scene.Define(stage, "/physicsScene")
    for env_id in (0, 1):
        UsdGeom.Xform.Define(stage, f"/World/envs/env_{env_id}")
    return stage


def test_legacy_authoring_remains_available_before_manager_barrier(monkeypatch) -> None:
    """Standalone USD cloning can still author legacy groups before managed assembly."""
    stage = _stage()
    monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", False)

    filter_collisions(
        stage,
        "/physicsScene",
        "/World/collisions",
        ["/World/envs/env_0", "/World/envs/env_1"],
    )

    assert stage.GetPrimAtPath("/World/collisions/group0").IsA(UsdPhysics.CollisionGroup)
    assert stage.GetPrimAtPath("/World/collisions/group1").IsA(UsdPhysics.CollisionGroup)
    assert stage.GetPrimAtPath("/physicsScene").GetAttribute("physxScene:invertCollisionGroupFilter").Get()


def test_legacy_authoring_rejects_post_barrier_mutation_atomically(monkeypatch) -> None:
    """A late legacy call cannot layer collision groups over manager-owned policy."""
    stage = _stage()
    root_before = stage.GetRootLayer().ExportToString()
    monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", True)

    with pytest.raises(RuntimeError, match=r"cannot run after PhysicsManager\.apply_collision_filter"):
        filter_collisions(
            stage,
            "/physicsScene",
            "/World/collisions",
            ["/World/envs/env_0", "/World/envs/env_1"],
        )

    assert stage.GetRootLayer().ExportToString() == root_before


def test_legacy_scene_filter_warns_pre_barrier_and_rejects_post_barrier(monkeypatch) -> None:
    """The deprecated scene API preserves its early return without bypassing the barrier guard."""
    scene = object.__new__(InteractiveScene)
    scene.stage = _stage()
    UsdGeom.Scope.Define(scene.stage, "/World/collisions")
    monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", False)

    with pytest.warns(DeprecationWarning, match=r"InteractiveScene\.filter_collisions\(\) is deprecated"):
        scene.filter_collisions()

    monkeypatch.setattr(PhysicsManager, "_collision_filter_applied", True)

    with pytest.raises(RuntimeError, match=r"cannot run after PhysicsManager\.apply_collision_filter"):
        scene.filter_collisions()
