# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collision-filter validation and articulation-root edge cases with pure USD."""

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.assets import ArticulationCfg, Asset
from isaaclab.sim import use_stage

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


def _author_robot(stage, path="/Robot", root="palm"):
    UsdGeom.Xform.Define(stage, path)
    for name in ("palm", "thumb", "tip"):
        body = UsdGeom.Xform.Define(stage, f"{path}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{path}/root_joint")
    fixed.CreateBody1Rel().SetTargets([f"{path}/palm"])
    for left, right in (("palm", "thumb"), ("thumb", "tip")):
        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{path}/{left}_{right}")
        joint.CreateBody0Rel().SetTargets([f"{path}/{left}"])
        joint.CreateBody1Rel().SetTargets([f"{path}/{right}"])
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(f"{path}/{root}"))


def _cfg(**kwargs):
    return ArticulationCfg(prim_path="/Robot", actuators={}, self_collision_filter_pairs=[("palm", "tip")], **kwargs)


@pytest.mark.parametrize(
    "root,asset_path",
    [("palm", "/Robot"), ("palm", "/Robot/palm"), ("root_joint", "/Robot/root_joint")],
)
def test_body_and_joint_roots_include_sibling_links(root, asset_path):
    """Articulation membership follows joints, including links outside the root prim's subtree."""
    stage = Usd.Stage.CreateInMemory()
    _author_robot(stage, root=root)
    cfg = _cfg()
    cfg.prim_path = asset_path
    with use_stage(stage):
        Asset(cfg)
    palm = stage.GetPrimAtPath("/Robot/palm")
    tip = stage.GetPrimAtPath("/Robot/tip")
    assert (
        tip.GetPath() in UsdPhysics.FilteredPairsAPI(palm).GetFilteredPairsRel().GetTargets()
        or palm.GetPath() in UsdPhysics.FilteredPairsAPI(tip).GetFilteredPairsRel().GetTargets()
    )


@pytest.mark.parametrize("selector", ["missing", "[", []], ids=["unmatched", "invalid-regex", "empty-group"])
def test_invalid_selectors_do_not_partially_author_filters(selector):
    """A bad later entry must not leave a partially applied filter configuration."""
    stage = Usd.Stage.CreateInMemory()
    _author_robot(stage)
    cfg = _cfg()
    cfg.self_collision_filter_pairs.append((selector, "palm"))
    with use_stage(stage), pytest.raises(ValueError):
        Asset(cfg)
    assert not stage.GetPrimAtPath("/Robot/palm").HasAPI(UsdPhysics.FilteredPairsAPI)
    assert not stage.GetPrimAtPath("/Robot/tip").HasAPI(UsdPhysics.FilteredPairsAPI)


def test_ambiguous_articulations_require_an_explicit_root():
    """Reject an asset containing multiple articulations without an explicit selection."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Robot")
    _author_robot(stage, "/Robot/Hand")
    _author_robot(stage, "/Robot/Other")
    with use_stage(stage), pytest.raises(ValueError, match="articulation"):
        Asset(_cfg())


def test_instance_proxy_bodies_raise_before_authoring():
    """Filters on immutable body instances report an error instead of silently doing nothing."""
    stage = Usd.Stage.CreateInMemory()
    _author_robot(stage, "/Template")
    robot = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    robot.GetReferences().AddInternalReference("/Template")
    robot.SetInstanceable(True)
    with use_stage(stage), pytest.raises(ValueError, match="instance-proxy"):
        Asset(_cfg())
    assert not stage.GetPrimAtPath("/Template/palm").HasAPI(UsdPhysics.FilteredPairsAPI)
