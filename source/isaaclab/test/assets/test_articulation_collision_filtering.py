# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Public articulation configuration contracts for USD collision exclusions."""

import pytest

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.assets import ArticulationCfg, Asset
from isaaclab.sim import use_stage

pytestmark = pytest.mark.unit


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
    return ArticulationCfg(
        prim_path="/Robot", actuators={}, self_collision_filter_pairs=[(["thumb", "tip"], "palm")], **kwargs
    )


@pytest.mark.parametrize(
    "root,asset_path",
    [("palm", "/Robot"), ("root_joint", "/Robot"), ("palm", "/Robot/palm"), ("root_joint", "/Robot/root_joint")],
)
def test_body_and_joint_roots_include_sibling_links(root, asset_path):
    """Articulation membership follows joints, including links outside the root prim's subtree."""
    stage = Usd.Stage.CreateInMemory()
    _author_robot(stage, root=root)
    cfg = _cfg()
    cfg.prim_path = asset_path
    with use_stage(stage):
        Asset(cfg)
    palm = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath("/Robot/palm"))
    targets = set(palm.GetFilteredPairsRel().GetTargets())
    # Exclusions are symmetric, so either endpoint can own the authored relationship.
    for link in ("thumb", "tip"):
        link_targets = (
            UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(f"/Robot/{link}")).GetFilteredPairsRel().GetTargets()
        )
        assert Sdf.Path(f"/Robot/{link}") in targets or Sdf.Path("/Robot/palm") in link_targets


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
    assert not stage.GetPrimAtPath("/Robot/thumb").HasAPI(UsdPhysics.FilteredPairsAPI)


def test_explicit_articulation_root_isolates_overlapping_groups():
    """An explicit root selects one graph; same-group exclusions omit self-pairs and preserve targets."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Robot")
    _author_robot(stage, "/Robot/Hand")
    _author_robot(stage, "/Robot/Other")
    palm = stage.GetPrimAtPath("/Robot/Hand/palm")
    UsdPhysics.FilteredPairsAPI.Apply(palm).CreateFilteredPairsRel().AddTarget("/External/Body")
    cfg = _cfg(articulation_root_prim_path="/Hand/palm")
    cfg.self_collision_filter_pairs = [(["thumb", ".*"], ".*"), ("thumb", "palm")]
    with use_stage(stage):
        Asset(cfg)
        Asset(cfg)
    pairs = set()
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot/Hand")):
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            assert target != prim.GetPath()
            pairs.add(frozenset((str(prim.GetPath()), str(target))))
    assert pairs == {
        frozenset(("/Robot/Hand/palm", "/External/Body")),
        frozenset(("/Robot/Hand/palm", "/Robot/Hand/thumb")),
        frozenset(("/Robot/Hand/palm", "/Robot/Hand/tip")),
        frozenset(("/Robot/Hand/thumb", "/Robot/Hand/tip")),
    }
    assert all(
        not prim.HasAPI(UsdPhysics.FilteredPairsAPI) for prim in Usd.PrimRange(stage.GetPrimAtPath("/Robot/Other"))
    )


def test_multiple_articulations_require_an_explicit_root():
    """A broad asset path cannot silently select one of several independent articulations."""
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
