# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for authoring the world fixed joint with pure USD.

These tests run on an in-memory USD stage and intentionally do NOT launch Isaac Sim /
Kit. :func:`create_world_fixed_joint` is the single fixed-root-link authoring path for
every backend, so passing here also covers kitless backends (e.g. Newton) where
``omni.physx`` is unavailable.
"""

import math

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.schemas.schemas import create_world_fixed_joint


def _make_root_prim(stage: Usd.Stage, path: str, translation: tuple[float, float, float]) -> Usd.Prim:
    """Create an xform articulation-root prim with a rigid body and a world translation."""
    xform = UsdGeom.Xform.Define(stage, path)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
    prim = xform.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(prim)
    UsdPhysics.RigidBodyAPI.Apply(prim)
    return prim


def _find_fixed_joint(stage: Usd.Stage) -> UsdPhysics.FixedJoint | None:
    """Return the first ``UsdPhysics.FixedJoint`` authored on the stage."""
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.FixedJoint):
            return UsdPhysics.FixedJoint(prim)
    return None


def test_create_world_fixed_joint_authors_world_anchored_joint():
    """Authoring must materialize a ``UsdPhysics.FixedJoint`` anchored to the world.

    A world-attached fixed joint targets only ``body1`` (the root link), leaving ``body0``
    empty, and anchors ``localPos0`` at the root link's world translation.
    """
    stage = Usd.Stage.CreateInMemory()
    prim = _make_root_prim(stage, "/World/Robot", translation=(1.0, 2.0, 3.0))

    create_world_fixed_joint(prim, stage)

    joint = _find_fixed_joint(stage)
    assert joint is not None, "no UsdPhysics.FixedJoint was authored"
    # world-attached joint: body0 empty, body1 -> root link
    assert joint.GetBody0Rel().GetTargets() == []
    assert joint.GetBody1Rel().GetTargets() == [prim.GetPath()]
    # localPos0 anchors the joint at the root link's world pose
    local_pos0 = joint.GetLocalPos0Attr().Get()
    assert math.isclose(local_pos0[0], 1.0) and math.isclose(local_pos0[1], 2.0) and math.isclose(local_pos0[2], 3.0)
    # the joint must be effectively unbreakable
    assert joint.GetBreakForceAttr().Get() > 1e37
    assert joint.GetBreakTorqueAttr().Get() > 1e37


def test_create_world_fixed_joint_skips_instanceable_root():
    """The joint must be authored on a writable ancestor when the root prim is instanceable.

    Instanceable/instance-proxy/prototype prims are not authorable, so the joint prim must
    be created under the first writable ancestor rather than under the instanceable root.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    prim = _make_root_prim(stage, "/World/Robot", translation=(0.0, 0.0, 0.0))
    prim.SetInstanceable(True)
    assert prim.IsInstanceable()

    create_world_fixed_joint(prim, stage)

    joint = _find_fixed_joint(stage)
    assert joint is not None, "no UsdPhysics.FixedJoint was authored"
    # the joint prim must not be parented under the instanceable root
    joint_path = joint.GetPrim().GetPath().pathString
    assert not joint_path.startswith("/World/Robot/"), f"joint authored under instanceable root: {joint_path}"
    # it still targets the root link
    assert joint.GetBody1Rel().GetTargets() == [prim.GetPath()]
