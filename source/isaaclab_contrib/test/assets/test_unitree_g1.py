# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Composition and collision contracts for the contributed G1 robot asset."""

from pxr import Usd, UsdPhysics

from isaaclab_contrib.assets import UNITREE_G1_29DOF_BOX_FOOT_CFG


def test_g1_asset_composes_with_foot_colliders():
    """The public asset resolves its payloads and retains 29 actuated joints and both foot colliders."""
    cfg = UNITREE_G1_29DOF_BOX_FOOT_CFG
    stage = Usd.Stage.Open(cfg.spawn.usd_path)
    assert stage is not None
    assert not stage.GetCompositionErrors()
    assert stage.GetDefaultPrim().IsValid()
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    assert len(joints) == 29
    for side in ("left", "right"):
        foot = next(
            p
            for p in stage.Traverse()
            if p.GetName() == f"{side}_ankle_roll_link" and p.HasAPI(UsdPhysics.RigidBodyAPI)
        )
        colliders = [p for p in Usd.PrimRange(foot) if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert colliders, f"Missing {side} robot foot collider"
        assert all(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is not False for p in colliders)
