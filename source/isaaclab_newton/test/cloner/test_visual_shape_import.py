# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""The Newton cloner imports visual-only geometry only when it is drawn."""

import newton
import pytest
from isaaclab_newton.cloner.newton_clone_utils import build_source_builders
from newton import ShapeFlags

from pxr import Usd, UsdGeom, UsdPhysics


@pytest.mark.parametrize("load_visual_shapes", [True, False])
@pytest.mark.parametrize("dynamic", [True, False])
def test_visual_shapes_hide_only_their_own_colliders(load_visual_shapes, dynamic):
    """Nested colliders stay hidden beside their visuals, including disabled rigid bodies."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    path = "/World/Asset"
    root = UsdGeom.Xform.Define(stage, path)
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim()).CreateRigidBodyEnabledAttr(dynamic)
    collider = UsdGeom.Cube.Define(stage, path + "/Collisions/collision")
    UsdPhysics.CollisionAPI.Apply(collider.GetPrim())
    UsdGeom.Cube.Define(stage, path + "/Visuals/visual")

    builder = build_source_builders(stage, [path], newton.ModelBuilder, [], load_visual_shapes=load_visual_shapes)[path]
    flags = dict(zip(builder.shape_label, builder.shape_flags, strict=True))
    assert len(flags) == 1 + load_visual_shapes
    assert flags[path + "/Collisions/collision"] & ShapeFlags.COLLIDE_SHAPES
    assert bool(flags[path + "/Collisions/collision"] & ShapeFlags.VISIBLE) is not load_visual_shapes
    if load_visual_shapes:
        assert flags[path + "/Visuals/visual"] & ShapeFlags.VISIBLE
