# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural collision tests for the cable-routing YAM and fixture assets."""

import math
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    BOARD_SIZE,
    BOARD_THICKNESS,
    BOARD_USD_PATH,
    CABLE_CONTACT_FRICTION,
    CONTACT_DAMPING,
    CONTACT_STIFFNESS,
    PEG_RADIUS,
    PEG_SHAFT_RADIUS,
    ROUND_PEG_USD_PATH,
    YAM_USD_PATH,
)


def _open_yam_stage() -> Usd.Stage:
    """Open the packaged Robot Menagerie YAM entry layer."""
    stage = Usd.Stage.Open(str(Path(YAM_USD_PATH)))
    assert stage is not None
    return stage


def _open_stage(path: str) -> Usd.Stage:
    """Open one packaged USD and require successful composition."""
    stage = Usd.Stage.Open(str(Path(path)))
    assert stage is not None
    return stage


def _authored_api_schemas(prim: Usd.Prim) -> set[str]:
    """Return composed authored API tokens, including schemas not registered kitlessly."""
    schemas = prim.GetMetadata("apiSchemas")
    assert schemas is not None
    return set(schemas.GetAddedOrExplicitItems())


def test_manipulationnet_fixtures_use_visual_meshes_and_primitive_colliders() -> None:
    """The rendered board and peg retain their simple collision proxies."""
    stage = _open_stage(BOARD_USD_PATH)
    root = stage.GetDefaultPrim()
    assert root.GetPath() == Sdf.Path("/MNetBoard")
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
    assert UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1.0

    visual_meshes = [prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    assert {prim.GetName() for prim in visual_meshes} == {"UpperLeft", "UpperRight", "LowerLeft", "LowerRight"}
    assert all(not prim.HasAPI(UsdPhysics.CollisionAPI) for prim in visual_meshes)
    assert all(not prim.HasAPI(UsdPhysics.MeshCollisionAPI) for prim in visual_meshes)

    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 1
    collider = colliders[0]
    assert collider.GetPath() == Sdf.Path("/MNetBoard/Collisions/ActiveWorkspace")
    assert collider.IsA(UsdGeom.Cube)
    assert UsdGeom.Imageable(collider).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
    assert UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
    cube = UsdGeom.Cube(collider)
    assert cube.GetSizeAttr().Get() == 1.0
    collider_scale = UsdGeom.Xformable(collider).GetOrderedXformOps()[0].Get()
    assert all(
        math.isclose(actual, expected, abs_tol=1.0e-7)
        for actual, expected in zip(collider_scale, (*BOARD_SIZE, BOARD_THICKNESS))
    )
    # The F1 render mesh must not seal or replace its primitive spool proxy.
    stage = _open_stage(ROUND_PEG_USD_PATH)
    root = stage.GetDefaultPrim()
    assert root.GetPath() == Sdf.Path("/RoundPeg")
    assert root.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()

    visual = stage.GetPrimAtPath("/RoundPeg/Visuals/F1")
    assert visual.IsA(UsdGeom.Mesh)
    assert not visual.HasAPI(UsdPhysics.CollisionAPI)
    assert not visual.HasAPI(UsdPhysics.MeshCollisionAPI)

    colliders = {
        prim.GetName(): UsdGeom.Cylinder(prim) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    assert colliders.keys() == {"LowerFlange", "Shaft", "UpperFlange"}
    assert all(collider for collider in colliders.values())
    assert all(collider.GetAxisAttr().Get() == UsdGeom.Tokens.z for collider in colliders.values())
    assert all(
        UsdGeom.Imageable(collider.GetPrim()).GetPurposeAttr().Get() == UsdGeom.Tokens.guide
        for collider in colliders.values()
    )
    assert math.isclose(colliders["LowerFlange"].GetRadiusAttr().Get(), PEG_RADIUS)
    assert math.isclose(colliders["LowerFlange"].GetHeightAttr().Get(), 0.0025)
    assert math.isclose(colliders["Shaft"].GetRadiusAttr().Get(), PEG_SHAFT_RADIUS)
    assert math.isclose(colliders["Shaft"].GetHeightAttr().Get(), 0.0200)
    assert math.isclose(colliders["UpperFlange"].GetRadiusAttr().Get(), PEG_RADIUS)
    assert math.isclose(colliders["UpperFlange"].GetHeightAttr().Get(), 0.0010)


def test_yam_menagerie_asset_uses_native_newton_primitive_colliders() -> None:
    """Test the official asset supplies complete primitive collision geometry."""
    stage = _open_yam_stage()
    colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]

    assert len(colliders) == 37
    assert sum(prim.IsA(UsdGeom.Capsule) for prim in colliders) == 21
    assert sum(prim.IsA(UsdGeom.Cube) for prim in colliders) == 4
    assert sum(prim.IsA(UsdGeom.Sphere) for prim in colliders) == 12
    assert all("NewtonCollisionAPI" in _authored_api_schemas(prim) for prim in colliders)
    assert all(prim.GetAttribute("newton:contactGap").IsValid() for prim in colliders)
    assert all(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() for prim in colliders)
    assert all(not prim.IsA(UsdGeom.Mesh) for prim in colliders)

    finger_collider = next(prim for prim in colliders if "link_left_finger" in str(prim.GetPath()))
    arm_collider = next(
        prim
        for prim in colliders
        if "link_left_finger" not in str(prim.GetPath()) and "link_right_finger" not in str(prim.GetPath())
    )
    finger_material, _ = UsdShade.MaterialBindingAPI(finger_collider).ComputeBoundMaterial(materialPurpose="physics")
    arm_material, _ = UsdShade.MaterialBindingAPI(arm_collider).ComputeBoundMaterial(materialPurpose="physics")
    assert finger_material.GetPrim().GetAttribute("physics:dynamicFriction").Get() == CABLE_CONTACT_FRICTION
    assert arm_material.GetPrim().GetAttribute("physics:dynamicFriction").Get() == 5.0
    for material in (finger_material, arm_material):
        assert material.GetPrim().GetAttribute("newton:contactStiffness").Get() == CONTACT_STIFFNESS
        assert math.isclose(
            material.GetPrim().GetAttribute("newton:contactDamping").Get(), CONTACT_DAMPING, rel_tol=1.0e-6
        )

    # Each caging fingertip keeps a detailed open geometry instead of being filled by
    # a visual-mesh convex hull: three capsules, two boxes, and six small contact beads.
    for leaf_name in ("lf_down", "rf_down"):
        leaf = next(prim for prim in stage.Traverse() if prim.GetName() == leaf_name)
        leaf_colliders = [prim for prim in Usd.PrimRange(leaf) if prim != leaf and prim.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(leaf_colliders) == 11
        assert sum(prim.IsA(UsdGeom.Capsule) for prim in leaf_colliders) == 3
        assert sum(prim.IsA(UsdGeom.Cube) for prim in leaf_colliders) == 2
        assert sum(prim.IsA(UsdGeom.Sphere) for prim in leaf_colliders) == 6


def test_yam_menagerie_asset_mirrors_the_parallel_gripper_for_newton() -> None:
    """Test Newton receives the signed finger mimic relation and physical limits."""
    stage = _open_yam_stage()
    left_path = Sdf.Path(
        "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/link_left_finger/left_finger"
    )
    right_path = Sdf.Path(
        "/i2rt_yam/Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6/link_right_finger/right_finger"
    )
    left = stage.GetPrimAtPath(left_path)
    right = stage.GetPrimAtPath(right_path)

    assert left.IsA(UsdPhysics.PrismaticJoint)
    assert right.IsA(UsdPhysics.PrismaticJoint)
    assert "NewtonMimicAPI" in _authored_api_schemas(left)
    assert "MjcEqualityJointAPI" in _authored_api_schemas(left)
    assert left.GetRelationship("newton:mimicJoint").GetTargets() == [right_path]
    assert left.GetAttribute("newton:mimicCoef1").Get() == -1.0
    assert left.GetAttribute("mjc:coef1").Get() == -1.0

    assert math.isclose(left.GetAttribute("physics:lowerLimit").Get(), -0.00205, abs_tol=1.0e-9)
    assert math.isclose(left.GetAttribute("physics:upperLimit").Get(), 0.037524, abs_tol=1.0e-9)
    assert math.isclose(right.GetAttribute("physics:lowerLimit").Get(), -0.037524, abs_tol=1.0e-9)
    assert math.isclose(right.GetAttribute("physics:upperLimit").Get(), 0.00205, abs_tol=1.0e-9)
