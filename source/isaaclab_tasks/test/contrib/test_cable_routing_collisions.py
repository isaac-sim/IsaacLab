# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Structural collision tests for the cable-routing YAM and fixture assets."""

import hashlib
import math
from pathlib import Path

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_tasks.contrib.cable_routing.cable_routing_env_cfg import (
    BOARD_SIZE,
    BOARD_THICKNESS,
    BOARD_USD_PATH,
    PEG_HEIGHT,
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


def test_manipulationnet_board_uses_visual_meshes_and_one_primitive_collider() -> None:
    """Test board triangles render while only the active-workspace box collides."""
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

    visual_points = [point for prim in visual_meshes for point in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
    assert math.isclose(min(point[0] for point in visual_points), -0.18, abs_tol=1.0e-6)
    assert math.isclose(max(point[0] for point in visual_points), 0.15, abs_tol=1.0e-6)
    assert math.isclose(min(point[1] for point in visual_points), -0.20, abs_tol=1.0e-6)
    assert math.isclose(max(point[1] for point in visual_points), 0.20, abs_tol=1.0e-6)


def test_manipulationnet_round_peg_uses_a_primitive_spool_collider() -> None:
    """Test the F1 render mesh cannot seal or replace its primitive spool proxy."""
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

    points = UsdGeom.Mesh(visual).GetPointsAttr().Get()
    assert math.isclose(
        max(point[2] for point in points) - min(point[2] for point in points), PEG_HEIGHT, abs_tol=1.0e-7
    )


def test_manipulationnet_assets_record_pinned_source_provenance() -> None:
    """Test both object-form assets retain their exact Apache source revision."""
    expected_commit = "2745ccc6099fb3b65e89cbdbaf7af6521bf8dd29"
    expected_hashes = {
        "board.usdc": {
            "board_segment_bottom_left.stl": "1256f953cd5a9e18000f107310b265ed63b6a984252413c4be5a427f9a097585",
            "board_segment_bottom_right.stl": "6de8c5362d04f6a99a15b00f7655c5d706112103e9a5c8546f0e5306253be62c",
            "board_segment_upper_left.stl": "fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f",
            "board_segment_upper_right.stl": "fa90f8e015401c743b9dd967166023e66c14b8883d9808e0675a915072a9442f",
        },
        "round_peg.usdc": {"round_peg.stl": "29d8169aaf13374e7f3ebcbba5f85ef95592408498315686483a9c62b87230e7"},
    }
    expected_output_hashes = {
        "board.usdc": "4c8056e1826857dbfd04eee69d407b12ce2fccaf3acb703d138f04c09b2472ca",
        "round_peg.usdc": "aa25d088c1e4e339664e22f58f4998cef306c96bb7413786db802ac07ab79d7c",
    }
    for path in (Path(BOARD_USD_PATH), Path(ROUND_PEG_USD_PATH)):
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_output_hashes[path.name]
        stage = _open_stage(str(path))
        metadata = stage.GetRootLayer().customLayerData
        assert metadata["sourceRepository"] == "https://github.com/ManipulationNet/mnet_client"
        assert metadata["sourceBranch"] == "ros_2"
        assert metadata["sourceCommit"] == expected_commit
        assert metadata["sourceUnitsToMeters"] == 0.01
        assert metadata["sourceHashes"] == expected_hashes[path.name]
    assert (Path(BOARD_USD_PATH).parent / "LICENSE").is_file()


def test_yam_menagerie_asset_has_fixed_base_and_expected_articulation_topology() -> None:
    """Test the official fixed-base model exposes its canonical bodies and joints."""
    stage = _open_yam_stage()
    assert stage.GetDefaultPrim().GetPath() == Sdf.Path("/i2rt_yam")
    assert stage.GetDefaultPrim().GetVariantSets().GetNames() == []

    rigid_bodies = {prim.GetName(): prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
    assert set(rigid_bodies) == {
        "arm",
        "link_1",
        "link_2",
        "link_3",
        "link_4",
        "link_5",
        "link_6",
        "link_left_finger",
        "lf_rot",
        "lf_down",
        "link_right_finger",
        "rf_rot",
        "rf_down",
    }
    assert rigid_bodies["arm"].HasAPI(UsdPhysics.ArticulationRootAPI)

    movable_joints = {
        prim.GetName(): prim
        for prim in stage.Traverse()
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint)
    }
    assert set(movable_joints) == {
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "left_finger",
        "right_finger",
    }
    fixed_joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed_joints) == 5
    base_joint = stage.GetPrimAtPath("/i2rt_yam/Geometry/arm/PhysicsFixedJoint")
    assert base_joint.IsA(UsdPhysics.FixedJoint)
    assert UsdPhysics.Joint(base_joint).GetBody0Rel().GetTargets() == [Sdf.Path("/i2rt_yam")]
    assert UsdPhysics.Joint(base_joint).GetBody1Rel().GetTargets() == [Sdf.Path("/i2rt_yam/Geometry/arm")]


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
