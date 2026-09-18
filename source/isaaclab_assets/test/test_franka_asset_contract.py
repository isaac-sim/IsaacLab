# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Contract tests for the canonical Franka Panda asset."""

from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

from pxr import Usd, UsdPhysics  # noqa: E402

from isaaclab_assets import (  # noqa: E402
    FRANKA_PANDA_CFG,
    FRANKA_PANDA_LEGACY_CFG,
    FRANKA_PANDA_MENAGERIE_CFG,
)

_GRIPPER_COLLIDERS = {"left_finger_pad", "right_finger_pad", "hand_capsule"}
_PRIMITIVE_ARM_COLLIDERS = {
    "link0_capsule",
    "link1_box",
    "link2_capsule",
    "link3_capsule",
    "link4_capsule",
    "link5_capsule_0",
    "link5_capsule_1",
    "link6_capsule",
    "link7_capsule_0",
    "link7_capsule_1",
}
_CONVEX_ARM_COLLIDERS = {
    "hand_c",
    "link0_c",
    "link1_c",
    "link2_c",
    "link3_c",
    "link4_c",
    "link5_c0",
    "link5_c1",
    "link5_c2",
    "link6_c",
    "link7_c",
}


def _enabled_colliders(stage: Usd.Stage) -> set[str]:
    return {
        prim.GetName()
        for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
    }


def test_franka_configs_select_the_canonical_and_legacy_assets() -> None:
    """The public config uses the flat asset while the old asset remains an explicit escape hatch."""
    assert FRANKA_PANDA_CFG.spawn.usd_path.endswith("/Robots/FrankaEmika/franka_panda.usda")
    assert FRANKA_PANDA_CFG.spawn.variants == {"Physics": "physx", "Colliders": "gripper_only"}
    assert FRANKA_PANDA_LEGACY_CFG.spawn.usd_path.endswith("/Robots/FrankaEmika/Legacy/panda_instanceable.usd")
    assert FRANKA_PANDA_MENAGERIE_CFG.to_dict() == FRANKA_PANDA_CFG.to_dict()


def test_franka_flat_asset_collider_and_visual_contract() -> None:
    """The canonical asset exposes stable collider choices without nested visual instances."""
    stage = Usd.Stage.Open(FRANKA_PANDA_CFG.spawn.usd_path)
    assert stage is not None
    robot = stage.GetDefaultPrim()
    variants = robot.GetVariantSets()
    assert set(variants.GetVariantSet("Colliders").GetVariantNames()) >= {
        "gripper_only",
        "primitives",
        "convex_hulls",
    }
    assert set(variants.GetVariantSet("Physics").GetVariantNames()) >= {"physx", "mujoco"}

    expected_colliders = {
        "gripper_only": _GRIPPER_COLLIDERS,
        "primitives": _GRIPPER_COLLIDERS | _PRIMITIVE_ARM_COLLIDERS,
        "convex_hulls": _GRIPPER_COLLIDERS | _CONVEX_ARM_COLLIDERS,
    }
    for physics_variant in ("physx", "mujoco"):
        variants.SetSelection("Physics", physics_variant)
        finger_joints = {
            prim.GetName(): prim
            for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            if prim.GetName() in {"panda_finger_joint1", "panda_finger_joint2"}
        }
        assert set(finger_joints) == {"panda_finger_joint1", "panda_finger_joint2"}
        leader_schemas = set(finger_joints["panda_finger_joint1"].GetAppliedSchemas())
        follower_schemas = set(finger_joints["panda_finger_joint2"].GetAppliedSchemas())
        if physics_variant == "physx":
            assert "PhysxMimicJointAPI:linear" in follower_schemas
        else:
            assert "MjcEqualityJointAPI" in leader_schemas
            assert "PhysxMimicJointAPI:linear" not in follower_schemas
        for collider_variant, expected in expected_colliders.items():
            variants.SetSelection("Colliders", collider_variant)
            assert _enabled_colliders(stage) == expected
            assert all(
                prim.GetName() != "physx_visuals" for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
            )
