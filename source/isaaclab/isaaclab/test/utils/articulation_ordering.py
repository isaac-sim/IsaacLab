# Copyright (c) 2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assertions for comparing articulation traces across joint orderings."""

from typing import Any

import torch

ANYMAL_C_PHYSX_JOINT_NAMES = (
    "LF_HAA",
    "LH_HAA",
    "RF_HAA",
    "RH_HAA",
    "LF_HFE",
    "LH_HFE",
    "RF_HFE",
    "RH_HFE",
    "LF_KFE",
    "LH_KFE",
    "RF_KFE",
    "RH_KFE",
)

PANDA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)

PANDA_BODY_NAMES = (
    "panda_link0",
    "panda_link1",
    "panda_link2",
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "panda_hand",
    "panda_leftfinger",
    "panda_rightfinger",
)
PANDA_ROOT_PRESERVING_REVERSED_BODY_NAMES = (PANDA_BODY_NAMES[0], *reversed(PANDA_BODY_NAMES[1:]))

BRANCHING_PHYSX_JOINT_NAMES = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow")
BRANCHING_MJWARP_JOINT_NAMES = ("left_shoulder", "left_elbow", "right_shoulder", "right_elbow")
BRANCHING_PHYSX_BODY_NAMES = ("base", "left_upper", "right_upper", "left_tip", "right_tip")
BRANCHING_MJWARP_BODY_NAMES = ("base", "left_upper", "left_tip", "right_upper", "right_tip")

# PhysX order of :func:`author_unsorted_sibling_articulation`: breadth-first, siblings in authored joint-prim order.
UNSORTED_SIBLINGS_PHYSX_JOINT_NAMES = ("thumb_joint", "index_joint", "thumb_tip_joint", "index_tip_joint")
UNSORTED_SIBLINGS_PHYSX_BODY_NAMES = ("palm", "thumb", "index", "middle", "thumb_tip", "index_tip")


def author_unsorted_sibling_articulation(usd_path: str, floating_base: bool = False) -> None:
    """Author an articulation whose sibling joints are not authored in path order.

    The palm's children are joined, in authored joint-prim order, by ``thumb_joint`` (revolute),
    ``index_joint`` (prismatic) and ``middle_joint`` (fixed). The joint-prim order differs from
    the sorted-path order, from the body-prim order and from the grandchild joint order.
    ``thumb_tip_joint`` is authored under its parent body instead of the joints scope.

    Args:
        usd_path: File path the stage is saved to.
        floating_base: Whether to omit the fixed joint that attaches the palm to the world.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: PLC0415

    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    stage.SetDefaultPrim(robot.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())

    positions = {
        "palm": (0.0, 0.0, 1.0),
        "middle": (0.0, 0.2, 1.0),
        "index": (0.2, 0.0, 1.0),
        "thumb": (-0.2, 0.0, 1.0),
        "thumb_tip": (-0.4, 0.0, 1.0),
        "index_tip": (0.4, 0.0, 1.0),
    }
    for name, position in positions.items():
        body = UsdGeom.Xform.Define(stage, f"/Robot/{name}")
        body.AddTranslateOp().Set(Gf.Vec3d(*position))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(0.1)
        collider = UsdGeom.Sphere.Define(stage, f"/Robot/{name}/collision")
        collider.CreateRadiusAttr(0.05)
        UsdPhysics.CollisionAPI.Apply(collider.GetPrim())

    joints = [
        ("/Robot/joints/thumb_joint", UsdPhysics.RevoluteJoint, "palm", "thumb"),
        ("/Robot/joints/index_joint", UsdPhysics.PrismaticJoint, "palm", "index"),
        ("/Robot/joints/middle_joint", UsdPhysics.FixedJoint, "palm", "middle"),
        ("/Robot/joints/index_tip_joint", UsdPhysics.RevoluteJoint, "index", "index_tip"),
        ("/Robot/thumb/thumb_tip_joint", UsdPhysics.RevoluteJoint, "thumb", "thumb_tip"),
    ]
    if not floating_base:
        joints.append(("/Robot/joints/root_joint", UsdPhysics.FixedJoint, None, "palm"))
    for path, joint_type, parent, child in joints:
        joint = joint_type.Define(stage, path)
        child_position = Gf.Vec3f(*positions[child])
        if parent is None:
            joint.CreateLocalPos0Attr(child_position)
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        else:
            joint.CreateBody0Rel().SetTargets([Sdf.Path(f"/Robot/{parent}")])
            half_offset = (child_position - Gf.Vec3f(*positions[parent])) * 0.5
            joint.CreateLocalPos0Attr(half_offset)
            joint.CreateLocalPos1Attr(-half_offset)
        joint.CreateBody1Rel().SetTargets([Sdf.Path(f"/Robot/{child}")])
        if joint_type is not UsdPhysics.FixedJoint:
            joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    stage.Save()


_ORDERING_TRACE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "computed_effort",
    "applied_effort",
    "adapter_applied_effort",
)
_ORDERING_TRACE_TOLERANCES = {
    "joint_pos": (2e-3, 1e-3),
    "joint_vel": (1e-2, 1e-2),
    "computed_effort": (1e-3, 1e-3),
    "applied_effort": (1e-3, 1e-3),
    "adapter_applied_effort": (1e-3, 1e-3),
}


def assert_articulation_ordering_trace_matches(
    identity_result: dict[str, Any],
    reordered_result: dict[str, Any],
    requested_joint_names: tuple[str, ...],
) -> None:
    """Assert that two traces command and observe the same physical joints.

    Args:
        identity_result: Trace recorded with backend joint ordering.
        reordered_result: Trace recorded with an explicit public joint ordering.
        requested_joint_names: Explicit public joint-name ordering used for the reordered trace.
    """
    assert identity_result["joint_names"] == identity_result["backend_joint_names"]
    assert reordered_result["joint_names"] == requested_joint_names
    assert reordered_result["backend_joint_names"] == identity_result["backend_joint_names"]

    installed_ordering = reordered_result["joint_ordering"]
    assert installed_ordering is not None
    assert installed_ordering["user_names"] == requested_joint_names
    assert installed_ordering["backend_names"] == identity_result["backend_joint_names"]
    expected_user_to_backend = tuple(
        identity_result["backend_joint_names"].index(name) for name in requested_joint_names
    )
    expected_backend_to_user = tuple(
        requested_joint_names.index(name) for name in identity_result["backend_joint_names"]
    )
    assert installed_ordering["user_to_backend_indices"] == expected_user_to_backend
    assert installed_ordering["backend_to_user_indices"] == expected_backend_to_user

    canonical_joint_names = tuple(identity_result["backend_joint_names"])
    identity_result = _canonicalize_ordering_result(identity_result, canonical_joint_names)
    reordered_result = _canonicalize_ordering_result(reordered_result, canonical_joint_names)

    assert identity_result["joint_names"] == reordered_result["joint_names"] == canonical_joint_names
    for field_name in ("target_pos", "target_vel", "effort_target"):
        identity_values = identity_result[field_name]
        reordered_values = reordered_result[field_name]
        if identity_values is None or reordered_values is None:
            assert identity_values is reordered_values
            continue
        torch.testing.assert_close(
            identity_values,
            reordered_values,
            rtol=0.0,
            atol=0.0,
            msg=f"{field_name} does not request the same physical command",
        )

    for field_name in _ORDERING_TRACE_FIELDS:
        atol, rtol = _ORDERING_TRACE_TOLERANCES[field_name]
        for step_index, (identity_values, reordered_values) in enumerate(
            zip(identity_result[field_name], reordered_result[field_name], strict=True)
        ):
            torch.testing.assert_close(
                identity_values,
                reordered_values,
                atol=atol,
                rtol=rtol,
                msg=f"{field_name} diverged at step {step_index}",
            )


def _canonicalize_ordering_result(result: dict[str, Any], canonical_joint_names: tuple[str, ...]) -> dict[str, Any]:
    """Gather public and adapter traces into one physical joint-name order."""
    canonical_result = dict(result)
    for field_name in _ORDERING_TRACE_FIELDS:
        source_names = result["adapter_joint_names"] if field_name.startswith("adapter_") else result["joint_names"]
        source_indices = tuple(source_names.index(name) for name in canonical_joint_names)
        canonical_result[field_name] = [
            values.index_select(
                1,
                torch.tensor(source_indices, dtype=torch.long, device=values.device),
            )
            for values in result[field_name]
        ]

    public_indices = tuple(result["joint_names"].index(name) for name in canonical_joint_names)
    for field_name in ("target_pos", "target_vel", "effort_target"):
        values = result[field_name]
        if values is not None:
            canonical_result[field_name] = values.index_select(
                1,
                torch.tensor(public_indices, dtype=torch.long, device=values.device),
            )

    canonical_result["joint_names"] = canonical_joint_names
    canonical_result["backend_joint_names"] = canonical_joint_names
    canonical_result["adapter_joint_names"] = canonical_joint_names
    return canonical_result
