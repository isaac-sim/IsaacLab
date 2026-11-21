# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the Unitree G1 OEM package corrections."""

import math

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_assets.robots.unitree_g1_oem import correct_g1_oem_stage


def test_correct_g1_oem_stage_makes_unitree_colliders_and_limits_standard():
    """Collision APIs authored on Xforms become gprim colliders, and the ankle limit matches the Unitree URDF."""
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/g1").GetPrim()
    stage.SetDefaultPrim(root)
    link = UsdGeom.Xform.Define(stage, "/g1/left_ankle_roll_link").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link)
    # Unitree's layout: the API sits on a `collisions` Xform above the geometry.
    meshes = UsdGeom.Xform.Define(stage, "/g1/left_ankle_roll_link/collisions").GetPrim()
    UsdPhysics.CollisionAPI.Apply(meshes)
    UsdPhysics.MeshCollisionAPI.Apply(meshes).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
    mesh = UsdGeom.Mesh.Define(stage, "/g1/left_ankle_roll_link/collisions/mesh")
    mesh.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(1, 0, 0), Gf.Vec3f(0, 1, 0), Gf.Vec3f(0, 0, 1)])
    mesh.CreateFaceVertexCountsAttr([3, 3, 3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3])
    spheres = UsdGeom.Xform.Define(stage, "/g1/left_ankle_roll_link/sole").GetPrim()
    UsdPhysics.CollisionAPI.Apply(spheres)
    UsdGeom.Sphere.Define(stage, "/g1/left_ankle_roll_link/sole/mesh_0/sphere").CreateRadiusAttr(0.005)
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/g1/joints/left_ankle_pitch_joint").GetPrim()
    UsdPhysics.DriveAPI.Apply(joint, "angular").CreateMaxForceAttr(50.0)
    joint.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(math.degrees(37.0))

    correct_g1_oem_stage(stage)

    shapes = UsdPhysics.UsdPhysicsLoadStageFromPrimRange(stage, ["/g1"])
    assert len(shapes[UsdPhysics.ObjectType.MeshShape][0]) == 1
    assert len(shapes[UsdPhysics.ObjectType.SphereShape][0]) == 1
    assert not meshes.HasAPI(UsdPhysics.CollisionAPI) and not spheres.HasAPI(UsdPhysics.CollisionAPI)
    assert UsdPhysics.MeshCollisionAPI(mesh.GetPrim()).GetApproximationAttr().Get() == UsdPhysics.Tokens.convexHull
    assert UsdPhysics.DriveAPI.Get(joint, "angular").GetMaxForceAttr().Get() == 35.0
    assert math.isclose(math.radians(joint.GetAttribute("physxJoint:maxJointVelocity").Get()), 30.0, rel_tol=1e-6)
