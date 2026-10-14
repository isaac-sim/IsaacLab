# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests of how the PhysX-dialect USD export groups bodies into articulations."""

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.sim.usd_export.physx_usd_exporter import physics_prims


@pytest.mark.parametrize("root", ["/Robot", "/Robot/Pelvis", "/Robot/RootJoint"])
def test_articulation_links_are_not_free_bodies(root):
    """Links belong to the articulation whether its root is on the robot prim, a link, or the root joint."""
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/Robot")
    for link in ("/Robot/Pelvis", "/Robot/Torso", "/Box"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, link).GetPrim())
    UsdPhysics.FixedJoint.Define(stage, "/Robot/RootJoint").CreateBody1Rel().SetTargets(["/Robot/Pelvis"])
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(root))

    assert physics_prims(stage) == ([root], ["/Box"])
