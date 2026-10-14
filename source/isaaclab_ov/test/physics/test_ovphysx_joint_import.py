# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native proof that legacy USD joint settings survive OVPhysX stage import and replacement."""

import math
from contextlib import closing

import numpy as np
import pytest

pytest.importorskip("ovphysx")

from isaaclab_ov.physics import OvPhysxCfg, OvPhysxManager
from isaaclab_ov.sim.views import OvPhysxView

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaaclab.sim import SimulationCfg, build_simulation_context


@pytest.mark.integration
def test_legacy_joint_units_and_armature_survive_stage_replacement():
    """Read solver properties before any Lab actuator can overwrite imported defaults."""
    cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", gravity=(0.0, 0.0, 0.0))
    with build_simulation_context(device="cpu", sim_cfg=cfg, auto_add_lighting=False) as sim:
        stage = sim.stage
        root = UsdGeom.Xform.Define(stage, "/World/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for name in ("Base", "Arm", "Slider"):
            body = UsdGeom.Cube.Define(stage, f"/World/Robot/{name}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(1.0)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Robot/Anchor")
        fixed.CreateBody1Rel().SetTargets(["/World/Robot/Base"])
        for name, joint_type, body0, body1, velocity in (
            ("Angular", UsdPhysics.RevoluteJoint, "Base", "Arm", 90.0),
            ("Linear", UsdPhysics.PrismaticJoint, "Arm", "Slider", 2.0),
        ):
            joint = joint_type.Define(stage, f"/World/Robot/{name}")
            joint.CreateBody0Rel().SetTargets([f"/World/Robot/{body0}"])
            joint.CreateBody1Rel().SetTargets([f"/World/Robot/{body1}"])
            prim = joint.GetPrim()
            prim.AddAppliedSchema("PhysxJointAPI")
            prim.CreateAttribute("physxJoint:maxJointVelocity", Sdf.ValueTypeNames.Float).Set(velocity)
            prim.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(0.25)
        for _ in range(2):
            sim.reset()
            with closing(
                OvPhysxView(OvPhysxManager.get_physx_instance(), prim_paths=["/World/Robot"], device="cpu")
            ) as view:
                names = view.binding_for("articulation_dof_max_velocity").dof_names
                assert set(names) == {"Angular", "Linear"}
                velocity = view.get_attribute("articulation_dof_max_velocity").numpy()[0]
                armature = view.get_attribute("articulation_dof_armature").numpy()[0]
                np.testing.assert_allclose(velocity, [math.pi / 2 if name == "Angular" else 2.0 for name in names])
                np.testing.assert_allclose(armature, 0.25)
            # Exercise real scene-data discovery and release its read sessions before teardown.
            data = OvPhysxManager._scene_data_backend
            assert set(data.transform_paths) == {f"/World/Robot/{name}" for name in ("Base", "Arm", "Slider")}
            sim.step()
            OvPhysxManager._warmup_done = False
        for name in ("Angular", "Linear"):
            attr = stage.GetPrimAtPath(f"/World/Robot/{name}").GetAttribute("newton:velocityLimit")
            assert not attr or not attr.HasAuthoredValueOpinion()
