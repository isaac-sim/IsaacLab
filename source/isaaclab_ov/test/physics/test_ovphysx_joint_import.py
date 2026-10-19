# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Native proof that legacy USD joint settings survive OVPhysX stage import."""

import math
from contextlib import closing

import numpy as np
import pytest

pytest.importorskip("ovphysx")

from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_ov.physics.ovphysx_compat import OVPHYSX_USES_TASKS
from isaaclab_ov.sim.views import OvPhysxView

from pxr import Gf, Sdf, UsdGeom, UsdPhysics

from isaaclab.sim import SimulationCfg, build_simulation_context


def _build_articulation(stage, root, body_names, body_mass):
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, root).GetPrim())
    for name in body_names:
        body = UsdGeom.Cube.Define(stage, f"{root}/{name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        mass = UsdPhysics.MassAPI.Apply(body)
        mass.CreateMassAttr(body_mass)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1))
    fixed = UsdPhysics.FixedJoint.Define(stage, f"{root}/Anchor")
    fixed.CreateBody1Rel().SetTargets([f"{root}/Base"])


@pytest.mark.integration
def test_legacy_joint_properties_reach_native_bindings():
    """Read solver properties before any Lab actuator can overwrite imported defaults."""
    replacements = {"Angular": (0.12, 0.03, 0.01), "Linear": (0.2, 0.1, 0.01)}
    cfg = SimulationCfg(
        physics=OvPhysxCfg(legacy_joint_friction_replacements=replacements),
        device="cpu",
        gravity=(0.0, 0.0, 0.0),
    )
    with build_simulation_context(device="cpu", sim_cfg=cfg, auto_add_lighting=False) as sim:
        stage = sim.stage
        _build_articulation(stage, "/World/Robot", ("Base", "Arm", "Slider"), 1.0)
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
            prim.CreateAttribute("physxJoint:jointFriction", Sdf.ValueTypeNames.Float).Set(0.5)
        sim.reset()
        with closing(
            OvPhysxView(sim.backend_manager.get_physx_instance(), prim_paths=["/World/Robot"], device="cpu")
        ) as view:
            names = view.binding_for("articulation_dof_max_velocity").dof_names
            assert set(names) == {"Angular", "Linear"}
            velocity = view.get_attribute("articulation_dof_max_velocity").numpy()[0]
            armature = view.get_attribute("articulation_dof_armature").numpy()[0]
            np.testing.assert_allclose(velocity, [math.pi / 2 if name == "Angular" else 2.0 for name in names])
            np.testing.assert_allclose(armature, 0.25)
            if OVPHYSX_USES_TASKS:
                friction = view.get_attribute("articulation_dof_friction_properties").numpy()[0]
                np.testing.assert_allclose(friction, [replacements[name] for name in names], rtol=1e-5)


@pytest.mark.integration
def test_joint_friction_is_an_effort():
    """Dry friction is an effort, not a load-dependent coefficient, on both supported runtimes."""
    cfg = SimulationCfg(physics=OvPhysxCfg(), device="cpu", dt=0.001)
    with build_simulation_context(device="cpu", sim_cfg=cfg, auto_add_lighting=False) as sim:
        for index, body_mass in enumerate((1.0, 4.0)):
            root = f"/World/Robot{index}"
            _build_articulation(sim.stage, root, ("Base", "Slider"), body_mass)
            joint = UsdPhysics.PrismaticJoint.Define(sim.stage, f"{root}/Joint")
            joint.CreateAxisAttr("X")
            joint.CreateBody0Rel().SetTargets([f"{root}/Base"])
            joint.CreateBody1Rel().SetTargets([f"{root}/Slider"])
        sim.reset()
        with closing(
            OvPhysxView(sim.backend_manager.get_physx_instance(), prim_paths=["/World/Robot*"], device="cpu")
        ) as view:
            view.set_attribute(
                "articulation_dof_friction_properties",
                np.array([[[0.2, 0.2, 0.0]], [[0.2, 0.2, 0.0]]], dtype=np.float32),
            )
            view.set_attribute("articulation_dof_velocity", np.ones((2, 1), dtype=np.float32))
            for _ in range(100):
                sim.step()
            velocity = view.get_attribute("articulation_dof_velocity").numpy()[:, 0]
            # F = m*a: quadrupling mass quarters deceleration.
            np.testing.assert_allclose((1.0 - velocity) / 0.1, [0.2, 0.05], atol=0.002)
