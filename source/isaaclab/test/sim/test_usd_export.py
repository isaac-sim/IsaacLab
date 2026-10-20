# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Kit-less tests of how the USD export maps engine values and actuators to prims."""

from types import SimpleNamespace

import pytest
import torch

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab.sim.usd_export.physx_usd_exporter import PhysxUsdExporter


class _Exporter(PhysxUsdExporter):
    def _read_rigid_body(self, path):
        return {}


def _articulation(joint_names, body_names):
    """An articulation of one environment whose data names joints and bodies without their prim paths."""

    def rows(*shape):
        return SimpleNamespace(torch=torch.zeros(1, *shape))

    dofs, bodies = len(joint_names), len(body_names)
    data = SimpleNamespace(
        joint_pos_limits=rows(dofs, 2),
        **{
            name: rows(dofs)
            for name in (
                "joint_stiffness",
                "joint_damping",
                "joint_effort_limits",
                "joint_armature",
                "joint_vel_limits",
                "joint_friction_coeff",
                "joint_dynamic_friction_coeff",
                "joint_viscous_friction_coeff",
            )
        },
        body_mass=rows(bodies),
        body_com_pose_b=rows(bodies, 7),
        body_inertia=rows(bodies, 9),
    )
    cfg = SimpleNamespace(prim_path="/Robot")
    return SimpleNamespace(cfg=cfg, joint_names=joint_names, body_names=body_names, data=data)


def _scene(**assets):
    """A scene of one environment, before its first physics step."""
    assets = {"articulations": {}, "rigid_objects": {}, "rigid_object_collections": {}, **assets}
    return SimpleNamespace(num_envs=1, sim=SimpleNamespace(get_physics_step_count=lambda: 0), **assets)


@pytest.mark.parametrize(
    "joint_names,body_names,name",
    [([], ["Base", "Foot"], "Foot"), (["Ankle"], ["Base"], "Ankle")],
    ids=["link name matches two prims", "joint name matches no prim"],
)
def test_names_must_match_one_prim(joint_names, body_names, name):
    """A name that matches no prim or several prims fails the export instead of writing another prim's values."""
    stage = Usd.Stage.CreateInMemory()
    UsdPhysics.ArticulationRootAPI.Apply(UsdGeom.Xform.Define(stage, "/Robot").GetPrim())
    for link in ("/Robot/Base", "/Robot/Left/Foot", "/Robot/Right/Foot"):
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, link).GetPrim())
    scene = _scene(articulations={"robot": _articulation(joint_names, body_names)})

    with pytest.raises(RuntimeError, match=f"one prim named '{name}'"):
        _Exporter(scene).compute_diff(stage)


@pytest.mark.parametrize(
    "joint_type,converted",
    [(UsdPhysics.RevoluteJoint, True), (UsdPhysics.SphericalJoint, False)],
    ids=["revolute joint", "spherical joint"],
)
def test_pd_actuator_becomes_a_drive_of_single_axis_joints_only(joint_type, converted):
    """A PD actuator becomes a revolute or prismatic joint's drive; on other joints it stays an actuator prim."""
    stage = Usd.Stage.CreateInMemory()
    joint = joint_type.Define(stage, "/Robot/Joint").GetPrim()
    actuator = stage.DefinePrim("/Robot/Actuator", "NewtonActuator")
    actuator.AddAppliedSchema("NewtonPDControlAPI")
    actuator.CreateRelationship("newton:targets").SetTargets([joint.GetPath()])
    actuator.CreateAttribute("newton:kp", Sdf.ValueTypeNames.Float).Set(80.0)

    _Exporter(_scene())._write_pd_actuators_as_drives(stage)

    assert actuator.IsActive() != converted
    assert bool([schema for schema in joint.GetAppliedSchemas() if schema.startswith("PhysicsDriveAPI")]) == converted
