# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Controlled procedural scenes for simulation parameter-validation tests."""

import math

from pxr import Gf, Sdf, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.schemas import UsdPhysicsDriveCfg, apply_drive

FREE_BODY_PRIM_PATH = "/World/Env_0/Object"
SINGLE_DOF_PRIM_PATH = "/World/Env_0/Robot"

JOINT_MASS = {"revolute": 1.0, "prismatic": 2.0}
JOINT_INERTIA = (0.1, 0.1, 0.05)
JOINT_EFFECTIVE_INERTIA = {"revolute": JOINT_INERTIA[2], "prismatic": JOINT_MASS["prismatic"]}
Q_REF = 0.2

PROBE_TARGET = 0.45
LIMIT_LOWER = -1.0
ACTIVE_UPPER = 0.3
INACTIVE_UPPER = 2.0

FREE_BODY_SIZE = 0.2
FREE_BODY_MASS = 2.0
FREE_BODY_INERTIA = (0.08, 0.12, 0.16)
FREE_BODY_COM = (0.0, 0.0, 0.0)

_RAD2DEG = 180.0 / math.pi


def build_single_dof(
    joint_type: str,
    *,
    usd_stiffness: float,
    usd_drive_damping: float = 0.0,
    usd_armature: float = 0.0,
    usd_passive_damping: float = 0.0,
    usd_lower: float | None = None,
    usd_upper: float | None = None,
) -> None:
    """Author a fixed-base single-DOF articulation on the current stage."""
    base_path = f"{SINGLE_DOF_PRIM_PATH}/base"
    link_path = f"{SINGLE_DOF_PRIM_PATH}/link"
    stage = sim_utils.get_current_stage()

    sim_utils.create_prim("/World/Env_0", "Xform")
    sim_utils.create_prim(SINGLE_DOF_PRIM_PATH, "Xform")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(SINGLE_DOF_PRIM_PATH))

    sim_utils.create_prim(base_path, "Sphere", attributes={"radius": 0.05})
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(base_path))
    base_mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(base_path))
    base_mass.CreateMassAttr().Set(1.0)
    base_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(1.0e-4, 1.0e-4, 1.0e-4))
    UsdPhysics.FixedJoint.Define(stage, f"{base_path}/FixedJoint").CreateBody0Rel().SetTargets([base_path])

    sim_utils.create_prim(link_path, "Cube", attributes={"size": FREE_BODY_SIZE})
    UsdPhysics.RigidBodyAPI.Apply(stage.GetPrimAtPath(link_path))
    link_mass = UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(link_path))
    link_mass.CreateMassAttr().Set(JOINT_MASS[joint_type])
    link_mass.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*JOINT_INERTIA))
    link_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

    joint_path = f"{link_path}/joint"
    if joint_type == "revolute":
        joint = UsdPhysics.RevoluteJoint.Define(stage, joint_path)
        axis = "Z"
        limit_conversion = _RAD2DEG
        damping_conversion = 1.0 / _RAD2DEG
    elif joint_type == "prismatic":
        joint = UsdPhysics.PrismaticJoint.Define(stage, joint_path)
        axis = "X"
        limit_conversion = 1.0
        damping_conversion = 1.0
    else:
        raise ValueError(f"Unsupported joint type: {joint_type}")

    joint.CreateBody0Rel().SetTargets([base_path])
    joint.CreateBody1Rel().SetTargets([link_path])
    joint.CreateAxisAttr().Set(axis)
    if usd_lower is not None and usd_upper is not None:
        joint.CreateLowerLimitAttr().Set(float(usd_lower * limit_conversion))
        joint.CreateUpperLimitAttr().Set(float(usd_upper * limit_conversion))
    if usd_armature > 0.0:
        joint.GetPrim().CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(usd_armature))
    joint.GetPrim().CreateAttribute("newton:damping", Sdf.ValueTypeNames.Float).Set(
        float(usd_passive_damping * damping_conversion)
    )
    apply_drive(
        UsdPhysicsDriveCfg(
            drive_type="force",
            stiffness=float(usd_stiffness),
            damping=float(usd_drive_damping),
            max_force=1.0e9,
        ),
        joint_path,
        stage,
    )


def make_single_dof_cfg(
    stiffness: float | None,
    damping: float | None,
    armature: float | None,
    *,
    joint_position: float = 0.0,
    joint_velocity: float = 0.0,
) -> ArticulationCfg:
    """Create an articulation config over the procedural single-DOF prims."""
    return ArticulationCfg(
        prim_path="/World/Env_.*/Robot",
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={".*": joint_position},
            joint_vel={".*": joint_velocity},
        ),
        actuators={
            "joint": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=stiffness,
                damping=damping,
                armature=armature,
            )
        },
    )


def build_free_body_usd(
    *,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mass: float = FREE_BODY_MASS,
    inertia: tuple[float, float, float] = FREE_BODY_INERTIA,
    principal_axes: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    center_of_mass: tuple[float, float, float] = FREE_BODY_COM,
) -> None:
    """Author one collision-free rigid body with explicit state and inertial properties.

    ``angular_velocity`` is specified in [rad/s] and converted to the [deg/s]
    convention of :class:`pxr.UsdPhysics.RigidBodyAPI`.
    """
    stage = sim_utils.get_current_stage()
    if not stage.GetPrimAtPath("/World/Env_0").IsValid():
        sim_utils.create_prim("/World/Env_0", "Xform")
    sim_utils.create_prim(
        FREE_BODY_PRIM_PATH,
        "Cube",
        translation=position,
        orientation=orientation,
        attributes={"size": FREE_BODY_SIZE},
    )
    prim = stage.GetPrimAtPath(FREE_BODY_PRIM_PATH)

    rigid_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid_api.CreateVelocityAttr().Set(Gf.Vec3f(*linear_velocity))
    rigid_api.CreateAngularVelocityAttr().Set(Gf.Vec3f(*(value * _RAD2DEG for value in angular_velocity)))

    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(float(mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    mass_api.CreatePrincipalAxesAttr().Set(Gf.Quatf(principal_axes[0], Gf.Vec3f(*principal_axes[1:])))
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))


def make_free_body_cfg(
    *,
    spawn: sim_utils.SpawnerCfg | None = None,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> RigidObjectCfg:
    """Create a rigid-object config for the controlled free-body prim."""
    return RigidObjectCfg(
        prim_path="/World/Env_.*/Object",
        spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,
            rot=orientation,
            lin_vel=linear_velocity,
            ang_vel=angular_velocity,
        ),
    )
