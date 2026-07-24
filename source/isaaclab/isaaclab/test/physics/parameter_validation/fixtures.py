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
from isaaclab.sim.schemas import (
    CollisionBaseCfg,
    MassPropertiesCfg,
    RigidBodyBaseCfg,
    UsdPhysicsDriveCfg,
    activate_contact_sensors,
    apply_drive,
    define_collision_properties,
    define_mass_properties,
    define_rigid_body_properties,
)
from isaaclab.sim.spawners.materials import UsdPhysicsRigidBodyMaterialCfg
from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
from isaaclab.sim.utils import bind_physics_material

FREE_BODY_PRIM_PATH = "/World/Env_0/Object"
SINGLE_DOF_PRIM_PATH = "/World/Env_0/Robot"
CONTACT_GROUND_PRIM_PATH = "/World/Env_0/Ground"
CONTACT_OBJECT_PRIM_PATH = "/World/Env_0/Object"

JOINT_MASS = {"revolute": 1.0, "prismatic": 2.0}
JOINT_INERTIA = (0.1, 0.1, 0.05)
JOINT_EFFECTIVE_INERTIA = {"revolute": JOINT_INERTIA[2], "prismatic": JOINT_MASS["prismatic"]}
Q_REF = 0.2

PROBE_TARGET = 0.6
LIMIT_LOWER = -1.0
ACTIVE_UPPER = 0.3
INACTIVE_UPPER = 2.0
ACTIVE_LOWER = -0.3
INACTIVE_LOWER = -2.0

FREE_BODY_SIZE = 0.2
FREE_BODY_MASS = 2.0
FREE_BODY_INERTIA = (0.08, 0.12, 0.16)
FREE_BODY_COM = (0.0, 0.0, 0.0)

CONTACT_BOX_SIZE = (0.2, 0.2, 0.2)
CONTACT_SPHERE_RADIUS = 0.1
CONTACT_MASS = 1.0
_CONTACT_PHYSICS_MATERIAL_NAME = "physicsMaterial"
_MIN_MU = 1e-5

_RAD2DEG = 180.0 / math.pi


def make_contact_material(mu: float, restitution: float = 0.0) -> UsdPhysicsRigidBodyMaterialCfg:
    """Create a backend-neutral rigid contact material."""
    return UsdPhysicsRigidBodyMaterialCfg(
        static_friction=mu,
        dynamic_friction=mu,
        restitution=restitution,
    )


def make_contact_box_cfg(
    prim_path: str,
    *,
    size: tuple[float, float, float] = CONTACT_BOX_SIZE,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mu: float = 0.0,
    restitution: float = 0.0,
    mass: float = CONTACT_MASS,
    kinematic: bool = False,
    disable_gravity: bool = False,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
    spawn: bool = True,
) -> RigidObjectCfg:
    """Create a controlled cuboid contact fixture configuration."""
    spawn_cfg = None
    if spawn:
        spawn_cfg = sim_utils.CuboidCfg(
            size=size,
            rigid_props=RigidBodyBaseCfg(
                kinematic_enabled=kinematic,
                disable_gravity=disable_gravity,
            ),
            collision_props=CollisionBaseCfg(
                collision_enabled=True,
                rest_offset=rest_offset,
                contact_offset=contact_offset,
            ),
            mass_props=MassPropertiesCfg(mass=mass),
            physics_material=make_contact_material(mu, restitution),
            activate_contact_sensors=True,
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=orientation),
    )


def make_contact_sphere_cfg(
    prim_path: str,
    *,
    radius: float = CONTACT_SPHERE_RADIUS,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mu: float = 0.0,
    restitution: float = 0.0,
    disable_gravity: bool = False,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
    spawn: bool = True,
) -> RigidObjectCfg:
    """Create a controlled spherical contact fixture configuration."""
    spawn_cfg = None
    if spawn:
        spawn_cfg = sim_utils.SphereCfg(
            radius=radius,
            rigid_props=RigidBodyBaseCfg(
                disable_gravity=disable_gravity,
            ),
            collision_props=CollisionBaseCfg(
                collision_enabled=True,
                rest_offset=rest_offset,
                contact_offset=contact_offset,
            ),
            mass_props=MassPropertiesCfg(mass=CONTACT_MASS),
            physics_material=make_contact_material(mu, restitution),
            activate_contact_sensors=True,
        )
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=spawn_cfg,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
    )


def _scalar_first_to_xyzw(orientation: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a scalar-first quaternion to the ``(x, y, z, w)`` layout used by ``create_prim``."""
    return (orientation[1], orientation[2], orientation[3], orientation[0])


def _build_contact_shape_usd(
    prim_path: str,
    *,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mesh_prim_type: str,
    mesh_attributes: dict[str, float],
    mesh_scale: tuple[float, float, float] | None = None,
    mass: float = CONTACT_MASS,
    kinematic: bool = False,
    disable_gravity: bool = False,
    mu: float = 0.0,
    restitution: float = 0.0,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
) -> None:
    """Author one rigid contact shape on the current stage."""
    stage = sim_utils.get_current_stage()
    if stage.GetPrimAtPath(prim_path).IsValid():
        raise ValueError(f"A prim already exists at path: '{prim_path}'.")

    sim_utils.create_prim(
        prim_path,
        "Xform",
        translation=position,
        orientation=_scalar_first_to_xyzw(orientation),
        stage=stage,
    )
    geom_prim_path = f"{prim_path}/geometry"
    mesh_prim_path = f"{geom_prim_path}/mesh"
    sim_utils.create_prim(
        mesh_prim_path,
        mesh_prim_type,
        scale=mesh_scale,
        attributes=mesh_attributes,
        stage=stage,
    )

    define_collision_properties(
        mesh_prim_path,
        CollisionBaseCfg(
            collision_enabled=True,
            rest_offset=rest_offset,
            contact_offset=contact_offset,
        ),
        stage=stage,
    )
    material_path = f"{geom_prim_path}/{_CONTACT_PHYSICS_MATERIAL_NAME}"
    spawn_physics_material(material_path, make_contact_material(mu, restitution), stage=stage)
    bind_physics_material(mesh_prim_path, material_path, stage=stage)

    define_mass_properties(prim_path, MassPropertiesCfg(mass=mass), stage=stage)
    define_rigid_body_properties(
        prim_path,
        RigidBodyBaseCfg(
            kinematic_enabled=kinematic,
            disable_gravity=disable_gravity,
        ),
        stage=stage,
    )
    activate_contact_sensors(prim_path, stage=stage)


def build_contact_box_usd(
    prim_path: str,
    *,
    size: tuple[float, float, float] = CONTACT_BOX_SIZE,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    mu: float = 0.0,
    restitution: float = 0.0,
    mass: float = CONTACT_MASS,
    kinematic: bool = False,
    disable_gravity: bool = False,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
) -> None:
    """Author a controlled cuboid contact fixture on the current stage."""
    cube_size = min(size)
    mesh_scale = (size[0] / cube_size, size[1] / cube_size, size[2] / cube_size)
    _build_contact_shape_usd(
        prim_path,
        position=position,
        orientation=orientation,
        mesh_prim_type="Cube",
        mesh_attributes={"size": cube_size},
        mesh_scale=mesh_scale,
        mass=mass,
        kinematic=kinematic,
        disable_gravity=disable_gravity,
        mu=mu,
        restitution=restitution,
        rest_offset=rest_offset,
        contact_offset=contact_offset,
    )


def build_contact_sphere_usd(
    prim_path: str,
    *,
    radius: float = CONTACT_SPHERE_RADIUS,
    position: tuple[float, float, float] = (0.0, 0.0, 1.0),
    mu: float = 0.0,
    restitution: float = 0.0,
    disable_gravity: bool = False,
    rest_offset: float = 0.0,
    contact_offset: float = 0.01,
) -> None:
    """Author a controlled spherical contact fixture on the current stage."""
    _build_contact_shape_usd(
        prim_path,
        position=position,
        mesh_prim_type="Sphere",
        mesh_attributes={"radius": radius},
        disable_gravity=disable_gravity,
        mu=mu,
        restitution=restitution,
        rest_offset=rest_offset,
        contact_offset=contact_offset,
    )


def build_single_dof(
    joint_type: str,
    *,
    usd_stiffness: float,
    usd_drive_damping: float = 0.0,
    usd_armature: float = 0.0,
    usd_passive_damping: float = 0.0,
    usd_lower: float | None = None,
    usd_upper: float | None = None,
    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0),
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
    link_mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))

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
