# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import contextlib
import math
from collections.abc import Sequence

import carb
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
import omni.kit
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.prims import GeometryPrim
from pxr import Gf, PhysxSchema, UsdPhysics, UsdShade


def create_ground_plane(
    prim_path: str,
    z_position: float = 0.0,
    static_friction: float = 1.0,
    dynamic_friction: float = 1.0,
    restitution: float = 0.0,
    color: Sequence[float] | None = (0.065, 0.0725, 0.080),
    **kwargs,
):
    """Spawns a ground plane into the scene.

    This method spawns the default ground plane (grid plane) from Isaac Sim into the scene.
    It applies a physics material to the ground plane and sets the color of the ground plane.

    Args:
        prim_path: The prim path to spawn the ground plane at.
        z_position: The z-location of the plane. Defaults to 0.
        static_friction: The static friction coefficient. Defaults to 1.0.
        dynamic_friction: The dynamic friction coefficient. Defaults to 1.0.
        restitution: The coefficient of restitution. Defaults to 0.0.
        color: The color of the ground plane.
            Defaults to (0.065, 0.0725, 0.080).

    Keyword Args:
        usd_path: The USD path to the ground plane. Defaults to the asset path
            `Isaac/Environments/Grid/default_environment.usd` on the Isaac Sim Nucleus server.
        improve_patch_friction: Whether to enable patch friction. Defaults to False.
        combine_mode: Determines the way physics materials will be combined during collisions.
            Available options are `average`, `min`, `multiply`, `multiply`, and `max`. Defaults to `average`.
        light_intensity: The power intensity of the light source. Defaults to 1e7.
        light_radius: The radius of the light source. Defaults to 50.0.
    """
    # Retrieve path to the plane
    if "usd_path" in kwargs:
        usd_path = kwargs["usd_path"]
    else:
        # get path to the nucleus server
        assets_root_path = nucleus_utils.get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Unable to access the Isaac Sim assets folder on Nucleus server.")
            return
        # prepend path to the grid plane
        usd_path = f"{assets_root_path}/Isaac/Environments/Grid/default_environment.usd"
    # Spawn Ground-plane
    prim_utils.create_prim(prim_path, usd_path=usd_path, translation=(0.0, 0.0, z_position))
    # Create physics material
    material = PhysicsMaterial(
        f"{prim_path}/groundMaterial",
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
    )
    # Apply PhysX Rigid Material schema
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material.prim)
    # Set patch friction property
    improve_patch_friction = kwargs.get("improve_patch_friction", False)
    physx_material_api.CreateImprovePatchFrictionAttr().Set(improve_patch_friction)
    # Set combination mode for coefficients
    combine_mode = kwargs.get("friciton_combine_mode", "multiply")
    physx_material_api.CreateFrictionCombineModeAttr().Set(combine_mode)
    physx_material_api.CreateRestitutionCombineModeAttr().Set(combine_mode)
    # Apply physics material to ground plane
    collision_prim_path = prim_utils.get_prim_path(
        prim_utils.get_first_matching_child_prim(
            prim_path, predicate=lambda x: prim_utils.get_prim_type_name(x) == "Plane"
        )
    )
    geom_prim = GeometryPrim(collision_prim_path, disable_stablization=False, collision=True)
    geom_prim.apply_physics_material(material)
    # Change the color of the plane
    # Warning: This is specific to the default grid plane asset.
    if color is not None:
        omni.kit.commands.execute(
            "ChangeProperty",
            prop_path=f"{prim_path}/Looks/theGrid.inputs:diffuse_tint",
            value=Gf.Vec3f(*color),
            prev=None,
        )
    # Add light source
    # By default, the one from Isaac Sim is too dim for large number of environments.
    # Warning: This is specific to the default grid plane asset.
    ambient_light = kwargs.get("ambient_light", True)
    if ambient_light:
        attributes = {"intensity": 600.0}
        attributes = {f"inputs:{k}": v for k, v in attributes.items()}
        # create light prim
        prim_utils.create_prim(f"{prim_path}/AmbientLight", "DistantLight", attributes=attributes)


def move_nested_prims(source_ns: str, target_ns: str):
    """Moves all prims from source namespace to target namespace.

    This function also moves all the references inside the source prim path
    to the target prim path.

    Args:
        source_ns: The source prim path.
        target_ns: The target prim path.
    """
    # check if target namespace exists
    prim_utils.define_prim(target_ns)
    # move all children prim from source namespace
    prim = prim_utils.get_prim_at_path(source_ns)
    for children in prim.GetChildren():
        orig_prim_path = prim_utils.get_prim_path(children)
        new_prim_path = orig_prim_path.replace(source_ns, target_ns)
        prim_utils.move_prim(orig_prim_path, new_prim_path)


def set_drive_dof_properties(
    prim_path: str,
    dof_name: str,
    stiffness: float | None = None,
    damping: float | None = None,
    max_velocity: float | None = None,
    max_force: float | None = None,
) -> None:
    """Set the DOF properties of a drive on an articulation.

    Args:
        prim_path: The prim path to the articulation root.
        dof_name: The name of the DOF/joint.
        stiffness: The stiffness of the drive.
        damping: The damping of the drive.
        max_velocity: The max velocity of the drive.
        max_force: The max effort of the drive.

    Raises:
        ValueError: When no joint of given name found under the provided prim path.
    """
    # find matching prim path for the dof name
    dof_prim = prim_utils.get_first_matching_child_prim(prim_path, lambda x: dof_name in x)
    if not dof_prim.IsValid():
        raise ValueError(f"No joint named '{dof_name}' found in articulation '{prim_path}'.")
    # obtain the dof drive type
    if dof_prim.IsA(UsdPhysics.RevoluteJoint):
        drive_type = "angular"
    elif dof_prim.IsA(UsdPhysics.PrismaticJoint):
        drive_type = "linear"
    else:
        # get joint USD prim
        dof_prim_path = prim_utils.get_prim_path(dof_prim)
        raise ValueError(f"The joint at path '{dof_prim_path}' is not linear or angular.")

    # convert to USD Physics drive
    if dof_prim.HasAPI(UsdPhysics.DriveAPI):
        drive_api = UsdPhysics.DriveAPI(dof_prim, drive_type)
    else:
        drive_api = UsdPhysics.DriveAPI.Apply(dof_prim, drive_type)
    # convert DOF type to be force
    if not drive_api.GetTypeAttr():
        drive_api.CreateTypeAttr().Set("force")
    else:
        drive_api.GetTypeAttr().Set("force")

    # set stiffness of the drive
    if stiffness is not None:
        # convert from radians to degrees
        # note: gains have units "per degrees"
        if drive_type == "angular":
            stiffness = stiffness * math.pi / 180
        # set property
        if not drive_api.GetStiffnessAttr():
            drive_api.CreateStiffnessAttr(stiffness)
        else:
            drive_api.GetStiffnessAttr().Set(stiffness)
    # set damping of the drive
    if damping is not None:
        # convert from radians to degrees
        # note: gains have units "per degrees"
        if drive_type == "angular":
            damping = damping * math.pi / 180
        # set property
        if not drive_api.GetDampingAttr():
            drive_api.CreateDampingAttr(damping)
        else:
            drive_api.GetDampingAttr().Set(damping)
    # set maximum force
    if max_force is not None:
        if not drive_api.GetMaxForceAttr():
            drive_api.CreateMaxForceAttr(max_force)
        else:
            drive_api.GetMaxForceAttr().Set(max_force)

    # convert to physx schema
    drive_schema = PhysxSchema.PhysxJointAPI(dof_prim)
    # set maximum velocity
    if max_velocity is not None:
        # convert from radians to degrees
        if drive_type == "angular":
            max_velocity = math.degrees(max_velocity)
        # set property
        if not drive_schema.GetMaxJointVelocityAttr():
            drive_schema.CreateMaxJointVelocityAttr(max_velocity)
        else:
            drive_schema.GetMaxJointVelocityAttr().Set(max_velocity)


def set_articulation_properties(
    prim_path: str,
    articulation_enabled: bool | None = None,
    solver_position_iteration_count: int | None = None,
    solver_velocity_iteration_count: int | None = None,
    sleep_threshold: float | None = None,
    stabilization_threshold: float | None = None,
    enable_self_collisions: bool | None = None,
) -> None:
    """Set PhysX parameters for an articulation prim.

    Args:
        prim_path: The prim path to the articulation root.
        articulation_enabled: Whether the articulation should be enabled/disabled.
        solver_position_iteration_count: Solver position iteration counts for the body.
        solver_velocity_iteration_count: Solver velocity iteration counts for the body.
        sleep_threshold: Mass-normalized kinetic energy threshold below which an
            actor may go to sleep.
        stabilization_threshold: The mass-normalized kinetic energy threshold below
            which an articulation may participate in stabilization.
        enable_self_collisions: Boolean defining whether self collisions should be
            enabled or disabled.

    Raises:
        ValueError: When no articulation schema found at specified articulation path.
    """
    # get articulation USD prim
    articulation_prim = prim_utils.get_prim_at_path(prim_path)
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        raise ValueError(f"No articulation schema present for prim '{prim_path}'.")
    # retrieve the articulation api
    physx_articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
    if not physx_articulation_api:
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)
    # set enable/disable rigid body API
    if articulation_enabled is not None:
        physx_articulation_api.GetArticulationEnabledAttr().Set(articulation_enabled)
    # set solver position iteration
    if solver_position_iteration_count is not None:
        physx_articulation_api.GetSolverPositionIterationCountAttr().Set(solver_position_iteration_count)
    # set solver velocity iteration
    if solver_velocity_iteration_count is not None:
        physx_articulation_api.GetSolverVelocityIterationCountAttr().Set(solver_velocity_iteration_count)
    # set sleep threshold
    if sleep_threshold is not None:
        physx_articulation_api.GetSleepThresholdAttr().Set(sleep_threshold)
    # set stabilization threshold
    if stabilization_threshold is not None:
        physx_articulation_api.GetStabilizationThresholdAttr().Set(stabilization_threshold)
    # set self collisions
    if enable_self_collisions is not None:
        physx_articulation_api.GetEnabledSelfCollisionsAttr().Set(enable_self_collisions)


def set_rigid_body_properties(
    prim_path: str,
    rigid_body_enabled: bool | None = None,
    solver_position_iteration_count: int | None = None,
    solver_velocity_iteration_count: int | None = None,
    linear_damping: float | None = None,
    angular_damping: float | None = None,
    max_linear_velocity: float | None = None,
    max_angular_velocity: float | None = None,
    sleep_threshold: float | None = None,
    stabilization_threshold: float | None = None,
    max_depenetration_velocity: float | None = None,
    max_contact_impulse: float | None = None,
    enable_gyroscopic_forces: bool | None = None,
    disable_gravity: bool | None = None,
    retain_accelerations: bool | None = None,
):
    """Set PhysX parameters for a rigid body prim.

    Args:
        prim_path: The prim path to the rigid body.
        rigid_body_enabled: Whether to enable or disable rigid body API.
        solver_position_iteration_count: Solver position iteration counts for the body.
        solver_velocity_iteration_count: Solver velocity iteration counts for the body.
        linear_damping: Linear damping coefficient.
        angular_damping: Angular damping coefficient.
        max_linear_velocity: Max allowable linear velocity for rigid body (in m/s).
        max_angular_velocity: Max allowable angular velocity for rigid body (in rad/s).
        sleep_threshold: Mass-normalized kinetic energy threshold below which an actor
            may go to sleep.
        stabilization_threshold: Mass-normalized kinetic energy threshold below which
            an actor may participate in stabilization.
        max_depenetration_velocity: The maximum depenetration velocity permitted to
            be introduced by the solver (in m/s).
        max_contact_impulse: The limit on the impulse that may be applied at a contact.
        enable_gyroscopic_forces: Enables computation of gyroscopic forces on the
            rigid body.
        disable_gravity: Disable gravity for the actor.
        retain_accelerations: Carries over forces/accelerations over sub-steps.

    Raises:
        ValueError: When no rigid-body schema found at specified prim path.
    """
    # get rigid-body USD prim
    rigid_body_prim = prim_utils.get_prim_at_path(prim_path)
    # check if prim has rigid-body applied on it
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        raise ValueError(f"No rigid body schema present for prim '{prim_path}'.")
    # retrieve the USD rigid-body api
    usd_rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
    # retrieve the physx rigid-body api
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(rigid_body_prim)
    if not physx_rigid_body_api:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_body_prim)
    # set enable/disable rigid body API
    if rigid_body_enabled is not None:
        usd_rigid_body_api.GetRigidBodyEnabledAttr().Set(rigid_body_enabled)
    # set solver position iteration
    if solver_position_iteration_count is not None:
        physx_rigid_body_api.GetSolverPositionIterationCountAttr().Set(solver_position_iteration_count)
    # set solver velocity iteration
    if solver_velocity_iteration_count is not None:
        physx_rigid_body_api.GetSolverVelocityIterationCountAttr().Set(solver_velocity_iteration_count)
    # set linear damping
    if linear_damping is not None:
        physx_rigid_body_api.GetLinearDampingAttr().Set(linear_damping)
    # set angular damping
    if angular_damping is not None:
        physx_rigid_body_api.GetAngularDampingAttr().Set(angular_damping)
    # set max linear velocity
    if max_linear_velocity is not None:
        physx_rigid_body_api.GetMaxLinearVelocityAttr().Set(max_linear_velocity)
    # set max angular velocity
    if max_angular_velocity is not None:
        max_angular_velocity = math.degrees(max_angular_velocity)
        physx_rigid_body_api.GetMaxAngularVelocityAttr().Set(max_angular_velocity)
    # set sleep threshold
    if sleep_threshold is not None:
        physx_rigid_body_api.GetSleepThresholdAttr().Set(sleep_threshold)
    # set stabilization threshold
    if stabilization_threshold is not None:
        physx_rigid_body_api.GetStabilizationThresholdAttr().Set(stabilization_threshold)
    # set max depenetration velocity
    if max_depenetration_velocity is not None:
        physx_rigid_body_api.GetMaxDepenetrationVelocityAttr().Set(max_depenetration_velocity)
    # set max contact impulse
    if max_contact_impulse is not None:
        physx_rigid_body_api.GetMaxContactImpulseAttr().Set(max_contact_impulse)
    # set enable gyroscopic forces
    if enable_gyroscopic_forces is not None:
        physx_rigid_body_api.GetEnableGyroscopicForcesAttr().Set(enable_gyroscopic_forces)
    # set disable gravity
    if disable_gravity is not None:
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
    # set retain accelerations
    if retain_accelerations is not None:
        physx_rigid_body_api.GetRetainAccelerationsAttr().Set(retain_accelerations)


def set_collision_properties(
    prim_path: str,
    collision_enabled: bool | None = None,
    contact_offset: float | None = None,
    rest_offset: float | None = None,
    torsional_patch_radius: float | None = None,
    min_torsional_patch_radius: float | None = None,
):
    """Set PhysX properties of collider prim.

    Args:
        prim_path: The prim path of parent.
        collision_enabled: Whether to enable/disable collider.
        contact_offset: Contact offset of a collision shape (in m).
        rest_offset: Rest offset of a collision shape (in m).
        torsional_patch_radius: Defines the radius of the contact patch
            used to apply torsional friction (in m).
        min_torsional_patch_radius: Defines the minimum radius of the
            contact patch used to apply torsional friction (in m).

    Raises:
        ValueError:  When no collision schema found at specified prim path.
    """
    # get USD prim
    collider_prim = prim_utils.get_prim_at_path(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        raise ValueError(f"No collider schema present for prim '{prim_path}'.")
    # retrieve the collision api
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
    if not physx_collision_api:
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)
    # set enable/disable collision API
    if collision_enabled is not None:
        physx_collision_api.GetCollisionEnabledAttr().Set(collision_enabled)
    # set contact offset
    if contact_offset is not None:
        physx_collision_api.GetContactOffsetAttr().Set(contact_offset)
    # set rest offset
    if rest_offset is not None:
        physx_collision_api.GetRestOffsetAttr().Set(rest_offset)
    # set torsional patch radius
    if torsional_patch_radius is not None:
        physx_collision_api.GetTorsionalPatchRadiusAttr().Set(torsional_patch_radius)
    # set min torsional patch radius
    if min_torsional_patch_radius is not None:
        physx_collision_api.GetMinTorsionalPatchRadiusAttr().Set(min_torsional_patch_radius)


def apply_physics_material(prim_path: str, material_path: str, weaker_than_descendants: bool = False):
    """Apply a physics material to a prim.

    Physics material can be applied only to a prim with physics-enabled on them. This includes having
    a collision APIs, or deformable body APIs, or being a particle system.

    Args:
        prim_path: The prim path of parent.
        material_path: The prim path of the material to apply.

    Raises:
        ValueError: If the material path does not exist on stage.
        ValueError: When prim at specified path is not physics-enabled.
    """
    # check if material exists
    if not prim_utils.is_prim_path_valid(material_path):
        raise ValueError(f"Physics material '{material_path}' does not exist.")
    # get USD prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # check if prim has collision applied on it
    has_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_deformable_body = prim.HasAPI(PhysxSchema.PhysxDeformableBodyAPI)
    has_particle_system = prim.IsA(PhysxSchema.PhysxParticleSystem)
    if not (has_collider or has_deformable_body or has_particle_system):
        raise ValueError(
            f"Cannot apply physics material on prim '{prim_path}'. It is neither a collider,"
            " nor a deformable body, nor a particle system."
        )
    # obtain material binding API
    if prim.HasAPI(UsdShade.MaterialBindingAPI):
        material_binding_api = UsdShade.MaterialBindingAPI(prim)
    else:
        material_binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
    # obtain the material prim
    material = UsdShade.Material(prim_utils.get_prim_at_path(material_path))
    # resolve token for weaker than descendants
    if weaker_than_descendants:
        binding_strength = UsdShade.Tokens.weakerThanDescendants
    else:
        binding_strength = UsdShade.Tokens.strongerThanDescendants
    # apply the material
    material_binding_api.Bind(material, bindingStrength=binding_strength, materialPurpose="physics")


def set_nested_articulation_properties(prim_path: str, **kwargs) -> None:
    """Set PhysX parameters on all articulations under specified prim-path.

    Note:
        Check the method meth:`set_articulation_properties` for keyword arguments.

    Args:
        prim_path: The prim path under which to search and apply articulation properties.

    Keyword Args:
        articulation_enabled: Whether the articulation should be enabled/disabled.
        solver_position_iteration_count: Solver position iteration counts for the body.
        solver_velocity_iteration_count: Solver velocity iteration counts for the body.
        sleep_threshold: Mass-normalized kinetic energy threshold below which an
            actor may go to sleep.
        stabilization_threshold: The mass-normalized kinetic energy threshold below
            which an articulation may participate in stabilization.
        enable_self_collisions: Boolean defining whether self collisions should be
            enabled or disabled.
    """
    # get USD prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # set articulation properties
        with contextlib.suppress(ValueError):
            set_articulation_properties(prim_utils.get_prim_path(child_prim), **kwargs)
        # add all children to tree
        all_prims += child_prim.GetChildren()


def set_nested_rigid_body_properties(prim_path: str, **kwargs):
    """Set PhysX parameters on all rigid bodies under specified prim-path.

    Note:
        Check the method meth:`set_rigid_body_properties` for keyword arguments.

    Args:
        prim_path: The prim path under which to search and apply rigid-body properties.

    Keyword Args:
        rigid_body_enabled: Whether to enable or disable rigid body API.
        solver_position_iteration_count: Solver position iteration counts for the body.
        solver_velocity_iteration_count: Solver velocity iteration counts for the body.
        linear_damping: Linear damping coefficient.
        angular_damping: Angular damping coefficient.
        max_linear_velocity: Max allowable linear velocity for rigid body (in m/s).
        max_angular_velocity: Max allowable angular velocity for rigid body (in rad/s).
        sleep_threshold: Mass-normalized kinetic energy threshold below which an actor
            may go to sleep.
        stabilization_threshold: Mass-normalized kinetic energy threshold below which
            an actor may participate in stabilization.
        max_depenetration_velocity: The maximum depenetration velocity permitted to
            be introduced by the solver (in m/s).
        max_contact_impulse: The limit on the impulse that may be applied at a contact.
        enable_gyroscopic_forces: Enables computation of gyroscopic forces on the
            rigid body.
        disable_gravity: Disable gravity for the actor.
        retain_accelerations: Carries over forces/accelerations over sub-steps.
    """
    # get USD prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # set rigid-body properties
        with contextlib.suppress(ValueError):
            set_rigid_body_properties(prim_utils.get_prim_path(child_prim), **kwargs)
        # add all children to tree
        all_prims += child_prim.GetChildren()


def set_nested_collision_properties(prim_path: str, **kwargs):
    """Set the collider properties of all meshes under a specified prim path.

    Note:
        Check the method meth:`set_collision_properties` for keyword arguments.

    Args:
        prim_path: The prim path under which to search and apply collider properties.

    Keyword Args:
        collision_enabled: Whether to enable/disable collider.
        contact_offset: Contact offset of a collision shape (in m).
        rest_offset: Rest offset of a collision shape (in m).
        torsional_patch_radius: Defines the radius of the contact patch
            used to apply torsional friction (in m).
        min_torsional_patch_radius: Defines the minimum radius of the
            contact patch used to apply torsional friction (in m).
    """
    # get USD prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # set collider properties
        with contextlib.suppress(ValueError):
            set_collision_properties(prim_utils.get_prim_path(child_prim), **kwargs)
        # add all children to tree
        all_prims += child_prim.GetChildren()


def apply_nested_physics_material(prim_path: str, material_path: str, weaker_than_descendants: bool = False):
    """Apply the physics material on all meshes under a specified prim path.

    Physics material can be applied only to a prim with physics-enabled on them. This includes having
    a collision APIs, or deformable body APIs, or being a particle system.

    Args:
        prim_path: The prim path under which to search and apply physics material.
        material_path: The path to the physics material to apply.
        weaker_than_descendants: Whether the material should override the
            descendants materials. Defaults to False.

    Raises:
        ValueError: If the material path does not exist on stage.
    """
    # check if material exists
    if not prim_utils.is_prim_path_valid(material_path):
        raise ValueError(f"Physics material '{material_path}' does not exist.")
    # get USD prim
    prim = prim_utils.get_prim_at_path(prim_path)
    # iterate over all prims under prim-path
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # set physics material
        with contextlib.suppress(ValueError):
            apply_physics_material(prim_utils.get_prim_path(child_prim), material_path, weaker_than_descendants)
        # add all children to tree
        all_prims += child_prim.GetChildren()
