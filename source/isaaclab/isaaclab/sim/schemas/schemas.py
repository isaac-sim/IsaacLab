# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import isaacsim.core.utils.stage as stage_utils
import omni.log
import omni.physx.scripts.utils as physx_utils
from omni.physx.scripts import deformableUtils as deformable_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from ..utils import (
    apply_nested,
    find_global_fixed_joint_prim,
    get_all_matching_child_prims,
    safe_set_attribute_on_usd_schema,
)
from . import schemas_cfg

"""
Articulation root properties.
"""


def define_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the articulation root schema on the input prim and set its properties.

    See :func:`modify_articulation_root_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the articulation root schema.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get articulation USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(prim):
        UsdPhysics.ArticulationRootAPI.Apply(prim)
    # set articulation root properties
    modify_articulation_root_properties(prim_path, cfg, stage)


@apply_nested
def modify_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for an articulation root prim.

    The `articulation root`_ marks the root of an articulation tree. For floating articulations, this should be on
    the root body. For fixed articulations, this API can be on a direct or indirect parent of the root joint
    which is fixed to the world.

    The schema comprises of attributes that belong to the `ArticulationRootAPI`_ and `PhysxArticulationAPI`_.
    schemas. The latter contains the PhysX parameters for the articulation root.

    The properties are applied to the articulation root prim. The common properties (such as solver position
    and velocity iteration counts, sleep threshold, stabilization threshold) take precedence over those specified
    in the rigid body schemas for all the rigid bodies in the articulation.

    .. caution::
        When the attribute :attr:`schemas_cfg.ArticulationRootPropertiesCfg.fix_root_link` is set to True,
        a fixed joint is created between the root link and the world frame (if it does not already exist). However,
        to deal with physics parser limitations, the articulation root schema needs to be applied to the parent of
        the root link.

    .. note::
        This function is decorated with :func:`apply_nested` that set the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _articulation root: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Articulations.html
    .. _ArticulationRootAPI: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html

    Args:
        prim_path: The prim path to the articulation root.
        cfg: The configuration for the articulation root.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        NotImplementedError: When the root prim is not a rigid body and a fixed joint is to be created.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get articulation USD prim
    articulation_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        return False
    # retrieve the articulation api
    physx_articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
    if not physx_articulation_api:
        physx_articulation_api = PhysxSchema.PhysxArticulationAPI.Apply(articulation_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # extract non-USD properties
    fix_root_link = cfg.pop("fix_root_link", None)

    # set into physx api
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_articulation_api, attr_name, value, camel_case=True)

    # fix root link based on input
    # we do the fixed joint processing later to not interfere with setting other properties
    if fix_root_link is not None:
        # check if a global fixed joint exists under the root prim
        existing_fixed_joint_prim = find_global_fixed_joint_prim(prim_path)

        # if we found a fixed joint, enable/disable it based on the input
        # otherwise, create a fixed joint between the world and the root link
        if existing_fixed_joint_prim is not None:
            omni.log.info(
                f"Found an existing fixed joint for the articulation: '{prim_path}'. Setting it to: {fix_root_link}."
            )
            existing_fixed_joint_prim.GetJointEnabledAttr().Set(fix_root_link)
        elif fix_root_link:
            omni.log.info(f"Creating a fixed joint for the articulation: '{prim_path}'.")

            # note: we have to assume that the root prim is a rigid body,
            #   i.e. we don't handle the case where the root prim is not a rigid body but has articulation api on it
            # Currently, there is no obvious way to get first rigid body link identified by the PhysX parser
            if not articulation_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise NotImplementedError(
                    f"The articulation prim '{prim_path}' does not have the RigidBodyAPI applied."
                    " To create a fixed joint, we need to determine the first rigid body link in"
                    " the articulation tree. However, this is not implemented yet."
                )

            # create a fixed joint between the root link and the world frame
            physx_utils.createJoint(stage=stage, joint_type="Fixed", from_prim=None, to_prim=articulation_prim)

            # Having a fixed joint on a rigid body is not treated as "fixed base articulation".
            # instead, it is treated as a part of the maximal coordinate tree.
            # Moving the articulation root to the parent solves this issue. This is a limitation of the PhysX parser.
            # get parent prim
            parent_prim = articulation_prim.GetParent()
            # apply api to parent
            UsdPhysics.ArticulationRootAPI.Apply(parent_prim)
            PhysxSchema.PhysxArticulationAPI.Apply(parent_prim)

            # copy the attributes
            # -- usd attributes
            usd_articulation_api = UsdPhysics.ArticulationRootAPI(articulation_prim)
            for attr_name in usd_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_prim.GetAttribute(attr_name).Set(attr.Get())
            # -- physx attributes
            physx_articulation_api = PhysxSchema.PhysxArticulationAPI(articulation_prim)
            for attr_name in physx_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_prim.GetAttribute(attr_name).Set(attr.Get())

            # remove api from root
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            articulation_prim.RemoveAPI(PhysxSchema.PhysxArticulationAPI)

    # success
    return True


"""
Rigid body properties.
"""


def define_rigid_body_properties(
    prim_path: str, cfg: schemas_cfg.RigidBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the rigid body schema on the input prim and set its properties.

    See :func:`modify_rigid_body_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the rigid body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        TypeError: When the prim already has conflicting API schemas.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has rigid body applied on it
    if not UsdPhysics.RigidBodyAPI(prim):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    # set rigid body properties
    modify_rigid_body_properties(prim_path, cfg, stage)


@apply_nested
def modify_rigid_body_properties(
    prim_path: str, cfg: schemas_cfg.RigidBodyPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a rigid body prim.

    A `rigid body`_ is a single body that can be simulated by PhysX. It can be either dynamic or kinematic.
    A dynamic body responds to forces and collisions. A `kinematic body`_ can be moved by the user, but does not
    respond to forces. They are similar to having static bodies that can be moved around.

    The schema comprises of attributes that belong to the `RigidBodyAPI`_ and `PhysxRigidBodyAPI`_.
    schemas. The latter contains the PhysX parameters for the rigid body.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html

    Args:
        prim_path: The prim path to the rigid body.
        cfg: The configuration for the rigid body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get rigid-body USD prim
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has rigid-body applied on it
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False
    # retrieve the USD rigid-body api
    usd_rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
    # retrieve the physx rigid-body api
    physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(rigid_body_prim)
    if not physx_rigid_body_api:
        physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(rigid_body_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["rigid_body_enabled", "kinematic_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_rigid_body_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_rigid_body_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Collision properties.
"""


def define_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the collision schema on the input prim and set its properties.

    See :func:`modify_collision_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(prim):
        UsdPhysics.CollisionAPI.Apply(prim)
    # set collision properties
    modify_collision_properties(prim_path, cfg, stage)


@apply_nested
def modify_collision_properties(
    prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX properties of collider prim.

    These properties are based on the `UsdPhysics.CollisionAPI`_ and `PhysxSchema.PhysxCollisionAPI`_ schemas.
    For more information on the properties, please refer to the official documentation.

    Tuning these parameters influence the contact behavior of the rigid body. For more information on
    tune them and their effect on the simulation, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/AdvancedCollisionDetection.html>`__.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. _PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    Args:
        prim_path: The prim path of parent.
        cfg: The configuration for the collider.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # retrieve the USD collision api
    usd_collision_api = UsdPhysics.CollisionAPI(collider_prim)
    # retrieve the collision api
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(collider_prim)
    if not physx_collision_api:
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["collision_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_collision_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_collision_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Mass properties.
"""


def define_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None):
    """Apply the mass schema on the input prim and set its properties.

    See :func:`modify_mass_properties` for more details on how the properties are set.

    Args:
        prim_path: The prim path where to apply the rigid body schema.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # check if prim has mass applied on it
    if not UsdPhysics.MassAPI(prim):
        UsdPhysics.MassAPI.Apply(prim)
    # set mass properties
    modify_mass_properties(prim_path, cfg, stage)


@apply_nested
def modify_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage | None = None) -> bool:
    """Set properties for the mass of a rigid body prim.

    These properties are based on the `UsdPhysics.MassAPI` schema. If the mass is not defined, the density is used
    to compute the mass. However, in that case, a collision approximation of the rigid body is used to
    compute the density. For more information on the properties, please refer to the
    `documentation <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`__.

    .. caution::

        The mass of an object can be specified in multiple ways and have several conflicting settings
        that are resolved based on precedence. Please make sure to understand the precedence rules
        before using this property.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    Args:
        prim_path: The prim path of the rigid body.
        cfg: The configuration for the mass properties.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    rigid_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has mass API applied on it
    if not UsdPhysics.MassAPI(rigid_prim):
        return False
    # retrieve the USD mass api
    usd_physics_mass_api = UsdPhysics.MassAPI(rigid_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["mass", "density"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_mass_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Contact sensor.
"""


def activate_contact_sensors(prim_path: str, threshold: float = 0.0, stage: Usd.Stage = None):
    """Activate the contact sensor on all rigid bodies under a specified prim path.

    This function adds the PhysX contact report API to all rigid bodies under the specified prim path.
    It also sets the force threshold beyond which the contact sensor reports the contact. The contact
    reporting API can only be added to rigid bodies.

    Args:
        prim_path: The prim path under which to search and prepare contact sensors.
        threshold: The threshold for the contact sensor. Defaults to 0.0.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: If the input prim path is not valid.
        ValueError: If there are no rigid bodies under the prim path.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get prim
    prim: Usd.Prim = stage.GetPrimAtPath(prim_path)
    # check if prim is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")
    # iterate over all children
    num_contact_sensors = 0
    all_prims = [prim]
    while len(all_prims) > 0:
        # get current prim
        child_prim = all_prims.pop(0)
        # check if prim is a rigid body
        # nested rigid bodies are not allowed by SDK so we can safely assume that
        # if a prim has a rigid body API, it is a rigid body and we don't need to
        # check its children
        if child_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # set sleep threshold to zero
            rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPrimPath())
            rb.CreateSleepThresholdAttr().Set(0.0)
            # add contact report API with threshold of zero
            if not child_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
                omni.log.verbose(f"Adding contact report API to prim: '{child_prim.GetPrimPath()}'")
                cr_api = PhysxSchema.PhysxContactReportAPI.Apply(child_prim)
            else:
                omni.log.verbose(f"Contact report API already exists on prim: '{child_prim.GetPrimPath()}'")
                cr_api = PhysxSchema.PhysxContactReportAPI.Get(stage, child_prim.GetPrimPath())
            # set threshold to zero
            cr_api.CreateThresholdAttr().Set(threshold)
            # increment number of contact sensors
            num_contact_sensors += 1
        else:
            # add all children to tree
            all_prims += child_prim.GetChildren()
    # check if no contact sensors were found
    if num_contact_sensors == 0:
        raise ValueError(
            f"No contact sensors added to the prim: '{prim_path}'. This means that no rigid bodies"
            " are present under this prim. Please check the prim path."
        )
    # success
    return True


"""
Joint drive properties.
"""


@apply_nested
def modify_joint_drive_properties(
    prim_path: str, drive_props: schemas_cfg.JointDrivePropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a joint prim.

    This function checks if the input prim is a prismatic or revolute joint and applies the joint drive schema
    on it. If the joint is a tendon (i.e., it has the `PhysxTendonAxisAPI`_ schema applied on it), then the joint
    drive schema is not applied.

    Based on the configuration, this method modifies the properties of the joint drive. These properties are
    based on the `UsdPhysics.DriveAPI`_ schema. For more information on the properties, please refer to the
    official documentation.

    .. caution::

        We highly recommend modifying joint properties of articulations through the functionalities in the
        :mod:`isaaclab.actuators` module. The methods here are for setting simulation low-level
        properties only.

    .. _UsdPhysics.DriveAPI: https://openusd.org/dev/api/class_usd_physics_drive_a_p_i.html
    .. _PhysxTendonAxisAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_a_p_i.html

    Args:
        prim_path: The prim path where to apply the joint drive schema.
        drive_props: The configuration for the joint drive.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # check if prim has joint drive applied on it
    if prim.IsA(UsdPhysics.RevoluteJoint):
        drive_api_name = "angular"
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        drive_api_name = "linear"
    else:
        return False
    # check that prim is not a tendon child prim
    # note: root prim is what "controls" the tendon so we still want to apply the drive to it
    if prim.HasAPI(PhysxSchema.PhysxTendonAxisAPI) and not prim.HasAPI(PhysxSchema.PhysxTendonAxisRootAPI):
        return False

    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name)
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_api_name)

    # change the drive type to input
    if drive_props.drive_type is not None:
        usd_drive_api.CreateTypeAttr().Set(drive_props.drive_type)

    return True


"""
Fixed tendon properties.
"""


@apply_nested
def modify_fixed_tendon_properties(
    prim_path: str, cfg: schemas_cfg.FixedTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a fixed tendon attachment prim.

    A `fixed tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _fixed tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationFixedTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html

    Args:
        prim_path: The prim path to the tendon attachment.
        cfg: The configuration for the tendon attachment.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has fixed tendon applied on it
    has_root_fixed_tendon = tendon_prim.HasAPI(PhysxSchema.PhysxTendonAxisRootAPI)
    if not has_root_fixed_tendon:
        return False

    # resolve all available instances of the schema since it is multi-instance
    for schema_name in tendon_prim.GetAppliedSchemas():
        # only consider the fixed tendon schema
        if "PhysxTendonAxisRootAPI" not in schema_name:
            continue
        # retrieve the USD tendon api
        instance_name = schema_name.split(":")[-1]
        physx_tendon_axis_api = PhysxSchema.PhysxTendonAxisRootAPI(tendon_prim, instance_name)

        # convert to dict
        cfg = cfg.to_dict()
        # set into PhysX API
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_schema(physx_tendon_axis_api, attr_name, value, camel_case=True)
    # success
    return True


"""
Deformable body properties.
"""


def define_deformable_body_properties(
    prim_path: str, cfg: schemas_cfg.DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the deformable body schema on the input prim and set its properties.

    See :func:`modify_deformable_body_properties` for more details on how the properties are set.

    .. note::
        If the input prim is not a mesh, this function will traverse the prim and find the first mesh
        under it. If no mesh or multiple meshes are found, an error is raised. This is because the deformable
        body schema can only be applied to a single mesh.

    Args:
        prim_path: The prim path where to apply the deformable body schema.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Raises:
        ValueError: When the prim path is not valid.
        ValueError: When the prim has no mesh or multiple meshes.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    # traverse the prim and get the mesh
    matching_prims = get_all_matching_child_prims(prim_path, lambda p: p.GetTypeName() == "Mesh")
    # check if the mesh is valid
    if len(matching_prims) == 0:
        raise ValueError(f"Could not find any mesh in '{prim_path}'. Please check asset.")
    if len(matching_prims) > 1:
        # get list of all meshes found
        mesh_paths = [p.GetPrimPath() for p in matching_prims]
        raise ValueError(
            f"Found multiple meshes in '{prim_path}': {mesh_paths}."
            " Deformable body schema can only be applied to one mesh."
        )

    # get deformable-body USD prim
    mesh_prim = matching_prims[0]
    # check if prim has deformable-body applied on it
    if not PhysxSchema.PhysxDeformableBodyAPI(mesh_prim):
        PhysxSchema.PhysxDeformableBodyAPI.Apply(mesh_prim)
    # set deformable body properties
    modify_deformable_body_properties(mesh_prim.GetPrimPath(), cfg, stage)


@apply_nested
def modify_deformable_body_properties(
    prim_path: str, cfg: schemas_cfg.DeformableBodyPropertiesCfg, stage: Usd.Stage | None = None
):
    """Modify PhysX parameters for a deformable body prim.

    A `deformable body`_ is a single body that can be simulated by PhysX. Unlike rigid bodies, deformable bodies
    support relative motion of the nodes in the mesh. Consequently, they can be used to simulate deformations
    under applied forces.

    PhysX soft body simulation employs Finite Element Analysis (FEA) to simulate the deformations of the mesh.
    It uses two tetrahedral meshes to represent the deformable body:

    1. **Simulation mesh**: This mesh is used for the simulation and is the one that is deformed by the solver.
    2. **Collision mesh**: This mesh only needs to match the surface of the simulation mesh and is used for
       collision detection.

    For most applications, we assume that the above two meshes are computed from the "render mesh" of the deformable
    body. The render mesh is the mesh that is visible in the scene and is used for rendering purposes. It is composed
    of triangles and is the one that is used to compute the above meshes based on PhysX cookings.

    The schema comprises of attributes that belong to the `PhysxDeformableBodyAPI`_. schemas containing the PhysX
    parameters for the deformable body.

    .. caution::
        The deformable body schema is still under development by the Omniverse team. The current implementation
        works with the PhysX schemas shipped with Isaac Sim 4.0.0 onwards. It may change in future releases.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _deformable body: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html
    .. _PhysxDeformableBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_deformable_a_p_i.html

    Args:
        prim_path: The prim path to the deformable body.
        cfg: The configuration for the deformable body.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()

    # get deformable-body USD prim
    deformable_body_prim = stage.GetPrimAtPath(prim_path)

    # check if the prim is valid and has the deformable-body API
    if not deformable_body_prim.IsValid() or not PhysxSchema.PhysxDeformableBodyAPI(deformable_body_prim):
        return False

    # retrieve the physx deformable-body api
    physx_deformable_body_api = PhysxSchema.PhysxDeformableBodyAPI(deformable_body_prim)
    # retrieve the physx deformable api
    physx_deformable_api = PhysxSchema.PhysxDeformableAPI(physx_deformable_body_api)

    # convert to dict
    cfg = cfg.to_dict()
    # set into deformable body API
    attr_kwargs = {
        attr_name: cfg.pop(attr_name)
        for attr_name in [
            "kinematic_enabled",
            "collision_simplification",
            "collision_simplification_remeshing",
            "collision_simplification_remeshing_resolution",
            "collision_simplification_target_triangle_count",
            "collision_simplification_force_conforming",
            "simulation_hexahedral_resolution",
            "solver_position_iteration_count",
            "vertex_velocity_damping",
            "sleep_damping",
            "sleep_threshold",
            "settling_threshold",
            "self_collision",
            "self_collision_filter_distance",
        ]
    }
    status = deformable_utils.add_physx_deformable_body(stage, prim_path=prim_path, **attr_kwargs)
    # check if the deformable body was successfully added
    if not status:
        return False

    # obtain the PhysX collision API (this is set when the deformable body is added)
    physx_collision_api = PhysxSchema.PhysxCollisionAPI(deformable_body_prim)

    # set into PhysX API
    for attr_name, value in cfg.items():
        if attr_name in ["rest_offset", "contact_offset"]:
            safe_set_attribute_on_usd_schema(physx_collision_api, attr_name, value, camel_case=True)
        else:
            safe_set_attribute_on_usd_schema(physx_deformable_api, attr_name, value, camel_case=True)

    # success
    return True
