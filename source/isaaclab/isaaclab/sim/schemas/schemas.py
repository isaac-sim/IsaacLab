# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# needed to import for allowing type-hinting: Usd.Stage | None
from __future__ import annotations

import logging
import math
from typing import Any

import omni.physx.scripts.utils as physx_utils
from omni.physx.scripts import deformableUtils as deformable_utils
from pxr import Usd, UsdPhysics

from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils.string import to_camel_case

from ..utils import (
    apply_nested,
    find_global_fixed_joint_prim,
    get_all_matching_child_prims,
    safe_set_attribute_on_usd_prim,
    safe_set_attribute_on_usd_schema,
)
from . import schemas_cfg

# import logger
logger = logging.getLogger(__name__)


"""
Constants.
"""

# Mapping from string names to USD/PhysX tokens for mesh collision approximation
# Refer to omniverse documentation
# https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/collision.html#mesh-geometry-colliders
# for available tokens.
MESH_APPROXIMATION_TOKENS = {
    "boundingCube": UsdPhysics.Tokens.boundingCube,
    "boundingSphere": UsdPhysics.Tokens.boundingSphere,
    "convexDecomposition": UsdPhysics.Tokens.convexDecomposition,
    "convexHull": UsdPhysics.Tokens.convexHull,
    "none": UsdPhysics.Tokens.none,
    "meshSimplification": UsdPhysics.Tokens.meshSimplification,
    "sdf": "sdf",  # PhysX SDF mesh token; use string (pxr.Tf.Token not available in all envs)
}


PHYSX_MESH_COLLISION_CFGS = [
    schemas_cfg.ConvexDecompositionPropertiesCfg,
    schemas_cfg.ConvexHullPropertiesCfg,
    schemas_cfg.TriangleMeshPropertiesCfg,
    schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
    schemas_cfg.SDFMeshPropertiesCfg,
]

USD_MESH_COLLISION_CFGS = [
    schemas_cfg.BoundingCubePropertiesCfg,
    schemas_cfg.BoundingSpherePropertiesCfg,
    schemas_cfg.ConvexDecompositionPropertiesCfg,
    schemas_cfg.ConvexHullPropertiesCfg,
    schemas_cfg.TriangleMeshSimplificationPropertiesCfg,
]


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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get articulation USD prim
    articulation_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has articulation applied on it
    if not UsdPhysics.ArticulationRootAPI(articulation_prim):
        return False
    # ensure PhysX articulation API is applied
    applied_schemas = articulation_prim.GetAppliedSchemas()
    if "PhysxArticulationAPI" not in applied_schemas:
        articulation_prim.AddAppliedSchema("PhysxArticulationAPI")

    # convert to dict
    cfg = cfg.to_dict()
    # extract non-USD properties
    fix_root_link = cfg.pop("fix_root_link", None)

    # set into physx api (prim attributes under physxArticulation:*)
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_prim(
            articulation_prim, f"physxArticulation:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
        )

    # fix root link based on input
    # we do the fixed joint processing later to not interfere with setting other properties
    if fix_root_link is not None:
        # check if a global fixed joint exists under the root prim
        existing_fixed_joint_prim = find_global_fixed_joint_prim(prim_path)

        # if we found a fixed joint, enable/disable it based on the input
        # otherwise, create a fixed joint between the world and the root link
        if existing_fixed_joint_prim is not None:
            logger.info(
                f"Found an existing fixed joint for the articulation: '{prim_path}'. Setting it to: {fix_root_link}."
            )
            existing_fixed_joint_prim.GetJointEnabledAttr().Set(fix_root_link)
        elif fix_root_link:
            logger.info(f"Creating a fixed joint for the articulation: '{prim_path}'.")

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
            parent_applied = parent_prim.GetAppliedSchemas()
            if "PhysxArticulationAPI" not in parent_applied:
                parent_prim.AddAppliedSchema("PhysxArticulationAPI")

            # copy the attributes
            # -- usd attributes
            usd_articulation_api = UsdPhysics.ArticulationRootAPI(articulation_prim)
            for attr_name in usd_articulation_api.GetSchemaAttributeNames():
                attr = articulation_prim.GetAttribute(attr_name)
                parent_attr = parent_prim.GetAttribute(attr_name)
                if not parent_attr:
                    parent_attr = parent_prim.CreateAttribute(attr_name, attr.GetTypeName())
                parent_attr.Set(attr.Get())
            # -- physx attributes (copy by name prefix)
            for attr in articulation_prim.GetAttributes():
                aname = attr.GetName()
                if aname.startswith("physxArticulation:"):
                    parent_attr = parent_prim.GetAttribute(aname)
                    if not parent_attr:
                        parent_attr = parent_prim.CreateAttribute(aname, attr.GetTypeName())
                    parent_attr.Set(attr.Get())

            # remove api from root
            articulation_prim.RemoveAppliedSchema("PhysxArticulationAPI")
            articulation_prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get rigid-body USD prim
    rigid_body_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has rigid-body applied on it
    if not UsdPhysics.RigidBodyAPI(rigid_body_prim):
        return False
    # retrieve the USD rigid-body api
    usd_rigid_body_api = UsdPhysics.RigidBodyAPI(rigid_body_prim)
    # ensure PhysX rigid body API is applied
    applied_schemas = rigid_body_prim.GetAppliedSchemas()
    if "PhysxRigidBodyAPI" not in applied_schemas:
        rigid_body_prim.AddAppliedSchema("PhysxRigidBodyAPI")

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["rigid_body_enabled", "kinematic_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_rigid_body_api, attr_name, value, camel_case=True)
    # set into PhysX API (prim attributes under physxRigidBody:*)
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_prim(
            rigid_body_prim, f"physxRigidBody:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
        )
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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # retrieve the USD collision api
    usd_collision_api = UsdPhysics.CollisionAPI(collider_prim)
    # ensure PhysX collision API is applied
    applied_schemas = collider_prim.GetAppliedSchemas()
    if "PhysxCollisionAPI" not in applied_schemas:
        collider_prim.AddAppliedSchema("PhysxCollisionAPI")

    mesh_collision_cfg = getattr(cfg, "mesh_collision_property", None)
    if mesh_collision_cfg is not None:
        modify_mesh_collision_properties(prim_path, mesh_collision_cfg, stage)
    # convert to dict
    cfg = cfg.to_dict()
    # pop the mesh_collision_property since it is already set
    cfg.pop("mesh_collision_property", None)
    # set into USD API
    for attr_name in ["collision_enabled"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_collision_api, attr_name, value, camel_case=True)
    # set into PhysX API (prim attributes under physxCollision:*)
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_prim(
            collider_prim, f"physxCollision:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
        )
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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
            child_applied = child_prim.GetAppliedSchemas()
            if "PhysxRigidBodyAPI" not in child_applied:
                child_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            safe_set_attribute_on_usd_prim(child_prim, "physxRigidBody:sleepThreshold", 0.0, camel_case=False)
            # add contact report API with threshold of zero
            if "PhysxContactReportAPI" not in child_applied:
                logger.debug(f"Adding contact report API to prim: '{child_prim.GetPrimPath()}'")
                child_prim.AddAppliedSchema("PhysxContactReportAPI")
            safe_set_attribute_on_usd_prim(child_prim, "physxContactReport:threshold", threshold, camel_case=False)
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
    prim_path: str, cfg: schemas_cfg.JointDrivePropertiesCfg, stage: Usd.Stage | None = None
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
        cfg: The configuration for the joint drive.
        stage: The stage where to find the prim. Defaults to None, in which case the
            current stage is used.

    Returns:
        True if the properties were successfully set, False otherwise.

    Raises:
        ValueError: If the input prim path is not valid.
    """
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    applied_schemas_str = str(prim.GetAppliedSchemas())
    if "PhysxTendonAxisAPI" in applied_schemas_str and "PhysxTendonAxisRootAPI" not in applied_schemas_str:
        return False

    # check if prim has joint drive applied on it
    usd_drive_api = UsdPhysics.DriveAPI(prim, drive_api_name)
    if not usd_drive_api:
        usd_drive_api = UsdPhysics.DriveAPI.Apply(prim, drive_api_name)
    # ensure PhysX joint API is applied
    if "PhysxJointAPI" not in applied_schemas_str:
        prim.AddAppliedSchema("PhysxJointAPI")

    # mapping from configuration name to USD attribute name
    cfg_to_usd_map = {
        "max_velocity": "max_joint_velocity",
        "max_effort": "max_force",
        "drive_type": "type",
    }
    # convert to dict
    cfg = cfg.to_dict()

    # check if linear drive
    is_linear_drive = prim.IsA(UsdPhysics.PrismaticJoint)
    # convert values for angular drives from radians to degrees units
    if not is_linear_drive:
        if cfg["max_velocity"] is not None:
            # rad / s --> deg / s
            cfg["max_velocity"] = cfg["max_velocity"] * 180.0 / math.pi
        if cfg["stiffness"] is not None:
            # N-m/rad --> N-m/deg
            cfg["stiffness"] = cfg["stiffness"] * math.pi / 180.0
        if cfg["damping"] is not None:
            # N-m-s/rad --> N-m-s/deg
            cfg["damping"] = cfg["damping"] * math.pi / 180.0

    # set into PhysX API (prim attributes under physxJoint:*)
    for attr_name in ["max_velocity"]:
        value = cfg.pop(attr_name, None)
        usd_attr_name = cfg_to_usd_map[attr_name]
        safe_set_attribute_on_usd_prim(
            prim, f"physxJoint:{to_camel_case(usd_attr_name, 'cC')}", value, camel_case=False
        )
    # set into USD API
    for attr_name, attr_value in cfg.items():
        attr_name = cfg_to_usd_map.get(attr_name, attr_name)
        safe_set_attribute_on_usd_schema(usd_drive_api, attr_name, attr_value, camel_case=True)

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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has fixed tendon applied on it
    applied_schemas = tendon_prim.GetAppliedSchemas()
    if not any("PhysxTendonAxisRootAPI" in s for s in applied_schemas):
        return False

    # resolve all available instances of the schema since it is multi-instance
    cfg = cfg.to_dict()
    for schema_name in applied_schemas:
        if "PhysxTendonAxisRootAPI" not in schema_name:
            continue
        # set into PhysX API by attribute prefix schema_name: (e.g. PhysxTendonAxisRootAPI:default:stiffness)
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_prim(
                tendon_prim,
                f"{schema_name}:{to_camel_case(attr_name, 'cC')}",
                value,
                camel_case=False,
            )
    # success
    return True


"""
Spatial tendon properties.
"""


@apply_nested
def modify_spatial_tendon_properties(
    prim_path: str, cfg: schemas_cfg.SpatialTendonPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Modify PhysX parameters for a spatial tendon attachment prim.

    A `spatial tendon`_ can be used to link multiple degrees of freedom of articulation joints
    through length and limit constraints. For instance, it can be used to set up an equality constraint
    between a driven and passive revolute joints.

    The schema comprises of attributes that belong to the `PhysxTendonAxisRootAPI`_ schema.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _spatial tendon: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxArticulationSpatialTendon.html
    .. _PhysxTendonAxisRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_axis_root_a_p_i.html
    .. _PhysxTendonAttachmentRootAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_root_a_p_i.html
    .. _PhysxTendonAttachmentLeafAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_tendon_attachment_leaf_a_p_i.html

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
        stage = get_current_stage()
    # get USD prim
    tendon_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has spatial tendon applied on it
    applied_schemas = tendon_prim.GetAppliedSchemas()
    has_spatial = any(
        "PhysxTendonAttachmentRootAPI" in s or "PhysxTendonAttachmentLeafAPI" in s for s in applied_schemas
    )
    if not has_spatial:
        return False

    cfg = cfg.to_dict()
    for schema_name in applied_schemas:
        if "PhysxTendonAttachmentRootAPI" not in schema_name and "PhysxTendonAttachmentLeafAPI" not in schema_name:
            continue
        for attr_name, value in cfg.items():
            safe_set_attribute_on_usd_prim(
                tendon_prim,
                f"{schema_name}:{to_camel_case(attr_name, 'cC')}",
                value,
                camel_case=False,
            )
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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

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
    # ensure PhysX deformable body API is applied
    mesh_applied = mesh_prim.GetAppliedSchemas()
    if "PhysxDeformableBodyAPI" not in mesh_applied:
        mesh_prim.AddAppliedSchema("PhysxDeformableBodyAPI")
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
    # get stage handle
    if stage is None:
        stage = get_current_stage()

    # get deformable-body USD prim
    deformable_body_prim = stage.GetPrimAtPath(prim_path)

    # check if the prim is valid and has the deformable-body API
    if not deformable_body_prim.IsValid():
        return False
    if "PhysxDeformableBodyAPI" not in deformable_body_prim.GetAppliedSchemas():
        return False

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

    # ensure PhysX deformable API is applied (set when add_physx_deformable_body runs; apply if missing)
    deformable_applied = deformable_body_prim.GetAppliedSchemas()
    if "PhysxCollisionAPI" not in deformable_applied:
        deformable_body_prim.AddAppliedSchema("PhysxCollisionAPI")
    if "PhysxDeformableAPI" not in deformable_applied:
        deformable_body_prim.AddAppliedSchema("PhysxDeformableAPI")

    # set into PhysX API (prim attributes: physxCollision:* for rest/contact offset, physxDeformable:* for rest)
    for attr_name, value in cfg.items():
        if attr_name in ["rest_offset", "contact_offset"]:
            safe_set_attribute_on_usd_prim(
                deformable_body_prim, f"physxCollision:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
            )
        else:
            safe_set_attribute_on_usd_prim(
                deformable_body_prim, f"physxDeformable:{to_camel_case(attr_name, 'cC')}", value, camel_case=False
            )

    # success
    return True


"""
Collision mesh properties.
"""


def _get_physx_collision_namespace(schema_name: str) -> str:
    """Convert PhysX schema name to attribute namespace used on the prim."""
    if not schema_name:
        raise ValueError("PhysX schema name must be provided for mesh collision properties.")
    schema_name = schema_name.removesuffix("API")
    return schema_name[0].lower() + schema_name[1:]


def _get_usd_mesh_collision_api(api_name: str):
    """Resolve the USD mesh collision API from a string name."""
    if not api_name:
        raise ValueError("USD schema name must be provided for mesh collision properties.")
    usd_api = getattr(UsdPhysics, api_name, None)
    if usd_api is None:
        raise ValueError(f"USD schema '{api_name}' not found in UsdPhysics.")
    return usd_api


def extract_mesh_collision_api_and_attrs(
    cfg: schemas_cfg.MeshCollisionPropertiesCfg,
) -> tuple[tuple[str, Any], dict[str, Any]]:
    """Extract the mesh collision API type/value and custom attributes from the configuration.

    Args:
        cfg: The configuration for the mesh collision properties.

    Returns:
        A tuple of ((api_type, api_value), custom_attrs). api_type is "usd" or "physx";
        api_value is the USD API class (callable) or PhysX schema name string.

    Raises:
        ValueError: When neither USD nor PhysX API can be determined to be used.
    """
    custom_attrs = {
        key: value
        for key, value in cfg.to_dict().items()
        if value is not None and key not in ["usd_api", "physx_api", "mesh_approximation_name"]
    }

    use_usd_api = False
    use_physx_api = False

    if len(custom_attrs) > 0 and type(cfg) in PHYSX_MESH_COLLISION_CFGS:
        use_physx_api = True
    elif len(custom_attrs) == 0:
        if type(cfg) in USD_MESH_COLLISION_CFGS:
            use_usd_api = True
        else:
            use_physx_api = True
    elif len(custom_attrs) > 0 and type(cfg) in USD_MESH_COLLISION_CFGS:
        raise ValueError("Args are specified but the USD Mesh API doesn't support them!")

    if use_usd_api and getattr(cfg, "usd_api", None):
        return ("usd", cfg.usd_api), custom_attrs
    if use_physx_api and getattr(cfg, "physx_api", None):
        return ("physx", cfg.physx_api), custom_attrs
    raise ValueError("Either USD or PhysX API should be used for modifying mesh collision attributes!")


def define_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionPropertiesCfg, stage: Usd.Stage | None = None
):
    """Apply the mesh collision schema on the input prim and set its properties.
    See :func:`modify_collision_mesh_properties` for more details on how the properties are set.
    Args:
        prim_path : The prim path where to apply the mesh collision schema.
        cfg : The configuration for the mesh collision properties.
        stage : The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    Raises:
        ValueError: When the prim path is not valid.
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim path is valid
    if not prim.IsValid():
        raise ValueError(f"Prim path '{prim_path}' is not valid.")

    (api_type, api_value), _ = extract_mesh_collision_api_and_attrs(cfg=cfg)

    if api_type == "usd":
        usd_api_class = _get_usd_mesh_collision_api(api_value)
        if not usd_api_class(prim):
            usd_api_class.Apply(prim)
    else:
        if api_value not in prim.GetAppliedSchemas():
            prim.AddAppliedSchema(api_value)

    modify_mesh_collision_properties(prim_path=prim_path, cfg=cfg, stage=stage)


@apply_nested
def modify_mesh_collision_properties(
    prim_path: str, cfg: schemas_cfg.MeshCollisionPropertiesCfg, stage: Usd.Stage | None = None
) -> bool:
    """Set properties for the mesh collision of a prim.
    These properties are based on either the `Phsyx the `UsdPhysics.MeshCollisionAPI` schema.
    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.
    .. UsdPhysics.MeshCollisionAPI: https://openusd.org/release/api/class_usd_physics_mesh_collision_a_p_i.html
    Args:
        prim_path : The prim path of the rigid body. This prim should be a Mesh prim.
        cfg : The configuration for the mesh collision properties.
        stage : The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    Returns:
        True if the properties were successfully set, False otherwise.
    Raises:
        ValueError: When the mesh approximation name is invalid.
    """
    # obtain stage
    if stage is None:
        stage = get_current_stage()
    # get USD prim
    prim = stage.GetPrimAtPath(prim_path)

    # we need MeshCollisionAPI to set mesh collision approximation attribute
    if not UsdPhysics.MeshCollisionAPI(prim):
        UsdPhysics.MeshCollisionAPI.Apply(prim)
    # convert mesh approximation string to token
    approximation_name = cfg.mesh_approximation_name
    if approximation_name not in MESH_APPROXIMATION_TOKENS:
        raise ValueError(
            f"Invalid mesh approximation name: '{approximation_name}'. "
            f"Valid options are: {list(MESH_APPROXIMATION_TOKENS.keys())}"
        )
    approximation_token = MESH_APPROXIMATION_TOKENS[approximation_name]
    safe_set_attribute_on_usd_schema(
        UsdPhysics.MeshCollisionAPI(prim), "Approximation", approximation_token, camel_case=False
    )

    (api_type, api_value), custom_attrs = extract_mesh_collision_api_and_attrs(cfg=cfg)

    if api_type == "usd":
        usd_api_class = _get_usd_mesh_collision_api(api_value)
        mesh_collision_api = usd_api_class(prim)
        if not mesh_collision_api:
            return False
        for attr_name, value in custom_attrs.items():
            camel_case = attr_name != "Attribute"
            safe_set_attribute_on_usd_schema(mesh_collision_api, attr_name, value, camel_case=camel_case)
    else:
        if api_value not in prim.GetAppliedSchemas():
            return False
        attr_namespace = _get_physx_collision_namespace(api_value)
        for attr_name, value in custom_attrs.items():
            attr_token = attr_name if attr_name == "Attribute" else to_camel_case(attr_name, "cC")
            safe_set_attribute_on_usd_prim(prim, f"{attr_namespace}:{attr_token}", value, camel_case=False)

    # success
    return True
