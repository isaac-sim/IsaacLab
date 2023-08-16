# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import omni.isaac.core.utils.stage as stage_utils
from pxr import PhysxSchema, Usd, UsdPhysics

from ..utils import apply_nested, safe_set_attribute_on_usd_schema
from . import schemas_cfg


@apply_nested
def set_articulation_root_properties(
    prim_path: str, cfg: schemas_cfg.ArticulationRootPropertiesCfg, stage: Usd.Stage = None
):
    """Set PhysX parameters for an articulation root prim.

    The `articulation root`_ marks the root of an articulation tree. For floating articulations, this should be on
    the root body. For fixed articulations, this API can be on a direct or indirect parent of the root joint
    which is fixed to the world.

    The schema comprises of attributes that belong to the `ArticulationRootAPI`_ and `PhysxArticulationAPI`_.
    schemas. The latter contains the PhysX parameters for the articulation root.

    The properties are applied to the articulation root prim. The common properties (such as solver position
    and velocity iteration counts, sleep threshold, stabilization threshold) take precedence over those specified
    in the rigid body schemas for all the rigid bodies in the articulation.

    .. note::
        This function is decorated with :func:`apply_nested` that set the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _articulation root: https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/Articulations.html
    .. _ArticulationRootAPI: https://openusd.org/dev/api/class_usd_physics_articulation_root_a_p_i.html
    .. _PhysxArticulationAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_articulation_a_p_i.html

    Args:
        prim_path (str): The prim path to the articulation root.
        cfg (schemas_cfg.ArticulationRootPropertiesCfg): The configuration for the articulation root.
        stage (Usd.Stage, optional): The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
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
    # set into physx api
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_articulation_api, attr_name, value)
    # success
    return True


@apply_nested
def set_rigid_body_properties(prim_path: str, cfg: schemas_cfg.RigidBodyPropertiesCfg, stage: Usd.Stage = None):
    """Set PhysX parameters for a rigid body prim.

    A `rigid body`_ is a single body that can be simulated by PhysX. It can be either dynamic or kinematic.
    A dynamic body responds to forces and collisions. A `kinematic body`_ can be moved by the user, but does not
    respond to forces. They are similar to having static bodies that can be moved around.

    The schema comprises of attributes that belong to the `RigidBodyAPI`_ and `PhysxRigidBodyAPI`_.
    schemas. The latter contains the PhysX parameters for the rigid body.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. _rigid body: https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/RigidBodyOverview.html
    .. _kinematic body: https://openusd.org/release/wp_rigid_body_physics.html#kinematic-bodies
    .. _RigidBodyAPI: https://openusd.org/dev/api/class_usd_physics_rigid_body_a_p_i.html
    .. _PhysxRigidBodyAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_rigid_body_a_p_i.html

    Args:
        prim_path (str): The prim path to the rigid body.
        cfg (schemas_cfg.RigidBodyPropertiesCfg): The configuration for the rigid body.
        stage (Usd.Stage, optional): The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
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
        safe_set_attribute_on_usd_schema(usd_rigid_body_api, attr_name, value)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_rigid_body_api, attr_name, value)
    # success
    return True


@apply_nested
def set_collision_properties(prim_path: str, cfg: schemas_cfg.CollisionPropertiesCfg, stage: Usd.Stage = None):
    """Set PhysX properties of collider prim.

    These properties are based on the `UsdPhysics.CollisionAPI` and `PhysxSchema.PhysxCollisionAPI`_ schemas.
    For more information on the properties, please refer to the official documentation.

    Tuning these parameters influence the contact behavior of the rigid body. For more information on
    tune them and their effect on the simulation, please refer to the
    `PhysX documentation <https://nvidia-omniverse.github.io/PhysX/physx/5.2.1/docs/AdvancedCollisionDetection.html>`_.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.CollisionAPI: https://openusd.org/dev/api/class_usd_physics_collision_a_p_i.html
    .. PhysxSchema.PhysxCollisionAPI: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/class_physx_schema_physx_collision_a_p_i.html

    Args:
        prim_path (str): The prim path of parent.
        cfg (schemas_cfg.CollisionPropertiesCfg): The configuration for the collider.
        stage (Usd.Stage, optional): The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    collider_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.CollisionAPI(collider_prim):
        return False
    # retrieve the USD rigid-body api
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
        safe_set_attribute_on_usd_schema(usd_collision_api, attr_name, value)
    # set into PhysX API
    for attr_name, value in cfg.items():
        safe_set_attribute_on_usd_schema(physx_collision_api, attr_name, value)
    # success
    return True


@apply_nested
def set_mass_properties(prim_path: str, cfg: schemas_cfg.MassPropertiesCfg, stage: Usd.Stage = None):
    """Set properties for the mass of a rigid body prim.

    These properties are based on the `UsdPhysics.MassAPI` schema. If the mass is not defined, the density is used
    to compute the mass. However, in that case, a collision approximation of the rigid body is used to
    compute the density. For more information on the properties, please refer to the
    `documentation <https://openusd.org/release/wp_rigid_body_physics.html#body-mass-properties>`_.

    .. caution::

        The mass of an object can be specified in multiple ways and have several conflicting settings
        that are resolved based on precedence. Please make sure to understand the precedence rules
        before using this property.

    .. note::
        This function is decorated with :func:`apply_nested` that sets the properties to all the prims
        (that have the schema applied on them) under the input prim path.

    .. UsdPhysics.MassAPI: https://openusd.org/dev/api/class_usd_physics_mass_a_p_i.html

    Args:
        prim_path (str): The prim path of the rigid body.
        cfg (schemas_cfg.MassPropertiesCfg): The configuration for the mass properties.
        stage (Usd.Stage, optional): The stage where to find the prim. Defaults to None, in which case the
            current stage is used.
    """
    # obtain stage
    if stage is None:
        stage = stage_utils.get_current_stage()
    # get USD prim
    rigid_prim = stage.GetPrimAtPath(prim_path)
    # check if prim has collision applied on it
    if not UsdPhysics.MassAPI(rigid_prim):
        return False
    # retrieve the USD rigid-body api
    usd_physics_mass_api = UsdPhysics.MassAPI(rigid_prim)

    # convert to dict
    cfg = cfg.to_dict()
    # set into USD API
    for attr_name in ["mass", "density"]:
        value = cfg.pop(attr_name, None)
        safe_set_attribute_on_usd_schema(usd_physics_mass_api, attr_name, value)
    # success
    return True
