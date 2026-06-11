# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

__all__ = [
    "MESH_APPROXIMATION_TOKENS",
    "PHYSX_MESH_COLLISION_CFGS",
    "USD_MESH_COLLISION_CFGS",
    "activate_contact_sensors",
    "define_actuator_properties",
    "define_articulation_root_properties",
    "define_collision_properties",
    "define_deformable_body_properties",
    "define_mass_properties",
    "define_mesh_collision_properties",
    "define_rigid_body_properties",
    "modify_articulation_root_properties",
    "modify_collision_properties",
    "modify_deformable_body_properties",
    "modify_fixed_tendon_properties",
    "modify_joint_drive_properties",
    "modify_mass_properties",
    "modify_mesh_collision_properties",
    "modify_rigid_body_properties",
    "modify_spatial_tendon_properties",
    "ArticulationRootBaseCfg",
    "BoundingCubePropertiesCfg",
    "BoundingSpherePropertiesCfg",
    "CollisionBaseCfg",
    "DeformableBodyPropertiesBaseCfg",
    "DeformableBodyPropertiesCfg",
    "JointDriveBaseCfg",
    "MassPropertiesCfg",
    "MeshCollisionBaseCfg",
    "MujocoJointDrivePropertiesCfg",
    "MujocoRigidBodyPropertiesCfg",
    "NewtonArticulationRootPropertiesCfg",
    "NewtonCollisionPropertiesCfg",
    "NewtonJointDrivePropertiesCfg",
    "NewtonMaterialPropertiesCfg",
    "NewtonMeshCollisionPropertiesCfg",
    "NewtonRigidBodyPropertiesCfg",
    "NewtonSDFCollisionPropertiesCfg",
    "RigidBodyBaseCfg",
]

from .schemas import (
    MESH_APPROXIMATION_TOKENS,
    PHYSX_MESH_COLLISION_CFGS,
    USD_MESH_COLLISION_CFGS,
    activate_contact_sensors,
    define_articulation_root_properties,
    define_collision_properties,
    define_deformable_body_properties,
    define_mass_properties,
    define_mesh_collision_properties,
    define_rigid_body_properties,
    modify_articulation_root_properties,
    modify_collision_properties,
    modify_deformable_body_properties,
    modify_fixed_tendon_properties,
    modify_joint_drive_properties,
    modify_mass_properties,
    modify_mesh_collision_properties,
    modify_rigid_body_properties,
    modify_spatial_tendon_properties,
)
from .schemas_actuators import (
    define_actuator_properties,
)
from .schemas_cfg import (
    ArticulationRootBaseCfg,
    BoundingCubePropertiesCfg,
    BoundingSpherePropertiesCfg,
    CollisionBaseCfg,
    DeformableBodyPropertiesBaseCfg,
    DeformableBodyPropertiesCfg,
    JointDriveBaseCfg,
    MassPropertiesCfg,
    MeshCollisionBaseCfg,
    RigidBodyBaseCfg,
)

# Forwarded to isaaclab_newton.sim.schemas via __getattr__ shim
MujocoJointDrivePropertiesCfg = ...
MujocoRigidBodyPropertiesCfg = ...
NewtonArticulationRootPropertiesCfg = ...
NewtonCollisionPropertiesCfg = ...
NewtonJointDrivePropertiesCfg = ...
NewtonMaterialPropertiesCfg = ...
NewtonMeshCollisionPropertiesCfg = ...
NewtonRigidBodyPropertiesCfg = ...
NewtonSDFCollisionPropertiesCfg = ...
