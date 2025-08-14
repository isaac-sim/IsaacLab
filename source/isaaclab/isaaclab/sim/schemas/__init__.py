# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing utilities for schemas used in Omniverse.

We wrap the USD schemas for PhysX and USD Physics in a more convenient API for setting the parameters from
Python. This is done so that configuration objects can define the schema properties to set and make it easier
to tune the physics parameters without requiring to open Omniverse Kit and manually set the parameters into
the respective USD attributes.

.. caution::

    Schema properties cannot be applied on prims that are prototypes as they are read-only prims. This
    particularly affects instanced assets where some of the prims (usually the visual and collision meshes)
    are prototypes so that the instancing can be done efficiently.

    In such cases, it is assumed that the prototypes have sim-ready properties on them that don't need to be modified.
    Trying to set properties into prototypes will throw a warning saying that the prim is a prototype and the
    properties cannot be set.

The schemas are defined in the following links:

* `UsdPhysics schema <https://openusd.org/dev/api/usd_physics_page_front.html>`_
* `PhysxSchema schema <https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/104.2/index.html>`_

Locally, the schemas are defined in the following files:

* ``_isaac_sim/extsPhysics/omni.usd.schema.physics/plugins/UsdPhysics/resources/UsdPhysics/schema.usda``
* ``_isaac_sim/extsPhysics/omni.usd.schema.physx/plugins/PhysxSchema/resources/generatedSchema.usda``

"""

from .schemas import (
    activate_contact_sensors,
    define_articulation_root_properties,
    define_collision_properties,
    define_deformable_body_properties,
    define_mass_properties,
    define_rigid_body_properties,
    modify_articulation_root_properties,
    modify_collision_properties,
    modify_deformable_body_properties,
    modify_fixed_tendon_properties,
    modify_joint_drive_properties,
    modify_mass_properties,
    modify_rigid_body_properties,
    modify_spatial_tendon_properties,
)
from .schemas_cfg import (
    ArticulationRootPropertiesCfg,
    CollisionPropertiesCfg,
    DeformableBodyPropertiesCfg,
    FixedTendonPropertiesCfg,
    JointDrivePropertiesCfg,
    MassPropertiesCfg,
    RigidBodyPropertiesCfg,
    SpatialTendonPropertiesCfg,
)
