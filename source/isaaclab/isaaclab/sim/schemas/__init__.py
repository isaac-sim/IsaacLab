# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

from isaaclab.utils.module import lazy_export

_stub_getattr, _stub_dir, __all__ = lazy_export()

# Names that moved out of this module into ``isaaclab_physx.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim.schemas`` does
# not require ``isaaclab_physx`` to be installed.
_PHYSX_FORWARDS = frozenset(
    {
        "RigidBodyPropertiesCfg",
        "JointDrivePropertiesCfg",
        "PhysxRigidBodyPropertiesCfg",
        "PhysxJointDrivePropertiesCfg",
        "CollisionPropertiesCfg",
        "PhysxCollisionPropertiesCfg",
        "DeformableBodyPropertiesCfg",
        "PhysxDeformableCollisionPropertiesCfg",
        "PhysxDeformableBodyPropertiesCfg",
        "ArticulationRootPropertiesCfg",
        "PhysxArticulationRootPropertiesCfg",
        "MeshCollisionPropertiesCfg",
        "ConvexHullPropertiesCfg",
        "ConvexDecompositionPropertiesCfg",
        "TriangleMeshPropertiesCfg",
        "TriangleMeshSimplificationPropertiesCfg",
        "SDFMeshPropertiesCfg",
        "PhysxConvexHullPropertiesCfg",
        "PhysxConvexDecompositionPropertiesCfg",
        "PhysxTriangleMeshPropertiesCfg",
        "PhysxTriangleMeshSimplificationPropertiesCfg",
        "PhysxSDFMeshPropertiesCfg",
        "FixedTendonPropertiesCfg",
        "SpatialTendonPropertiesCfg",
        "PhysxFixedTendonPropertiesCfg",
        "PhysxSpatialTendonPropertiesCfg",
    }
)

# Names that moved out of this module into ``isaaclab_newton.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim.schemas`` does
# not require ``isaaclab_newton`` to be installed.
_NEWTON_FORWARDS = frozenset(
    {
        "MujocoRigidBodyPropertiesCfg",
        "MujocoJointDrivePropertiesCfg",
        "NewtonRigidBodyPropertiesCfg",
        "NewtonJointDrivePropertiesCfg",
        "NewtonCollisionPropertiesCfg",
        "NewtonMeshCollisionPropertiesCfg",
        "NewtonMaterialPropertiesCfg",
        "NewtonArticulationRootPropertiesCfg",
        "NewtonSDFCollisionPropertiesCfg",
    }
)


def __getattr__(name):
    if name in _PHYSX_FORWARDS:
        try:
            from isaaclab_physx.sim.schemas import schemas_cfg as _physx_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.schemas.{name}' has moved to 'isaaclab_physx.sim.schemas'."
                " Install the isaaclab_physx extension or update your import. This forwarding"
                " shim is scheduled for removal in 4.0."
            ) from e
        return getattr(_physx_cfg, name)
    if name in _NEWTON_FORWARDS:
        try:
            from isaaclab_newton.sim.schemas import schemas_cfg as _newton_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.schemas.{name}' has moved to 'isaaclab_newton.sim.schemas'."
                " Install the isaaclab_newton extension or update your import. This forwarding"
                " shim is scheduled for removal in 4.0."
            ) from e
        return getattr(_newton_cfg, name)
    return _stub_getattr(name)


def __dir__():
    return sorted(set(_stub_dir()) | _PHYSX_FORWARDS | _NEWTON_FORWARDS)
