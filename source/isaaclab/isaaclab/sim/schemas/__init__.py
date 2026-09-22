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

import importlib

from isaaclab.utils.module import lazy_export

_stub_getattr, _stub_dir, __all__ = lazy_export()

# Cfg names that moved out of core into a backend package, resolved lazily on first access so
# importing ``isaaclab.sim.schemas`` does not require the backend to be installed.
_MOVED_CFGS: dict[str, frozenset[str]] = {
    "isaaclab_physx": frozenset(
        {
            "RigidBodyPropertiesCfg",
            "JointDrivePropertiesCfg",
            "PhysxRigidBodyPropertiesCfg",
            "PhysxJointDrivePropertiesCfg",
            "CollisionPropertiesCfg",
            "PhysxCollisionPropertiesCfg",
            "DeformableBodyPropertiesCfg",
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
    ),
    "isaaclab_newton": frozenset(
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
    ),
}
_MOVED_CFG_NAMES = frozenset().union(*_MOVED_CFGS.values())


def _import_moved_cfg(name: str, source: str):
    """Resolve a cfg that moved to a backend package, or return None when ``name`` is not forwarded.

    Args:
        name: The attribute name being looked up.
        source: The dotted module name the lookup happened on, for the error message.

    Raises:
        ImportError: If the owning backend package is not installed.
    """
    for package, names in _MOVED_CFGS.items():
        if name in names:
            try:
                module = importlib.import_module(f"{package}.sim.schemas.schemas_cfg")
            except ImportError as e:
                raise ImportError(
                    f"'{source}.{name}' has moved to '{package}.sim.schemas'. Install the {package} extension"
                    " or update your import. This forwarding shim is scheduled for removal in 4.0."
                ) from e
            return getattr(module, name)
    return None


def __getattr__(name):
    value = _import_moved_cfg(name, __name__)
    return _stub_getattr(name) if value is None else value


def __dir__():
    return sorted(set(_stub_dir()) | _MOVED_CFG_NAMES)
