# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package containing simulation-specific functionalities.

These include:

* Ability to spawn different objects and materials into Omniverse
* Define and modify various schemas on USD prims
* Converters to obtain USD file from other file formats (such as URDF, OBJ, STL, FBX)
* Utility class to control the simulator

.. note::
    Currently, only a subset of all possible schemas and prims in Omniverse are supported.
    We are expanding the these set of functions on a need basis. In case, there are
    specific prims or schemas that you would like to include, please open an issue on GitHub
    as a feature request elaborating on the required application.

To make it convenient to use the module, we recommend importing the module as follows:

.. code-block:: python

    import isaaclab.sim as sim_utils

"""

import importlib

from isaaclab.utils.module import lazy_export

_stub_getattr, _stub_dir, __all__ = lazy_export()

# Names that moved out of this package into ``isaaclab_physx.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim`` does not
# require ``isaaclab_physx`` to be installed.
_PHYSX_FORWARDS_SCHEMAS = frozenset(
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
)

# Names that moved out of this package into ``isaaclab_physx.sim.spawners.materials``.
_PHYSX_FORWARDS_MATERIALS = frozenset(
    {
        "DeformableBodyMaterialCfg",
        "RigidBodyMaterialCfg",
        "SurfaceDeformableBodyMaterialCfg",
        "PhysxRigidBodyMaterialCfg",
        "PhysxDeformableBodyMaterialCfg",
        "PhysxSurfaceDeformableBodyMaterialCfg",
    }
)

_PHYSX_FORWARDS = _PHYSX_FORWARDS_SCHEMAS | _PHYSX_FORWARDS_MATERIALS

# Names that moved out of this package into ``isaaclab_newton.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim`` does not
# require ``isaaclab_newton`` to be installed.
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


# (forwarded names, public package the names moved to, module defining them, extension to install)
_FORWARDS = (
    (_PHYSX_FORWARDS_SCHEMAS, "isaaclab_physx.sim.schemas", "isaaclab_physx.sim.schemas.schemas_cfg", "isaaclab_physx"),
    (
        _PHYSX_FORWARDS_MATERIALS,
        "isaaclab_physx.sim.spawners.materials",
        "isaaclab_physx.sim.spawners.materials.physics_materials_cfg",
        "isaaclab_physx",
    ),
    (_NEWTON_FORWARDS, "isaaclab_newton.sim.schemas", "isaaclab_newton.sim.schemas.schemas_cfg", "isaaclab_newton"),
)


def __getattr__(name):
    for names, package, module_name, extension in _FORWARDS:
        if name not in names:
            continue
        try:
            module = importlib.import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.{name}' has moved to '{package}'. Install the {extension} extension or update"
                " your import. This forwarding shim is scheduled for removal in 4.0."
            ) from e
        return getattr(module, name)
    return _stub_getattr(name)


def __dir__():
    return sorted(set(_stub_dir()) | _PHYSX_FORWARDS | _NEWTON_FORWARDS)
