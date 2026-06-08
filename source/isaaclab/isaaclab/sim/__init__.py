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

from isaaclab.utils.module import lazy_export

_stub_getattr, _stub_dir, __all__ = lazy_export()

# Names that moved out of this package into ``isaaclab_physx.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim`` does not
# require ``isaaclab_physx`` to be installed.
_PHYSX_FORWARDS_SCHEMAS = frozenset({
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
})

# Names that moved out of this package into ``isaaclab_physx.sim.spawners.materials``.
_PHYSX_FORWARDS_MATERIALS = frozenset({
    "DeformableBodyMaterialCfg",
    "RigidBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
    "PhysxRigidBodyMaterialCfg",
    "PhysxDeformableBodyMaterialCfg",
    "PhysxSurfaceDeformableBodyMaterialCfg",
})

_PHYSX_FORWARDS = _PHYSX_FORWARDS_SCHEMAS | _PHYSX_FORWARDS_MATERIALS

# Names that moved out of this package into ``isaaclab_newton.sim.schemas``.
# Resolved lazily on first access so importing ``isaaclab.sim`` does not
# require ``isaaclab_newton`` to be installed.
_NEWTON_FORWARDS = frozenset({
    "MujocoRigidBodyPropertiesCfg",
    "MujocoJointDrivePropertiesCfg",
    "NewtonRigidBodyPropertiesCfg",
    "NewtonJointDrivePropertiesCfg",
    "NewtonCollisionPropertiesCfg",
    "NewtonMeshCollisionPropertiesCfg",
    "NewtonMaterialPropertiesCfg",
    "NewtonArticulationRootPropertiesCfg",
})


def __getattr__(name):
    if name in _PHYSX_FORWARDS_SCHEMAS:
        try:
            from isaaclab_physx.sim.schemas import schemas_cfg as _physx_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.{name}' has moved to 'isaaclab_physx.sim.schemas'."
                " Install the isaaclab_physx extension or update your import. This forwarding"
                " shim is scheduled for removal in 4.0."
            ) from e
        return getattr(_physx_cfg, name)
    if name in _PHYSX_FORWARDS_MATERIALS:
        try:
            from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as _physx_mat_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.{name}' has moved to 'isaaclab_physx.sim.spawners.materials'."
                " Install the isaaclab_physx extension or update your import. This forwarding"
                " shim is scheduled for removal in 4.0."
            ) from e
        return getattr(_physx_mat_cfg, name)
    if name in _NEWTON_FORWARDS:
        try:
            from isaaclab_newton.sim.schemas import schemas_cfg as _newton_cfg
        except ImportError as e:
            raise ImportError(
                f"'isaaclab.sim.{name}' has moved to 'isaaclab_newton.sim.schemas'."
                " Install the isaaclab_newton extension or update your import. This forwarding"
                " shim is scheduled for removal in 4.0."
            ) from e
        return getattr(_newton_cfg, name)
    return _stub_getattr(name)


def __dir__():
    return sorted(set(_stub_dir()) | _PHYSX_FORWARDS | _NEWTON_FORWARDS)
