# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the forwarding shims resolve the PhysX cfgs that were relocated to
:mod:`isaaclab_physx`. Covers both the schema cfgs (in :mod:`isaaclab.sim.schemas`) and the
material cfgs (in :mod:`isaaclab.sim.spawners.materials`).

These tests do not require Isaac Sim — only Python import semantics.
"""

import warnings

import pytest
from isaaclab_physx.sim.schemas import schemas_cfg as physx_cfg
from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as physx_mat_cfg

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
import isaaclab.sim.schemas.schemas_cfg as schemas_cfg_submodule
import isaaclab.sim.spawners.materials as materials
import isaaclab.sim.spawners.materials.physics_materials_cfg as materials_cfg_submodule

FORWARDED_NAMES = [
    "RigidBodyPropertiesCfg",
    "JointDrivePropertiesCfg",
    "PhysxRigidBodyPropertiesCfg",
    "PhysxJointDrivePropertiesCfg",
    "CollisionPropertiesCfg",
    "PhysxCollisionPropertiesCfg",
    "DeformableBodyPropertiesCfg",
    "PhysxDeformableBodyPropertiesCfg",
    "PhysxDeformableCollisionPropertiesCfg",
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
]

DEPRECATED_FORWARDED_NAMES = [
    "RigidBodyPropertiesCfg",
    "JointDrivePropertiesCfg",
    "CollisionPropertiesCfg",
    "DeformableBodyPropertiesCfg",
    "ArticulationRootPropertiesCfg",
    "MeshCollisionPropertiesCfg",
    "ConvexHullPropertiesCfg",
    "ConvexDecompositionPropertiesCfg",
    "TriangleMeshPropertiesCfg",
    "TriangleMeshSimplificationPropertiesCfg",
    "SDFMeshPropertiesCfg",
    "FixedTendonPropertiesCfg",
    "SpatialTendonPropertiesCfg",
]

FORWARDED_MATERIAL_NAMES = [
    "DeformableBodyMaterialCfg",
    "RigidBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
    "PhysxRigidBodyMaterialCfg",
    "PhysxDeformableBodyMaterialCfg",
    "PhysxSurfaceDeformableBodyMaterialCfg",
]

DEPRECATED_FORWARDED_MATERIAL_NAMES = [
    "DeformableBodyMaterialCfg",
    "RigidBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
]


@pytest.mark.parametrize("name", FORWARDED_NAMES)
def test_schemas_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.schemas.<name>`` resolves to the same class object as the one in
    ``isaaclab_physx.sim.schemas.schemas_cfg``."""
    assert getattr(schemas, name) is getattr(physx_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_NAMES)
def test_sim_namespace_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.<name>`` (i.e. ``sim_utils.<name>``) resolves to the same class object."""
    assert getattr(sim_utils, name) is getattr(physx_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_NAMES)
def test_schemas_cfg_submodule_shim_resolves_to_physx_class(name):
    """``from isaaclab.sim.schemas.schemas_cfg import <name>`` (direct submodule import path)
    resolves to the same class object as the relocated definition."""
    assert getattr(schemas_cfg_submodule, name) is getattr(physx_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_MATERIAL_NAMES)
def test_materials_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.spawners.materials.<name>`` resolves to the same class object as the
    one in ``isaaclab_physx.sim.spawners.materials.physics_materials_cfg``."""
    assert getattr(materials, name) is getattr(physx_mat_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_MATERIAL_NAMES)
def test_materials_cfg_submodule_shim_resolves_to_physx_class(name):
    """``from isaaclab.sim.spawners.materials.physics_materials_cfg import <name>`` (direct
    submodule import path) resolves to the same class object as the relocated definition."""
    assert getattr(materials_cfg_submodule, name) is getattr(physx_mat_cfg, name)


@pytest.mark.parametrize("name", FORWARDED_MATERIAL_NAMES)
def test_sim_namespace_material_shim_resolves_to_physx_class(name):
    """``isaaclab.sim.<name>`` (i.e. ``sim_utils.<name>``) resolves to the relocated material class."""
    assert getattr(sim_utils, name) is getattr(physx_mat_cfg, name)


def test_deprecated_alias_emits_deprecation_warning():
    """Instantiating ``RigidBodyPropertiesCfg`` via the shim still emits ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schemas.RigidBodyPropertiesCfg()
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


@pytest.mark.parametrize("name", DEPRECATED_FORWARDED_NAMES)
def test_deprecated_aliases_emit_deprecation_warning(name):
    """Instantiating each deprecated forwarded alias via the shim emits exactly one
    ``DeprecationWarning``."""
    cls = getattr(schemas, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"{name}: expected one DeprecationWarning, got {len(deprecations)}"


@pytest.mark.parametrize("name", DEPRECATED_FORWARDED_MATERIAL_NAMES)
def test_deprecated_material_aliases_emit_deprecation_warning(name):
    """Instantiating a deprecated material alias via the shim still emits ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(materials, name)()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"{name}: expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


def test_new_class_does_not_emit_deprecation_warning():
    """Instantiating ``PhysxRigidBodyPropertiesCfg`` directly does NOT emit ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        schemas.PhysxRigidBodyPropertiesCfg()
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_new_material_class_does_not_emit_deprecation_warning():
    """Instantiating ``PhysxRigidBodyMaterialCfg`` directly does NOT emit ``DeprecationWarning``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        materials.PhysxRigidBodyMaterialCfg()
    assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_deformable_component_cfg_is_not_forwarded_from_core():
    """Component deformable cfgs are backend-owned and not forwarded from ``isaaclab``."""
    assert not hasattr(schemas, "PhysXDeformableBodyPropertiesCfg")
    assert not hasattr(sim_utils, "PhysXDeformableBodyPropertiesCfg")
    assert not hasattr(schemas_cfg_submodule, "PhysXDeformableBodyPropertiesCfg")


def test_dir_lists_forwarded_names():
    """``dir(isaaclab.sim.schemas)`` includes the forwarded names so IDE autocomplete works."""
    listing = dir(schemas)
    for name in FORWARDED_NAMES:
        assert name in listing


def test_dir_lists_forwarded_material_names():
    """``dir(isaaclab.sim.spawners.materials)`` includes the forwarded names."""
    listing = dir(materials)
    for name in FORWARDED_MATERIAL_NAMES:
        assert name in listing
