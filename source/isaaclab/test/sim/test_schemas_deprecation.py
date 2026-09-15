# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the legacy schema cfg classes and schema writers are deprecated but still work.

The fragment API (one ``@configclass`` per USD applied schema) replaces the inheritance-based
``*PropertiesCfg`` / ``*BaseCfg`` classes and the ``define_*`` / ``modify_*`` writers. Both APIs
coexist: the legacy names must emit a ``DeprecationWarning`` that names their concrete
replacement, and must keep behaving exactly as before.

These tests run on an in-memory USD stage and do not launch Isaac Sim / Kit.
"""

import dataclasses
import inspect
import warnings

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim.schemas as schemas
import isaaclab.sim.schemas.schemas_cfg as schemas_cfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


# Legacy cfg class -> a distinctive substring of the replacement named in its warning.
# Deformable cfgs are intentionally absent: their fragment families do not exist yet, so the
# legacy deformable path cannot be deprecated. Tendon and material cfgs are out of scope here.
DEPRECATED_CORE_CFGS = {
    "MassPropertiesCfg": "MassCfg",
    "RigidBodyBaseCfg": "UsdPhysicsRigidBodyCfg",
    "CollisionBaseCfg": "UsdPhysicsCollisionCfg",
    "ArticulationRootBaseCfg": "PhysxArticulationCfg",
    "JointDriveBaseCfg": "UsdPhysicsDriveCfg",
    "MeshCollisionBaseCfg": "UsdPhysicsMeshCollisionCfg",
    "BoundingCubePropertiesCfg": "boundingCube",
    "BoundingSpherePropertiesCfg": "boundingSphere",
}

DEPRECATED_PHYSX_CFGS = {
    "PhysxRigidBodyPropertiesCfg": "PhysxRigidBodyCfg",
    "RigidBodyPropertiesCfg": "PhysxRigidBodyCfg",
    "PhysxJointDrivePropertiesCfg": "PhysxJointCfg",
    "JointDrivePropertiesCfg": "PhysxJointCfg",
    "PhysxCollisionPropertiesCfg": "PhysxCollisionCfg",
    "CollisionPropertiesCfg": "PhysxCollisionCfg",
    "PhysxArticulationRootPropertiesCfg": "PhysxArticulationCfg",
    "ArticulationRootPropertiesCfg": "PhysxArticulationCfg",
    "MeshCollisionPropertiesCfg": "UsdPhysicsMeshCollisionCfg",
    "PhysxConvexHullPropertiesCfg": "PhysxConvexHullCfg",
    "ConvexHullPropertiesCfg": "PhysxConvexHullCfg",
    "PhysxConvexDecompositionPropertiesCfg": "PhysxConvexDecompositionCfg",
    "ConvexDecompositionPropertiesCfg": "PhysxConvexDecompositionCfg",
    "PhysxTriangleMeshPropertiesCfg": "PhysxTriangleMeshCfg",
    "TriangleMeshPropertiesCfg": "PhysxTriangleMeshCfg",
    "PhysxTriangleMeshSimplificationPropertiesCfg": "PhysxTriangleMeshSimplificationCfg",
    "TriangleMeshSimplificationPropertiesCfg": "PhysxTriangleMeshSimplificationCfg",
    "PhysxSDFMeshPropertiesCfg": "PhysxSDFMeshCfg",
    "SDFMeshPropertiesCfg": "PhysxSDFMeshCfg",
}

DEPRECATED_NEWTON_CFGS = {
    "NewtonRigidBodyPropertiesCfg": "MujocoRigidBodyCfg",
    "MujocoRigidBodyPropertiesCfg": "MujocoRigidBodyCfg",
    "NewtonJointDrivePropertiesCfg": "MujocoJointCfg",
    "MujocoJointDrivePropertiesCfg": "MujocoJointCfg",
    "NewtonCollisionPropertiesCfg": "NewtonCollisionCfg",
    "NewtonMeshCollisionPropertiesCfg": "NewtonMeshCollisionCfg",
    "NewtonSDFCollisionPropertiesCfg": "NewtonSDFCollisionCfg",
    "NewtonArticulationRootPropertiesCfg": "NewtonArticulationCfg",
}

# Replacement fragments, which must stay silent.
CURRENT_CORE_FRAGMENTS = [
    "MassCfg",
    "UsdPhysicsRigidBodyCfg",
    "UsdPhysicsCollisionCfg",
    "UsdPhysicsDriveCfg",
    "UsdPhysicsMeshCollisionCfg",
]

CURRENT_PHYSX_FRAGMENTS = [
    "PhysxRigidBodyCfg",
    "PhysxJointCfg",
    "PhysxCollisionCfg",
    "PhysxArticulationCfg",
    "PhysxConvexHullCfg",
    "PhysxConvexDecompositionCfg",
    "PhysxTriangleMeshCfg",
    "PhysxTriangleMeshSimplificationCfg",
    "PhysxSDFMeshCfg",
]

CURRENT_NEWTON_FRAGMENTS = [
    "MujocoRigidBodyCfg",
    "MujocoJointCfg",
    "NewtonCollisionCfg",
    "NewtonMeshCollisionCfg",
    "NewtonSDFCollisionCfg",
    "NewtonArticulationCfg",
]

# Legacy writer -> the fragment-based writer named in its warning. ``define_deformable_*`` /
# ``modify_deformable_body_properties`` are excluded for the same reason as the deformable cfgs.
DEPRECATED_WRITERS = {
    "define_articulation_root_properties": "apply_articulation_root_properties",
    "modify_articulation_root_properties": "apply_articulation_root_properties",
    "define_rigid_body_properties": "apply_rigid_body_properties",
    "modify_rigid_body_properties": "apply_rigid_body_properties",
    "define_collision_properties": "apply_collision_properties",
    "modify_collision_properties": "apply_collision_properties",
    "define_mass_properties": "apply_mass_properties",
    "modify_mass_properties": "apply_mass_properties",
    "modify_joint_drive_properties": "apply_joint_drive_properties",
    "modify_fixed_tendon_properties": "apply_fixed_tendon_properties",
    "modify_spatial_tendon_properties": "apply_spatial_tendon_properties",
    "define_mesh_collision_properties": "apply_mesh_collision_properties",
    "modify_mesh_collision_properties": "apply_mesh_collision_properties",
}

EXCLUDED_DEFORMABLE_SYMBOLS = [
    "DeformableBodyPropertiesBaseCfg",
    "define_deformable_body_properties",
    "modify_deformable_body_properties",
    "define_deformable_curve_properties",
]


def _physx_cfgs():
    """Import the PhysX schema cfg module, skipping the test when the extension is absent."""
    return pytest.importorskip("isaaclab_physx.sim.schemas.schemas_cfg")


def _newton_cfgs():
    """Import the Newton schema cfg module, skipping the test when the extension is absent."""
    return pytest.importorskip("isaaclab_newton.sim.schemas.schemas_cfg")


def _deprecations(func):
    """Call ``func`` and return the ``DeprecationWarning`` instances it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func()
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


def _assert_deprecated_once(cls, replacement: str) -> None:
    """Instantiating ``cls`` raises exactly one deprecation naming ``replacement`` and 4.0."""
    deprecations = _deprecations(cls)
    assert len(deprecations) == 1, f"{cls.__name__}: expected one DeprecationWarning, got {len(deprecations)}"
    message = str(deprecations[0].message)
    assert cls.__name__ in message
    assert replacement in message, f"{cls.__name__}: warning does not name '{replacement}': {message}"
    assert "4.0" in message, f"{cls.__name__}: warning does not state the removal version: {message}"


"""
Deprecated cfg classes.
"""


@pytest.mark.parametrize("name,replacement", sorted(DEPRECATED_CORE_CFGS.items()))
def test_legacy_core_cfg_warns_on_instantiation(name, replacement):
    """Each legacy core cfg warns once, naming its fragment replacement."""
    _assert_deprecated_once(getattr(schemas_cfg, name), replacement)


@pytest.mark.parametrize("name,replacement", sorted(DEPRECATED_PHYSX_CFGS.items()))
def test_legacy_physx_cfg_warns_on_instantiation(name, replacement):
    """Each legacy PhysX cfg warns once, naming its fragment replacement."""
    _assert_deprecated_once(getattr(_physx_cfgs(), name), replacement)


@pytest.mark.parametrize("name,replacement", sorted(DEPRECATED_NEWTON_CFGS.items()))
def test_legacy_newton_cfg_warns_on_instantiation(name, replacement):
    """Each legacy Newton cfg warns once, naming its fragment replacement."""
    _assert_deprecated_once(getattr(_newton_cfgs(), name), replacement)


@pytest.mark.parametrize("name", CURRENT_CORE_FRAGMENTS)
def test_core_fragment_does_not_warn(name):
    """The replacement core fragments must not warn."""
    assert _deprecations(getattr(schemas_cfg, name)) == []


@pytest.mark.parametrize("name", CURRENT_PHYSX_FRAGMENTS)
def test_physx_fragment_does_not_warn(name):
    """The replacement PhysX fragments must not warn."""
    assert _deprecations(getattr(_physx_cfgs(), name)) == []


@pytest.mark.parametrize("name", CURRENT_NEWTON_FRAGMENTS)
def test_newton_fragment_does_not_warn(name):
    """The replacement Newton fragments must not warn."""
    assert _deprecations(getattr(_newton_cfgs(), name)) == []


@pytest.mark.parametrize("name", EXCLUDED_DEFORMABLE_SYMBOLS)
def test_deformable_symbol_is_not_deprecated(name):
    """Deformable symbols stay undeprecated until their fragment families land."""
    symbol = getattr(schemas, name)
    if inspect.isclass(symbol):
        assert _deprecations(symbol) == []
    else:
        assert ".. deprecated::" not in (symbol.__doc__ or "")


def test_legacy_cfg_import_does_not_warn():
    """Importing the schema modules must not warn: only construction is deprecated."""
    import importlib  # noqa: PLC0415

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(schemas_cfg)
    assert [w for w in caught if issubclass(w.category, DeprecationWarning)] == []


"""
Deprecated cfg classes keep working.
"""


def test_legacy_cfg_keeps_field_values():
    """The deprecation wrapper must forward every constructor argument unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=2.5, density=1200.0)
    assert cfg.mass == 2.5
    assert cfg.density == 1200.0
    assert [f.name for f in dataclasses.fields(cfg)] == ["mass", "density"]


def test_legacy_cfg_keeps_configclass_helpers():
    """``to_dict`` / ``copy`` / ``replace`` must survive the deprecation wrapper."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=2.5, density=1200.0)
        assert cfg.to_dict()["mass"] == 2.5
        assert cfg.copy().density == 1200.0
        assert cfg.replace(mass=4.0).mass == 4.0
        assert dataclasses.replace(cfg, mass=6.0).mass == 6.0


def test_legacy_cfg_keeps_init_signature():
    """``inspect.signature`` must still report the dataclass fields, not ``*args, **kwargs``."""
    parameters = inspect.signature(schemas_cfg.MassPropertiesCfg.__init__).parameters
    assert list(parameters) == ["self", "mass", "density"]


def test_legacy_cfg_keeps_field_alias_forwarding():
    """The renamed-field aliases on the legacy joint-drive cfg still forward."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.JointDriveBaseCfg(max_effort=80.0, max_velocity=5.0)
    assert cfg.max_force == 80.0
    assert cfg.max_joint_velocity == 5.0


def test_legacy_cfg_subclass_warns_only_for_itself():
    """A legacy subclass warns once for its own name, not once per legacy base."""
    physx_cfg = _physx_cfgs()
    deprecations = _deprecations(physx_cfg.RigidBodyPropertiesCfg)
    assert len(deprecations) == 1
    assert "RigidBodyPropertiesCfg is deprecated" in str(deprecations[0].message)


"""
Deprecated writers.
"""


def _stage_with_rigid_body() -> tuple[Usd.Stage, str]:
    """Return an in-memory stage carrying one rigid-body prim, and that prim's path."""
    stage = Usd.Stage.CreateInMemory()
    prim_path = "/World/Body"
    prim = UsdGeom.Cube.Define(stage, prim_path).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim)
    return stage, prim_path


@pytest.mark.parametrize("name,replacement", sorted(DEPRECATED_WRITERS.items()))
def test_legacy_writer_documents_its_replacement(name, replacement):
    """Every legacy writer carries a ``.. deprecated::`` note naming its fragment writer."""
    doc = getattr(schemas, name).__doc__ or ""
    assert ".. deprecated:: 3.0" in doc, f"{name}: missing deprecation directive"
    assert replacement in doc, f"{name}: docstring does not name '{replacement}'"
    assert "removed" in doc and "4.0" in doc, f"{name}: docstring does not state the removal version"


def test_modify_mass_properties_warns_and_writes():
    """The legacy mass writer warns once and still authors ``physics:mass``."""
    stage, prim_path = _stage_with_rigid_body()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=3.0)

    deprecations = _deprecations(lambda: schemas.modify_mass_properties(prim_path, cfg, stage))
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    assert "modify_mass_properties is deprecated" in message
    assert "apply_mass_properties" in message
    assert "4.0" in message
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(3.0)


def test_define_mass_properties_warns_once_despite_delegation():
    """``define_*`` delegates to ``modify_*`` but the caller still sees a single warning."""
    stage, prim_path = _stage_with_rigid_body()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=7.0)

    deprecations = _deprecations(lambda: schemas.define_mass_properties(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert "define_mass_properties is deprecated" in str(deprecations[0].message)
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(7.0)


def test_modify_rigid_body_properties_warns_and_writes():
    """The legacy rigid-body writer warns once and still authors ``physics:kinematicEnabled``."""
    stage, prim_path = _stage_with_rigid_body()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.RigidBodyBaseCfg(kinematic_enabled=True)

    deprecations = _deprecations(lambda: schemas.modify_rigid_body_properties(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert "apply_rigid_body_properties" in str(deprecations[0].message)
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:kinematicEnabled").Get() is True


def test_apply_mass_properties_does_not_warn():
    """The fragment writer is the replacement and must stay silent."""
    stage, prim_path = _stage_with_rigid_body()
    fragment = schemas_cfg.MassCfg(mass=5.0)

    deprecations = _deprecations(lambda: schemas.apply_mass_properties(prim_path, [fragment], stage=stage))
    assert deprecations == []
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(5.0)
