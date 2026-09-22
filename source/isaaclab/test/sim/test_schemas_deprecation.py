# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests that the legacy schema cfg classes and schema writers are deprecated but still work.

The fragment API (one ``@configclass`` per USD applied schema) replaces the inheritance-based
``*PropertiesCfg`` / ``*BaseCfg`` classes and the ``define_*`` / ``modify_*`` writers. Both APIs
coexist: the legacy names must emit a ``DeprecationWarning`` that names their concrete
replacement, keep behaving exactly as before, and keep resolving through the forwarding shims for
the cfgs relocated to the backend packages.

These tests run on an in-memory USD stage and do not launch Isaac Sim / Kit.
"""

import dataclasses
import inspect
import json
import math
import subprocess
import sys
import warnings

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
import isaaclab.sim.schemas.schemas_cfg as schemas_cfg

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


# Legacy cfg class -> every fragment its warning must name, plus any legacy field that has no
# fragment at all and therefore has to be called out explicitly. A legacy class bundles several
# USD namespaces, so a warning naming only the backend-specific fragment would tell the user to
# drop the properties the class inherits -- these expectations are the whole point of the test.
# Deformable cfgs are intentionally absent: their fragment families do not exist yet, so the
# legacy deformable path cannot be deprecated. Tendon and material cfgs are out of scope here.
_RIGID_BODY = ("UsdPhysicsRigidBodyCfg", "PhysxRigidBodyCfg")
_COLLISION = ("UsdPhysicsCollisionCfg", "PhysxCollisionCfg", "mesh_collision_property")
_JOINT_DRIVE = ("UsdPhysicsDriveCfg", "PhysxJointCfg", "ensure_drives_exist")
_ARTICULATION = ("PhysxArticulationCfg", "fix_root_link")

DEPRECATED_CORE_CFGS = {
    "MassPropertiesCfg": ("MassCfg",),
    "RigidBodyBaseCfg": _RIGID_BODY,
    "CollisionBaseCfg": _COLLISION,
    "ArticulationRootBaseCfg": _ARTICULATION,
    "JointDriveBaseCfg": _JOINT_DRIVE,
    "MeshCollisionBaseCfg": ("UsdPhysicsMeshCollisionCfg",),
    "BoundingCubePropertiesCfg": ("UsdPhysicsMeshCollisionCfg", "boundingCube"),
    "BoundingSpherePropertiesCfg": ("UsdPhysicsMeshCollisionCfg", "boundingSphere"),
}

DEPRECATED_PHYSX_CFGS = {
    "PhysxRigidBodyPropertiesCfg": _RIGID_BODY,
    "RigidBodyPropertiesCfg": _RIGID_BODY,
    "PhysxJointDrivePropertiesCfg": _JOINT_DRIVE,
    "JointDrivePropertiesCfg": _JOINT_DRIVE,
    "PhysxCollisionPropertiesCfg": _COLLISION,
    "CollisionPropertiesCfg": _COLLISION,
    "PhysxArticulationRootPropertiesCfg": _ARTICULATION,
    "ArticulationRootPropertiesCfg": _ARTICULATION,
    "MeshCollisionPropertiesCfg": ("UsdPhysicsMeshCollisionCfg",),
    "PhysxConvexHullPropertiesCfg": ("PhysxConvexHullCfg",),
    "ConvexHullPropertiesCfg": ("PhysxConvexHullCfg",),
    "PhysxConvexDecompositionPropertiesCfg": ("PhysxConvexDecompositionCfg",),
    "ConvexDecompositionPropertiesCfg": ("PhysxConvexDecompositionCfg",),
    "PhysxTriangleMeshPropertiesCfg": ("PhysxTriangleMeshCfg",),
    "TriangleMeshPropertiesCfg": ("PhysxTriangleMeshCfg",),
    "PhysxTriangleMeshSimplificationPropertiesCfg": ("PhysxTriangleMeshSimplificationCfg",),
    "TriangleMeshSimplificationPropertiesCfg": ("PhysxTriangleMeshSimplificationCfg",),
    "PhysxSDFMeshPropertiesCfg": ("PhysxSDFMeshCfg",),
    "SDFMeshPropertiesCfg": ("PhysxSDFMeshCfg",),
}

DEPRECATED_NEWTON_CFGS = {
    "NewtonRigidBodyPropertiesCfg": _RIGID_BODY,
    "MujocoRigidBodyPropertiesCfg": _RIGID_BODY + ("MujocoRigidBodyCfg",),
    "NewtonJointDrivePropertiesCfg": _JOINT_DRIVE,
    "MujocoJointDrivePropertiesCfg": _JOINT_DRIVE + ("MujocoJointCfg",),
    "NewtonCollisionPropertiesCfg": _COLLISION + ("NewtonCollisionCfg",),
    "NewtonMeshCollisionPropertiesCfg": _COLLISION
    + ("NewtonCollisionCfg", "UsdPhysicsMeshCollisionCfg", "NewtonMeshCollisionCfg"),
    "NewtonSDFCollisionPropertiesCfg": _COLLISION + ("NewtonCollisionCfg", "NewtonSDFCollisionCfg"),
    "NewtonArticulationRootPropertiesCfg": _ARTICULATION + ("NewtonArticulationCfg",),
}

# Deprecated aliases that only exist as shims; their warning names no fragment.
DEPRECATED_SHIM_ONLY_CFGS = ["FixedTendonPropertiesCfg", "SpatialTendonPropertiesCfg", "DeformableBodyPropertiesCfg"]

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

# Cfg names that moved to a backend package but keep resolving through the core shims.
FORWARDED_PHYSX_NAMES = sorted(DEPRECATED_PHYSX_CFGS) + [
    "DeformableBodyPropertiesCfg",
    "PhysxDeformableBodyPropertiesCfg",
    "FixedTendonPropertiesCfg",
    "SpatialTendonPropertiesCfg",
    "PhysxFixedTendonPropertiesCfg",
    "PhysxSpatialTendonPropertiesCfg",
]
FORWARDED_NEWTON_NAMES = sorted(DEPRECATED_NEWTON_CFGS) + ["NewtonMaterialPropertiesCfg"]


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


def _legacy(cls, **kwargs):
    """Instantiate a legacy cfg without its deprecation warning polluting the test."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return cls(**kwargs)


def _assert_deprecated_once(cls, expected: tuple[str, ...]) -> None:
    """Instantiating ``cls`` raises one deprecation naming every entry of ``expected``, and the removal version."""
    deprecations = _deprecations(cls)
    assert len(deprecations) == 1, f"{cls.__name__}: expected one DeprecationWarning, got {len(deprecations)}"
    message = str(deprecations[0].message)
    assert cls.__name__ in message
    missing = [name for name in expected if name not in message]
    assert not missing, f"{cls.__name__}: warning omits {missing}: {message}"
    assert "3.2" in message, f"{cls.__name__}: warning does not state the removal version: {message}"


def _api_schemas(prim: Usd.Prim) -> set[str]:
    """Applied API schema names including unregistered token schemas."""
    return set(prim.GetPrimTypeInfo().GetAppliedAPISchemas())


"""
Deprecated cfg classes.
"""


@pytest.mark.parametrize("name,expected", sorted(DEPRECATED_CORE_CFGS.items()))
def test_legacy_core_cfg_warns_on_instantiation(name, expected):
    """Each legacy core cfg warns once, naming every fragment its fields need."""
    _assert_deprecated_once(getattr(schemas_cfg, name), expected)


@pytest.mark.parametrize("name,expected", sorted(DEPRECATED_PHYSX_CFGS.items()))
def test_legacy_physx_cfg_warns_on_instantiation(name, expected):
    """Each legacy PhysX cfg warns once, naming every fragment its fields need."""
    _assert_deprecated_once(getattr(_physx_cfgs(), name), expected)


@pytest.mark.parametrize("name,expected", sorted(DEPRECATED_NEWTON_CFGS.items()))
def test_legacy_newton_cfg_warns_on_instantiation(name, expected):
    """Each legacy Newton cfg warns once, naming every fragment its fields need."""
    _assert_deprecated_once(getattr(_newton_cfgs(), name), expected)


@pytest.mark.parametrize("name", DEPRECATED_SHIM_ONLY_CFGS)
def test_deprecated_shim_alias_warns_once(name):
    _physx_cfgs()
    deprecations = _deprecations(getattr(schemas, name))
    assert len(deprecations) == 1, f"{name}: expected one DeprecationWarning, got {len(deprecations)}"
    assert "3.2" in str(deprecations[0].message)


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


# Imports the schema cfg modules for the first time in a fresh interpreter and reports every
# ``DeprecationWarning`` raised while doing so. Run in a subprocess rather than via
# ``importlib.reload``: reloading the shared cfg module rebinds the base classes other already
# imported modules still reference, which would leave two incompatible generations of the class
# hierarchy alive and make the rest of the session order-dependent.
_IMPORT_SILENCE_PROBE = """
import importlib
import json
import sys
import warnings

import isaaclab  # noqa: F401  # warm up the package itself; the schema modules stay unimported

modules = [
    "isaaclab.sim.schemas.schemas_cfg",
    "isaaclab_physx.sim.schemas.schemas_cfg",
    "isaaclab_newton.sim.schemas.schemas_cfg",
]
assert not any(m in sys.modules for m in modules), "schema cfg modules were already imported"

messages = []
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for module in modules:
        try:
            importlib.import_module(module)
        except ImportError:
            continue
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]

print("RESULT " + json.dumps(messages))
"""


def test_legacy_cfg_import_does_not_warn():
    """Importing the schema modules must not warn: only construction is deprecated."""
    result = subprocess.run([sys.executable, "-c", _IMPORT_SILENCE_PROBE], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"probe failed:\n{result.stdout}\n{result.stderr}"
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("RESULT "))
    assert json.loads(line[len("RESULT ") :]) == []


"""
Deprecated cfg classes keep working.
"""


def test_legacy_cfg_keeps_dataclass_behaviour():
    """The deprecation wrapper forwards arguments and keeps the configclass helpers and signature."""
    cfg = _legacy(schemas_cfg.MassPropertiesCfg, mass=2.5, density=1200.0)
    assert (cfg.mass, cfg.density) == (2.5, 1200.0)
    assert [f.name for f in dataclasses.fields(cfg)] == ["mass", "density"]
    assert cfg.to_dict()["mass"] == 2.5
    assert cfg.copy().density == 1200.0
    assert cfg.replace(mass=4.0).mass == 4.0
    assert dataclasses.replace(cfg, mass=6.0).mass == 6.0
    assert list(inspect.signature(schemas_cfg.MassPropertiesCfg.__init__).parameters) == ["self", "mass", "density"]


def test_legacy_cfg_keeps_field_alias_forwarding():
    """The renamed-field aliases on the legacy joint-drive cfg still forward."""
    cfg = _legacy(schemas_cfg.JointDriveBaseCfg, max_effort=80.0, max_velocity=5.0)
    assert cfg.max_force == 80.0 and cfg.max_effort is None
    assert cfg.max_joint_velocity == 5.0 and cfg.max_velocity is None


def test_legacy_cfg_subclass_warns_only_for_itself():
    """A legacy subclass warns once for its own name, not once per legacy base."""
    deprecations = _deprecations(_physx_cfgs().RigidBodyPropertiesCfg)
    assert len(deprecations) == 1
    assert "RigidBodyPropertiesCfg is deprecated" in str(deprecations[0].message)


def test_usd_api_physx_api_attrs_deprecated():
    """Reading ``usd_api`` / ``physx_api`` on a mesh cfg warns and returns the legacy-mapped value."""
    cfg = _legacy(_physx_cfgs().PhysxConvexHullPropertiesCfg)
    with pytest.warns(DeprecationWarning, match="usd_api"):
        assert cfg.usd_api == "MeshCollisionAPI"
    with pytest.warns(DeprecationWarning, match="physx_api"):
        assert cfg.physx_api == "PhysxConvexHullCollisionAPI"
    with pytest.raises(AttributeError):
        cfg.no_such_attribute


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
    assert ".. deprecated:: 3.1" in doc, f"{name}: missing deprecation directive"
    assert replacement in doc, f"{name}: docstring does not name '{replacement}'"
    assert "removed" in doc and "3.2" in doc, f"{name}: docstring does not state the removal version"


@pytest.mark.parametrize("name", ["define_mass_properties", "modify_mass_properties"])
def test_legacy_mass_writer_warns_once_and_traverses_children(name):
    """Direct and delegated writers warn at the caller and retain subtree traversal."""
    stage, prim_path = _stage_with_rigid_body()
    child = UsdGeom.Cube.Define(stage, f"{prim_path}/Child").GetPrim()
    UsdPhysics.MassAPI.Apply(child)
    cfg = _legacy(schemas_cfg.MassPropertiesCfg, mass=3.0)

    deprecations = _deprecations(lambda: getattr(schemas, name)(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__
    message = str(deprecations[0].message)
    assert f"{name} is deprecated" in message and "apply_mass_properties" in message and "3.2" in message
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(3.0)
    assert child.GetAttribute("physics:mass").Get() == pytest.approx(3.0)


@pytest.mark.parametrize("name", ["define_collision_properties", "modify_collision_properties"])
def test_legacy_collision_writer_warns_once_despite_mesh_delegation(name):
    """Nested mesh configuration still writes through multiple legacy writer calls."""
    stage = Usd.Stage.CreateInMemory()
    prim_path = "/World/Mesh"
    prim = UsdGeom.Mesh.Define(stage, prim_path).GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    mesh_cfg = _legacy(schemas_cfg.MeshCollisionBaseCfg, mesh_approximation_name="boundingCube")
    cfg = _legacy(schemas_cfg.CollisionBaseCfg, mesh_collision_property=mesh_cfg)

    deprecations = _deprecations(lambda: getattr(schemas, name)(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert f"{name} is deprecated" in str(deprecations[0].message)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "boundingCube"


def test_modify_rigid_body_properties_warns_and_writes():
    """The legacy rigid-body writer warns once, still authors and raises on a missing prim."""
    stage, prim_path = _stage_with_rigid_body()
    cfg = _legacy(schemas_cfg.RigidBodyBaseCfg, kinematic_enabled=True)
    deprecations = _deprecations(lambda: schemas.modify_rigid_body_properties(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert "apply_rigid_body_properties" in str(deprecations[0].message)
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:kinematicEnabled").Get() is True
    with pytest.raises(ValueError), warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.modify_rigid_body_properties("/World/DoesNotExist", cfg, stage)


def test_apply_mass_properties_does_not_warn():
    """The fragment writer is the replacement and must stay silent."""
    stage, prim_path = _stage_with_rigid_body()
    fragment = schemas_cfg.MassCfg(mass=5.0)
    deprecations = _deprecations(lambda: schemas.apply_mass_properties(prim_path, [fragment], stage=stage))
    assert deprecations == []
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(5.0)


"""
Deprecated writers keep their USD routing.

Base-cfg fields whose USD attribute is PhysX-namespaced apply the PhysX schema only when set, so
Newton-targeted assets do not get PhysX schemas stamped on them.
"""


@pytest.fixture
def stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    return stage


def test_legacy_joint_drive_cfg_gates_physx_joint_schema(stage):
    """``max_joint_velocity`` on the base cfg authors ``physxJoint:maxJointVelocity`` in deg/s, and only then."""
    writer = inspect.unwrap(schemas.modify_joint_drive_properties)
    for name in ("j0", "j1"):
        UsdPhysics.RevoluteJoint.Define(stage, f"/World/{name}")
    with_limit = _legacy(
        schemas_cfg.JointDriveBaseCfg, drive_type="acceleration", max_force=80.0, max_joint_velocity=10.0
    )
    writer("/World/j0", with_limit, stage)
    j0 = stage.GetPrimAtPath("/World/j0")
    assert j0.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(math.degrees(10.0))
    assert j0.GetAttribute("drive:angular:physics:maxForce").Get() == pytest.approx(80.0)
    without_limit = _legacy(schemas_cfg.JointDriveBaseCfg, drive_type="acceleration", max_force=80.0, stiffness=10.0)
    writer("/World/j1", without_limit, stage)
    j1 = stage.GetPrimAtPath("/World/j1")
    assert "PhysxJointAPI" not in _api_schemas(j1)
    assert j1.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(math.radians(10.0))


def test_legacy_rigid_body_cfgs_gate_physx_schema(stage):
    physx_cfg = _physx_cfgs()
    UsdGeom.Cube.Define(stage, "/World/gravity")
    UsdGeom.Cube.Define(stage, "/World/plain")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.define_rigid_body_properties(
            "/World/gravity",
            _legacy(schemas_cfg.RigidBodyBaseCfg, rigid_body_enabled=True, disable_gravity=True),
            stage,
        )
        schemas.define_rigid_body_properties(
            "/World/plain", _legacy(physx_cfg.PhysxRigidBodyPropertiesCfg, rigid_body_enabled=True), stage
        )
    gravity = stage.GetPrimAtPath("/World/gravity")
    assert gravity.GetAttribute("physxRigidBody:disableGravity").Get() is True
    assert "PhysxRigidBodyAPI" in _api_schemas(gravity)
    plain = stage.GetPrimAtPath("/World/plain")
    assert plain.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert "PhysxRigidBodyAPI" not in _api_schemas(plain)


def test_legacy_collision_cfgs_gate_physx_schema(stage):
    physx_cfg = _physx_cfgs()
    cases = {
        "/World/offsets": (
            _legacy(schemas_cfg.CollisionBaseCfg, collision_enabled=True, contact_offset=0.05, rest_offset=0.001),
            {"physxCollision:contactOffset": 0.05, "physxCollision:restOffset": 0.001},
        ),
        "/World/plain": (_legacy(schemas_cfg.CollisionBaseCfg, collision_enabled=True), {}),
        "/World/torsional": (
            _legacy(physx_cfg.PhysxCollisionPropertiesCfg, torsional_patch_radius=1.0),
            {"physxCollision:torsionalPatchRadius": 1.0},
        ),
    }
    for path, (cfg, expected) in cases.items():
        UsdGeom.Cube.Define(stage, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            schemas.define_collision_properties(path, cfg, stage)
        prim = stage.GetPrimAtPath(path)
        assert ("PhysxCollisionAPI" in _api_schemas(prim)) is bool(expected), path
        for attr, value in expected.items():
            assert prim.GetAttribute(attr).Get() == pytest.approx(value), attr
    assert stage.GetPrimAtPath("/World/plain").GetAttribute("physics:collisionEnabled").Get() is True


def test_legacy_articulation_cfgs_gate_physx_schema_and_mirror_newton(stage):
    physx_cfg = _physx_cfgs()
    for path in ("/World/enabled", "/World/topology", "/World/self_collisions"):
        UsdGeom.Xform.Define(stage, path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.define_articulation_root_properties(
            "/World/enabled", _legacy(schemas_cfg.ArticulationRootBaseCfg, articulation_enabled=False), stage
        )
        schemas.define_articulation_root_properties(
            "/World/topology", _legacy(schemas_cfg.ArticulationRootBaseCfg, fix_root_link=False), stage
        )
        schemas.define_articulation_root_properties(
            "/World/self_collisions",
            _legacy(physx_cfg.PhysxArticulationRootPropertiesCfg, enabled_self_collisions=True),
            stage,
        )
    enabled = stage.GetPrimAtPath("/World/enabled")
    assert enabled.GetAttribute("physxArticulation:articulationEnabled").Get() is False
    assert "PhysxArticulationAPI" in _api_schemas(enabled)
    # the topology flag alone authors no PhysX attribute, so no PhysX schema is stamped
    assert "PhysxArticulationAPI" not in _api_schemas(stage.GetPrimAtPath("/World/topology"))
    mirrored = stage.GetPrimAtPath("/World/self_collisions")
    assert mirrored.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert mirrored.GetAttribute("newton:selfCollisionEnabled").Get() is True
    assert {"PhysxArticulationAPI", "NewtonArticulationRootAPI"} <= _api_schemas(mirrored)


def test_legacy_mesh_collision_cfgs_gate_cooking_schema(stage):
    physx_cfg = _physx_cfgs()
    cases = {
        "/World/base": (
            _legacy(schemas_cfg.MeshCollisionBaseCfg, mesh_approximation_name="boundingCube"),
            "boundingCube",
            False,
        ),
        "/World/tuned": (_legacy(physx_cfg.PhysxConvexHullPropertiesCfg, hull_vertex_limit=64), "convexHull", True),
        "/World/untuned": (_legacy(physx_cfg.PhysxConvexHullPropertiesCfg), "convexHull", False),
    }
    for path, (cfg, token, cooking_schema) in cases.items():
        UsdGeom.Mesh.Define(stage, path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            schemas.define_mesh_collision_properties(path, cfg, stage)
        prim = stage.GetPrimAtPath(path)
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), path
        assert prim.GetAttribute("physics:approximation").Get() == token, path
        assert ("PhysxConvexHullCollisionAPI" in _api_schemas(prim)) is cooking_schema, path
    assert stage.GetPrimAtPath("/World/tuned").GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 64


def test_legacy_nested_writers_reach_nested_rigid_bodies(stage):
    """Child links authored under their parent link (URDF importer) are all modified."""
    physx_cfg = _physx_cfgs()
    paths = ["/World/Robot/pelvis", "/World/Robot/pelvis/hip", "/World/Robot/pelvis/hip/knee"]
    for path in paths:
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.modify_rigid_body_properties(
            "/World/Robot", physx_cfg.RigidBodyPropertiesCfg(disable_gravity=True), stage
        )
        schemas.modify_mass_properties("/World/Robot", schemas_cfg.MassPropertiesCfg(mass=2.5), stage)
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        assert prim.GetAttribute("physxRigidBody:disableGravity").Get() is True, path
        assert prim.GetAttribute("physics:mass").Get() == pytest.approx(2.5), path


"""
Silencing the warnings, as documented in the 3.0 migration guide.
"""


def _surviving(probe, **filter_kwargs) -> int:
    """Return how many ``DeprecationWarning`` instances ``probe`` raises under ``filter_kwargs``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=DeprecationWarning, **filter_kwargs)
        probe()
    return len([w for w in caught if issubclass(w.category, DeprecationWarning)])


def _legacy_writer_probe():
    stage, prim_path = _stage_with_rigid_body()
    schemas.modify_mass_properties(prim_path, _legacy(schemas_cfg.MassPropertiesCfg, mass=3.0), stage)


def test_documented_message_filter_silences_legacy_cfgs_and_writers():
    """The message regex in the migration guide silences the cfg and writer deprecations."""
    assert _surviving(schemas_cfg.MassPropertiesCfg, message=r"\w+ is deprecated\. Use ") == 0
    assert _surviving(_legacy_writer_probe, message=r"\w+ is deprecated\. Use ") == 0


def test_documented_message_filter_silences_renamed_field_aliases():
    """The alias regex in the migration guide silences ``max_effort`` / ``max_velocity``."""

    def probe():
        schemas_cfg.JointDriveBaseCfg(max_effort=80.0, max_velocity=5.0)

    # Three deprecations fire: the legacy class plus one per alias. Only the aliases are filtered.
    assert _surviving(probe, message="no-such-warning") == 3
    assert _surviving(probe, message=r"\'\w+\' is deprecated; use ") == 1


def test_module_filter_cannot_silence_these_warnings():
    """``module=`` matches the caller's module, so it never matches the schema modules.

    The warnings intentionally use ``stacklevel`` to point at the user's call site. A filter
    keyed on ``isaaclab.sim.schemas`` therefore silences nothing, which is why the migration
    guide documents a ``message=`` filter instead.
    """
    assert _surviving(schemas_cfg.MassPropertiesCfg, module="isaaclab.sim.schemas.*") == 1
    assert _surviving(_legacy_writer_probe, module="isaaclab.sim.schemas.*") == 1


"""
Forwarding shims for the cfgs relocated to the backend packages.
"""


@pytest.mark.parametrize("name", FORWARDED_PHYSX_NAMES)
def test_physx_shims_resolve_to_relocated_class(name):
    """Every public access path resolves to the class object defined in ``isaaclab_physx``."""
    expected = getattr(_physx_cfgs(), name)
    assert getattr(schemas, name) is expected
    assert getattr(schemas_cfg, name) is expected
    assert getattr(sim_utils, name) is expected
    assert name in dir(schemas)


@pytest.mark.parametrize("name", FORWARDED_NEWTON_NAMES)
def test_newton_shims_resolve_to_relocated_class(name):
    """Every public access path resolves to the class object defined in ``isaaclab_newton``."""
    expected = getattr(_newton_cfgs(), name)
    assert getattr(schemas, name) is expected
    assert getattr(schemas_cfg, name) is expected
    assert getattr(sim_utils, name) is expected
    assert name in dir(schemas)


def test_shims_reject_unknown_and_backend_component_names():
    """Component deformable cfgs are backend-owned and not forwarded from ``isaaclab``."""
    for module in (schemas, schemas_cfg, sim_utils):
        assert not hasattr(module, "PhysXDeformableBodyPropertiesCfg")
    with pytest.raises(AttributeError):
        schemas_cfg.NoSuchCfg
