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
import json
import re
import subprocess
import sys
import warnings

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim.schemas as schemas
import isaaclab.sim.schemas.schemas_cfg as schemas_cfg
from isaaclab.sim.utils.stage import use_stage
from isaaclab.utils import clone, replace, to_dict

pytestmark = [pytest.mark.unit, pytest.mark.kitless]


# Legacy cfg class -> every fragment its warning must name, plus any legacy field that has no
# fragment at all and therefore has to be called out explicitly. A legacy class bundles several
# USD namespaces, so a warning naming only the backend-specific fragment would tell the user to
# drop the properties the class inherits -- these expectations are the whole point of the test.
# The legacy deformable cfgs do not encode the deformable type, so their warnings must also name
# both deformable slots. Tendon and material cfgs are out of scope here.
_RIGID_BODY = ("UsdPhysicsRigidBodyCfg", "PhysxRigidBodyCfg")
_COLLISION = ("UsdPhysicsCollisionCfg", "PhysxCollisionCfg", "mesh_collision_property")
_JOINT_DRIVE = ("UsdPhysicsDriveCfg", "PhysxJointCfg", "ensure_drives_exist")
_ARTICULATION = ("PhysxArticulationCfg", "fix_root_link")
_DEFORMABLE_SLOTS = ("volume_deformable_props", "surface_deformable_props")
_PHYSX_DEFORMABLE = (
    "OmniPhysicsDeformableBodyCfg",
    "PhysxDeformableBodyCfg",
    "PhysxSurfaceDeformableBodyCfg",
) + _DEFORMABLE_SLOTS

DEPRECATED_CORE_CFGS = {
    "MassPropertiesCfg": ("MassCfg",),
    "RigidBodyBaseCfg": _RIGID_BODY,
    "CollisionBaseCfg": _COLLISION,
    "ArticulationRootBaseCfg": _ARTICULATION,
    "JointDriveBaseCfg": _JOINT_DRIVE,
    "MeshCollisionBaseCfg": ("UsdPhysicsMeshCollisionCfg",),
    "BoundingCubePropertiesCfg": ("UsdPhysicsMeshCollisionCfg", "boundingCube"),
    "BoundingSpherePropertiesCfg": ("UsdPhysicsMeshCollisionCfg", "boundingSphere"),
    "DeformableBodyPropertiesBaseCfg": ("DeformableBodyFragment",) + _DEFORMABLE_SLOTS,
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
    "OmniPhysicsDeformableBodyPropertiesCfg": ("OmniPhysicsDeformableBodyCfg",) + _DEFORMABLE_SLOTS,
    "PhysXDeformableBodyPropertiesCfg": ("PhysxDeformableBodyCfg", "PhysxSurfaceDeformableBodyCfg") + _DEFORMABLE_SLOTS,
    "PhysxDeformableBodyPropertiesCfg": _PHYSX_DEFORMABLE,
    "DeformableBodyPropertiesCfg": _PHYSX_DEFORMABLE,
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
    # the class has no fields, so its cover is an empty slot of the matching deformable type
    "NewtonDeformableBodyPropertiesCfg": ("surface_deformable_props=[]", "volume_deformable_props=[]"),
}

# Replacement fragments, which must stay silent.
CURRENT_CORE_FRAGMENTS = [
    "MassCfg",
    "UsdPhysicsRigidBodyCfg",
    "UsdPhysicsCollisionCfg",
    "UsdPhysicsDriveCfg",
    "UsdPhysicsMeshCollisionCfg",
    "OmniPhysicsDeformableBodyCfg",
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
    "PhysxDeformableBodyCfg",
    "PhysxSurfaceDeformableBodyCfg",
]

CURRENT_NEWTON_FRAGMENTS = [
    "MujocoRigidBodyCfg",
    "MujocoJointCfg",
    "NewtonCollisionCfg",
    "NewtonMeshCollisionCfg",
    "NewtonSDFCollisionCfg",
    "NewtonArticulationCfg",
]

# Legacy writer -> the fragment-based writer(s) named in its warning. The deformable writers take
# the deformable type as an argument (or read it from the stage), so they name both family writers.
_DEFORMABLE_WRITERS = ("apply_volume_deformable_properties", "apply_surface_deformable_properties")
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
    "define_deformable_body_properties": _DEFORMABLE_WRITERS,
    "modify_deformable_body_properties": _DEFORMABLE_WRITERS,
}

# Deformable symbols that stay undeprecated. The curve writer has no fragment family to point at,
# and the material cfgs follow the rigid-body material cfgs, which are not deprecated either.
UNDEPRECATED_DEFORMABLE_WRITERS = ["define_deformable_curve_properties"]
UNDEPRECATED_DEFORMABLE_MATERIALS = {
    "isaaclab.sim.spawners.materials": [
        "DeformableBodyMaterialBaseCfg",
        "SurfaceDeformableBodyMaterialBaseCfg",
        "OmniPhysicsDeformableMaterialCfg",
        "OmniPhysicsSurfaceDeformableMaterialCfg",
    ],
    "isaaclab_physx.sim.spawners.materials.physics_materials_cfg": [
        "PhysxDeformableBodyMaterialCfg",
        "PhysxSurfaceDeformableBodyMaterialCfg",
        "PhysxDeformableMaterialCfg",
        "PhysxSurfaceDeformableMaterialCfg",
        "OmniPhysicsDeformableMaterialCfg",
        "OmniPhysicsSurfaceDeformableMaterialCfg",
        "PhysXDeformableMaterialCfg",
    ],
    "isaaclab_newton.sim.spawners.materials": [
        "NewtonDeformableBodyMaterialCfg",
        "NewtonSurfaceDeformableBodyMaterialCfg",
        "NewtonVolumeDeformableMaterialCfg",
        "NewtonSurfaceDeformableMaterialCfg",
        "NewtonDeformableMaterialCfg",
    ],
}
# The MPM particle path has no deformable-body cfg of its own, so it is untouched as well.
UNDEPRECATED_MPM_CFGS = {
    "isaaclab_newton.sim.spawners.mpm": ["MPMParticleMaterialCfg", "MPMGridCfg", "MPMPointsCfg"],
    "isaaclab_newton.assets.mpm_object.mpm_object_cfg": ["MPMObjectCfg"],
}


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


def _assert_deprecated_once(cls, expected: tuple[str, ...]) -> None:
    """Instantiating ``cls`` raises one deprecation naming every entry of ``expected``, and 5.0."""
    deprecations = _deprecations(cls)
    assert len(deprecations) == 1, f"{cls.__name__}: expected one DeprecationWarning, got {len(deprecations)}"
    message = str(deprecations[0].message)
    assert cls.__name__ in message
    missing = [name for name in expected if name not in message]
    assert not missing, f"{cls.__name__}: warning omits {missing}: {message}"
    assert "3.2" in message, f"{cls.__name__}: warning does not state the removal version: {message}"


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


# Every deformable-body fragment a legacy deformable cfg can migrate to, and every legacy
# deformable cfg. The cover test derives the fragments a warning must name from the fields.
_DEFORMABLE_FRAGMENTS = [
    ("isaaclab.sim.schemas.schemas_cfg", "OmniPhysicsDeformableBodyCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "PhysxDeformableBodyCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "PhysxSurfaceDeformableBodyCfg"),
]
_LEGACY_DEFORMABLE_CFGS = [
    ("isaaclab.sim.schemas.schemas_cfg", "DeformableBodyPropertiesBaseCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "OmniPhysicsDeformableBodyPropertiesCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "PhysXDeformableBodyPropertiesCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "PhysxDeformableBodyPropertiesCfg"),
    ("isaaclab_physx.sim.schemas.schemas_cfg", "DeformableBodyPropertiesCfg"),
    ("isaaclab_newton.sim.schemas.schemas_cfg", "NewtonDeformableBodyPropertiesCfg"),
]


def _field_names(cls) -> set[str]:
    """Return the authored field names of a schema cfg class (the applier callable excluded)."""
    return {field.name for field in dataclasses.fields(cls)} - {"func"}


@pytest.mark.parametrize("module_name,name", _LEGACY_DEFORMABLE_CFGS)
def test_legacy_deformable_cfg_warning_names_complete_fragment_cover(module_name, name):
    """The fragments a legacy deformable cfg warning names carry every field of that cfg, and both slots."""
    cls = getattr(pytest.importorskip(module_name), name)
    fragments = [getattr(pytest.importorskip(module), fragment) for module, fragment in _DEFORMABLE_FRAGMENTS]
    deprecations = _deprecations(cls)
    assert len(deprecations) == 1
    message = str(deprecations[0].message)
    named = [fragment for fragment in fragments if re.search(rf"\b{fragment.__name__}\b", message)]
    covered = set().union(*(_field_names(fragment) for fragment in named))
    uncovered = _field_names(cls) - covered
    assert not uncovered, f"{name}: warning names {[f.__name__ for f in named]}, which miss {sorted(uncovered)}"
    assert all(slot in message for slot in _DEFORMABLE_SLOTS), f"{name}: warning omits a deformable slot"


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


@pytest.mark.parametrize("name", UNDEPRECATED_DEFORMABLE_WRITERS)
def test_deformable_curve_writer_is_not_deprecated(name):
    """The curve writer has no fragment family, so it must neither warn nor document a deprecation."""
    writer = getattr(schemas, name)
    assert ".. deprecated::" not in (writer.__doc__ or "")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.BasisCurves.Define(stage, "/World/Cable")
    assert _deprecations(lambda: writer("/World/Cable", stage=stage)) == []
    # a token API schema is only visible through the prim type info
    applied = stage.GetPrimAtPath("/World/Cable").GetPrimTypeInfo().GetAppliedAPISchemas()
    assert "PhysicsCurvesDeformableSimAPI" in applied


@pytest.mark.parametrize(
    "module_name,name",
    [(module, name) for module, names in UNDEPRECATED_DEFORMABLE_MATERIALS.items() for name in names],
)
def test_deformable_material_cfg_is_not_deprecated(module_name, name):
    """Deformable material cfgs stay undeprecated, like the rigid-body material cfgs."""
    module = pytest.importorskip(module_name)
    assert _deprecations(getattr(module, name)) == []


@pytest.mark.parametrize(
    "module_name,name", [(module, name) for module, names in UNDEPRECATED_MPM_CFGS.items() for name in names]
)
def test_mpm_cfg_is_not_deprecated(module_name, name):
    """The MPM particle cfgs are not part of the legacy deformable-body API and stay undeprecated."""
    module = pytest.importorskip(module_name)
    assert _deprecations(getattr(module, name)) == []


# Imports the schema cfg modules for the first time in a fresh interpreter and reports every
# ``DeprecationWarning`` raised while doing so. Run in a subprocess rather than via
# ``importlib.reload``: reloading the shared cfg module rebinds the base classes other already
# imported modules still reference, which would leave two incompatible generations of the class
# hierarchy alive and make the rest of the session order-dependent.
_IMPORT_SILENCE_PROBE = """
import importlib
import json
import re
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
        assert to_dict(cfg)["mass"] == 2.5
        assert clone(cfg).density == 1200.0
        assert replace(cfg, mass=4.0).mass == 4.0
        assert dataclasses.replace(cfg, mass=6.0).mass == 6.0


def test_legacy_cfg_keeps_init_signature():
    """``inspect.signature`` must still report the dataclass fields, not ``*args, **kwargs``."""
    parameters = inspect.signature(schemas_cfg.MassPropertiesCfg.__init__).parameters
    assert list(parameters) == ["self", "mass", "density"]


def test_legacy_cfg_keeps_field_alias_forwarding():
    """The renamed-field aliases on the legacy joint-drive cfg still forward."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = schemas_cfg.JointDriveBaseCfg(max_effort=80.0, max_velocity=5.0)
    messages = [str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)]
    assert any("max_effort" in message for message in messages)
    assert any("max_velocity" in message for message in messages)
    assert cfg.max_force == 80.0
    assert cfg.max_joint_velocity == 5.0
    # the aliases are cleared once forwarded, so writers only see the canonical fields
    assert cfg.max_velocity is None and cfg.max_effort is None


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
    """Every legacy writer carries a ``.. deprecated::`` note naming its fragment writer(s)."""
    doc = getattr(schemas, name).__doc__ or ""
    assert ".. deprecated:: 3.1" in doc, f"{name}: missing deprecation directive"
    for writer in (replacement,) if isinstance(replacement, str) else replacement:
        assert writer in doc, f"{name}: docstring does not name '{writer}'"
    assert "removed" in doc and "3.2" in doc, f"{name}: docstring does not state the removal version"


@pytest.mark.parametrize("name", ["define_mass_properties", "modify_mass_properties"])
def test_legacy_mass_writer_warns_once_and_traverses_children(name):
    """Direct and delegated writers warn at the caller and retain subtree traversal."""
    stage, prim_path = _stage_with_rigid_body()
    child = UsdGeom.Cube.Define(stage, f"{prim_path}/Child").GetPrim()
    UsdPhysics.MassAPI.Apply(child)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=3.0)

    deprecations = _deprecations(lambda: getattr(schemas, name)(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__
    message = str(deprecations[0].message)
    assert f"{name} is deprecated" in message
    assert "apply_mass_properties" in message
    assert "3.2" in message
    assert stage.GetPrimAtPath(prim_path).GetAttribute("physics:mass").Get() == pytest.approx(3.0)
    assert child.GetAttribute("physics:mass").Get() == pytest.approx(3.0)


@pytest.mark.parametrize("name", ["define_collision_properties", "modify_collision_properties"])
def test_legacy_collision_writer_warns_once_despite_mesh_delegation(name):
    """Nested mesh configuration still writes through multiple legacy writer calls."""
    stage = Usd.Stage.CreateInMemory()
    prim_path = "/World/Mesh"
    prim = UsdGeom.Mesh.Define(stage, prim_path).GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.CollisionBaseCfg(
            mesh_collision_property=schemas_cfg.MeshCollisionBaseCfg(mesh_approximation_name="boundingCube")
        )

    deprecations = _deprecations(lambda: getattr(schemas, name)(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert f"{name} is deprecated" in str(deprecations[0].message)
    assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() == "boundingCube"


def _stage_with_surface_mesh() -> tuple[Usd.Stage, str]:
    """Return an in-memory stage carrying a body prim with one triangle-mesh child, and its path."""
    stage = Usd.Stage.CreateInMemory()
    prim_path = "/World/Cloth"
    UsdGeom.Xform.Define(stage, prim_path)
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/mesh")
    mesh.GetPointsAttr().Set([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 0, 2, 3])
    mesh.GetFaceVertexCountsAttr().Set([3, 3])
    return stage, prim_path


def test_define_deformable_body_properties_warns_once_and_authors():
    """The legacy deformable writer warns once at the caller, including its internal ``modify`` call."""
    newton_cfg = _newton_cfgs()
    stage, prim_path = _stage_with_surface_mesh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = newton_cfg.NewtonDeformableBodyPropertiesCfg()

    # the mesh lookup inside the writer reads the current stage, not the ``stage`` argument
    with use_stage(stage):
        deprecations = _deprecations(
            lambda: schemas.define_deformable_body_properties(prim_path, cfg, stage, deformable_type="surface")
        )
    assert len(deprecations) == 1
    assert deprecations[0].filename == __file__
    message = str(deprecations[0].message)
    assert "define_deformable_body_properties is deprecated" in message
    assert all(writer in message for writer in _DEFORMABLE_WRITERS)
    assert "3.2" in message
    assert "PhysicsDeformableBodyAPI" in stage.GetPrimAtPath(prim_path).GetPrimTypeInfo().GetAppliedAPISchemas()


def test_modify_deformable_body_properties_warns_and_writes():
    """The legacy deformable modifier warns once and still authors the ``omniphysics:*`` fields."""
    physx_cfg = _physx_cfgs()
    stage, prim_path = _stage_with_surface_mesh()
    stage.GetPrimAtPath(prim_path).AddAppliedSchema("PhysicsDeformableBodyAPI")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = physx_cfg.OmniPhysicsDeformableBodyPropertiesCfg(mass=2.0)

    deprecations = _deprecations(lambda: schemas.modify_deformable_body_properties(prim_path, cfg, stage))
    assert len(deprecations) == 1
    assert "modify_deformable_body_properties is deprecated" in str(deprecations[0].message)
    assert stage.GetPrimAtPath(prim_path).GetAttribute("omniphysics:mass").Get() == pytest.approx(2.0)


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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = schemas_cfg.MassPropertiesCfg(mass=3.0)
    schemas.modify_mass_properties(prim_path, cfg, stage)


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
