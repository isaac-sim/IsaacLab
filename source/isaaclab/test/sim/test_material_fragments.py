# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Rigid-body physics material fragments, their legacy cfgs, spawner routing and the material shims.

The tests author on in-memory USD stages and do not launch Isaac Sim / Kit.
"""

import dataclasses
import sys
import typing
import warnings

import pytest
from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg
from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonMaterialCfg
from isaaclab_physx.sim.spawners.materials import physics_materials_cfg as physx_mat_cfg
from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg, PhysxRigidBodyMaterialCfg

from pxr import Plug, Usd, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
import isaaclab.sim.spawners.materials as materials
from isaaclab.sim import SimulationCfg
from isaaclab.sim.schemas import SchemaFragment
from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg, GroundPlaneCfg
from isaaclab.sim.spawners.materials import physics_materials_cfg as materials_cfg
from isaaclab.sim.spawners.materials import (
    spawn_physics_material,
    spawn_rigid_body_material,
    spawn_rigid_body_material_from_fragments,
)
from isaaclab.sim.spawners.materials.physics_materials_cfg import (
    RigidBodyMaterialBaseCfg,
    RigidBodyMaterialFragment,
    UsdPhysicsRigidBodyMaterialCfg,
)
from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg, MeshCuboidCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg
from isaaclab.sim.utils import get_first_matching_child_prim
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.string import to_camel_case

pytestmark = pytest.mark.unit


def _register_physx_codeless_schemas() -> None:
    """Register OVPhysX's codeless schemas so ``PhysxMaterialAPI`` resolves without Kit.

    The USD schema registry is built once per process, so this must run before the first schema
    lookup; the ``physx_schemas`` fixture skips tests when the registration came too late.
    """
    try:
        import ovphysx
    except ImportError:
        return
    registry = Plug.Registry()
    registered = {plugin.name.casefold() for plugin in registry.GetAllPlugins()}
    paths = [str(p) for p in ovphysx.codeless_schema_paths() if p.parent.name.casefold() not in registered]
    if paths:
        registry.RegisterPlugins(paths)


_register_physx_codeless_schemas()


@pytest.fixture
def stage() -> Usd.Stage:
    """A fresh current stage, so spawners and explicit-stage writers author on the same stage."""
    sim_utils.create_new_stage()
    return sim_utils.get_current_stage()


@pytest.fixture
def physx_schemas() -> None:
    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("PhysxMaterialAPI") is None:
        pytest.skip("PhysX schemas are not registered in this process")


def _api_schemas(prim: Usd.Prim) -> set[str]:
    """Applied API schema names including unregistered token schemas."""
    return set(prim.GetPrimTypeInfo().GetAppliedAPISchemas())


def _cfg_attrs(cfg_type, exclude: frozenset[str] = frozenset()) -> set[str]:
    return {to_camel_case(f.name) for f in dataclasses.fields(cfg_type) if f.name not in exclude | {"func"}}


def _bound_physics_material(prim: Usd.Prim) -> Usd.Prim:
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose="physics")
    return material.GetPrim()


def _deprecations(func) -> list[warnings.WarningMessage]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        func()
    return [w for w in caught if issubclass(w.category, DeprecationWarning)]


"""
Fragments.
"""


@pytest.mark.parametrize(
    ("cfg", "namespace", "applied_schema"),
    [
        (UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7), "physics", None),
        (PhysxMaterialCfg(compliant_contact_stiffness=100.0), "physxMaterial", "PhysxMaterialAPI"),
    ],
)
def test_material_fragment_metadata(cfg, namespace, applied_schema):
    assert isinstance(cfg, RigidBodyMaterialFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == namespace
    assert type(cfg)._usd_applied_schema == applied_schema
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"


def test_spawn_from_fragments_composes_namespaces_and_leaves_none_unwritten(stage):
    """The writer spawns the material prim, applies the neutral anchor and composes every namespace."""
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat",
        [
            UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.1, density=1200.0),
            PhysxMaterialCfg(
                compliant_contact_stiffness=100.0,
                friction_combine_mode="max",
                damping_combine_mode="min",
                compliant_contact_acceleration_spring=True,
            ),
        ],
        stage,
    )
    assert prim.IsA(UsdShade.Material) and prim.HasAPI(UsdPhysics.MaterialAPI)
    for attr, value in {
        "physics:staticFriction": 0.7,
        "physics:dynamicFriction": 0.6,
        "physics:restitution": 0.1,
        "physics:density": 1200.0,
        "physxMaterial:compliantContactStiffness": 100.0,
    }.items():
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), attr
    assert "PhysxMaterialAPI" in _api_schemas(prim)
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"
    assert prim.GetAttribute("physxMaterial:dampingCombineMode").Get() == "min"
    assert prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get() is True

    # a bare fragment is accepted and None fields stay unauthored (partial update)
    single = spawn_rigid_body_material_from_fragments(
        "/World/Single", UsdPhysicsRigidBodyMaterialCfg(static_friction=0.3), stage
    )
    assert single.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.3)
    assert not single.GetAttribute("physics:dynamicFriction").HasAuthoredValue()


def test_spawn_physics_material_dispatches_fragments_and_legacy_cfgs(stage):
    """The shared dispatcher accepts fragment collections and legacy cfgs carrying their own ``func``."""
    frag_prim = spawn_physics_material("/World/MaterialA", (UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4),))
    assert frag_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert frag_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.4)

    # the legacy base authors only ``physics:*`` and never stamps the PhysX schema
    base = spawn_physics_material(
        "/World/MaterialB", RigidBodyMaterialBaseCfg(static_friction=0.7, dynamic_friction=0.6, density=800.0)
    )
    assert base.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert base.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.6)
    assert base.GetAttribute("physics:density").Get() == pytest.approx(800.0)
    assert "PhysxMaterialAPI" not in _api_schemas(base)
    unset = spawn_rigid_body_material("/World/MaterialC", RigidBodyMaterialBaseCfg())
    assert not unset.GetAttribute("physics:density").HasAuthoredValue()

    # the legacy PhysX cfg is metadata-driven off the same ``physxMaterial`` namespace as the fragment
    physx = spawn_physics_material(
        "/World/MaterialD",
        PhysxRigidBodyMaterialCfg(
            static_friction=0.9,
            compliant_contact_stiffness=100.0,
            damping_combine_mode="min",
            compliant_contact_acceleration_spring=True,
        ),
    )
    assert physx.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.9)
    assert "PhysxMaterialAPI" in _api_schemas(physx)
    assert physx.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100.0)
    assert physx.GetAttribute("physxMaterial:dampingCombineMode").Get() == "min"
    assert physx.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get() is True


def test_legacy_material_spawner_is_current_stage_bound(stage):
    """The legacy path raises on a different explicit stage instead of silently authoring on the current one."""
    other = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="current stage"):
        spawn_physics_material("/World/MatOther", PhysxRigidBodyMaterialCfg(), stage=other)
    assert not stage.GetPrimAtPath("/World/MatOther").IsValid()
    assert spawn_physics_material("/World/MatCurrent", PhysxRigidBodyMaterialCfg(), stage=stage).IsValid()


def test_fragment_writer_validates_inputs_before_authoring(stage):
    with pytest.raises(ValueError):
        spawn_rigid_body_material_from_fragments("/World/MatEmpty", [], stage)
    # a list mixing a fragment with a legacy cfg is not a valid fragment list
    with pytest.raises(TypeError):
        spawn_physics_material(
            "/World/MatMixed",
            [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4), PhysxRigidBodyMaterialCfg(static_friction=0.9)],
        )
    with pytest.raises(TypeError):
        spawn_rigid_body_material_from_fragments("/World/MatLegacy", PhysxRigidBodyMaterialCfg(), stage)
    with pytest.raises(TypeError):
        spawn_physics_material("/World/MatInvalid", object())
    for path in ("MatEmpty", "MatMixed", "MatLegacy", "MatInvalid"):
        assert not stage.GetPrimAtPath(f"/World/{path}").IsValid()


def test_material_cfgs_match_schema_attributes(physx_schemas):
    """Schema-drift guard: each interface covers exactly the attributes of the schema it authors."""
    usd_attrs = {name.split(":", 1)[1] for name in UsdPhysics.MaterialAPI.GetSchemaAttributeNames()}
    assert _cfg_attrs(UsdPhysicsRigidBodyMaterialCfg) == usd_attrs
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("PhysxMaterialAPI")
    physx_attrs = {str(name).split(":", 1)[1] for name in definition.GetPropertyNames()}
    assert _cfg_attrs(PhysxMaterialCfg) == physx_attrs
    base_fields = frozenset(f.name for f in dataclasses.fields(RigidBodyMaterialBaseCfg))
    assert _cfg_attrs(PhysxRigidBodyMaterialCfg, base_fields) == physx_attrs


"""
Spawner routing.
"""


@pytest.mark.parametrize(
    ("material", "attr", "value"),
    [
        (UsdPhysicsRigidBodyMaterialCfg(static_friction=0.65), "physics:staticFriction", 0.65),
        (PhysxRigidBodyMaterialCfg(static_friction=0.65), "physics:staticFriction", 0.65),
        (NewtonMaterialPropertiesCfg(torsional_friction=0.3), "newton:torsionalFriction", 0.3),
    ],
    ids=["fragment", "legacy_physx", "legacy_newton"],
)
def test_mesh_spawner_accepts_and_binds_rigid_materials(stage, material, attr, value):
    """The mesh spawner's rigid-vs-deformable guard admits fragments and every legacy rigid material cfg."""
    cfg = MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        physics_material=material,
    )
    assert cfg.func("/World/MeshCube", cfg, stage=stage).IsValid()
    material_prim = stage.GetPrimAtPath("/World/MeshCube/geometry/material")
    assert material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert material_prim.GetAttribute(attr).Get() == pytest.approx(value)
    assert _bound_physics_material(stage.GetPrimAtPath("/World/MeshCube/geometry/mesh")) == material_prim


def test_ground_plane_and_terrain_spawners_accept_fragment_materials(stage):
    cfg = GroundPlaneCfg(physics_material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.42))
    assert cfg.func("/World/groundPlane", cfg).IsValid()
    material_prim = stage.GetPrimAtPath("/World/groundPlane/physicsMaterial")
    assert material_prim.HasAPI(UsdPhysics.MaterialAPI)
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.42)
    plane = get_first_matching_child_prim("/World/groundPlane", lambda prim: prim.GetTypeName() == "Plane", stage=stage)
    assert _bound_physics_material(plane) == material_prim

    import trimesh

    create_prim_from_mesh(
        "/World/terrain",
        trimesh.creation.box(extents=(1.0, 1.0, 0.2)),
        physics_material=[UsdPhysicsRigidBodyMaterialCfg(static_friction=0.9), NewtonMaterialCfg(rolling_friction=0.1)],
    )
    terrain_material = stage.GetPrimAtPath("/World/terrain/physicsMaterial")
    assert terrain_material.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.9)
    assert terrain_material.GetAttribute("newton:rollingFriction").Get() == pytest.approx(0.1)


def test_public_default_material_types_remain_core_importable():
    """Default rigid materials must not require importing a physics-backend package."""
    defaults = (
        SimulationCfg().physics_material,
        GroundPlaneCfg().physics_material,
        TerrainImporterCfg(prim_path="/World/terrain").physics_material,
    )
    assert all(type(material) is RigidBodyMaterialBaseCfg for material in defaults)


def test_material_slot_unions_match_spawner_kind():
    """Rigid-only spawner slots admit the rigid base + fragments and exclude the deformable root."""

    def union_args(cls) -> set:
        # evaluate the annotation against its declaring module so the TYPE_CHECKING-only ``Usd``
        # import of the inherited ``func`` field never needs resolving
        annotation = cls.__dict__["__annotations__"]["physics_material"]
        return set(typing.get_args(eval(annotation, vars(sys.modules[cls.__module__]))))

    for cls in (ShapeCfg, GroundPlaneCfg, TerrainImporterCfg):
        args = union_args(cls)
        assert {materials_cfg.RigidBodyMaterialBaseCfg, materials_cfg.RigidBodyMaterialFragment} <= args, cls.__name__
        assert materials_cfg.PhysicsMaterialCfg not in args, f"{cls.__name__} is rigid-only"
    for cls in (FileCfg, MeshCfg):
        args = union_args(cls)
        assert {materials_cfg.PhysicsMaterialCfg, materials_cfg.RigidBodyMaterialFragment} <= args, cls.__name__


"""
Forwarding shims for the material cfgs relocated to isaaclab_physx.
"""

FORWARDED_MATERIAL_NAMES = [
    "DeformableBodyMaterialCfg",
    "RigidBodyMaterialCfg",
    "SurfaceDeformableBodyMaterialCfg",
    "PhysxRigidBodyMaterialCfg",
    "PhysxDeformableBodyMaterialCfg",
    "PhysxSurfaceDeformableBodyMaterialCfg",
]

DEPRECATED_FORWARDED_MATERIAL_NAMES = FORWARDED_MATERIAL_NAMES[:3]


@pytest.mark.parametrize("name", FORWARDED_MATERIAL_NAMES)
def test_material_shims_resolve_to_relocated_class(name):
    """Every public access path resolves to the class object defined in ``isaaclab_physx``."""
    expected = getattr(physx_mat_cfg, name)
    assert getattr(materials, name) is expected
    assert getattr(materials_cfg, name) is expected
    assert getattr(sim_utils, name) is expected
    assert name in dir(materials)


@pytest.mark.parametrize("name", DEPRECATED_FORWARDED_MATERIAL_NAMES)
def test_deprecated_material_alias_warns_once(name):
    deprecations = _deprecations(getattr(materials, name))
    assert len(deprecations) == 1
    assert "3.2" in str(deprecations[0].message)


@pytest.mark.parametrize("cls", [PhysxRigidBodyMaterialCfg, PhysxMaterialCfg, UsdPhysicsRigidBodyMaterialCfg])
def test_current_material_classes_do_not_warn(cls):
    assert _deprecations(cls) == []
