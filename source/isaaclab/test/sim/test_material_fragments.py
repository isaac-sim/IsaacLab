# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest

from pxr import UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

pytestmark = pytest.mark.integration

# -------------------------------------------------------------------------------------
# RigidBodyMaterialFragment marker + metadata
# -------------------------------------------------------------------------------------


def test_rigid_body_material_fragment_metadata_defaults():
    from isaaclab.sim.schemas import SchemaFragment
    from isaaclab.sim.spawners.materials.physics_materials_cfg import (
        RigidBodyMaterialFragment,
        UsdPhysicsRigidBodyMaterialCfg,
    )

    cfg = UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7)
    assert isinstance(cfg, RigidBodyMaterialFragment) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == "physics"
    assert type(cfg)._usd_applied_schema is None  # MaterialAPI applied by the family writer
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"
    assert cfg.static_friction == 0.7 and cfg.dynamic_friction is None


def test_physx_material_fragment_metadata_defaults():
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialFragment

    cfg = PhysxMaterialCfg(compliant_contact_stiffness=100.0)
    assert isinstance(cfg, RigidBodyMaterialFragment)
    assert type(cfg)._usd_namespace == "physxMaterial"
    assert type(cfg)._usd_applied_schema == "PhysxMaterialAPI"
    assert cfg.func == "isaaclab.sim.schemas:apply_namespaced"


# -------------------------------------------------------------------------------------
# spawn_rigid_body_material_from_fragments: spawn prim + anchor + multi-namespace compose
# -------------------------------------------------------------------------------------


def test_spawn_rigid_body_material_from_fragments_composes_namespaces():
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat",
        [
            UsdPhysicsRigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.1),
            PhysxMaterialCfg(compliant_contact_stiffness=100.0, friction_combine_mode="max"),
        ],
        stage,
    )
    assert prim.IsA(UsdShade.Material)
    assert bool(UsdPhysics.MaterialAPI(prim))  # neutral anchor applied by the writer
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.6)
    assert prim.GetAttribute("physics:restitution").Get() == pytest.approx(0.1)
    # the PhysX fragment applied its own schema and namespace
    assert "PhysxMaterialAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100.0)
    assert prim.GetAttribute("physxMaterial:frictionCombineMode").Get() == "max"


def test_spawn_rigid_body_material_from_fragments_accepts_single_fragment():
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat2", UsdPhysicsRigidBodyMaterialCfg(static_friction=0.3), stage
    )
    assert prim.IsA(UsdShade.Material)
    assert bool(UsdPhysics.MaterialAPI(prim))
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.3)


def test_spawn_physics_material_dispatches_fragments_and_legacy():
    """The shared dispatcher handles both a rigid-body fragment collection and a legacy material
    cfg carrying its own ``func``."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_physics_material
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))

    # tuple form is accepted by the low-level dispatcher alongside the list form used by cfg slots
    frag_prim = spawn_physics_material("/World/MaterialA", (UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4),))
    assert bool(UsdPhysics.MaterialAPI(frag_prim))
    assert frag_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.4)

    # legacy single-cfg form (RigidBodyMaterialBaseCfg subclass with its own spawner func)
    legacy_prim = spawn_physics_material("/World/MaterialB", PhysxRigidBodyMaterialCfg(static_friction=0.9))
    assert bool(UsdPhysics.MaterialAPI(legacy_prim))
    assert legacy_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.9)


def test_spawn_physics_material_rejects_non_current_stage_for_legacy():
    """The legacy path is current-stage-bound; an explicit different stage raises instead of
    silently authoring on the current stage. Passing the current stage explicitly stays valid
    (the in-tree spawners do so unconditionally)."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from pxr import Usd

    from isaaclab.sim.spawners.materials import spawn_physics_material

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    other = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError, match="current stage"):
        spawn_physics_material("/World/MatOther", PhysxRigidBodyMaterialCfg(), stage=other)
    # nothing leaked onto the current stage
    assert not sim_utils.get_current_stage().GetPrimAtPath("/World/MatOther").IsValid()
    # explicit current stage remains supported
    prim = spawn_physics_material("/World/MatCurrent", PhysxRigidBodyMaterialCfg(), stage=sim_utils.get_current_stage())
    assert prim.IsValid()


def test_fragment_writer_validates_inputs_before_authoring():
    """Direct and dispatched fragment calls share one validation contract and do not leave prims."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import (
        spawn_physics_material,
        spawn_rigid_body_material_from_fragments,
    )
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()

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


def test_spawn_rigid_body_material_from_fragments_leaves_none_fields_unwritten():
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/Mat3", [UsdPhysicsRigidBodyMaterialCfg(static_friction=0.5)], stage
    )
    # only the authored field is written; None fields are left unauthored (partial update)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.5)
    assert not prim.GetAttribute("physics:dynamicFriction").HasAuthoredValue()


# -------------------------------------------------------------------------------------
# UsdPhysicsRigidBodyMaterialCfg: density round-trip + physics:* schema parity
# -------------------------------------------------------------------------------------


def test_usd_physics_rigid_body_material_density_round_trips():
    """``physics:density`` participates in mass computation via material binding; it must author
    the same as the other ``UsdPhysics.MaterialAPI`` friction/restitution fields."""
    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/MatDensity", [UsdPhysicsRigidBodyMaterialCfg(density=1200.0)], stage
    )
    assert prim.GetAttribute("physics:density").Get() == pytest.approx(1200.0)


def test_usd_physics_rigid_body_material_fragment_matches_material_api_schema():
    """Schema-parity guard: the set of ``physics:*`` attrs the neutral fragment can author must
    equal the attribute set on ``UsdPhysics.MaterialAPI`` (4 attrs: static/dynamic friction,
    restitution, density). Catches drift if the schema gains/loses an attribute."""
    import dataclasses

    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.utils.string import to_camel_case

    fragment_fields = {f.name for f in dataclasses.fields(UsdPhysicsRigidBodyMaterialCfg) if f.name != "func"}
    schema_attr_names = {name.split(":", 1)[1] for name in UsdPhysics.MaterialAPI.GetSchemaAttributeNames()}
    fragment_attr_names = {to_camel_case(name, "cC") for name in fragment_fields}
    assert fragment_attr_names == schema_attr_names


# -------------------------------------------------------------------------------------
# PhysxMaterialCfg: damping-combine-mode + compliant-contact-acceleration-spring
# -------------------------------------------------------------------------------------


def test_physx_material_fragment_authors_damping_combine_mode_and_acceleration_spring():
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials import spawn_rigid_body_material_from_fragments

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    prim = spawn_rigid_body_material_from_fragments(
        "/World/MatPhysxExtra",
        [PhysxMaterialCfg(damping_combine_mode="min", compliant_contact_acceleration_spring=True)],
        stage,
    )
    assert "PhysxMaterialAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("physxMaterial:dampingCombineMode").Get() == "min"
    assert prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get() is True


# -------------------------------------------------------------------------------------
# Finding 4: mesh spawner must accept a fragment list for a rigid physics_material
# -------------------------------------------------------------------------------------


def test_spawn_mesh_with_rigid_props_accepts_fragment_list_physics_material():
    """Regression test: the rigid-vs-deformable material guard in the mesh spawner used to reject
    a fragment / fragment-list ``physics_material`` outright. A rigid-body fragment list must spawn
    and bind successfully."""
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    cfg = MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=[UsdPhysicsRigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.55)],
    )
    prim = cfg.func("/World/MeshCubeFrag", cfg, stage=stage)
    assert prim.IsValid()
    material_prim = stage.GetPrimAtPath("/World/MeshCubeFrag/geometry/material")
    assert material_prim.IsValid()
    assert bool(UsdPhysics.MaterialAPI(material_prim))
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.65)
    # material binding: the mesh prim carries a physics-purpose material binding
    binding_api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/MeshCubeFrag/geometry/mesh"))
    bound_material, _ = binding_api.ComputeBoundMaterial(materialPurpose="physics")
    assert bound_material.GetPath() == material_prim.GetPath()


# -------------------------------------------------------------------------------------
# Finding 5: ground-plane spawner must accept a fragment-list physics_material
# -------------------------------------------------------------------------------------


def test_spawn_ground_plane_accepts_fragment_list_physics_material():
    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    cfg = GroundPlaneCfg(physics_material=[UsdPhysicsRigidBodyMaterialCfg(static_friction=0.42)])
    prim = cfg.func("/World/groundPlane", cfg)
    assert prim.IsValid()
    material_prim = stage.GetPrimAtPath("/World/groundPlane/physicsMaterial")
    assert material_prim.IsValid()
    assert bool(UsdPhysics.MaterialAPI(material_prim))
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.42)
    # the collision (Plane) prim under the ground plane must bind to the spawned material
    from isaaclab.sim.utils import get_first_matching_child_prim

    collision_prim = get_first_matching_child_prim(
        "/World/groundPlane", predicate=lambda _prim: _prim.GetTypeName() == "Plane", stage=stage
    )
    assert collision_prim is not None
    binding_api = UsdShade.MaterialBindingAPI(collision_prim)
    bound_material, _ = binding_api.ComputeBoundMaterial(materialPurpose="physics")
    assert bound_material.GetPath() == material_prim.GetPath()


# -------------------------------------------------------------------------------------
# Regression: the mesh spawner's rigid-material guard must also accept legacy (non-fragment)
# rigid-body material cfgs, not just the deprecated ``RigidBodyMaterialCfg`` alias.
# -------------------------------------------------------------------------------------


def test_spawn_mesh_with_rigid_props_accepts_legacy_physx_rigid_body_material():
    """Regression test: the mesh guard used to check ``isinstance(cfg.physics_material,
    RigidBodyMaterialCfg)`` -- the deprecated PhysX leaf alias -- which rejected the canonical
    legacy :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` even though
    :func:`~isaaclab.sim.spawners.materials.spawn_physics_material` accepts it."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    cfg = MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=PhysxRigidBodyMaterialCfg(static_friction=0.65, dynamic_friction=0.55),
    )
    prim = cfg.func("/World/MeshCubeLegacyPhysx", cfg, stage=stage)
    assert prim.IsValid()
    material_prim = stage.GetPrimAtPath("/World/MeshCubeLegacyPhysx/geometry/material")
    assert material_prim.IsValid()
    assert bool(UsdPhysics.MaterialAPI(material_prim))
    assert material_prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.65)
    binding_api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/MeshCubeLegacyPhysx/geometry/mesh"))
    bound_material, _ = binding_api.ComputeBoundMaterial(materialPurpose="physics")
    assert bound_material.GetPath() == material_prim.GetPath()


def test_spawn_mesh_with_rigid_props_accepts_legacy_newton_material():
    """Same regression as above for Newton's legacy rigid-body material cfg, which also derives
    from :class:`~isaaclab.sim.spawners.materials.RigidBodyMaterialBaseCfg` (not the deprecated
    PhysX ``RigidBodyMaterialCfg`` alias) and must not be rejected by the mesh guard."""
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCuboidCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    cfg = MeshCuboidCfg(
        size=(1.0, 1.0, 1.0),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        physics_material=NewtonMaterialPropertiesCfg(torsional_friction=0.3, rolling_friction=0.001),
    )
    prim = cfg.func("/World/MeshCubeLegacyNewton", cfg, stage=stage)
    assert prim.IsValid()
    material_prim = stage.GetPrimAtPath("/World/MeshCubeLegacyNewton/geometry/material")
    assert material_prim.IsValid()
    assert bool(UsdPhysics.MaterialAPI(material_prim))
    assert material_prim.GetAttribute("newton:torsionalFriction").Get() == pytest.approx(0.3)
    assert material_prim.GetAttribute("newton:rollingFriction").Get() == pytest.approx(0.001)
    binding_api = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/World/MeshCubeLegacyNewton/geometry/mesh"))
    bound_material, _ = binding_api.ComputeBoundMaterial(materialPurpose="physics")
    assert bound_material.GetPath() == material_prim.GetPath()


# -------------------------------------------------------------------------------------
# PhysxRigidBodyMaterialCfg (legacy): damping-combine-mode + compliant-contact-acceleration-spring
# -------------------------------------------------------------------------------------


def test_legacy_physx_rigid_body_material_authors_damping_combine_mode_and_acceleration_spring():
    """The legacy :class:`~isaaclab_physx.sim.spawners.materials.PhysxRigidBodyMaterialCfg` must
    author the same two ``physxMaterial:*`` attributes as the
    :class:`~isaaclab_physx.sim.spawners.materials.PhysxMaterialCfg` fragment, since the legacy
    spawner is metadata-driven off the same ``physxMaterial`` namespace."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import PhysxRigidBodyMaterialCfg

    from isaaclab.sim.spawners.materials import spawn_rigid_body_material

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = PhysxRigidBodyMaterialCfg(damping_combine_mode="min", compliant_contact_acceleration_spring=True)
    prim = spawn_rigid_body_material("/World/MatLegacyPhysxExtra", cfg)
    assert "PhysxMaterialAPI" in prim.GetAppliedSchemas()
    assert prim.GetAttribute("physxMaterial:dampingCombineMode").Get() == "min"
    assert prim.GetAttribute("physxMaterial:compliantContactAccelerationSpring").Get() is True


# -------------------------------------------------------------------------------------
# Generated-terrain routing: create_prim_from_mesh must accept a fragment-list physics_material
# -------------------------------------------------------------------------------------


def test_create_prim_from_mesh_accepts_fragment_list():
    """Generated terrain routes its material through the dispatcher, so a fragment list works
    end-to-end instead of crashing on the legacy-only spawn path."""
    import trimesh
    from isaaclab_newton.sim.spawners.materials.physics_materials_cfg import NewtonMaterialCfg

    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.terrains.utils import create_prim_from_mesh

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    mesh = trimesh.creation.box(extents=(1.0, 1.0, 0.2))
    create_prim_from_mesh(
        "/World/terrainFrag",
        mesh,
        physics_material=[UsdPhysicsRigidBodyMaterialCfg(static_friction=0.9), NewtonMaterialCfg(rolling_friction=0.1)],
    )
    stage = sim_utils.get_current_stage()
    mat = stage.GetPrimAtPath("/World/terrainFrag/physicsMaterial")
    assert mat.IsValid()
    assert mat.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.9)
    assert mat.GetAttribute("newton:rollingFriction").Get() == pytest.approx(0.1)


def test_legacy_base_cfg_authors_density():
    """The legacy rigid material base authors ``physics:density``, matching the USD fragment.

    ``UsdPhysics.MaterialAPI`` defines four properties; explicitly set values must be available
    through both interfaces, including material density read by Newton's importer.
    """
    from isaaclab.sim.spawners.materials import spawn_rigid_body_material
    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    prim = spawn_rigid_body_material("/World/LegacyDensity", RigidBodyMaterialBaseCfg(density=800.0))
    assert prim.GetAttribute("physics:density").Get() == pytest.approx(800.0)
    # None default -> unauthored (backward compatible)
    prim2 = spawn_rigid_body_material("/World/LegacyDensityNone", RigidBodyMaterialBaseCfg())
    assert not prim2.GetAttribute("physics:density").HasAuthoredValue()


def test_public_default_material_types_remain_backward_compatible():
    """Fragment support must not silently replace the released default config objects."""
    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg

    from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

    defaults = (
        SimulationCfg().physics_material,
        GroundPlaneCfg().physics_material,
        TerrainImporterCfg(prim_path="/World/terrain").physics_material,
    )
    assert all(type(material) is RigidBodyMaterialCfg for material in defaults)


def test_physx_fragment_and_legacy_cfg_match_material_api_schema():
    """Both PhysX interfaces cover the properties declared by ``PhysxMaterialAPI``."""
    import dataclasses

    from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import (
        PhysxMaterialCfg,
        PhysxRigidBodyMaterialCfg,
    )

    from pxr import PhysxSchema

    from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialBaseCfg
    from isaaclab.utils.string import to_camel_case

    def fields(cls):
        return {f.name for f in dataclasses.fields(cls) if f.name != "func"}

    base = fields(RigidBodyMaterialBaseCfg)
    schema_attrs = {name.split(":", 1)[1] for name in PhysxSchema.PhysxMaterialAPI.GetSchemaAttributeNames()}
    fragment_attrs = {to_camel_case(name, "cC") for name in fields(PhysxMaterialCfg)}
    legacy_attrs = {to_camel_case(name, "cC") for name in fields(PhysxRigidBodyMaterialCfg) - base}
    assert fragment_attrs == schema_attrs
    assert legacy_attrs == schema_attrs


# -------------------------------------------------------------------------------------
# Slot-typing contract
# -------------------------------------------------------------------------------------


def test_material_slot_unions_match_spawner_kind():
    """Structural slot typing: rigid-only spawner slots admit the rigid base + fragments and
    exclude the deformable-admitting root; mixed spawners keep the broad root. New slots added
    to this list keep the contract enforced."""
    import sys
    import typing

    from isaaclab.sim.spawners.from_files.from_files_cfg import FileCfg, GroundPlaneCfg
    from isaaclab.sim.spawners.materials import physics_materials_cfg as mats
    from isaaclab.sim.spawners.meshes.meshes_cfg import MeshCfg
    from isaaclab.sim.spawners.shapes.shapes_cfg import ShapeCfg
    from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

    def union_args(cls):
        # typing.get_type_hints(cls) resolves annotations across the whole MRO, including the
        # inherited ``func: Callable[..., Usd.Prim]`` from ``SpawnerCfg``, whose ``Usd`` import is
        # TYPE_CHECKING-only and unresolvable at runtime. Evaluate the ``physics_material``
        # annotation directly against its declaring class's module globals instead -- each class
        # in this list declares the field itself, so this never needs the base class's namespace.
        annotation = cls.__dict__["__annotations__"]["physics_material"]
        resolved = eval(annotation, vars(sys.modules[cls.__module__]))
        return set(typing.get_args(resolved))

    for cls in (ShapeCfg, GroundPlaneCfg, TerrainImporterCfg):
        args = union_args(cls)
        assert mats.RigidBodyMaterialBaseCfg in args, f"{cls.__name__} must admit the rigid base"
        assert mats.RigidBodyMaterialFragment in args, f"{cls.__name__} must admit fragments"
        assert mats.PhysicsMaterialCfg not in args, f"{cls.__name__} is rigid-only"
    for cls in (FileCfg, MeshCfg):
        args = union_args(cls)
        assert mats.PhysicsMaterialCfg in args, f"{cls.__name__} spawns deformables too"
        assert mats.RigidBodyMaterialFragment in args
