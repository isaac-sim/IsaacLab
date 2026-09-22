# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Schema fragment writers: family dispatch, backend fragments, spawner routing and USD-asset targeting.

The tests author on in-memory USD stages and do not launch Isaac Sim / Kit. PhysX schema
definitions are registered from the OVPhysX codeless schemas when the PhysX runtime is absent, so
``GetAppliedSchemas`` still reports ``Physx*`` tokens. The PhysX root relocation performed through the
physics manager needs a live simulation and is covered in ``test_schemas.py``.
"""

import dataclasses
import inspect
import math
import os
from typing import ClassVar

import pytest
from isaaclab_newton.sim.schemas import (
    MujocoCollisionCfg,
    MujocoFixedTendonCfg,
    MujocoJointCfg,
    MujocoRigidBodyCfg,
    NewtonArticulationCfg,
    NewtonCollisionCfg,
    NewtonMeshCollisionCfg,
    NewtonSDFCollisionCfg,
    apply_mujoco_collision,
    apply_mujoco_fixed_tendon,
)
from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionCfg,
    PhysxCollisionPropertiesCfg,
    PhysxConvexDecompositionCfg,
    PhysxConvexHullCfg,
    PhysxDeformableBodyPropertiesCfg,
    PhysxFixedTendonPropertiesCfg,
    PhysxJointCfg,
    PhysxRigidBodyCfg,
    PhysxRigidBodyPropertiesCfg,
    PhysxSDFMeshCfg,
    PhysxSpatialTendonPropertiesCfg,
    PhysxTendonAttachmentRootCfg,
    PhysxTendonAxisCfg,
    PhysxTendonAxisRootCfg,
    PhysxTriangleMeshCfg,
    PhysxTriangleMeshSimplificationCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxSurfaceDeformableBodyMaterialCfg

from pxr import Gf, Plug, Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.physics import PhysicsManager
from isaaclab.sim.schemas import (
    ArticulationRootFragment,
    CollisionFragment,
    JointDriveFragment,
    MassCfg,
    MassFragment,
    MassPropertiesCfg,
    MeshCollisionFragment,
    RigidBodyFragment,
    SchemaFragment,
    UsdPhysicsCollisionCfg,
    UsdPhysicsDriveCfg,
    UsdPhysicsMeshCollisionCfg,
    UsdPhysicsRigidBodyCfg,
    _backend_hooks,
    apply_articulation_root_properties,
    apply_collision_properties,
    apply_drive,
    apply_fixed_tendon_properties,
    apply_joint_drive_properties,
    apply_mass_properties,
    apply_mesh_collision,
    apply_mesh_collision_properties,
    apply_namespaced,
    apply_rigid_body_properties,
    apply_spatial_tendon_properties,
    modify_fixed_tendon_properties,
    modify_spatial_tendon_properties,
)
from isaaclab.sim.schemas.schemas import create_world_fixed_joint
from isaaclab.sim.spawners._utils import fragment_mapping
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import safe_set_attribute_on_usd_prim
from isaaclab.utils import configclass
from isaaclab.utils.string import to_camel_case

pytestmark = pytest.mark.unit


@pytest.fixture
def stage() -> Usd.Stage:
    """A fresh current stage, so spawners and explicit-stage writers author on the same stage."""
    sim_utils.create_new_stage()
    return sim_utils.get_current_stage()


def _register_physx_codeless_schemas() -> None:
    """Register OVPhysX's codeless schemas so ``Physx*`` applied schemas resolve without Kit.

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
def physx_schemas() -> None:
    """Skip when the PhysX applied schemas are unknown to this process's USD schema registry."""
    if Usd.SchemaRegistry().FindAppliedAPIPrimDefinition("PhysxTendonAxisRootAPI") is None:
        pytest.skip("PhysX schemas are not registered in this process")


def _xform(stage: Usd.Stage, path: str, *apis) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    for api in apis:
        api.Apply(prim)
    return prim


def _joint(stage: Usd.Stage, path: str, joint_type=UsdPhysics.RevoluteJoint, body1: str | None = None) -> Usd.Prim:
    joint = joint_type.Define(stage, path)
    if body1 is not None:
        joint.CreateBody1Rel().SetTargets([body1])
    return joint.GetPrim()


def _instance_of(stage: Usd.Stage, source_path: str, instance_path: str) -> Usd.Prim:
    """Reference ``source_path`` from an instanceable prim so its children become read-only proxies."""
    instance = UsdGeom.Xform.Define(stage, instance_path).GetPrim()
    instance.GetReferences().AddInternalReference(source_path)
    instance.SetInstanceable(True)
    return instance


def _authored(prim: Usd.Prim, attr: str) -> bool:
    return prim.GetAttribute(attr).HasAuthoredValue()


def _api_schemas(prim: Usd.Prim) -> set[str]:
    """Applied API schema names including unregistered token schemas."""
    return set(prim.GetPrimTypeInfo().GetAppliedAPISchemas())


def _schema_attrs(schema_name: str) -> set[str]:
    """Attribute names (without namespace prefix) declared by a registered applied API schema."""
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_name)
    return {str(name).split(":", 1)[1] for name in definition.GetPropertyNames()}


def _cfg_attrs(cfg_type, exclude: frozenset[str] = frozenset()) -> set[str]:
    """camelCase names of a cfg's dataclass fields, minus ``func`` and ``exclude``."""
    return {to_camel_case(f.name) for f in dataclasses.fields(cfg_type) if f.name not in exclude | {"func"}}


"""
Fragment metadata.
"""


@pytest.mark.parametrize(
    ("cfg", "marker", "namespace", "applied_schema", "func"),
    [
        (UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), RigidBodyFragment, "physics", None, "apply_namespaced"),
        (UsdPhysicsCollisionCfg(collision_enabled=True), CollisionFragment, "physics", None, "apply_namespaced"),
        (MassCfg(mass=2.0), MassFragment, "physics", None, "apply_namespaced"),
        (UsdPhysicsMeshCollisionCfg(), MeshCollisionFragment, "physics", None, "apply_mesh_collision"),
        (UsdPhysicsDriveCfg(stiffness=10.0), JointDriveFragment, None, None, "apply_drive"),
        (
            PhysxArticulationCfg(articulation_enabled=True),
            ArticulationRootFragment,
            "physxArticulation",
            "PhysxArticulationAPI",
            "apply_namespaced",
        ),
    ],
)
def test_fragment_metadata(cfg, marker, namespace, applied_schema, func):
    """Core fragments carry their marker base, USD namespace, owned schema and default applier."""
    assert isinstance(cfg, marker) and isinstance(cfg, SchemaFragment)
    assert type(cfg)._usd_namespace == namespace
    assert type(cfg)._usd_applied_schema == applied_schema
    assert cfg.func == f"isaaclab.sim.schemas:{func}"


@pytest.mark.parametrize(
    ("cfg_type", "kwargs", "alias", "canonical"),
    [
        (UsdPhysicsDriveCfg, {"max_effort": 42.0}, "max_effort", "max_force"),
        (PhysxJointCfg, {"max_velocity": 10.0}, "max_velocity", "max_joint_velocity"),
    ],
)
def test_renamed_field_alias_forwards(cfg_type, kwargs, alias, canonical):
    with pytest.warns(DeprecationWarning, match=alias):
        cfg = cfg_type(**kwargs)
    assert getattr(cfg, canonical) == kwargs[alias]
    assert getattr(cfg, alias) is None


def test_fragment_mapping_normalizes_bare_fragment_and_list():
    """A bare fragment (or list) on a spawner field is shorthand for the anchor-prim mapping."""
    frag = UsdPhysicsRigidBodyCfg(rigid_body_enabled=True)
    a, b = MassCfg(mass=1.0), MassCfg(density=10.0)
    assert fragment_mapping(frag) == {"": [frag]}
    assert fragment_mapping([a, b]) == fragment_mapping((a, b)) == {"": [a, b]}
    mapping = {"/.*": [frag]}
    assert fragment_mapping(mapping) is mapping
    # legacy dataclass cfgs report None so callers route them to the legacy writers
    assert fragment_mapping(MassPropertiesCfg(mass=1.0)) is None
    assert fragment_mapping(None) is None


"""
apply_namespaced.
"""


@pytest.mark.parametrize(
    ("fragment", "expected", "unauthored"),
    [
        (
            UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
            {"physics:rigidBodyEnabled": True},
            ["physics:kinematicEnabled"],
        ),
        (
            PhysxRigidBodyCfg(linear_damping=0.1, disable_gravity=True),
            {"physxRigidBody:linearDamping": 0.1, "physxRigidBody:disableGravity": True},
            [],
        ),
        (MujocoRigidBodyCfg(gravcomp=1.0), {"mjc:gravcomp": 1.0}, []),
        (MujocoRigidBodyCfg(), {}, ["mjc:gravcomp"]),
        (UsdPhysicsCollisionCfg(collision_enabled=True), {"physics:collisionEnabled": True}, []),
        (
            PhysxCollisionCfg(contact_offset=0.02, rest_offset=0.0, torsional_patch_radius=0.1),
            {
                "physxCollision:contactOffset": 0.02,
                "physxCollision:restOffset": 0.0,
                "physxCollision:torsionalPatchRadius": 0.1,
            },
            [],
        ),
        (
            NewtonCollisionCfg(contact_margin=0.01, contact_gap=0.005),
            {"newton:contactMargin": 0.01, "newton:contactGap": 0.005},
            [],
        ),
        (MassCfg(mass=3.0), {"physics:mass": 3.0}, ["physics:density"]),
        (
            PhysxArticulationCfg(articulation_enabled=True, enabled_self_collisions=False, sleep_threshold=0.1),
            {
                "physxArticulation:articulationEnabled": True,
                "physxArticulation:enabledSelfCollisions": False,
                "physxArticulation:sleepThreshold": 0.1,
            },
            [],
        ),
        (NewtonArticulationCfg(self_collision_enabled=True), {"newton:selfCollisionEnabled": True}, []),
        (MujocoJointCfg(actuatorgravcomp=True), {"mjc:actuatorgravcomp": True}, []),
    ],
)
def test_apply_namespaced_writes_only_set_fields(stage, fragment, expected, unauthored):
    """Every non-None field lands under the fragment's namespace; None fields stay unauthored."""
    prim = _xform(stage, "/World/Body")
    apply_namespaced(fragment, "/World/Body", stage)
    for attr, value in expected.items():
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), attr
    for attr in unauthored:
        assert not _authored(prim, attr), attr


def test_apply_namespaced_rejects_invalid_prim_and_missing_namespace(stage):
    @configclass
    class _NoNamespaceFragment(RigidBodyFragment):
        _usd_namespace: ClassVar[str | None] = None
        rigid_body_enabled: bool | None = None

    with pytest.raises(ValueError):
        apply_namespaced(UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), "/World/DoesNotExist", stage)
    _xform(stage, "/World/NoNs", UsdPhysics.RigidBodyAPI)
    with pytest.raises(ValueError):
        apply_namespaced(_NoNamespaceFragment(rigid_body_enabled=True), "/World/NoNs", stage)


"""
Anchor-API family writers: rigid body, collision, mass, articulation root.
"""

FAMILIES = [
    pytest.param(
        apply_rigid_body_properties,
        UsdPhysics.RigidBodyAPI,
        lambda: PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
        "physxRigidBody:maxDepenetrationVelocity",
        5.0,
        id="rigid_body",
    ),
    pytest.param(
        apply_collision_properties,
        UsdPhysics.CollisionAPI,
        lambda: PhysxCollisionCfg(contact_offset=0.02),
        "physxCollision:contactOffset",
        0.02,
        id="collision",
    ),
    pytest.param(apply_mass_properties, UsdPhysics.MassAPI, lambda: MassCfg(mass=2.0), "physics:mass", 2.0, id="mass"),
    pytest.param(
        apply_articulation_root_properties,
        UsdPhysics.ArticulationRootAPI,
        lambda: PhysxArticulationCfg(solver_position_iteration_count=8),
        "physxArticulation:solverPositionIterationCount",
        8,
        id="articulation",
    ),
]


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_creates_on_every_matched_prim(stage, writer, api, make_fragment, attr, value):
    """``create_if_missing`` applies the anchor to every matched prim lacking it; the expression is trusted."""
    prims = [_xform(stage, path) for path in ("/World/Grp", "/World/Grp/a", "/World/Grp/b")]
    assert writer("/World/Grp(/.*)?", [make_fragment()], create_if_missing=True, stage=stage) is True
    for prim in prims:
        assert prim.HasAPI(api), prim.GetPath()
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), prim.GetPath()


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_targets_existing_carriers_only(stage, writer, api, make_fragment, attr, value):
    """Without creation only prims already carrying the anchor are modified, and only those matched."""
    left = _xform(stage, "/World/Bot/armL", api)
    right = _xform(stage, "/World/Bot/armR", api)
    bare = _xform(stage, "/World/Bot/frame")
    assert writer("/World/Bot/(armL|frame)", [make_fragment()], stage=stage) is True
    assert left.GetAttribute(attr).Get() == pytest.approx(value)
    assert not _authored(right, attr)
    assert not bare.HasAPI(api) and not _authored(bare, attr)


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_zero_targets_warn_and_empty_list_is_noop(stage, writer, api, make_fragment, attr, value, caplog):
    """No carrier and no creation: warn, author nothing, return False. No fragments: silent True."""
    bare = _xform(stage, "/World/Bare")
    with caplog.at_level("WARNING"):
        assert writer("/World/Bare", [make_fragment()], stage=stage) is False
    assert "/World/Bare" in caplog.text
    assert not bare.HasAPI(api)
    assert writer("/World/Bare", [], create_if_missing=True, stage=stage) is True
    assert not bare.HasAPI(api)


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_visits_every_target_and_aggregates_results(stage, writer, api, make_fragment, attr, value):
    """Every sibling carrier is visited in order and one failing applier makes the family result False."""
    for path in ("/World/A", "/World/B"):
        _xform(stage, path, api)
    visited = []

    def record(_cfg, path, _stage):
        visited.append(path)
        return path != "/World/B"

    failing = make_fragment()
    failing.func = record
    assert writer("/World/(A|B)", [failing], stage=stage) is False
    assert visited == ["/World/A", "/World/B"]
    assert writer("/World/(A|B)", [make_fragment()], stage=stage) is True


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_reaches_nested_carriers(stage, writer, api, make_fragment, attr, value, caplog):
    """Carriers nested under carriers (URDF-importer layouts) are all authored."""
    paths = ["/World/Robot/pelvis", "/World/Robot/pelvis/hip", "/World/Robot/pelvis/hip/knee"]
    prims = [_xform(stage, path, api) for path in paths]
    with caplog.at_level("WARNING"):
        assert writer("/World/Robot(/.*)?", [make_fragment()], stage=stage) is True
    for prim in prims:
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), prim.GetPath()
    if api is UsdPhysics.ArticulationRootAPI:
        # nested roots are the asset author's responsibility; the writer only warns
        assert "nested" in caplog.text.lower()


@pytest.mark.parametrize(("writer", "api", "make_fragment", "attr", "value"), FAMILIES)
def test_family_writer_skips_instanced_carriers(stage, writer, api, make_fragment, attr, value, caplog):
    """A carrier inside an instance is a read-only proxy: skipped with a warning and a False return."""
    _xform(stage, "/World/Source")
    _xform(stage, "/World/Source/body", api)
    instance = _instance_of(stage, "/World/Source", "/World/Asset")
    proxy = stage.GetPrimAtPath("/World/Asset/body")
    assert proxy.IsInstanceProxy() and proxy.HasAPI(api)
    with caplog.at_level("WARNING"):
        assert writer("/World/Asset(/.*)?", [make_fragment()], stage=stage) is False
    assert not instance.HasAPI(api)
    assert not _authored(proxy, attr)
    assert "/World/Asset/body" in caplog.text


@pytest.mark.parametrize(
    ("writer", "api", "fragments", "expected"),
    [
        pytest.param(
            apply_rigid_body_properties,
            UsdPhysics.RigidBodyAPI,
            [
                UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
                PhysxRigidBodyCfg(linear_damping=0.2),
                MujocoRigidBodyCfg(gravcomp=1.0),
            ],
            {"physics:rigidBodyEnabled": True, "physxRigidBody:linearDamping": 0.2, "mjc:gravcomp": 1.0},
            id="rigid_body",
        ),
        pytest.param(
            apply_collision_properties,
            UsdPhysics.CollisionAPI,
            [
                UsdPhysicsCollisionCfg(collision_enabled=True),
                PhysxCollisionCfg(contact_offset=0.02),
                NewtonCollisionCfg(contact_margin=0.01),
                MujocoCollisionCfg(condim=4),
            ],
            {
                "physics:collisionEnabled": True,
                "physxCollision:contactOffset": 0.02,
                "newton:contactMargin": 0.01,
                "mjc:condim": 4,
            },
            id="collision",
        ),
        pytest.param(
            apply_mass_properties,
            UsdPhysics.MassAPI,
            [MassCfg(mass=5.0, density=100.0)],
            {"physics:mass": 5.0, "physics:density": 100.0},
            id="mass",
        ),
        pytest.param(
            apply_articulation_root_properties,
            UsdPhysics.ArticulationRootAPI,
            [
                PhysxArticulationCfg(enabled_self_collisions=True, solver_position_iteration_count=8),
                NewtonArticulationCfg(self_collision_enabled=True),
            ],
            {
                "physxArticulation:enabledSelfCollisions": True,
                "physxArticulation:solverPositionIterationCount": 8,
                "newton:selfCollisionEnabled": True,
            },
            id="articulation",
        ),
    ],
)
def test_family_writer_composes_backend_namespaces(stage, writer, api, fragments, expected):
    """One fragment list authors the USD, PhysX and Newton namespaces on the same anchored prim."""
    prim = _xform(stage, "/World/Body")
    assert writer("/World/Body", fragments, create_if_missing=True, stage=stage) is True
    assert prim.HasAPI(api)
    for attr, value in expected.items():
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), attr


def test_mujoco_collision_fragment_authors_typed_arrays_and_validates(stage):
    prim = _xform(stage, "/World/C", UsdPhysics.CollisionAPI)
    cfg = MujocoCollisionCfg(
        condim=4, group=2, priority=3, solimp=(0.9, 0.99, 0.001, 0.5, 2.0), solmix=0.75, solref=(0.02, 1.0)
    )
    apply_mujoco_collision(cfg, "/World/C", stage)
    assert (prim.GetAttribute("mjc:condim").Get(), prim.GetAttribute("mjc:group").Get()) == (4, 2)
    assert prim.GetAttribute("mjc:priority").Get() == 3
    assert prim.GetAttribute("mjc:solmix").Get() == pytest.approx(0.75)
    for name, value in (("solimp", (0.9, 0.99, 0.001, 0.5, 2.0)), ("solref", (0.02, 1.0))):
        attr = prim.GetAttribute(f"mjc:{name}")
        assert attr.GetTypeName() == Sdf.ValueTypeNames.DoubleArray
        assert tuple(attr.Get()) == pytest.approx(value)
    # unset fields are not authored
    sparse = _xform(stage, "/World/Sparse", UsdPhysics.CollisionAPI)
    apply_mujoco_collision(MujocoCollisionCfg(condim=6), "/World/Sparse", stage)
    assert all(
        not sparse.GetAttribute(f"mjc:{n}").IsValid() for n in ("group", "priority", "solimp", "solmix", "solref")
    )
    for kwargs, message in (
        ({"condim": 2}, "'condim' must be one of"),
        ({"group": 6}, "'group' must be between"),
        ({"priority": -1}, "'priority' must be non-negative"),
        ({"solmix": -0.1}, "'solmix' must be non-negative"),
        ({"solimp": (0.9, 0.95, 0.001, 0.5)}, "'solimp' must contain exactly 5"),
        ({"solref": (0.02,)}, "'solref' must contain exactly 2"),
    ):
        with pytest.raises(ValueError, match=message):
            apply_mujoco_collision(MujocoCollisionCfg(**kwargs), "/World/C", stage)


"""
Articulation root: targeting and topology flags that do not need a physics manager.
"""


def test_articulation_writer_tunes_existing_child_root_without_duplicating(stage):
    """A root on a child prim (as in USD assets) is tuned in place; the top prim gains no second root."""
    top = _xform(stage, "/World/Asset")
    child = _xform(stage, "/World/Asset/base", UsdPhysics.ArticulationRootAPI)
    apply_articulation_root_properties(
        "/World/Asset(/.*)?", [PhysxArticulationCfg(solver_position_iteration_count=8)], stage
    )
    assert child.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert not top.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)] == [child]


def test_articulation_writer_rejects_non_fragments_and_topology_only_does_not_stamp_root(stage):
    _xform(stage, "/World/BadList")
    with pytest.raises(TypeError, match="ArticulationRootFragment"):
        apply_articulation_root_properties(
            "/World/BadList", [PhysxArticulationRootPropertiesCfg(solver_position_iteration_count=8)], stage
        )
    prim = _xform(stage, "/World/NoRoot", UsdPhysics.RigidBodyAPI)
    apply_articulation_root_properties("/World/NoRoot", [], stage, fix_root_link=False)
    assert not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    assert not prim.HasAPI(UsdPhysics.ArticulationRootAPI)


def _root_with_world_joint(stage: Usd.Stage, path: str, enabled: bool) -> UsdPhysics.FixedJoint:
    root = _xform(stage, path, UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    joint = UsdPhysics.FixedJoint.Define(stage, f"{path}/FixedJoint")
    joint.CreateBody1Rel().SetTargets([root.GetPath()])
    joint.CreateJointEnabledAttr(enabled)
    return joint


def test_fix_root_link_false_disables_existing_joint_on_the_explicit_stage(stage):
    """Fragment writes and the fixed-joint lookup both stay on the supplied stage."""
    current_joint = _root_with_world_joint(stage, "/World/Robot", enabled=True)
    other_stage = Usd.Stage.CreateInMemory()
    other_joint = _root_with_world_joint(other_stage, "/World/Robot", enabled=True)
    apply_articulation_root_properties(
        "/World/Robot", [PhysxArticulationCfg(solver_position_iteration_count=4)], other_stage, fix_root_link=False
    )
    other_root = other_stage.GetPrimAtPath("/World/Robot")
    assert other_root.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 4
    assert other_joint.GetJointEnabledAttr().Get() is False
    assert not _authored(stage.GetPrimAtPath("/World/Robot"), "physxArticulation:solverPositionIterationCount")
    assert current_joint.GetJointEnabledAttr().Get() is True


def test_fix_root_link_true_requires_active_simulation(stage):
    """Fixing the base resolves the backend from the live simulation, so it fails clearly without one."""
    _xform(stage, "/World/Robot", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    with pytest.raises(RuntimeError):
        apply_articulation_root_properties("/World/Robot", [], stage, fix_root_link=True)


def test_articulation_fragment_and_legacy_cfg_match_physx_schema(physx_schemas):
    """Both interfaces cover exactly the attributes registered by ``PhysxArticulationAPI``."""
    schema_attrs = _schema_attrs("PhysxArticulationAPI")
    assert _cfg_attrs(PhysxArticulationCfg) == schema_attrs
    assert _cfg_attrs(PhysxArticulationRootPropertiesCfg, frozenset({"fix_root_link"})) == schema_attrs


"""
World fixed joint authoring (pure USD, shared by every backend).
"""


def _fixed_joints(stage: Usd.Stage) -> list[UsdPhysics.FixedJoint]:
    return [UsdPhysics.FixedJoint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]


def test_create_world_fixed_joint_anchors_root_pose_and_avoids_instanceable_root():
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    root = _xform(stage, "/World/Robot", UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI)
    UsdGeom.Xform(root).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    create_world_fixed_joint(root, stage)
    (joint,) = _fixed_joints(stage)
    # world-attached: body0 empty, body1 the root link, anchored at the root's world translation
    assert joint.GetBody0Rel().GetTargets() == []
    assert joint.GetBody1Rel().GetTargets() == [root.GetPath()]
    assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(1.0, 2.0, 3.0)
    assert joint.GetBreakForceAttr().Get() > 1e37 and joint.GetBreakTorqueAttr().Get() > 1e37

    # an instanceable root is not authorable, so the joint goes under the first writable ancestor
    root.SetInstanceable(True)
    create_world_fixed_joint(root, stage)
    new_joint = [j for j in _fixed_joints(stage) if j.GetPrim() != joint.GetPrim()][0]
    assert not new_joint.GetPath().pathString.startswith("/World/Robot/")
    assert new_joint.GetBody1Rel().GetTargets() == [root.GetPath()]


def test_base_physics_manager_fix_articulation_root_is_idempotent_and_needs_rigid_body(stage):
    """The neutral capability pins the current pose, enables an existing joint and keeps the root in place."""
    _xform(stage, "/World/Robot")
    root = _xform(stage, "/World/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    UsdGeom.Xform(root).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    assert PhysicsManager.fix_articulation_root(root, stage) == root
    (joint,) = _fixed_joints(stage)
    assert list(joint.GetBody1Rel().GetTargets()) == [root.GetPath()]
    assert joint.GetLocalPos0Attr().Get() == Gf.Vec3f(1.0, 2.0, 3.0)
    assert not stage.GetPrimAtPath("/World/Robot").HasAPI(UsdPhysics.ArticulationRootAPI)
    joint.CreateJointEnabledAttr(False)
    assert PhysicsManager.fix_articulation_root(root, stage) == root
    assert joint.GetJointEnabledAttr().Get() is True and len(_fixed_joints(stage)) == 1

    no_body = _xform(stage, "/World/NoRB", UsdPhysics.ArticulationRootAPI)
    with pytest.raises(NotImplementedError):
        PhysicsManager.fix_articulation_root(no_body, stage)


"""
Mesh collision.
"""


@pytest.mark.parametrize(
    ("fragment", "expected", "token_schema"),
    [
        (
            PhysxConvexHullCfg(hull_vertex_limit=32, min_thickness=0.002),
            {"physxConvexHullCollision:hullVertexLimit": 32, "physxConvexHullCollision:minThickness": 0.002},
            None,
        ),
        (
            PhysxConvexDecompositionCfg(max_convex_hulls=8, shrink_wrap=True),
            {
                "physxConvexDecompositionCollision:maxConvexHulls": 8,
                "physxConvexDecompositionCollision:shrinkWrap": True,
            },
            None,
        ),
        (PhysxTriangleMeshCfg(weld_tolerance=0.01), {"physxTriangleMeshCollision:weldTolerance": 0.01}, None),
        (
            PhysxTriangleMeshSimplificationCfg(simplification_metric=0.7),
            {"physxTriangleMeshSimplificationCollision:simplificationMetric": 0.7},
            None,
        ),
        (
            PhysxSDFMeshCfg(sdf_resolution=128, sdf_margin=0.02),
            {"physxSDFMeshCollision:sdfResolution": 128, "physxSDFMeshCollision:sdfMargin": 0.02},
            None,
        ),
        (NewtonMeshCollisionCfg(max_hull_vertices=24), {"newton:maxHullVertices": 24}, "NewtonMeshCollisionAPI"),
        (
            NewtonSDFCollisionCfg(sdf_max_resolution=64, hydroelastic_enabled=True),
            {"newton:sdfMaxResolution": 64, "newton:hydroelasticEnabled": True},
            "NewtonSDFCollisionAPI",
        ),
    ],
)
def test_cooking_fragment_writes_its_namespace(stage, fragment, expected, token_schema):
    """Each cooking fragment writes its own namespace; the approximation name is never a namespaced attr."""
    prim = _xform(stage, "/World/Mesh", UsdPhysics.MeshCollisionAPI)
    apply_namespaced(fragment, "/World/Mesh", stage)
    for attr, value in expected.items():
        assert prim.GetAttribute(attr).Get() == pytest.approx(value), attr
    namespace = type(fragment)._usd_namespace
    assert not prim.HasAttribute(f"{namespace}:meshApproximationName")
    if token_schema is not None:
        # Newton ships unregistered token schemas, visible only through the prim type info
        assert token_schema in _api_schemas(prim)


def test_mesh_collision_writer_applies_anchor_and_composes_token(stage):
    """The anchor is implicit and the last non-"none" cooking fragment sets ``physics:approximation``."""
    prim = _xform(stage, "/World/M0")
    assert apply_mesh_collision_properties(
        "/World/M0", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="boundingCube")], stage
    )
    assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
    assert prim.GetAttribute("physics:approximation").Get() == "boundingCube"
    # fragments may come from any iterable; the convex-hull cooking fragment implies its token
    fragments = (
        f
        for f in [
            UsdPhysicsMeshCollisionCfg(),
            PhysxConvexHullCfg(hull_vertex_limit=48),
            NewtonMeshCollisionCfg(max_hull_vertices=48),
        ]
    )
    assert apply_mesh_collision_properties("/World/M0", fragments, stage)
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 48
    assert prim.GetAttribute("newton:maxHullVertices").Get() == 48


def test_apply_mesh_collision_writes_namespace_and_implied_token(stage):
    prim = _xform(stage, "/World/Mfunc", UsdPhysics.MeshCollisionAPI)
    assert apply_mesh_collision(PhysxConvexHullCfg(hull_vertex_limit=16), "/World/Mfunc", stage)
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 16
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"


@pytest.mark.parametrize(
    "writer", [apply_mesh_collision, lambda cfg, path, stage: apply_mesh_collision_properties(path, [cfg], stage)]
)
def test_mesh_collision_writers_reject_invalid_token_and_prim(stage, writer):
    _xform(stage, "/World/Mesh")
    with pytest.raises(ValueError):
        writer(UsdPhysicsMeshCollisionCfg(mesh_approximation_name="notAToken"), "/World/Mesh", stage)
    with pytest.raises(ValueError):
        writer(UsdPhysicsMeshCollisionCfg(), "/World/DoesNotExist", stage)


def test_mesh_collision_writer_aggregates_fragment_results(stage):
    _xform(stage, "/World/Magg")
    failing, ok = UsdPhysicsMeshCollisionCfg(), UsdPhysicsMeshCollisionCfg()
    failing.func = lambda cfg, prim_path, stage=None: False
    ok.func = lambda cfg, prim_path, stage=None: True
    assert apply_mesh_collision_properties("/World/Magg", [failing, ok], stage) is False
    assert apply_mesh_collision_properties("/World/Magg", [ok], stage) is True


def test_mesh_collision_fragments_author_on_every_matched_collider(stage):
    """Through the collision family writer, mesh fragments author on all matched colliders."""
    colliders = [
        _xform(stage, "/World/Grp/agg", UsdPhysics.CollisionAPI),
        UsdGeom.Cube.Define(stage, "/World/Grp/box").GetPrim(),
    ]
    UsdPhysics.CollisionAPI.Apply(colliders[1])
    assert apply_collision_properties(
        "/World/Grp(/.*)?", [UsdPhysicsMeshCollisionCfg(mesh_approximation_name="convexHull")], stage=stage
    )
    for prim in colliders:
        assert prim.HasAPI(UsdPhysics.MeshCollisionAPI), prim.GetPath()
        assert prim.GetAttribute("physics:approximation").Get() == "convexHull", prim.GetPath()


"""
Joint drives.
"""


@pytest.mark.parametrize(
    ("joint_type", "instance", "scale"),
    [(UsdPhysics.RevoluteJoint, "angular", math.pi / 180.0), (UsdPhysics.PrismaticJoint, "linear", 1.0)],
)
def test_apply_drive_selects_instance_and_converts_angular_gains(stage, joint_type, instance, scale):
    """Angular drives store gains per degree; linear drives are written as authored."""
    prim = _joint(stage, "/World/Art/joint", joint_type)
    assert apply_drive(
        UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0, damping=0.1),
        "/World/Art/joint",
        stage,
    )
    assert prim.HasAPI(UsdPhysics.DriveAPI, instance)
    assert prim.GetAttribute(f"drive:{instance}:physics:type").Get() == "acceleration"
    assert prim.GetAttribute(f"drive:{instance}:physics:maxForce").Get() == pytest.approx(80.0)
    assert prim.GetAttribute(f"drive:{instance}:physics:stiffness").Get() == pytest.approx(10.0 * scale)
    assert prim.GetAttribute(f"drive:{instance}:physics:damping").Get() == pytest.approx(0.1 * scale)
    _xform(stage, "/World/NotAJoint")
    assert apply_drive(UsdPhysicsDriveCfg(stiffness=1.0), "/World/NotAJoint", stage) is False


def test_joint_drive_writer_composes_namespaces_and_gates_drive_api(stage):
    """``DriveAPI`` is applied only when a drive fragment is present; PhysX converts angular velocity to deg/s."""
    revolute = _joint(stage, "/World/Art/j0")
    prismatic = _joint(stage, "/World/Art/j1", UsdPhysics.PrismaticJoint)
    assert apply_joint_drive_properties("/World/Art(/.*)?", [PhysxJointCfg(max_joint_velocity=5.0)], stage)
    assert not revolute.HasAPI(UsdPhysics.DriveAPI, "angular")
    assert revolute.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(math.degrees(5.0))
    assert prismatic.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(5.0)

    fragments = [
        UsdPhysicsDriveCfg(drive_type="acceleration", max_force=80.0, stiffness=10.0),
        PhysxJointCfg(max_joint_velocity=3.0),
        MujocoJointCfg(actuatorgravcomp=True),
    ]
    assert apply_joint_drive_properties("/World/Art/j0", fragments, stage)
    assert revolute.HasAPI(UsdPhysics.DriveAPI, "angular")
    assert revolute.GetAttribute("drive:angular:physics:maxForce").Get() == pytest.approx(80.0)
    assert revolute.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(math.radians(10.0))
    assert revolute.GetAttribute("physxJoint:maxJointVelocity").Get() == pytest.approx(math.degrees(3.0))
    assert revolute.GetAttribute("mjc:actuatorgravcomp").Get() is True


def test_joint_drive_writer_ensure_drives_exist_and_create_if_missing(stage):
    """``ensure_drives_exist`` seeds a minimal stiffness on passive drives; ``create_if_missing`` applies the API."""
    seeded = _joint(stage, "/World/Art/seeded")
    assert apply_joint_drive_properties(
        "/World/Art/seeded", [UsdPhysicsDriveCfg(max_force=1.0)], stage, ensure_drives_exist=True
    )
    assert seeded.GetAttribute("drive:angular:physics:stiffness").Get() == pytest.approx(math.radians(1e-3))

    bare = _joint(stage, "/World/Art/bare")
    assert not bare.HasAPI(UsdPhysics.DriveAPI, "angular")
    assert apply_joint_drive_properties(
        "/World/Art/bare", [PhysxJointCfg(max_joint_velocity=5.0)], stage, create_if_missing=True
    )
    assert bare.HasAPI(UsdPhysics.DriveAPI, "angular")


@pytest.mark.parametrize("skip_all", [True, False])
def test_joint_drive_writer_honors_registered_skip_predicates(stage, monkeypatch, skip_all):
    """Backends exclude joints through registered predicates; an empty registry skips nothing."""
    monkeypatch.setattr(_backend_hooks, "_JOINT_DRIVE_SKIP_PREDICATES", [])
    if skip_all:
        _backend_hooks.register_joint_drive_skip_predicate(lambda prim: True)
    joint = _joint(stage, "/World/Art/j0")
    apply_joint_drive_properties("/World/Art(/.*)?", [UsdPhysicsDriveCfg(stiffness=10.0)], stage)
    assert joint.HasAPI(UsdPhysics.DriveAPI, "angular") is not skip_all


def test_joint_drive_writer_skips_physx_tendon_child_joints(stage, physx_schemas):
    """A tendon-child joint (axis API without the root API) receives no drive and no PhysX joint attrs."""
    joint = _joint(stage, "/World/Art/j0")
    joint.AddAppliedSchema("PhysxTendonAxisAPI:axis0")
    apply_joint_drive_properties(
        "/World/Art(/.*)?", [UsdPhysicsDriveCfg(stiffness=10.0), PhysxJointCfg(max_joint_velocity=5.0)], stage
    )
    assert not joint.HasAPI(UsdPhysics.DriveAPI, "angular")
    assert not _authored(joint, "physxJoint:maxJointVelocity")


def test_joint_drive_writer_instanced_joints_warn_once_and_fail(stage, caplog):
    _xform(stage, "/World/Src")
    _joint(stage, "/World/Src/j0")
    _instance_of(stage, "/World/Src", "/World/Bot")
    with caplog.at_level("WARNING"):
        assert apply_joint_drive_properties("/World/Bot(/.*)?", [PhysxJointCfg(max_joint_velocity=5.0)], stage) is False
    assert "Skipping fragment updates on instanced prims" in caplog.text
    assert "Could not apply joint-drive properties" not in caplog.text


def test_mujoco_joint_actuatorgravcomp_enables_unset_body_gravcomp(stage):
    """``actuatorgravcomp`` is inert without body gravcomp, so the applier enables it on each joint's child body."""
    for link in ("link_a", "link_b", "link_c"):
        UsdGeom.Cube.Define(stage, f"/World/Art/{link}")
    _joint(stage, "/World/Art/j0", body1="/World/Art/link_a")
    _joint(stage, "/World/Art/j1", UsdPhysics.PrismaticJoint, body1="/World/Art/link_b")
    _joint(stage, "/World/Art/j2", body1="/World/Art/link_c")
    authored_body = stage.GetPrimAtPath("/World/Art/link_c")
    safe_set_attribute_on_usd_prim(authored_body, "mjc:gravcomp", 0.5, camel_case=False)
    apply_joint_drive_properties("/World/Art(/.*)?", [MujocoJointCfg(actuatorgravcomp=True)], stage)
    for link in ("link_a", "link_b"):
        assert stage.GetPrimAtPath(f"/World/Art/{link}").GetAttribute("mjc:gravcomp").Get() == pytest.approx(1.0)
    # an explicitly authored body value is preserved
    assert authored_body.GetAttribute("mjc:gravcomp").Get() == pytest.approx(0.5)


def test_mujoco_joint_without_actuatorgravcomp_authors_nothing(stage):
    UsdGeom.Cube.Define(stage, "/World/Art/body1")
    joint = _joint(stage, "/World/Art/j0", body1="/World/Art/body1")
    apply_joint_drive_properties("/World/Art(/.*)?", [MujocoJointCfg()], stage)
    assert not _authored(joint, "mjc:actuatorgravcomp")
    assert stage.GetPrimAtPath("/World/Art/body1").GetAttribute("mjc:gravcomp").Get() is None


"""
Tendons (tune-not-apply families).
"""


def _prim_with_schema_tokens(stage: Usd.Stage, path: str, tokens: list[str]) -> Usd.Prim:
    prim = _xform(stage, path)
    token_op = Sdf.TokenListOp()
    token_op.explicitItems = tokens
    prim.SetMetadata("apiSchemas", token_op)
    return prim


@pytest.mark.parametrize(
    ("instance_names", "selected"),
    [("t0", {"t0"}), (None, {"t0", "t1", "t2"}), (["t0", "t2"], {"t0", "t2"})],
)
def test_fixed_tendon_root_fragment_selects_instances(stage, physx_schemas, instance_names, selected):
    prim = _prim_with_schema_tokens(stage, "/World/FT", [f"PhysxTendonAxisRootAPI:{i}" for i in ("t0", "t1", "t2")])
    cfg = PhysxTendonAxisRootCfg(instance_names=instance_names, stiffness=9.0, lower_limit=-0.2)
    assert apply_fixed_tendon_properties("/World/FT", [cfg], stage)
    for instance in ("t0", "t1", "t2"):
        attr = prim.GetAttribute(f"physxTendon:{instance}:stiffness")
        assert attr.HasAuthoredValue() is (instance in selected)
        if instance in selected:
            assert attr.Get() == pytest.approx(9.0)
            assert prim.GetAttribute(f"physxTendon:{instance}:lowerLimit").Get() == pytest.approx(-0.2)


def test_fixed_tendon_axis_fragment_targets_root_and_child_axes(stage, physx_schemas):
    _xform(stage, "/World/Hand")
    root = _prim_with_schema_tokens(
        stage, "/World/Hand/root", ["PhysxTendonAxisRootAPI:index", "PhysxTendonAxisRootAPI:shared"]
    )
    child = _prim_with_schema_tokens(stage, "/World/Hand/child", ["PhysxTendonAxisAPI:index"])
    cfg = PhysxTendonAxisCfg(instance_names="index", gearing=[-0.5], force_coefficient=[2.0], joint_axis=["rotX"])
    assert apply_fixed_tendon_properties("/World/Hand(/.*)?", [cfg], stage)
    for prim in (root, child):
        assert list(prim.GetAttribute("physxTendon:index:gearing").Get()) == pytest.approx([-0.5])
        assert list(prim.GetAttribute("physxTendon:index:forceCoefficient").Get()) == pytest.approx([2.0])
        assert list(prim.GetAttribute("physxTendon:index:jointAxis").Get()) == ["rotX"]
    assert not _authored(root, "physxTendon:shared:gearing")


@pytest.mark.parametrize(
    ("cfg_type", "schema_type"),
    [
        (PhysxTendonAxisRootCfg, "PhysxTendonAxisRootAPI"),
        (PhysxTendonAxisCfg, "PhysxTendonAxisAPI"),
        (PhysxTendonAttachmentRootCfg, "PhysxTendonAttachmentRootAPI"),
    ],
)
def test_tendon_fragment_fields_belong_to_schema(physx_schemas, cfg_type, schema_type):
    definition = Usd.SchemaRegistry().FindAppliedAPIPrimDefinition(schema_type)
    schema_properties = {
        str(Usd.SchemaRegistry.GetMultipleApplyNameTemplateBaseName(str(name)))
        for name in definition.GetPropertyNames()
        if "__INSTANCE_NAME__" in str(name)
    }
    assert _cfg_attrs(cfg_type, frozenset({"instance_names"})) <= schema_properties


@pytest.mark.parametrize(
    ("writer", "cfg", "schema"),
    [
        (apply_fixed_tendon_properties, PhysxTendonAxisRootCfg(stiffness=5.0), "PhysxTendonAxisRootAPI:t0"),
        (
            apply_spatial_tendon_properties,
            PhysxTendonAttachmentRootCfg(stiffness=5.0),
            "PhysxTendonAttachmentRootAPI:t0",
        ),
    ],
)
def test_tendon_writers_descend_to_child_prims_and_dispatch_multiple_fragments(
    stage, physx_schemas, writer, cfg, schema
):
    _xform(stage, "/World/Robot")
    child = _prim_with_schema_tokens(stage, "/World/Robot/joint", [schema])
    assert writer("/World/Robot(/.*)?", [cfg, cfg.replace(stiffness=None, damping=0.75)], stage)
    assert child.GetAttribute("physxTendon:t0:stiffness").Get() == pytest.approx(5.0)
    assert child.GetAttribute("physxTendon:t0:damping").Get() == pytest.approx(0.75)


def test_spatial_tendon_writers_select_root_instances_and_skip_leaves(stage, physx_schemas):
    tokens = ["PhysxTendonAttachmentRootAPI:r0", "PhysxTendonAttachmentRootAPI:r1", "PhysxTendonAttachmentLeafAPI:l0"]
    prim = _prim_with_schema_tokens(stage, "/World/ST", tokens)
    cfg = PhysxTendonAttachmentRootCfg(instance_names="r0", stiffness=4.0, limit_stiffness=0.25)
    assert apply_spatial_tendon_properties("/World/ST", [cfg], stage)
    assert prim.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(4.0)
    assert prim.GetAttribute("physxTendon:r0:limitStiffness").Get() == pytest.approx(0.25)
    assert not _authored(prim, "physxTendon:r1:stiffness")
    assert not prim.GetAttribute("physxTendon:l0:stiffness").IsValid()
    # the legacy writer tunes every root instance and also skips leaves
    legacy = _prim_with_schema_tokens(stage, "/World/STlegacy", tokens)
    legacy_cfg = PhysxSpatialTendonPropertiesCfg(stiffness=6.0)
    assert inspect.unwrap(modify_spatial_tendon_properties)("/World/STlegacy", legacy_cfg, stage)
    assert legacy.GetAttribute("physxTendon:r0:stiffness").Get() == pytest.approx(6.0)
    assert legacy.GetAttribute("physxTendon:r1:stiffness").Get() == pytest.approx(6.0)
    assert not legacy.GetAttribute("physxTendon:l0:stiffness").IsValid()


def test_legacy_and_fragment_fixed_tendon_writers_author_identically(stage, physx_schemas):
    for root in ("/World/legacy", "/World/fragment"):
        _xform(stage, root)
        _prim_with_schema_tokens(stage, f"{root}/J0", ["PhysxTendonAxisRootAPI:t0", "PhysxTendonAxisRootAPI:t1"])
        _prim_with_schema_tokens(stage, f"{root}/nested/J1", ["PhysxTendonAxisRootAPI:t0"])
    with pytest.warns(DeprecationWarning):
        modify_fixed_tendon_properties(
            "/World/legacy", PhysxFixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1), stage
        )
    apply_fixed_tendon_properties(
        "/World/fragment(/.*)?", [PhysxTendonAxisRootCfg(limit_stiffness=30.0, damping=0.1)], stage
    )

    def collect(root: str) -> dict[str, float]:
        attrs = {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith("physxTendon:") and name.rsplit(":", 1)[1] in ("limitStiffness", "damping"):
                    if attr.HasAuthoredValue():
                        attrs[f"{prim.GetPath().pathString[len(root) :]}|{name}"] = attr.Get()
        return attrs

    legacy = collect("/World/legacy")
    assert legacy, "legacy writer authored no tendon attributes (test would be vacuous)"
    assert collect("/World/fragment") == pytest.approx(legacy)


def test_mujoco_fixed_tendon_writers_target_mjc_tendons_only(stage):
    tendon = stage.DefinePrim("/World/MjcT", "MjcTendon")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0, damping=0.25), "/World/MjcT", stage) is True
    assert tendon.GetAttribute("mjc:stiffness").Get() == pytest.approx(2.0)
    assert tendon.GetAttribute("mjc:damping").Get() == pytest.approx(0.25)
    other = _xform(stage, "/World/NotMjc")
    assert apply_mujoco_fixed_tendon(MujocoFixedTendonCfg(stiffness=2.0), "/World/NotMjc", stage) is False
    assert not other.HasAttribute("mjc:stiffness")
    # the legacy PhysX cfg shares only stiffness/damping with MuJoCo tendons
    legacy = stage.DefinePrim("/World/LegacyMjc", "MjcTendon")
    cfg = PhysxFixedTendonPropertiesCfg(stiffness=2.0, damping=0.25, lower_limit=-1.0, upper_limit=1.0)
    assert inspect.unwrap(modify_fixed_tendon_properties)("/World/LegacyMjc", cfg, stage)
    assert legacy.GetAttribute("mjc:stiffness").Get() == pytest.approx(2.0)
    assert not legacy.HasAttribute("mjc:lowerLimit") and not legacy.HasAttribute("mjc:upperLimit")


"""
Spawner routing.
"""


def test_shape_spawner_routes_fragment_slots(stage):
    """Bare fragments, lists and mappings author on the shape's anchor prims; an empty list is a no-op."""
    cfg = sim_utils.CuboidCfg(
        size=(1, 1, 1),
        rigid_props=[UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), PhysxRigidBodyCfg(linear_damping=0.3)],
        collision_props={"": [UsdPhysicsCollisionCfg(collision_enabled=True), PhysxCollisionCfg(contact_offset=0.03)]},
        mass_props=MassCfg(mass=4.0),
    )
    cfg.func("/World/Cube", cfg)
    prim = stage.GetPrimAtPath("/World/Cube")
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.HasAPI(UsdPhysics.MassAPI)
    assert prim.GetAttribute("physxRigidBody:linearDamping").Get() == pytest.approx(0.3)
    assert prim.GetAttribute("physics:mass").Get() == pytest.approx(4.0)
    mesh = stage.GetPrimAtPath("/World/Cube/geometry/mesh")
    assert mesh.HasAPI(UsdPhysics.CollisionAPI)
    assert mesh.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.03)

    empty = sim_utils.CuboidCfg(
        size=(1, 1, 1), rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True), mass_props={"": []}
    )
    empty.func("/World/Empty", empty)
    assert not _authored(stage.GetPrimAtPath("/World/Empty"), "physics:mass")


def test_deformable_collision_props_land_on_simulation_mesh_and_reject_legacy_cfg(stage):
    """Collision fragments author on the simulation mesh (the ``CollisionAPI`` carrier), never the body prim."""
    cfg = sim_utils.MeshCuboidCfg(
        size=(0.3, 0.04, 0.04),
        deformable_props=PhysxDeformableBodyPropertiesCfg(),
        collision_props=[PhysxCollisionCfg(contact_offset=0.005, rest_offset=0.0005)],
        # the surface branch needs no tetrahedralization dependency
        physics_material=PhysxSurfaceDeformableBodyMaterialCfg(),
    )
    cfg.func("/World/beam", cfg)
    sim_mesh = stage.GetPrimAtPath("/World/beam/sim_mesh")
    assert "PhysxCollisionAPI" in _api_schemas(sim_mesh)
    assert sim_mesh.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.005)
    assert sim_mesh.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(0.0005)
    assert not _authored(stage.GetPrimAtPath("/World/beam"), "physxCollision:restOffset")

    with pytest.warns(DeprecationWarning):
        legacy = sim_utils.MeshCuboidCfg(
            size=(0.1, 0.1, 0.1),
            deformable_props=PhysxDeformableBodyPropertiesCfg(),
            collision_props=PhysxCollisionPropertiesCfg(rest_offset=0.0005),
        )
    with pytest.raises(ValueError, match="collision fragments"):
        legacy.func("/World/beam_legacy", legacy)


"""
USD-asset targeting through the from-files spawner.
"""

LINK_REL_PATHS = ("link1", "link2", "link2/link3")
"""Link prim paths relative to the robot root; ``link2/link3`` is a nested child link."""


def _author_robot_usd(path: str) -> None:
    """A robot layout: the default prim carries the root, links carry rigid-body + mass, colliders collision."""
    asset = Usd.Stage.CreateNew(path)
    robot = UsdGeom.Xform.Define(asset, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for link_name in LINK_REL_PATHS:
        _xform(asset, f"/Robot/{link_name}", UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI)
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(asset, f"/Robot/{link_name}/collider").GetPrim())
    asset.SetDefaultPrim(robot.GetPrim())
    asset.Save()


def _spawn_usd(stage: Usd.Stage, tmp_path, prim_path: str, author=_author_robot_usd, **cfg_kwargs) -> Usd.Stage:
    usd_path = os.path.join(tmp_path, "asset.usda")
    author(usd_path)
    _spawn_from_usd_file(prim_path, usd_path, UsdFileCfg(usd_path=usd_path, **cfg_kwargs))
    return stage


@pytest.mark.parametrize(
    ("slot", "fragment", "api", "attr", "value", "carrier_suffix"),
    [
        (
            "rigid_props",
            PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
            UsdPhysics.RigidBodyAPI,
            "physxRigidBody:maxDepenetrationVelocity",
            5.0,
            "",
        ),
        (
            "collision_props",
            PhysxCollisionCfg(contact_offset=0.02),
            UsdPhysics.CollisionAPI,
            "physxCollision:contactOffset",
            0.02,
            "/collider",
        ),
        ("mass_props", MassCfg(mass=2.0), UsdPhysics.MassAPI, "physics:mass", 2.0, ""),
    ],
)
def test_fragments_target_existing_carriers_on_usd_asset(
    stage, tmp_path, slot, fragment, api, attr, value, carrier_suffix
):
    """Fragments modify the asset's existing carriers in place; the spawn prim gains no anchor."""
    _spawn_usd(stage, tmp_path, "/World/Robot", **{slot: {"(/.*)?": [fragment]}})
    spawn_prim = stage.GetPrimAtPath("/World/Robot")
    assert spawn_prim.HasAPI(UsdPhysics.ArticulationRootAPI) and not spawn_prim.HasAPI(api)
    assert not _authored(spawn_prim, attr)
    for link_name in LINK_REL_PATHS:
        carrier = stage.GetPrimAtPath(f"/World/Robot/{link_name}{carrier_suffix}")
        assert carrier.GetAttribute(attr).Get() == pytest.approx(value), link_name


def test_fragment_and_legacy_paths_place_apis_identically_on_usd_asset(stage, tmp_path):
    """On a real asset topology the fragment path places schemas exactly where the legacy path does."""
    usd_path = os.path.join(tmp_path, "robot.usda")
    _author_robot_usd(usd_path)
    with pytest.warns(DeprecationWarning):
        legacy_cfg = UsdFileCfg(
            usd_path=usd_path, rigid_props=PhysxRigidBodyPropertiesCfg(max_depenetration_velocity=5.0)
        )
    frag_cfg = UsdFileCfg(
        usd_path=usd_path, rigid_props={"(/.*)?": [PhysxRigidBodyCfg(max_depenetration_velocity=5.0)]}
    )
    _spawn_from_usd_file("/World/Legacy", usd_path, legacy_cfg)
    _spawn_from_usd_file("/World/Frag", usd_path, frag_cfg)

    def snapshot(root: str) -> dict[str, tuple[set[str], dict[str, object]]]:
        return {
            p.GetPath().pathString.removeprefix(root): (
                set(p.GetAppliedSchemas()),
                {a.GetName(): a.Get() for a in p.GetAttributes() if a.HasAuthoredValue()},
            )
            for p in Usd.PrimRange(stage.GetPrimAtPath(root))
        }

    assert snapshot("/World/Legacy") == snapshot("/World/Frag")


@pytest.mark.parametrize(
    ("rigid_props", "expected"),
    [
        (
            {"/link2(/.*)?": [PhysxRigidBodyCfg(max_depenetration_velocity=5.0)]},
            {"link1": None, "link2": 5.0, "link2/link3": 5.0},
        ),
        (
            {
                "(/.*)?": [PhysxRigidBodyCfg(max_depenetration_velocity=5.0)],
                "/link2(/.*)?": [PhysxRigidBodyCfg(max_depenetration_velocity=1.0)],
            },
            {"link1": 5.0, "link2": 1.0, "link2/link3": 1.0},
        ),
    ],
    ids=["narrowing_key", "insertion_order_override"],
)
def test_fragment_mapping_keys_narrow_and_override_in_order(stage, tmp_path, rigid_props, expected):
    _spawn_usd(stage, tmp_path, "/World/Robot", rigid_props=rigid_props)
    for link_name, value in expected.items():
        attr = stage.GetPrimAtPath(f"/World/Robot/{link_name}").GetAttribute("physxRigidBody:maxDepenetrationVelocity")
        if value is None:
            assert not attr.HasAuthoredValue(), link_name
        else:
            assert attr.Get() == pytest.approx(value), link_name


def test_bare_fragment_on_usd_asset_reaches_nested_bodies(stage, tmp_path):
    """The convenience form keeps the legacy nested writer's reach instead of pinning to the spawn prim."""
    _spawn_usd(
        stage,
        tmp_path,
        "/World/Robot",
        rigid_props=PhysxRigidBodyCfg(max_depenetration_velocity=5.0),
        mass_props=MassCfg(mass=2.0),
    )
    for link_name in LINK_REL_PATHS:
        link = stage.GetPrimAtPath(f"/World/Robot/{link_name}")
        assert link.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(5.0)
        assert link.GetAttribute("physics:mass").Get() == pytest.approx(2.0)


def _author_prop_usd(path: str) -> None:
    """An art asset without physics schemas: the spawner has to turn it into a single rigid body."""
    asset = Usd.Stage.CreateNew(path)
    prop = UsdGeom.Xform.Define(asset, "/Prop")
    UsdGeom.Cube.Define(asset, "/Prop/geometry")
    asset.SetDefaultPrim(prop.GetPrim())
    asset.Save()


def test_bare_fragments_make_a_schema_free_asset_a_single_rigid_body(stage, tmp_path):
    _spawn_usd(
        stage,
        tmp_path,
        "/World/Prop",
        author=_author_prop_usd,
        rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True),
        mass_props=MassCfg(mass=0.05),
        collision_props=UsdPhysicsCollisionCfg(collision_enabled=True),
    )
    prop = stage.GetPrimAtPath("/World/Prop")
    assert (
        prop.HasAPI(UsdPhysics.RigidBodyAPI)
        and prop.HasAPI(UsdPhysics.MassAPI)
        and prop.HasAPI(UsdPhysics.CollisionAPI)
    )
    assert prop.GetAttribute("physics:rigidBodyEnabled").Get() is True
    assert prop.GetAttribute("physics:mass").Get() == pytest.approx(0.05)
    assert prop.GetAttribute("physics:collisionEnabled").Get() is True
    # the geometry below the spawn prim must not become a second, nested body
    assert not stage.GetPrimAtPath("/World/Prop/geometry").HasAPI(UsdPhysics.RigidBodyAPI)


def test_spawn_from_file_with_empty_tendon_lists_is_noop(stage, tmp_path):
    usd_path = str(tmp_path / "mini.usda")
    asset = Usd.Stage.CreateNew(usd_path)
    asset.SetDefaultPrim(UsdGeom.Xform.Define(asset, "/Root").GetPrim())
    asset.Save()
    cfg = sim_utils.UsdFileCfg(
        usd_path=usd_path, fixed_tendons_props={"(/.*)?": []}, spatial_tendons_props={"(/.*)?": []}
    )
    cfg.func("/World/Asset", cfg)
    assert stage.GetPrimAtPath("/World/Asset").IsValid()
