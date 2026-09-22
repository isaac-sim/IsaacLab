# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Schema writers that need a live PhysX simulation: legacy writers on robot assets and root fixing.

Everything that only needs a USD stage lives in the kitless ``test_schema_fragments.py`` and
``test_schemas_deprecation.py``. This file covers the legacy writers against real robot assets,
that the authored schemas simulate, and the PhysX root relocation performed through the physics
manager when ``fix_root_link`` is set.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
import os
import warnings
from types import SimpleNamespace

import pytest
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab_physx.sim.schemas import (
    PhysxArticulationCfg,
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxJointDrivePropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.string import to_camel_case

pytestmark = [pytest.mark.integration, pytest.mark.isaacsim_ci]


@pytest.fixture
def sim():
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.1))
    yield sim
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


@pytest.fixture(scope="module")
def legacy_cfgs() -> SimpleNamespace:
    """Legacy cfgs with every field set so the validation helpers check every authored attribute."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return SimpleNamespace(
            articulation=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                articulation_enabled=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                sleep_threshold=1.0,
                stabilization_threshold=5.0,
                fix_root_link=False,
            ),
            rigid=PhysxRigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.1,
                angular_damping=0.5,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=10.0,
                max_contact_impulse=10.0,
                enable_gyroscopic_forces=True,
                retain_accelerations=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=1,
                sleep_threshold=1.0,
                stabilization_threshold=6.0,
            ),
            collision=PhysxCollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.05,
                rest_offset=0.001,
                min_torsional_patch_radius=0.1,
                torsional_patch_radius=1.0,
            ),
            mass=schemas.MassPropertiesCfg(mass=1.0, density=100.0),
            joint=PhysxJointDrivePropertiesCfg(
                drive_type="acceleration", max_force=80.0, max_joint_velocity=10.0, stiffness=10.0, damping=0.1
            ),
        )


def _xform(stage: Usd.Stage, path: str, *apis) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    for api in apis:
        api.Apply(prim)
    return prim


def _articulation_roots(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]


def _fixed_joints(stage: Usd.Stage) -> list[Usd.Prim]:
    return [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.FixedJoint)]


def _api_schemas(prim: Usd.Prim) -> set[str]:
    """Applied API schema names including unregistered token schemas."""
    return set(prim.GetPrimTypeInfo().GetAppliedAPISchemas())


"""
Legacy writers on robot assets.
"""


def _cfg_items(cfg, skip: tuple[str, ...] = ()) -> list[tuple[str, object]]:
    """Cfg fields that author a USD attribute: skips class metadata, ``func`` and ``skip``."""
    return [(k, v) for k, v in cfg.__dict__.items() if not k.startswith("_") and k not in ("func", *skip)]


def _assert_articulation_properties(prim_path: str, cfg, has_default_fixed_root: bool) -> None:
    """Check the PhysX articulation attributes and the world fixed joint state on the root prim."""
    stage = sim_utils.get_current_stage()
    root_prim = stage.GetPrimAtPath(prim_path)
    for name, value in _cfg_items(cfg, skip=("fix_root_link",)):
        attr = root_prim.GetAttribute(f"physxArticulation:{to_camel_case(name)}")
        assert attr.Get() == pytest.approx(value, abs=1e-5), attr.GetName()
    fixed_joint = sim_utils.find_global_fixed_joint_prim(prim_path)
    if cfg.fix_root_link is None:
        return
    if has_default_fixed_root:
        # the asset ships with a world joint: the flag toggles it
        assert fixed_joint is not None
        assert fixed_joint.GetJointEnabledAttr().Get() == cfg.fix_root_link
    else:
        assert (fixed_joint is not None) == cfg.fix_root_link


def _assert_namespaced_properties(prim_path: str, api, namespace: str, cfg, skip: tuple[str, ...] = ()) -> None:
    """Check ``<namespace>:<camelCase(field)>`` on every authorable prim under ``prim_path`` carrying ``api``.

    Prims inside instances are read-only, so the writers skip them and so does this check.
    """
    root = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    carriers = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(api)]
    assert carriers, f"no prim under {prim_path} carries {api.__name__}"
    for prim in carriers:
        for name, value in _cfg_items(cfg, skip=skip):
            attr = prim.GetAttribute(f"{namespace}:{to_camel_case(name)}")
            assert attr.Get() == pytest.approx(value, abs=1e-5), f"{prim.GetPath()} {attr.GetName()}"


RIGID_SKIP = ("rigid_body_enabled", "kinematic_enabled")
COLLISION_SKIP = ("collision_enabled", "mesh_collision_property")


def _assert_joint_drive_properties(prim_path: str, cfg) -> None:
    """Check the drive attributes on every joint, converting the degree-based angular values back."""
    root = sim_utils.get_current_stage().GetPrimAtPath(prim_path)
    joints = [
        joint
        for link in root.GetAllChildren()
        for joint in link.GetChildren()
        if joint.IsA(UsdPhysics.PrismaticJoint) or joint.IsA(UsdPhysics.RevoluteJoint)
    ]
    assert joints, "asset has no revolute or prismatic joints"
    for joint in joints:
        assert joint.HasAPI(UsdPhysics.DriveAPI)
        drive = "linear" if joint.IsA(UsdPhysics.PrismaticJoint) else "angular"
        angular = drive == "angular"
        for name, value in _cfg_items(cfg, skip=("ensure_drives_exist", "max_effort", "max_velocity")):
            if name == "drive_type":
                assert joint.GetAttribute(f"drive:{drive}:physics:type").Get() == value
                continue
            if name == "max_joint_velocity":
                authored = joint.GetAttribute("physxJoint:maxJointVelocity").Get()
                authored = math.radians(authored) if angular else authored
            else:
                authored = joint.GetAttribute(f"drive:{drive}:physics:{to_camel_case(name)}").Get()
                if angular and name in ("stiffness", "damping"):
                    authored = math.degrees(authored)
            assert authored == pytest.approx(value, abs=1e-5), f"{joint.GetPath()} {name}"


@pytest.mark.parametrize(
    ("asset", "root_suffix", "has_default_fixed_root"),
    [
        pytest.param("Robots/ANYbotics/anymal_c/anymal_c.usd", "/base", False, id="anymal_instanced"),
        pytest.param("Robots/FrankaRobotics/FrankaPanda/franka.usd", "", True, id="franka"),
    ],
)
def test_legacy_writers_on_robot_asset(sim, legacy_cfgs, asset, root_suffix, has_default_fixed_root):
    """The legacy nested writers author every schema on a real (possibly instanced) robot asset.

    Collision schemas are covered on spawned shapes instead: robot collision meshes live inside
    instanceable geometry, which the writers cannot author on.
    """
    prim_path = "/World/asset"
    sim_utils.create_prim(prim_path, usd_path=f"{ISAAC_NUCLEUS_DIR}/{asset}", translation=(0.0, 0.0, 0.62))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.modify_articulation_root_properties(prim_path, legacy_cfgs.articulation)
        schemas.modify_rigid_body_properties(prim_path, legacy_cfgs.rigid)
        schemas.modify_mass_properties(prim_path, legacy_cfgs.mass)
        schemas.modify_joint_drive_properties(prim_path, legacy_cfgs.joint)
    _assert_articulation_properties(prim_path + root_suffix, legacy_cfgs.articulation, has_default_fixed_root)
    _assert_namespaced_properties(prim_path, UsdPhysics.RigidBodyAPI, "physxRigidBody", legacy_cfgs.rigid, RIGID_SKIP)
    _assert_namespaced_properties(prim_path, UsdPhysics.MassAPI, "physics", legacy_cfgs.mass)
    _assert_joint_drive_properties(prim_path, legacy_cfgs.joint)

    # fixing the root afterwards must not fail on an already-authored asset
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fixed = legacy_cfgs.articulation.replace(fix_root_link=True)
        schemas.modify_articulation_root_properties(prim_path, fixed)
    if has_default_fixed_root:
        _assert_articulation_properties(prim_path, fixed, has_default_fixed_root)


def test_legacy_defined_schemas_simulate(sim, legacy_cfgs):
    """Schemas defined from scratch on an articulation and rigid bodies produce a simulatable scene."""
    sim_utils.create_prim("/World/parent", prim_type="Xform")
    sim_utils.create_prim("/World/parent/child", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    sim_utils.create_prim("/World/cube", prim_type="Cube", translation=(1.0, 1.0, 0.62))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        schemas.define_articulation_root_properties("/World/parent", legacy_cfgs.articulation)
        for path in ("/World/parent/child", "/World/cube"):
            schemas.define_rigid_body_properties(path, legacy_cfgs.rigid)
            schemas.define_collision_properties(path, legacy_cfgs.collision)
            schemas.define_mass_properties(path, legacy_cfgs.mass)
    _assert_articulation_properties("/World/parent", legacy_cfgs.articulation, has_default_fixed_root=False)
    _assert_namespaced_properties("/World", UsdPhysics.RigidBodyAPI, "physxRigidBody", legacy_cfgs.rigid, RIGID_SKIP)
    _assert_namespaced_properties(
        "/World", UsdPhysics.CollisionAPI, "physxCollision", legacy_cfgs.collision, COLLISION_SKIP
    )
    _assert_namespaced_properties("/World", UsdPhysics.MassAPI, "physics", legacy_cfgs.mass)
    sim.reset()
    for _ in range(100):
        sim.step()


"""
fix_root_link through the PhysX physics manager.

PhysX treats a fixed joint on a rigid body as part of a maximal-coordinate tree, so the manager
relocates the articulation root to the parent prim and every root schema has to move with it.
"""


@pytest.mark.parametrize("existing_joint", [False, True], ids=["create_joint", "enable_existing_joint"])
def test_fix_root_link_creates_or_enables_joint_and_relocates_root(sim, existing_joint):
    stage = sim_utils.get_current_stage()
    parent = _xform(stage, "/World/Robot")
    root = _xform(stage, "/World/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    if existing_joint:
        joint = UsdPhysics.FixedJoint.Define(stage, "/World/Robot/base/FixedJoint")
        joint.CreateBody1Rel().SetTargets([root.GetPath()])
        joint.CreateJointEnabledAttr(False)

    assert schemas.apply_articulation_root_properties(
        "/World/Robot(/.*)?", [PhysxArticulationCfg(articulation_enabled=True)], stage, fix_root_link=True
    )

    (fixed_joint,) = _fixed_joints(stage)
    assert UsdPhysics.FixedJoint(fixed_joint).GetJointEnabledAttr().Get() is True
    assert _articulation_roots(stage) == [parent]
    assert parent.GetAttribute("physxArticulation:articulationEnabled").Get() is True


def test_fix_root_link_requires_rigid_body_root(sim):
    """Without a rigid body there is no link to anchor the world joint to."""
    stage = sim_utils.get_current_stage()
    _xform(stage, "/World/Robot", UsdPhysics.ArticulationRootAPI)
    with pytest.raises(NotImplementedError):
        schemas.apply_articulation_root_properties(
            "/World/Robot", [PhysxArticulationCfg(articulation_enabled=True)], stage, fix_root_link=True
        )


def test_fix_root_link_moves_backend_root_schemas_with_the_root(sim):
    """Fragments land on the relocated root and a pre-authored Newton root API moves along with its value.

    Leaving ``NewtonArticulationRootAPI`` on the former root link would keep a second root alive, since
    that API composes ``PhysicsArticulationRootAPI``.
    """
    stage = sim_utils.get_current_stage()
    parent = _xform(stage, "/World/Robot")
    child = _xform(stage, "/World/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    child.AddAppliedSchema("NewtonArticulationRootAPI")
    child.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    schemas.apply_articulation_root_properties(
        "/World/Robot(/.*)?",
        [PhysxArticulationCfg(solver_position_iteration_count=8), NewtonArticulationCfg(self_collision_enabled=True)],
        stage,
        fix_root_link=True,
    )

    assert _articulation_roots(stage) == [parent]
    assert parent.GetAttribute("physxArticulation:solverPositionIterationCount").Get() == 8
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is True
    assert "NewtonArticulationRootAPI" in _api_schemas(parent)
    assert "NewtonArticulationRootAPI" not in _api_schemas(child)


def test_fix_root_link_preserves_complete_authored_property_spec(sim):
    """Relocation moves sampled, metadata and connection opinions rather than one default value."""
    stage = sim_utils.get_current_stage()
    parent = _xform(stage, "/World/Robot")
    child = _xform(
        stage, "/World/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI
    )
    child.AddAppliedSchema("PhysxArticulationAPI")
    driver_attr = _xform(stage, "/World/Driver").CreateAttribute("output", Sdf.ValueTypeNames.Float)
    driver_attr.Set(0.5)
    source_attr = child.GetAttribute("physxArticulation:sleepThreshold")
    source_attr.Set(0.1, Usd.TimeCode(1.0))
    source_attr.Set(0.2, Usd.TimeCode(2.0))
    source_attr.SetMetadata("documentation", "sampled sleep threshold")
    source_attr.AddConnection(driver_attr.GetPath())

    schemas.apply_articulation_root_properties("/World/Robot(/.*)?", [], stage, fix_root_link=True)

    assert _articulation_roots(stage) == [parent]
    assert "PhysxArticulationAPI" in _api_schemas(parent) and "PhysxArticulationAPI" not in _api_schemas(child)
    moved_attr = parent.GetAttribute("physxArticulation:sleepThreshold")
    assert not stage.GetRootLayer().GetAttributeAtPath(moved_attr.GetPath()).HasInfo("default")
    assert moved_attr.GetTimeSamples() == [1.0, 2.0]
    assert moved_attr.Get(Usd.TimeCode(1.0)) == pytest.approx(0.1)
    assert moved_attr.Get(Usd.TimeCode(2.0)) == pytest.approx(0.2)
    assert moved_attr.GetMetadata("documentation") == "sampled sleep threshold"
    assert moved_attr.GetConnections() == [driver_attr.GetPath()]
    # only the root schemas move; the body schemas stay on the link
    assert child.HasAPI(UsdPhysics.RigidBodyAPI) and child.HasAPI(UsdPhysics.MassAPI)
    assert not parent.HasAPI(UsdPhysics.RigidBodyAPI) and not parent.HasAPI(UsdPhysics.MassAPI)


def test_legacy_fix_root_link_relocates_root_with_its_newton_mirror(sim):
    """The legacy writer relocates the root itself, carrying the mirrored Newton self-collision flag along."""
    stage = sim_utils.get_current_stage()
    parent = _xform(stage, "/World/Robot")
    child = _xform(stage, "/World/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    child.AddAppliedSchema("NewtonArticulationRootAPI")
    child.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = PhysxArticulationRootPropertiesCfg(enabled_self_collisions=True, fix_root_link=True)
        schemas.modify_articulation_root_properties(child.GetPath(), cfg, stage)

    assert _articulation_roots(stage) == [parent]
    assert len(_fixed_joints(stage)) == 1
    assert parent.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert parent.GetAttribute("newton:selfCollisionEnabled").Get() is True
    assert "NewtonArticulationRootAPI" in _api_schemas(parent)
    assert "NewtonArticulationRootAPI" not in _api_schemas(child)
    assert not child.GetAttribute("newton:selfCollisionEnabled").HasAuthoredValue()


def _author_child_root_robot_usd(path: str) -> None:
    """A robot whose articulation root sits on a rigid-body child link, as ANYmal-style assets do."""
    asset = Usd.Stage.CreateNew(path)
    robot = UsdGeom.Xform.Define(asset, "/Robot")
    _xform(asset, "/Robot/base", UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI)
    asset.SetDefaultPrim(robot.GetPrim())
    asset.Save()


@pytest.mark.parametrize(
    ("articulation_props", "expected_attrs"),
    [
        pytest.param(None, {}, id="none"),
        pytest.param({}, {}, id="empty_mapping"),
        pytest.param([], {}, id="empty_list"),
        pytest.param(
            {
                "(/.*)?": [
                    PhysxArticulationCfg(solver_position_iteration_count=8),
                    NewtonArticulationCfg(self_collision_enabled=True),
                ]
            },
            {"physxArticulation:solverPositionIterationCount": 8, "newton:selfCollisionEnabled": True},
            id="composed_fragments",
        ),
    ],
)
def test_spawn_from_usd_file_fixes_root_on_child_link(sim, tmp_path, articulation_props, expected_attrs):
    """The from-files spawner honors ``fix_root_link`` for every slot form and composes fragments on the moved root.

    An empty slot carries no targeting intent, so the spawner sweeps the spawn prim's subtree to
    reach a root on a child link instead of pinning the expression to the schema-free spawn prim.
    """
    usd_path = os.path.join(tmp_path, "robot.usda")
    _author_child_root_robot_usd(usd_path)
    cfg = UsdFileCfg(usd_path=usd_path, articulation_props=articulation_props, fix_root_link=True)
    _spawn_from_usd_file("/World/Robot", usd_path, cfg)

    stage = sim_utils.get_current_stage()
    root = stage.GetPrimAtPath("/World/Robot")
    assert len(_fixed_joints(stage)) == 1
    assert _articulation_roots(stage) == [root]
    for attr, value in expected_attrs.items():
        assert root.GetAttribute(attr).Get() == value, attr
