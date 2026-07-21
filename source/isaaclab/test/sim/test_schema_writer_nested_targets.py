# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import os

import pytest
from isaaclab_physx.sim.schemas import PhysxCollisionCfg, PhysxRigidBodyCfg

from pxr import Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.schemas import MassCfg

pytestmark = pytest.mark.integration


LINK_REL_PATHS = ("link1", "link2", "link2/link3")
"""Link prim paths relative to the robot root; ``link2/link3`` is a nested child link."""


def _author_robot_usd(path: str) -> None:
    """Author a minimal articulated-robot USD layout.

    Mirrors how real robot assets are structured: the default (spawn) prim carries
    ``ArticulationRootAPI``, the link prims carry ``RigidBodyAPI`` and ``MassAPI``, and the
    collider prims under the links carry ``CollisionAPI``. The spawn prim itself carries no
    rigid-body, collision, or mass schema. ``link3`` is authored under ``link2`` the way the
    URDF importer nests child links under their parent link.
    """
    stage = Usd.Stage.CreateNew(path)
    robot = UsdGeom.Xform.Define(stage, "/Robot")
    UsdPhysics.ArticulationRootAPI.Apply(robot.GetPrim())
    for link_name in LINK_REL_PATHS:
        link = UsdGeom.Xform.Define(stage, f"/Robot/{link_name}").GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(link)
        UsdPhysics.MassAPI.Apply(link)
        collider = UsdGeom.Cube.Define(stage, f"/Robot/{link_name}/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
    stage.SetDefaultPrim(robot.GetPrim())
    stage.Save()


def _spawn_robot(tmp_path, prim_path: str, **cfg_kwargs):
    """Author the robot asset and spawn it at *prim_path* through the production USD spawn path."""
    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    usd_path = os.path.join(tmp_path, "robot.usda")
    _author_robot_usd(usd_path)
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    cfg = UsdFileCfg(usd_path=usd_path, **cfg_kwargs)
    _spawn_from_usd_file(prim_path, usd_path, cfg)
    return sim_utils.get_current_stage()


# -------------------------------------------------------------------------------------
# Regression: fragment writers must target the asset's existing schema-bearing prims,
# not force-anchor the defining API on the spawn prim.
# -------------------------------------------------------------------------------------


def test_rigid_body_fragments_target_existing_bodies_on_usd_asset(tmp_path):
    """Fragment ``rigid_props`` on a USD robot must modify the existing link bodies in place.

    Force-applying ``RigidBodyAPI`` on the spawn prim would invent a body the asset's joints
    never reference and change the asset's dynamics. The fragment path must match the legacy
    nested writer: author onto the prims that already carry the API and leave the spawn prim
    untouched.
    """
    stage = _spawn_robot(
        tmp_path,
        "/World/RobotA",
        rigid_props=[PhysxRigidBodyCfg(max_depenetration_velocity=5.0)],
    )
    spawn_prim = stage.GetPrimAtPath("/World/RobotA")
    # the spawn prim keeps its articulation root and gains NO rigid-body anchor or attrs
    assert spawn_prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    assert not spawn_prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not spawn_prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").HasAuthoredValue()
    # every existing link body received the fragment's attribute, including the nested one
    for link_name in LINK_REL_PATHS:
        link = stage.GetPrimAtPath(f"/World/RobotA/{link_name}")
        assert link.HasAPI(UsdPhysics.RigidBodyAPI)
        assert link.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(5.0), link_name


def test_collision_fragments_target_existing_colliders_on_usd_asset(tmp_path):
    """Fragment ``collision_props`` must modify the asset's existing collider prims in place."""
    stage = _spawn_robot(
        tmp_path,
        "/World/RobotB",
        collision_props=[PhysxCollisionCfg(contact_offset=0.02)],
    )
    spawn_prim = stage.GetPrimAtPath("/World/RobotB")
    assert not spawn_prim.HasAPI(UsdPhysics.CollisionAPI)
    assert not spawn_prim.GetAttribute("physxCollision:contactOffset").HasAuthoredValue()
    for link_name in LINK_REL_PATHS:
        collider = stage.GetPrimAtPath(f"/World/RobotB/{link_name}/collider")
        assert collider.HasAPI(UsdPhysics.CollisionAPI)
        assert collider.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.02), link_name


def test_mass_fragments_target_existing_mass_prims_on_usd_asset(tmp_path):
    """Fragment ``mass_props`` must modify the asset's existing mass-bearing link prims in place."""
    stage = _spawn_robot(
        tmp_path,
        "/World/RobotC",
        mass_props=[MassCfg(mass=2.0)],
    )
    spawn_prim = stage.GetPrimAtPath("/World/RobotC")
    assert not spawn_prim.HasAPI(UsdPhysics.MassAPI)
    assert not spawn_prim.GetAttribute("physics:mass").HasAuthoredValue()
    for link_name in LINK_REL_PATHS:
        link = stage.GetPrimAtPath(f"/World/RobotC/{link_name}")
        assert link.GetAttribute("physics:mass").Get() == pytest.approx(2.0), link_name


def test_fragment_and_legacy_paths_place_apis_identically_on_usd_asset(tmp_path):
    """On a real asset topology, the fragment path must place APIs exactly where legacy does.

    Spawns two copies of the same robot asset; configures one through the legacy single-cfg
    path and one through the fragment path with equivalent values; asserts every prim in both
    subtrees carries the same applied-schema set and the same authored attribute values.
    """
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

    from isaaclab.sim.spawners.from_files.from_files import _spawn_from_usd_file
    from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

    usd_path = os.path.join(tmp_path, "robot.usda")
    _author_robot_usd(usd_path)
    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    legacy_cfg = UsdFileCfg(usd_path=usd_path, rigid_props=PhysxRigidBodyPropertiesCfg(max_depenetration_velocity=5.0))
    frag_cfg = UsdFileCfg(usd_path=usd_path, rigid_props=[PhysxRigidBodyCfg(max_depenetration_velocity=5.0)])
    _spawn_from_usd_file("/World/Legacy", usd_path, legacy_cfg)
    _spawn_from_usd_file("/World/Frag", usd_path, frag_cfg)
    stage = sim_utils.get_current_stage()

    legacy_root = stage.GetPrimAtPath("/World/Legacy")
    frag_root = stage.GetPrimAtPath("/World/Frag")
    legacy_prims = {p.GetPath().pathString.removeprefix("/World/Legacy"): p for p in Usd.PrimRange(legacy_root)}
    frag_prims = {p.GetPath().pathString.removeprefix("/World/Frag"): p for p in Usd.PrimRange(frag_root)}
    assert legacy_prims.keys() == frag_prims.keys()
    for rel_path, legacy_prim in legacy_prims.items():
        frag_prim = frag_prims[rel_path]
        assert set(legacy_prim.GetAppliedSchemas()) == set(frag_prim.GetAppliedSchemas()), rel_path
        legacy_attrs = {a.GetName(): a.Get() for a in legacy_prim.GetAttributes() if a.HasAuthoredValue()}
        frag_attrs = {a.GetName(): a.Get() for a in frag_prim.GetAttributes() if a.HasAuthoredValue()}
        assert legacy_attrs == frag_attrs, rel_path


# -------------------------------------------------------------------------------------
# Preserved behavior: bare prims (shape/mesh spawners) still get a fresh anchor, and
# nested rigid-body trees (URDF-importer layouts) have every body modified.
# -------------------------------------------------------------------------------------


def test_rigid_body_fragments_create_on_bare_prim():
    """With ``create_if_missing`` and one non-carrier match, the writer anchors that prim."""
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Bare")
    result = apply_rigid_body_properties(
        "/World/Bare", [PhysxRigidBodyCfg(max_depenetration_velocity=3.0)], create_if_missing=True, stage=stage
    )
    prim = stage.GetPrimAtPath("/World/Bare")
    assert result is True
    assert prim.HasAPI(UsdPhysics.RigidBodyAPI)
    assert prim.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(3.0)


def test_rigid_body_fragments_create_errors_on_multiple_candidates():
    """Creating rigid bodies on more than one matched prim is never intentional."""
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    for path in ("/World/Grp", "/World/Grp/a", "/World/Grp/b"):
        UsdGeom.Xform.Define(stage, path)
    with pytest.raises(ValueError, match="RigidBodyAPI"):
        apply_rigid_body_properties(
            "/World/Grp/**", [PhysxRigidBodyCfg(max_depenetration_velocity=3.0)], create_if_missing=True, stage=stage
        )


def test_rigid_body_fragments_pattern_narrows_targets():
    """An expression targets only the carriers it matches."""
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    for path in ("/World/Bot/armL", "/World/Bot/armR"):
        prim = UsdGeom.Xform.Define(stage, path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    result = apply_rigid_body_properties(
        "/World/Bot/armL", [PhysxRigidBodyCfg(max_depenetration_velocity=7.0)], stage=stage
    )
    assert result is True
    left = stage.GetPrimAtPath("/World/Bot/armL")
    right = stage.GetPrimAtPath("/World/Bot/armR")
    assert left.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(7.0)
    assert not right.GetAttribute("physxRigidBody:maxDepenetrationVelocity").HasAuthoredValue()


def test_rigid_body_fragments_zero_targets_warn_and_return_false(caplog):
    """No carrier matched and no creation requested: warn, author nothing, report failure."""
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Bare")
    with caplog.at_level("WARNING"):
        result = apply_rigid_body_properties(
            "/World/Bare", [PhysxRigidBodyCfg(max_depenetration_velocity=3.0)], stage=stage
        )
    assert result is False
    assert not stage.GetPrimAtPath("/World/Bare").HasAPI(UsdPhysics.RigidBodyAPI)
    assert "/World/Bare" in caplog.text


def test_rigid_body_and_mass_fragments_modify_nested_bodies():
    """Every body in a nested rigid-body tree is modified, not just the outermost one.

    The URDF importer authors child links under their parent link prims, so carriers of
    ``RigidBodyAPI`` (and ``MassAPI``) legitimately nest. A trailing ``**`` expression must
    reach all of them, matching the legacy writers' full-subtree traversal for these families.
    """
    from isaaclab.sim.schemas import apply_mass_properties, apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    body_paths = ["/World/Robot/pelvis", "/World/Robot/pelvis/hip", "/World/Robot/pelvis/hip/knee"]
    UsdGeom.Xform.Define(stage, "/World/Robot")
    for body_path in body_paths:
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        UsdPhysics.MassAPI.Apply(body)

    rigid_result = apply_rigid_body_properties(
        "/World/Robot/**", [PhysxRigidBodyCfg(max_depenetration_velocity=7.0)], stage=stage
    )
    mass_result = apply_mass_properties("/World/Robot/**", [MassCfg(mass=2.5)], stage=stage)

    assert rigid_result is True
    assert mass_result is True
    for body_path in body_paths:
        body = stage.GetPrimAtPath(body_path)
        assert body.GetAttribute("physxRigidBody:maxDepenetrationVelocity").Get() == pytest.approx(7.0), body_path
        assert body.GetAttribute("physics:mass").Get() == pytest.approx(2.5), body_path


def test_rigid_body_fragments_skip_instanced_carriers(caplog):
    """A carrier inside an instance is a read-only proxy: skipped with a warning and a False return.

    Creation is not requested, so the instance root gains no rigid-body anchor either.
    """
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    source = UsdGeom.Xform.Define(stage, "/World/Source").GetPrim()
    source_body = UsdGeom.Xform.Define(stage, "/World/Source/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(source_body)
    instance = UsdGeom.Xform.Define(stage, "/World/Asset").GetPrim()
    instance.GetReferences().AddInternalReference(source.GetPath())
    instance.SetInstanceable(True)
    proxy_body = stage.GetPrimAtPath("/World/Asset/body")
    assert proxy_body.IsInstanceProxy() and proxy_body.HasAPI(UsdPhysics.RigidBodyAPI)

    with caplog.at_level("WARNING"):
        result = apply_rigid_body_properties(
            "/World/Asset/**", [PhysxRigidBodyCfg(max_depenetration_velocity=5.0)], stage=stage
        )

    assert result is False
    assert not instance.HasAPI(UsdPhysics.RigidBodyAPI)
    assert not proxy_body.GetAttribute("physxRigidBody:maxDepenetrationVelocity").HasAuthoredValue()
    assert "/World/Asset/body" in caplog.text


def test_rigid_body_fragments_empty_list_authors_nothing():
    """An empty fragment list is a no-op: no anchor API is authored and the writer reports success."""
    from isaaclab.sim.schemas import apply_rigid_body_properties

    sim_utils.create_new_stage()
    SimulationContext(SimulationCfg(dt=0.01))
    stage = sim_utils.get_current_stage()
    UsdGeom.Xform.Define(stage, "/World/Bare")
    result = apply_rigid_body_properties("/World/Bare", [], stage=stage)
    prim = stage.GetPrimAtPath("/World/Bare")
    assert result is True
    assert not prim.HasAPI(UsdPhysics.RigidBodyAPI)
