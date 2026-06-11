# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for Newton and MuJoCo schema cfg classes in isaaclab_newton."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import pytest
from isaaclab_newton.sim.schemas import (
    MujocoJointDrivePropertiesCfg,
    MujocoRigidBodyPropertiesCfg,
    NewtonArticulationRootPropertiesCfg,
    NewtonCollisionPropertiesCfg,
    NewtonJointDrivePropertiesCfg,
    NewtonMaterialPropertiesCfg,
    NewtonMeshCollisionPropertiesCfg,
    NewtonRigidBodyPropertiesCfg,
    NewtonSDFCollisionPropertiesCfg,
)

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials import spawn_rigid_body_material


@pytest.fixture
def setup_sim():
    """Fixture to set up and tear down the simulation context."""
    sim_utils.create_new_stage()
    sim = SimulationContext(SimulationCfg(dt=0.1))
    yield sim
    sim._disable_app_control_on_stop_handle = True
    sim.stop()
    sim.clear_instance()


def _has_authored_api_schema(prim, schema_name: str) -> bool:
    """Return whether a schema name is applied or authored in ``apiSchemas`` metadata."""
    if schema_name in prim.GetAppliedSchemas():
        return True
    api_schemas = prim.GetMetadata("apiSchemas")
    if api_schemas is None:
        return False
    return any(
        schema_name in getattr(api_schemas, item_list)
        for item_list in ("explicitItems", "prependedItems", "appendedItems", "addedItems")
    )


# ---------------------------------------------------------------------------
# MuJoCo rigid body gravity compensation
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_mujoco_gravcomp_written(setup_sim):
    """gravcomp=0.5 must write mjc:gravcomp=0.5 on the prim."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/body_gc", prim_type="Cube", translation=(0.0, 0.0, 0.5))
    schemas.define_rigid_body_properties("/World/body_gc", MujocoRigidBodyPropertiesCfg(gravcomp=0.5))
    attr = stage.GetPrimAtPath("/World/body_gc").GetAttribute("mjc:gravcomp")
    assert attr.IsValid(), "mjc:gravcomp was not authored"
    assert attr.Get() == pytest.approx(0.5)


@pytest.mark.isaacsim_ci
def test_mujoco_gravcomp_not_written_when_none(setup_sim):
    """gravcomp=None must not write mjc:gravcomp."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/body_gc2", prim_type="Cube", translation=(1.0, 0.0, 0.5))
    schemas.define_rigid_body_properties("/World/body_gc2", MujocoRigidBodyPropertiesCfg())
    attr = stage.GetPrimAtPath("/World/body_gc2").GetAttribute("mjc:gravcomp")
    assert not attr.IsValid(), "mjc:gravcomp should not be authored when gravcomp=None"


# ---------------------------------------------------------------------------
# MuJoCo joint actuator gravity comp
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_mujoco_actuatorgravcomp_written(setup_sim):
    """actuatorgravcomp=True must write mjc:actuatorgravcomp=True on the joint prim."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/art_gc", prim_type="Xform")
    sim_utils.create_prim("/World/art_gc/body0", prim_type="Cube")
    sim_utils.create_prim("/World/art_gc/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/art_gc/joint0")
    schemas.modify_joint_drive_properties("/World/art_gc", MujocoJointDrivePropertiesCfg(actuatorgravcomp=True))
    attr = stage.GetPrimAtPath("/World/art_gc/joint0").GetAttribute("mjc:actuatorgravcomp")
    assert attr.IsValid(), "mjc:actuatorgravcomp was not authored"
    assert attr.Get() is True


@pytest.mark.isaacsim_ci
def test_mujoco_actuatorgravcomp_not_written_when_none(setup_sim):
    """actuatorgravcomp=None must not write mjc:actuatorgravcomp."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/art_gc2", prim_type="Xform")
    sim_utils.create_prim("/World/art_gc2/body0", prim_type="Cube")
    sim_utils.create_prim("/World/art_gc2/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/art_gc2/joint0")
    schemas.modify_joint_drive_properties("/World/art_gc2", MujocoJointDrivePropertiesCfg())
    attr = stage.GetPrimAtPath("/World/art_gc2/joint0").GetAttribute("mjc:actuatorgravcomp")
    assert not attr.IsValid(), "mjc:actuatorgravcomp should not be authored when None"


# ---------------------------------------------------------------------------
# Newton collision
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_collision_contact_margin_written(setup_sim):
    """contact_margin=0.01 must write newton:contactMargin and apply NewtonCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/col_newton", prim_type="Cube", translation=(2.0, 0.0, 0.5))
    schemas.define_collision_properties("/World/col_newton", NewtonCollisionPropertiesCfg(contact_margin=0.01))
    prim = stage.GetPrimAtPath("/World/col_newton")
    assert prim.GetAttribute("newton:contactMargin").Get() == pytest.approx(0.01)
    assert "NewtonCollisionAPI" in prim.GetAppliedSchemas()


@pytest.mark.isaacsim_ci
def test_newton_collision_no_schema_when_none(setup_sim):
    """NewtonCollisionPropertiesCfg() with all None must NOT apply NewtonCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/col_newton2", prim_type="Cube", translation=(3.0, 0.0, 0.5))
    schemas.define_collision_properties("/World/col_newton2", NewtonCollisionPropertiesCfg())
    applied = stage.GetPrimAtPath("/World/col_newton2").GetAppliedSchemas()
    assert "NewtonCollisionAPI" not in applied


# ---------------------------------------------------------------------------
# Newton material
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_material_properties_written(setup_sim):
    """torsional_friction and rolling_friction must be written and NewtonMaterialAPI applied."""
    mat_cfg = NewtonMaterialPropertiesCfg(torsional_friction=0.3, rolling_friction=0.001)
    prim = spawn_rigid_body_material("/World/newton_mat", mat_cfg)
    assert prim.GetAttribute("newton:torsionalFriction").Get() == pytest.approx(0.3)
    assert prim.GetAttribute("newton:rollingFriction").Get() == pytest.approx(0.001)
    assert "NewtonMaterialAPI" in prim.GetAppliedSchemas()


@pytest.mark.isaacsim_ci
def test_newton_material_no_schema_when_none(setup_sim):
    """NewtonMaterialPropertiesCfg() with all Newton fields None must NOT apply NewtonMaterialAPI."""
    mat_cfg = NewtonMaterialPropertiesCfg()
    prim = spawn_rigid_body_material("/World/newton_mat2", mat_cfg)
    assert "NewtonMaterialAPI" not in prim.GetAppliedSchemas()


# ---------------------------------------------------------------------------
# Newton articulation root
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_articulation_self_collision_written(setup_sim):
    """self_collision_enabled=True must write newton:selfCollisionEnabled and apply the API."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/nart", prim_type="Xform")
    sim_utils.create_prim("/World/nart/body0", prim_type="Cube")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/nart"))
    schemas.modify_articulation_root_properties(
        "/World/nart",
        NewtonArticulationRootPropertiesCfg(self_collision_enabled=True),
    )
    prim = stage.GetPrimAtPath("/World/nart")
    assert prim.GetAttribute("newton:selfCollisionEnabled").Get() is True
    assert "NewtonArticulationRootAPI" in prim.GetAppliedSchemas()


@pytest.mark.isaacsim_ci
def test_newton_articulation_no_schema_when_none(setup_sim):
    """NewtonArticulationRootPropertiesCfg() with None must NOT apply NewtonArticulationRootAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/nart2", prim_type="Xform")
    sim_utils.create_prim("/World/nart2/body0", prim_type="Cube")
    UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/World/nart2"))
    schemas.modify_articulation_root_properties(
        "/World/nart2",
        NewtonArticulationRootPropertiesCfg(),
    )
    applied = stage.GetPrimAtPath("/World/nart2").GetAppliedSchemas()
    assert "NewtonArticulationRootAPI" not in applied


# ---------------------------------------------------------------------------
# Newton mesh collision (max_hull_vertices, NewtonMeshCollisionAPI)
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_mesh_collision_max_hull_vertices_written(setup_sim):
    """max_hull_vertices=64 must write newton:maxHullVertices and apply NewtonMeshCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/mesh_col", prim_type="Cube", translation=(4.0, 0.0, 0.5))
    schemas.define_mesh_collision_properties(
        "/World/mesh_col",
        NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="convexHull", max_hull_vertices=64),
    )
    prim = stage.GetPrimAtPath("/World/mesh_col")
    assert prim.GetAttribute("newton:maxHullVertices").Get() == 64
    assert "NewtonMeshCollisionAPI" in prim.GetAppliedSchemas()


@pytest.mark.isaacsim_ci
def test_newton_mesh_collision_no_schema_when_none(setup_sim):
    """NewtonMeshCollisionPropertiesCfg() with max_hull_vertices=None must NOT apply NewtonMeshCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/mesh_col2", prim_type="Cube", translation=(5.0, 0.0, 0.5))
    schemas.define_mesh_collision_properties(
        "/World/mesh_col2",
        NewtonMeshCollisionPropertiesCfg(mesh_approximation_name="convexHull"),
    )
    applied = stage.GetPrimAtPath("/World/mesh_col2").GetAppliedSchemas()
    assert "NewtonMeshCollisionAPI" not in applied


# ---------------------------------------------------------------------------
# Newton SDF collision
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_sdf_collision_properties_written(setup_sim):
    """SDF fields must write newton:* attributes and apply NewtonSDFCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/sdf_col", prim_type="Cube", translation=(6.0, 0.0, 0.5))
    schemas.define_collision_properties(
        "/World/sdf_col",
        NewtonSDFCollisionPropertiesCfg(
            sdf_max_resolution=64,
            sdf_narrow_band_inner=-0.02,
            sdf_narrow_band_outer=0.03,
            sdf_target_voxel_size=0.005,
            sdf_texture_format="float32",
            sdf_padding=0.01,
            hydroelastic_enabled=True,
            hydroelastic_stiffness=1.0e8,
        ),
    )
    prim = stage.GetPrimAtPath("/World/sdf_col")
    assert prim.GetAttribute("newton:sdfMaxResolution").Get() == 64
    assert prim.GetAttribute("newton:sdfNarrowBandInner").Get() == pytest.approx(-0.02)
    assert prim.GetAttribute("newton:sdfNarrowBandOuter").Get() == pytest.approx(0.03)
    assert prim.GetAttribute("newton:sdfTargetVoxelSize").Get() == pytest.approx(0.005)
    assert prim.GetAttribute("newton:sdfTextureFormat").Get() == "float32"
    assert prim.GetAttribute("newton:sdfPadding").Get() == pytest.approx(0.01)
    assert prim.GetAttribute("newton:hydroelasticEnabled").Get() is True
    assert prim.GetAttribute("newton:hydroelasticStiffness").Get() == pytest.approx(1.0e8)
    applied = prim.GetAppliedSchemas()
    assert _has_authored_api_schema(prim, "NewtonSDFCollisionAPI")
    assert "NewtonMeshCollisionAPI" not in applied


@pytest.mark.isaacsim_ci
def test_newton_sdf_collision_no_schema_when_only_base_fields_set(setup_sim):
    """Base Newton collision fields must not apply NewtonSDFCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/sdf_base_only", prim_type="Cube", translation=(7.0, 0.0, 0.5))
    schemas.define_collision_properties(
        "/World/sdf_base_only",
        NewtonSDFCollisionPropertiesCfg(contact_margin=0.005),
    )
    prim = stage.GetPrimAtPath("/World/sdf_base_only")
    applied = prim.GetAppliedSchemas()
    assert prim.GetAttribute("newton:contactMargin").Get() == pytest.approx(0.005)
    assert "NewtonCollisionAPI" in applied
    assert not _has_authored_api_schema(prim, "NewtonSDFCollisionAPI")


@pytest.mark.isaacsim_ci
def test_newton_sdf_collision_no_schema_when_none(setup_sim):
    """NewtonSDFCollisionPropertiesCfg() with all None must NOT apply NewtonSDFCollisionAPI."""
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/sdf_col2", prim_type="Cube", translation=(8.0, 0.0, 0.5))
    schemas.define_collision_properties("/World/sdf_col2", NewtonSDFCollisionPropertiesCfg())
    prim = stage.GetPrimAtPath("/World/sdf_col2")
    applied = prim.GetAppliedSchemas()
    assert "NewtonCollisionAPI" not in applied
    assert not _has_authored_api_schema(prim, "NewtonSDFCollisionAPI")


# ---------------------------------------------------------------------------
# Class hierarchy contract: Mujoco IS-A Newton
# ---------------------------------------------------------------------------


def test_mujoco_isinstance_newton():
    """MujocoXxxCfg instances must be isinstance of their Newton parent.

    The auto-enable spawner logic and any future polymorphic dispatch on
    ``isinstance(cfg, NewtonRigidBodyPropertiesCfg)`` depends on this contract.
    """
    mjc_rigid = MujocoRigidBodyPropertiesCfg(gravcomp=0.5)
    assert isinstance(mjc_rigid, NewtonRigidBodyPropertiesCfg)

    mjc_joint = MujocoJointDrivePropertiesCfg(actuatorgravcomp=True)
    assert isinstance(mjc_joint, NewtonJointDrivePropertiesCfg)


# ---------------------------------------------------------------------------
# Multi-namespace mixed write — verify per-declaring-class MRO routing keeps
# fields owned by different classes in different namespaces on the same prim.
# ---------------------------------------------------------------------------


@pytest.mark.isaacsim_ci
def test_newton_mesh_collision_mixed_namespace_write(setup_sim):
    """A NewtonMeshCollisionPropertiesCfg with both contact_margin (declared on
    NewtonCollisionPropertiesCfg) and max_hull_vertices (declared on
    NewtonMeshCollisionPropertiesCfg) must write each under its declaring class's
    namespace and apply both schemas.
    """
    stage = sim_utils.get_current_stage()
    sim_utils.create_prim("/World/mesh_mixed", prim_type="Cube", translation=(9.0, 0.0, 0.5))
    schemas.define_mesh_collision_properties(
        "/World/mesh_mixed",
        NewtonMeshCollisionPropertiesCfg(
            mesh_approximation_name="convexHull",
            max_hull_vertices=32,
            contact_margin=0.005,
        ),
    )
    prim = stage.GetPrimAtPath("/World/mesh_mixed")
    # Both attributes share the newton namespace but are gated on different applied
    # schemas (NewtonCollisionAPI for contact_margin, NewtonMeshCollisionAPI for
    # max_hull_vertices); per-declaring-class routing applies the right schema for each.
    assert prim.GetAttribute("newton:contactMargin").Get() == pytest.approx(0.005)
    assert prim.GetAttribute("newton:maxHullVertices").Get() == 32
    applied = prim.GetAppliedSchemas()
    assert "NewtonCollisionAPI" in applied
    assert "NewtonMeshCollisionAPI" in applied
