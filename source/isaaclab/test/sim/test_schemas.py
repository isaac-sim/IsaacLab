# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math
import warnings

import pytest
from isaaclab_physx.sim.schemas import (
    ArticulationRootPropertiesCfg as ArticulationRootDeprecatedAliasCfg,
)
from isaaclab_physx.sim.schemas import (
    CollisionPropertiesCfg as PhysxCollisionPropertiesCfgAlias,
)
from isaaclab_physx.sim.schemas import (
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionPropertiesCfg,
    PhysxJointDrivePropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg, RigidBodyMaterialCfg

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.sim.schemas as schemas
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg, spawn_rigid_body_material
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.string import to_camel_case


@pytest.fixture
def setup_simulation():
    """Fixture to set up and tear down the simulation context."""
    # Create a new stage
    sim_utils.create_new_stage()
    # Simulation time-step
    dt = 0.1
    # Load kit helper
    sim = SimulationContext(SimulationCfg(dt=dt))
    # Set some default values for test
    arti_cfg = schemas.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,
        articulation_enabled=True,
        solver_position_iteration_count=4,
        solver_velocity_iteration_count=1,
        sleep_threshold=1.0,
        stabilization_threshold=5.0,
        fix_root_link=False,
    )
    rigid_cfg = PhysxRigidBodyPropertiesCfg(
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
    )
    collision_cfg = schemas.CollisionPropertiesCfg(
        collision_enabled=True,
        contact_offset=0.05,
        rest_offset=0.001,
        min_torsional_patch_radius=0.1,
        torsional_patch_radius=1.0,
    )
    mass_cfg = schemas.MassPropertiesCfg(mass=1.0, density=100.0)
    joint_cfg = PhysxJointDrivePropertiesCfg(
        drive_type="acceleration", max_force=80.0, max_joint_velocity=10.0, stiffness=10.0, damping=0.1
    )
    yield sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg
    # Teardown
    sim._disable_app_control_on_stop_handle = True  # prevent timeout
    sim.stop()
    sim.clear_instance()


@pytest.mark.isaacsim_ci
def test_valid_properties_cfg(setup_simulation):
    """Test that all the config instances have non-None values.

    This is to ensure that we check that all the properties of the schema are set.
    """
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    # deprecation aliases are nulled by __post_init__ after forwarding to the canonical
    # field; exclude them from the all-non-None check.
    deprecation_aliases = {"max_velocity", "max_effort"}
    for cfg in [arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg]:
        for k, v in cfg.__dict__.items():
            # skip class-metadata keys (``_usd_*``) and deprecation aliases nulled in __post_init__
            if k.startswith("_") or k in deprecation_aliases:
                continue
            assert v is not None, f"{cfg.__class__.__name__}:{k} is None. Please make sure schemas are valid."


@pytest.mark.isaacsim_ci
def test_max_joint_velocity_on_base_cfg(setup_simulation):
    """Setting ``max_joint_velocity`` on the base ``JointDriveBaseCfg`` must author
    ``physxJoint:maxJointVelocity`` on the prim, identical to setting it on
    the deprecated PhysX subclass.

    Regression test for the Path 2 placement rule: ``max_joint_velocity`` is the
    only USD path to ``Model.joint_velocity_limit`` and lives on the base.
    """
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.JointDriveBaseCfg(
        drive_type="acceleration",
        max_force=80.0,
        max_joint_velocity=10.0,
        stiffness=10.0,
        damping=0.1,
    )

    # spawn a minimal articulation with a revolute joint, then write properties.
    sim_utils.create_prim("/World/Articulation", prim_type="Xform")
    sim_utils.create_prim("/World/Articulation/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Articulation/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/joint_0")

    prim_path = "/World/Articulation/joint_0"
    # use unwrapped function (no parent traversal) so this returns the inner bool
    schemas.modify_joint_drive_properties.__wrapped__(prim_path, base_cfg)

    # Revolute drives convert rad/s -> deg/s; check the authored value.
    attr = stage.GetPrimAtPath(prim_path).GetAttribute("physxJoint:maxJointVelocity")
    assert attr.IsValid(), "physxJoint:maxJointVelocity was not authored on the prim"
    expected_deg_per_sec = 10.0 * 180.0 / math.pi
    assert attr.Get() == pytest.approx(expected_deg_per_sec, rel=1e-6)


@pytest.mark.isaacsim_ci
def test_max_velocity_deprecation_alias(setup_simulation):
    """Legacy ``max_velocity`` kwarg must forward to ``max_joint_velocity`` and emit
    a ``DeprecationWarning``. Behavior must match setting ``max_joint_velocity`` directly.
    """
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    with pytest.warns(DeprecationWarning, match="max_velocity"):
        base_cfg = schemas.JointDriveBaseCfg(
            drive_type="acceleration",
            max_force=80.0,
            max_velocity=10.0,
            stiffness=10.0,
            damping=0.1,
        )

    assert base_cfg.max_joint_velocity == 10.0
    assert base_cfg.max_velocity is None

    sim_utils.create_prim("/World/Articulation_dep", prim_type="Xform")
    sim_utils.create_prim("/World/Articulation_dep/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Articulation_dep/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation_dep/joint_0")
    prim_path = "/World/Articulation_dep/joint_0"
    schemas.modify_joint_drive_properties.__wrapped__(prim_path, base_cfg)

    attr = stage.GetPrimAtPath(prim_path).GetAttribute("physxJoint:maxJointVelocity")
    assert attr.IsValid()
    assert attr.Get() == pytest.approx(10.0 * 180.0 / math.pi, rel=1e-6)


@pytest.mark.isaacsim_ci
def test_max_effort_deprecation_alias(setup_simulation):
    """Legacy ``max_effort`` kwarg must forward to ``max_force`` and emit
    a ``DeprecationWarning``. Behavior must match setting ``max_force`` directly.
    """
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    with pytest.warns(DeprecationWarning, match="max_effort"):
        base_cfg = schemas.JointDriveBaseCfg(
            drive_type="acceleration",
            max_effort=42.0,
            stiffness=10.0,
            damping=0.1,
        )

    assert base_cfg.max_force == 42.0
    assert base_cfg.max_effort is None

    sim_utils.create_prim("/World/Articulation_eff", prim_type="Xform")
    sim_utils.create_prim("/World/Articulation_eff/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Articulation_eff/body1", prim_type="Cube")
    UsdPhysics.PrismaticJoint.Define(stage, "/World/Articulation_eff/joint_0")
    prim_path = "/World/Articulation_eff/joint_0"
    schemas.modify_joint_drive_properties.__wrapped__(prim_path, base_cfg)

    attr = stage.GetPrimAtPath(prim_path).GetAttribute("drive:linear:physics:maxForce")
    assert attr.IsValid()
    assert attr.Get() == pytest.approx(42.0, rel=1e-6)


@pytest.mark.isaacsim_ci
def test_joint_drive_base_no_physx_schema_when_max_joint_velocity_unset(setup_simulation):
    """Regression: setting only UsdPhysics drive fields on JointDriveBaseCfg
    must NOT cause PhysxJointAPI to be applied to the prim. Without this,
    Newton-targeted users get PhysX schemas stamped on every joint."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.JointDriveBaseCfg(
        drive_type="acceleration",
        max_force=80.0,
        stiffness=10.0,
        damping=0.1,
        # max_joint_velocity intentionally left None
    )
    sim_utils.create_prim("/World/Articulation", prim_type="Xform")
    sim_utils.create_prim("/World/Articulation/body0", prim_type="Cube")
    sim_utils.create_prim("/World/Articulation/body1", prim_type="Cube")
    UsdPhysics.RevoluteJoint.Define(stage, "/World/Articulation/joint_0")

    prim_path = "/World/Articulation/joint_0"
    schemas.modify_joint_drive_properties.__wrapped__(prim_path, base_cfg)

    applied = stage.GetPrimAtPath(prim_path).GetAppliedSchemas()
    assert "PhysxJointAPI" not in applied, (
        f"PhysxJointAPI should not be applied when max_velocity is None; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_disable_gravity_on_base_cfg(setup_simulation):
    """Setting disable_gravity on the base RigidBodyBaseCfg must author
    physxRigidBody:disableGravity on the prim. PhysX honors per-body;
    Newton currently honors at scene level (partial), documented in field
    docstring. Regression test for the consumption-gated placement rule."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.RigidBodyBaseCfg(
        rigid_body_enabled=True,
        kinematic_enabled=False,
        disable_gravity=True,
    )
    sim_utils.create_prim("/World/cube_dg", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_rigid_body_properties("/World/cube_dg", base_cfg)

    prim_path = "/World/cube_dg"
    attr = stage.GetPrimAtPath(prim_path).GetAttribute("physxRigidBody:disableGravity")
    assert attr.IsValid(), "physxRigidBody:disableGravity was not authored on the prim"
    assert attr.Get() is True
    applied = stage.GetPrimAtPath(prim_path).GetAppliedSchemas()
    assert "PhysxRigidBodyAPI" in applied, (
        f"PhysxRigidBodyAPI must be applied when disable_gravity is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_rigid_body_no_physx_schema_when_all_physx_fields_none(setup_simulation):
    """Regression: PhysxRigidBodyPropertiesCfg with all PhysX-specific fields
    left as None must NOT cause PhysxRigidBodyAPI to be applied to the prim.
    The user only authored UsdPhysics-standard fields; the PhysX schema
    should not be stamped onto a Newton-targeted asset."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    cfg = PhysxRigidBodyPropertiesCfg(
        rigid_body_enabled=True,
        kinematic_enabled=False,
        # every PhysX field intentionally left None
    )
    sim_utils.create_prim("/World/cube_no_physx", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_rigid_body_properties("/World/cube_no_physx", cfg)

    prim_path = "/World/cube_no_physx"
    applied = stage.GetPrimAtPath(prim_path).GetAppliedSchemas()
    assert "PhysxRigidBodyAPI" not in applied, (
        f"PhysxRigidBodyAPI should not be applied when no PhysX fields are set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_rigid_body_material_base_cfg(setup_simulation):
    """Setting only UsdPhysics fields on RigidBodyMaterialBaseCfg must author the
    three friction/restitution attrs and must NOT apply PhysxMaterialAPI."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    cfg = RigidBodyMaterialBaseCfg(static_friction=0.7, dynamic_friction=0.6, restitution=0.1)
    prim_path = "/World/Looks/BaseMaterial"
    spawn_rigid_body_material.__wrapped__(prim_path, cfg)

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physics:dynamicFriction").Get() == pytest.approx(0.6)
    assert prim.GetAttribute("physics:restitution").Get() == pytest.approx(0.1)
    applied = prim.GetAppliedSchemas()
    assert "PhysxMaterialAPI" not in applied, (
        f"PhysxMaterialAPI must not be applied for the base cfg; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_rigid_body_material_cfg(setup_simulation):
    """Setting a PhysX-namespaced field on PhysxRigidBodyMaterialCfg must author the
    namespaced attribute AND apply PhysxMaterialAPI."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    cfg = PhysxRigidBodyMaterialCfg(static_friction=0.7, compliant_contact_stiffness=100.0)
    prim_path = "/World/Looks/PhysxMaterial"
    spawn_rigid_body_material.__wrapped__(prim_path, cfg)

    prim = stage.GetPrimAtPath(prim_path)
    assert prim.GetAttribute("physics:staticFriction").Get() == pytest.approx(0.7)
    assert prim.GetAttribute("physxMaterial:compliantContactStiffness").Get() == pytest.approx(100.0)
    applied = prim.GetAppliedSchemas()
    assert "PhysxMaterialAPI" in applied, (
        f"PhysxMaterialAPI must be applied when a PhysX field is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_rigid_body_material_deprecation_alias(setup_simulation):
    """Instantiating the legacy ``RigidBodyMaterialCfg`` name emits exactly one
    ``DeprecationWarning`` whose message references the 5.0 removal target."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RigidBodyMaterialCfg()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_collision_base_cfg_writes_physx_namespaced_attrs(setup_simulation):
    """Setting ``contact_offset`` / ``rest_offset`` on the base ``CollisionBaseCfg`` must
    author the ``physxCollision:*`` attributes AND apply ``PhysxCollisionAPI``. Newton's
    importer consumes them via the PhysX bridge resolver."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.CollisionBaseCfg(collision_enabled=True, contact_offset=0.05, rest_offset=0.001)
    sim_utils.create_prim("/World/cube_co", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_collision_properties("/World/cube_co", base_cfg)

    prim = stage.GetPrimAtPath("/World/cube_co")
    assert prim.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.05)
    assert prim.GetAttribute("physxCollision:restOffset").Get() == pytest.approx(0.001)
    applied = prim.GetAppliedSchemas()
    assert "PhysxCollisionAPI" in applied, (
        f"PhysxCollisionAPI must be applied when contact_offset/rest_offset are set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_collision_base_cfg_no_physx_schema_when_only_usd_field_set(setup_simulation):
    """Regression: setting only ``collision_enabled`` on ``CollisionBaseCfg`` must NOT
    cause ``PhysxCollisionAPI`` to be applied. The user only authored a UsdPhysics-standard
    field; the PhysX schema should not be stamped onto a Newton-targeted prim."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.CollisionBaseCfg(collision_enabled=True)
    sim_utils.create_prim("/World/cube_co_only", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_collision_properties("/World/cube_co_only", base_cfg)

    prim = stage.GetPrimAtPath("/World/cube_co_only")
    assert prim.GetAttribute("physics:collisionEnabled").Get() is True
    applied = prim.GetAppliedSchemas()
    assert "PhysxCollisionAPI" not in applied, (
        f"PhysxCollisionAPI should not be applied when only collision_enabled is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_collision_cfg_writes_torsional_patch(setup_simulation):
    """Setting ``torsional_patch_radius`` on ``PhysxCollisionPropertiesCfg`` must author
    the ``physxCollision:torsionalPatchRadius`` attribute AND apply ``PhysxCollisionAPI``."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    cfg = PhysxCollisionPropertiesCfg(torsional_patch_radius=1.0)
    sim_utils.create_prim("/World/cube_tpr", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_collision_properties("/World/cube_tpr", cfg)

    prim = stage.GetPrimAtPath("/World/cube_tpr")
    assert prim.GetAttribute("physxCollision:torsionalPatchRadius").Get() == pytest.approx(1.0)
    applied = prim.GetAppliedSchemas()
    assert "PhysxCollisionAPI" in applied


@pytest.mark.isaacsim_ci
def test_collision_deprecation_alias(setup_simulation):
    """Instantiating the legacy ``CollisionPropertiesCfg`` name emits exactly one
    ``DeprecationWarning`` whose message references the 5.0 removal target."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        PhysxCollisionPropertiesCfgAlias()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_articulation_root_base_cfg_writes_articulation_enabled(setup_simulation):
    """Setting ``articulation_enabled`` on the base ``ArticulationRootBaseCfg`` must author
    ``physxArticulation:articulationEnabled`` AND apply ``PhysxArticulationAPI``. The
    PhysX namespace is honored at sim time by PhysX and as a spawn-time guard by the IL
    Newton wrapper."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.ArticulationRootBaseCfg(articulation_enabled=False)
    sim_utils.create_prim("/World/arti_ae", prim_type="Xform")
    schemas.define_articulation_root_properties("/World/arti_ae", base_cfg)

    prim = stage.GetPrimAtPath("/World/arti_ae")
    assert prim.GetAttribute("physxArticulation:articulationEnabled").Get() is False
    applied = prim.GetAppliedSchemas()
    assert "PhysxArticulationAPI" in applied, (
        f"PhysxArticulationAPI must be applied when articulation_enabled is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_articulation_root_base_no_physx_schema_when_only_fix_root_link_set(setup_simulation):
    """Regression: setting only ``fix_root_link`` on ``ArticulationRootBaseCfg`` must NOT
    cause ``PhysxArticulationAPI`` to be applied. ``fix_root_link`` is a writer-side flag
    materializing ``UsdPhysics.FixedJoint``; it does not author any PhysX-namespaced
    attribute. Newton-targeted prims that only set ``fix_root_link`` should not receive
    ``PhysxArticulationAPI`` stamping."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    base_cfg = schemas.ArticulationRootBaseCfg(fix_root_link=False)
    sim_utils.create_prim("/World/arti_frl", prim_type="Xform")
    schemas.define_articulation_root_properties("/World/arti_frl", base_cfg)

    prim = stage.GetPrimAtPath("/World/arti_frl")
    applied = prim.GetAppliedSchemas()
    assert "PhysxArticulationAPI" not in applied, (
        f"PhysxArticulationAPI should not be applied when only fix_root_link is set; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_articulation_root_writes_self_collisions(setup_simulation):
    """Setting ``enabled_self_collisions`` on ``PhysxArticulationRootPropertiesCfg`` must
    author ``physxArticulation:enabledSelfCollisions`` AND apply ``PhysxArticulationAPI``."""
    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    cfg = PhysxArticulationRootPropertiesCfg(enabled_self_collisions=True)
    sim_utils.create_prim("/World/arti_sc", prim_type="Xform")
    schemas.define_articulation_root_properties("/World/arti_sc", cfg)

    prim = stage.GetPrimAtPath("/World/arti_sc")
    assert prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    applied = prim.GetAppliedSchemas()
    assert "PhysxArticulationAPI" in applied


@pytest.mark.isaacsim_ci
def test_articulation_root_deprecation_alias(setup_simulation):
    """Instantiating the legacy ``ArticulationRootPropertiesCfg`` name emits exactly one
    ``DeprecationWarning`` whose message references the 5.0 removal target."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ArticulationRootDeprecatedAliasCfg()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected exactly one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_mesh_collision_base_cfg_writes_approximation_token(setup_simulation):
    """``MeshCollisionBaseCfg(mesh_approximation_name="boundingCube")`` authors
    ``physics:approximation`` via ``UsdPhysics.MeshCollisionAPI``. No PhysX cooking schema is
    applied because the base class declares no PhysX namespace."""
    from pxr import UsdGeom

    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    UsdGeom.Mesh.Define(stage, "/World/mesh_base")
    cfg = schemas.MeshCollisionBaseCfg(mesh_approximation_name="boundingCube")
    schemas.define_mesh_collision_properties("/World/mesh_base", cfg)

    prim = stage.GetPrimAtPath("/World/mesh_base")
    assert prim.GetAttribute("physics:approximation").Get() == "boundingCube"
    applied = prim.GetAppliedSchemas()
    # The standard UsdPhysics.MeshCollisionAPI is registered under
    # ``PhysicsMeshCollisionAPI`` in the prim's applied-schema list.
    assert any("MeshCollisionAPI" in s for s in applied), (
        f"a MeshCollisionAPI schema must be applied; got {list(applied)}"
    )
    # no PhysX cooking schema applied for the base class
    assert not any(s.startswith("Physx") and "Mesh" in s for s in applied), (
        f"no PhysX mesh schema should be applied for the base class; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_physx_convex_hull_writes_tuning_attrs(setup_simulation):
    """Setting tuning fields on ``PhysxConvexHullPropertiesCfg`` authors the
    ``physxConvexHullCollision:*`` namespaced attributes AND applies
    ``PhysxConvexHullCollisionAPI``."""
    from isaaclab_physx.sim.schemas import PhysxConvexHullPropertiesCfg

    from pxr import UsdGeom

    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    UsdGeom.Mesh.Define(stage, "/World/mesh_ch")
    cfg = PhysxConvexHullPropertiesCfg(hull_vertex_limit=64, min_thickness=0.001)
    schemas.define_mesh_collision_properties("/World/mesh_ch", cfg)

    prim = stage.GetPrimAtPath("/World/mesh_ch")
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    assert prim.GetAttribute("physxConvexHullCollision:hullVertexLimit").Get() == 64
    assert prim.GetAttribute("physxConvexHullCollision:minThickness").Get() == pytest.approx(0.001)
    applied = prim.GetAppliedSchemas()
    assert "PhysxConvexHullCollisionAPI" in applied


@pytest.mark.isaacsim_ci
def test_physx_convex_hull_no_physx_schema_when_no_tuning_fields_set(setup_simulation):
    """Regression: ``PhysxConvexHullPropertiesCfg()`` with all tuning fields None must NOT
    apply ``PhysxConvexHullCollisionAPI``. The approximation token is still authored on the
    standard ``UsdPhysics.MeshCollisionAPI``."""
    from isaaclab_physx.sim.schemas import PhysxConvexHullPropertiesCfg

    from pxr import UsdGeom

    sim, _, _, _, _, _ = setup_simulation
    stage = sim_utils.get_current_stage()

    UsdGeom.Mesh.Define(stage, "/World/mesh_ch_default")
    cfg = PhysxConvexHullPropertiesCfg()
    schemas.define_mesh_collision_properties("/World/mesh_ch_default", cfg)

    prim = stage.GetPrimAtPath("/World/mesh_ch_default")
    assert prim.GetAttribute("physics:approximation").Get() == "convexHull"
    applied = prim.GetAppliedSchemas()
    assert "PhysxConvexHullCollisionAPI" not in applied, (
        f"PhysxConvexHullCollisionAPI should not be applied without tuning fields; got {list(applied)}"
    )


@pytest.mark.isaacsim_ci
def test_bounding_cube_default_token(setup_simulation):
    """``BoundingCubePropertiesCfg()`` defaults to the ``boundingCube`` token."""
    cfg = schemas.BoundingCubePropertiesCfg()
    assert cfg.mesh_approximation_name == "boundingCube"


@pytest.mark.isaacsim_ci
@pytest.mark.parametrize(
    "name",
    [
        "MeshCollisionPropertiesCfg",
        "ConvexHullPropertiesCfg",
        "ConvexDecompositionPropertiesCfg",
        "TriangleMeshPropertiesCfg",
        "TriangleMeshSimplificationPropertiesCfg",
        "SDFMeshPropertiesCfg",
    ],
)
def test_mesh_collision_deprecation_aliases(setup_simulation, name):
    """Each legacy mesh-collision class name emits exactly one DeprecationWarning on
    instantiation and the warning message references the 5.0 removal target."""
    from isaaclab_physx.sim.schemas import schemas_cfg as physx_cfg

    cls = getattr(physx_cfg, name)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"{name}: expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_physx_fixed_tendon_relocation(setup_simulation):
    """``PhysxFixedTendonPropertiesCfg`` is importable from
    :mod:`isaaclab_physx.sim.schemas` and round-trips its fields."""
    from isaaclab_physx.sim.schemas import PhysxFixedTendonPropertiesCfg

    cfg = PhysxFixedTendonPropertiesCfg(
        tendon_enabled=True,
        stiffness=10.0,
        damping=0.5,
        limit_stiffness=1.0,
        offset=0.1,
        rest_length=0.2,
    )
    assert cfg.tendon_enabled is True
    assert cfg.stiffness == 10.0
    assert cfg.damping == 0.5
    assert cfg.limit_stiffness == 1.0
    assert cfg.offset == 0.1
    assert cfg.rest_length == 0.2


@pytest.mark.isaacsim_ci
def test_fixed_tendon_deprecation_alias(setup_simulation):
    """Instantiating the legacy ``FixedTendonPropertiesCfg`` (via the shim) emits exactly
    one ``DeprecationWarning`` whose message references the 5.0 removal target."""
    cls = schemas.FixedTendonPropertiesCfg
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_physx_spatial_tendon_relocation(setup_simulation):
    """``PhysxSpatialTendonPropertiesCfg`` is importable from
    :mod:`isaaclab_physx.sim.schemas` and round-trips its fields."""
    from isaaclab_physx.sim.schemas import PhysxSpatialTendonPropertiesCfg

    cfg = PhysxSpatialTendonPropertiesCfg(
        tendon_enabled=True,
        stiffness=20.0,
        damping=0.25,
        limit_stiffness=2.0,
        offset=0.05,
    )
    assert cfg.tendon_enabled is True
    assert cfg.stiffness == 20.0
    assert cfg.damping == 0.25
    assert cfg.limit_stiffness == 2.0
    assert cfg.offset == 0.05


@pytest.mark.isaacsim_ci
def test_spatial_tendon_deprecation_alias(setup_simulation):
    """Instantiating the legacy ``SpatialTendonPropertiesCfg`` (via the shim) emits exactly
    one ``DeprecationWarning`` whose message references the 5.0 removal target."""
    cls = schemas.SpatialTendonPropertiesCfg
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cls()
    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1, f"expected one DeprecationWarning, got {len(deprecations)}"
    assert "5.0" in str(deprecations[0].message)


@pytest.mark.isaacsim_ci
def test_usd_api_physx_api_attrs_deprecated(setup_simulation):
    """Reading ``cfg.usd_api`` and ``cfg.physx_api`` on the new mesh cfgs emits a
    DeprecationWarning and returns the legacy-mapped string value."""
    from isaaclab_physx.sim.schemas import PhysxConvexHullPropertiesCfg

    cfg = PhysxConvexHullPropertiesCfg()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        usd_api_value = cfg.usd_api
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert usd_api_value == "MeshCollisionAPI"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        physx_api_value = cfg.physx_api
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert physx_api_value == "PhysxConvexHullCollisionAPI"


@pytest.mark.isaacsim_ci
def test_modify_properties_on_invalid_prim(setup_simulation):
    """Test modifying properties on a prim that does not exist."""
    sim, _, rigid_cfg, _, _, _ = setup_simulation
    # set properties
    with pytest.raises(ValueError):
        schemas.modify_rigid_body_properties("/World/asset_xyz", rigid_cfg)


@pytest.mark.isaacsim_ci
def test_modify_properties_on_articulation_instanced_usd(setup_simulation):
    """Test modifying properties on articulation instanced usd.

    In this case, modifying collision properties on the articulation instanced usd will fail.
    """
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    # spawn asset to the stage
    asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_c/anymal_c.usd"
    if "4.5" in ISAAC_NUCLEUS_DIR:
        asset_usd_file = asset_usd_file.replace("http", "https").replace("4.5", "5.0")
    sim_utils.create_prim("/World/asset_instanced", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

    # set properties on the asset and check all properties are set
    schemas.modify_articulation_root_properties("/World/asset_instanced", arti_cfg)
    schemas.modify_rigid_body_properties("/World/asset_instanced", rigid_cfg)
    schemas.modify_mass_properties("/World/asset_instanced", mass_cfg)
    schemas.modify_joint_drive_properties("/World/asset_instanced", joint_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset_instanced/base", arti_cfg, False)
    _validate_rigid_body_properties_on_prim("/World/asset_instanced", rigid_cfg)
    _validate_mass_properties_on_prim("/World/asset_instanced", mass_cfg)
    _validate_joint_drive_properties_on_prim("/World/asset_instanced", joint_cfg)

    # make a fixed joint
    arti_cfg.fix_root_link = True
    schemas.modify_articulation_root_properties("/World/asset_instanced", arti_cfg)


@pytest.mark.isaacsim_ci
def test_modify_properties_on_articulation_usd(setup_simulation):
    """Test setting properties on articulation usd."""
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, joint_cfg = setup_simulation
    # spawn asset to the stage
    asset_usd_file = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    if "4.5" in ISAAC_NUCLEUS_DIR:
        asset_usd_file = asset_usd_file.replace("http", "https").replace("4.5", "5.0")
    sim_utils.create_prim("/World/asset", usd_path=asset_usd_file, translation=(0.0, 0.0, 0.62))

    # set properties on the asset and check all properties are set
    schemas.modify_articulation_root_properties("/World/asset", arti_cfg)
    schemas.modify_rigid_body_properties("/World/asset", rigid_cfg)
    schemas.modify_collision_properties("/World/asset", collision_cfg)
    schemas.modify_mass_properties("/World/asset", mass_cfg)
    schemas.modify_joint_drive_properties("/World/asset", joint_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset", arti_cfg, True)
    _validate_rigid_body_properties_on_prim("/World/asset", rigid_cfg)
    _validate_collision_properties_on_prim("/World/asset", collision_cfg)
    _validate_mass_properties_on_prim("/World/asset", mass_cfg)
    _validate_joint_drive_properties_on_prim("/World/asset", joint_cfg)

    # make a fixed joint
    arti_cfg.fix_root_link = True
    schemas.modify_articulation_root_properties("/World/asset", arti_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/asset", arti_cfg, True)


@pytest.mark.isaacsim_ci
def test_defining_rigid_body_properties_on_prim(setup_simulation):
    """Test defining rigid body properties on a prim."""
    sim, _, rigid_cfg, collision_cfg, mass_cfg, _ = setup_simulation
    # create a prim
    sim_utils.create_prim("/World/parent", prim_type="XForm")
    # spawn a prim
    sim_utils.create_prim("/World/cube1", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    # set properties on the asset and check all properties are set
    schemas.define_rigid_body_properties("/World/cube1", rigid_cfg)
    schemas.define_collision_properties("/World/cube1", collision_cfg)
    schemas.define_mass_properties("/World/cube1", mass_cfg)
    # validate the properties
    _validate_rigid_body_properties_on_prim("/World/cube1", rigid_cfg)
    _validate_collision_properties_on_prim("/World/cube1", collision_cfg)
    _validate_mass_properties_on_prim("/World/cube1", mass_cfg)

    # spawn another prim
    sim_utils.create_prim("/World/cube2", prim_type="Cube", translation=(1.0, 1.0, 0.62))
    # set properties on the asset and check all properties are set
    schemas.define_rigid_body_properties("/World/cube2", rigid_cfg)
    schemas.define_collision_properties("/World/cube2", collision_cfg)
    # validate the properties
    _validate_rigid_body_properties_on_prim("/World/cube2", rigid_cfg)
    _validate_collision_properties_on_prim("/World/cube2", collision_cfg)

    # check if we can play
    sim.reset()
    for _ in range(100):
        sim.step()


@pytest.mark.isaacsim_ci
def test_defining_articulation_properties_on_prim(setup_simulation):
    """Test defining articulation properties on a prim."""
    sim, arti_cfg, rigid_cfg, collision_cfg, mass_cfg, _ = setup_simulation
    # create a parent articulation
    sim_utils.create_prim("/World/parent", prim_type="Xform")
    schemas.define_articulation_root_properties("/World/parent", arti_cfg)
    # validate the properties
    _validate_articulation_properties_on_prim("/World/parent", arti_cfg, False)

    # create a child articulation
    sim_utils.create_prim("/World/parent/child", prim_type="Cube", translation=(0.0, 0.0, 0.62))
    schemas.define_rigid_body_properties("/World/parent/child", rigid_cfg)
    schemas.define_mass_properties("/World/parent/child", mass_cfg)

    # check if we can play
    sim.reset()
    for _ in range(100):
        sim.step()


@pytest.mark.isaacsim_ci
def test_multi_instance_schema_detection_on_tendon_joints(setup_simulation):
    """Test that multi-instance PhysX tendon schemas are correctly detected via substring matching.

    Multi-instance schemas (e.g. PhysxTendonAxisAPI, PhysxTendonAxisRootAPI) appear in
    GetAppliedSchemas() as 'SchemaName:instanceName' (e.g. 'PhysxTendonAxisAPI:inst0').
    An exact ``in list`` check fails because 'PhysxTendonAxisAPI' != 'PhysxTendonAxisAPI:inst0'.
    This test ensures the substring-based detection used by modify_joint_drive_properties
    and modify_fixed_tendon_properties handles multi-instance schemas correctly.

    We call the unwrapped functions directly (via ``__wrapped__``) to bypass the
    ``@apply_nested`` decorator, which traverses children and does not return the
    inner function's bool result.
    """
    sim, _, _, _, _, joint_cfg = setup_simulation
    stage = sim_utils.get_current_stage()

    # unwrap to get the raw functions that return bool
    _modify_joint_drive = schemas.modify_joint_drive_properties.__wrapped__
    _modify_fixed_tendon = schemas.modify_fixed_tendon_properties.__wrapped__

    # -- set up two body prims connected by a revolute joint
    sim_utils.create_prim("/World/tendon_test", prim_type="Xform")
    sim_utils.create_prim("/World/tendon_test/body0", prim_type="Cube")
    sim_utils.create_prim("/World/tendon_test/body1", prim_type="Cube")
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/tendon_test/body1/joint0")
    joint_prim = joint.GetPrim()

    # -- 1) Joint with only tendon child schema (no root) -> drive should be SKIPPED
    joint_prim.AddAppliedSchema("PhysxTendonAxisAPI:inst0")
    applied = joint_prim.GetAppliedSchemas()
    assert any("PhysxTendonAxisAPI" in s for s in applied), "Multi-instance schema not found via substring"
    assert "PhysxTendonAxisAPI" not in applied, "Exact match should NOT find multi-instance schema"

    result = _modify_joint_drive(joint_prim.GetPrimPath().pathString, joint_cfg)
    assert result is False, "Tendon child joint should be skipped (return False)"

    # -- 2) Joint with both child AND root tendon schema -> drive should NOT be skipped
    joint_prim.AddAppliedSchema("PhysxTendonAxisRootAPI:inst0")
    applied = joint_prim.GetAppliedSchemas()
    assert any("PhysxTendonAxisRootAPI" in s for s in applied)
    assert "PhysxTendonAxisRootAPI" not in applied, "Exact match should NOT find multi-instance schema"

    result = _modify_joint_drive(joint_prim.GetPrimPath().pathString, joint_cfg)
    assert result is True, "Tendon root joint should NOT be skipped"

    # -- 3) modify_fixed_tendon_properties should detect multi-instance root schema
    tendon_cfg = schemas.FixedTendonPropertiesCfg(stiffness=10.0, damping=0.1)
    result = _modify_fixed_tendon(joint_prim.GetPrimPath().pathString, tendon_cfg)
    assert result is True, "Prim with PhysxTendonAxisRootAPI:inst0 should be detected"

    # -- 4) Prim WITHOUT any tendon root schema -> modify_fixed_tendon should return False
    sim_utils.create_prim("/World/tendon_test/body2", prim_type="Cube")
    no_tendon_joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/tendon_test/body2/joint1")
    result = _modify_fixed_tendon(no_tendon_joint.GetPrim().GetPrimPath().pathString, tendon_cfg)
    assert result is False, "Prim without tendon root schema should return False"


"""
Helper functions.
"""


def _validate_articulation_properties_on_prim(
    prim_path: str, arti_cfg, has_default_fixed_root: bool, verbose: bool = False
):
    """Validate the articulation properties on the prim.

    If :attr:`has_default_fixed_root` is True, then the asset already has a fixed root link. This is used to check the
    expected behavior of the fixed root link configuration.
    """
    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check articulation properties are set correctly
    for attr_name, attr_value in arti_cfg.__dict__.items():
        # skip class metadata and names we know are not present
        if attr_name.startswith("_") or attr_name == "func":
            continue
        # handle fixed root link
        if attr_name == "fix_root_link" and attr_value is not None:
            # obtain the fixed joint prim
            fixed_joint_prim = sim_utils.find_global_fixed_joint_prim(prim_path)
            # if asset does not have a fixed root link then check if the joint is created
            if not has_default_fixed_root:
                if attr_value:
                    assert fixed_joint_prim is not None
                else:
                    assert fixed_joint_prim is None
            else:
                # check a joint exists
                assert fixed_joint_prim is not None
                # check if the joint is enabled or disabled
                is_enabled = fixed_joint_prim.GetJointEnabledAttr().Get()
                assert is_enabled == attr_value
            # skip the rest of the checks
            continue
        # convert attribute name in prim to cfg name
        prim_prop_name = f"physxArticulation:{to_camel_case(attr_name, to='cC')}"
        # validate the values
        assert root_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, abs=1e-5), (
            f"Failed setting for {prim_prop_name}"
        )


def _validate_rigid_body_properties_on_prim(prim_path: str, rigid_cfg, verbose: bool = False):
    """Validate the rigid body properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check rigid body properties are set correctly
    for link_prim in root_prim.GetChildren():
        if UsdPhysics.RigidBodyAPI(link_prim):
            for attr_name, attr_value in rigid_cfg.__dict__.items():
                # skip class metadata and names we know are not present
                if attr_name.startswith("_") or attr_name in [
                    "func",
                    "rigid_body_enabled",
                    "kinematic_enabled",
                ]:
                    continue
                # convert attribute name in prim to cfg name
                prim_prop_name = f"physxRigidBody:{to_camel_case(attr_name, to='cC')}"
                # validate the values
                assert link_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, abs=1e-5), (
                    f"Failed setting for {prim_prop_name}"
                )
        elif verbose:
            print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a rigid body.")


def _validate_collision_properties_on_prim(prim_path: str, collision_cfg, verbose: bool = False):
    """Validate the collision properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check collision properties are set correctly
    for link_prim in root_prim.GetChildren():
        for mesh_prim in link_prim.GetChildren():
            if UsdPhysics.CollisionAPI(mesh_prim):
                for attr_name, attr_value in collision_cfg.__dict__.items():
                    # skip names we know are not present and class-metadata keys
                    if attr_name.startswith("_") or attr_name in ["func", "collision_enabled"]:
                        continue
                    # convert attribute name in prim to cfg name
                    prim_prop_name = f"physxCollision:{to_camel_case(attr_name, to='cC')}"
                    # validate the values
                    assert mesh_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, abs=1e-5), (
                        f"Failed setting for {prim_prop_name}"
                    )
            elif verbose:
                print(f"Skipping prim {mesh_prim.GetPrimPath()} as it is not a collision mesh.")


def _validate_mass_properties_on_prim(prim_path: str, mass_cfg, verbose: bool = False):
    """Validate the mass properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check rigid body mass properties are set correctly
    for link_prim in root_prim.GetChildren():
        if UsdPhysics.MassAPI(link_prim):
            for attr_name, attr_value in mass_cfg.__dict__.items():
                # skip names we know are not present and class-metadata keys
                if attr_name in ["func"] or attr_name.startswith("_"):
                    continue
                # print(link_prim.GetProperties())
                prim_prop_name = f"physics:{to_camel_case(attr_name, to='cC')}"
                # validate the values
                assert link_prim.GetAttribute(prim_prop_name).Get() == pytest.approx(attr_value, abs=1e-5), (
                    f"Failed setting for {prim_prop_name}"
                )
        elif verbose:
            print(f"Skipping prim {link_prim.GetPrimPath()} as it is not a mass api.")


def _validate_joint_drive_properties_on_prim(prim_path: str, joint_cfg, verbose: bool = False):
    """Validate the mass properties on the prim.

    Note:
        Right now this function exploits the hierarchy in the asset to check the properties. This is not a
        fool-proof way of checking the properties.
    """
    # Obtain stage handle
    stage = sim_utils.get_current_stage()
    # the root prim
    root_prim = stage.GetPrimAtPath(prim_path)
    # check joint drive properties are set correctly
    for link_prim in root_prim.GetAllChildren():
        for joint_prim in link_prim.GetChildren():
            if joint_prim.IsA(UsdPhysics.PrismaticJoint) or joint_prim.IsA(UsdPhysics.RevoluteJoint):
                # check it has drive API
                assert joint_prim.HasAPI(UsdPhysics.DriveAPI)
                # iterate over the joint properties
                for attr_name, attr_value in joint_cfg.__dict__.items():
                    # skip class metadata and names we know are not present on the USD prim
                    if attr_name.startswith("_") or attr_name in ["func", "ensure_drives_exist"]:
                        continue
                    # resolve the drive (linear or angular)
                    drive_model = "linear" if joint_prim.IsA(UsdPhysics.PrismaticJoint) else "angular"

                    # manually check joint type since it is a string type
                    if attr_name == "drive_type":
                        prim_attr_name = f"drive:{drive_model}:physics:type"
                        # check the value
                        assert attr_value == joint_prim.GetAttribute(prim_attr_name).Get()
                        continue

                    # non-string attributes
                    if attr_name == "max_joint_velocity":
                        prim_attr_name = "physxJoint:maxJointVelocity"
                    else:
                        prim_attr_name = f"drive:{drive_model}:physics:{to_camel_case(attr_name, to='cC')}"

                    # obtain value from USD API (for angular, these follow degrees unit)
                    prim_attr_value = joint_prim.GetAttribute(prim_attr_name).Get()

                    # for angular drives, we expect user to set in radians
                    # the values reported by USD are in degrees
                    if drive_model == "angular":
                        if attr_name == "max_joint_velocity":
                            # deg / s --> rad / s
                            prim_attr_value = prim_attr_value * math.pi / 180.0
                        elif attr_name in ["stiffness", "damping"]:
                            # N-m/deg or N-m-s/deg --> N-m/rad or N-m-s/rad
                            prim_attr_value = prim_attr_value * 180.0 / math.pi

                    # validate the values
                    assert prim_attr_value == pytest.approx(attr_value, abs=1e-5), (
                        f"Failed setting for {prim_attr_name}"
                    )
            elif verbose:
                print(f"Skipping prim {joint_prim.GetPrimPath()} as it is not a joint drive api.")
