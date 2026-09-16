# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD parity between the legacy single-cfg schema writers and their fragment replacements.

Each test authors the legacy cfg on one prim and the fragment list that replaces it on a
sibling prim of the same type, then diffs the applied API schemas and every authored
attribute. A fragment list is only a faithful replacement when the resulting USD is
byte-for-byte equivalent, so any dropped attribute, dropped applied schema, or changed unit
conversion shows up as a non-empty diff.

The tests run on an in-memory USD stage and intentionally do NOT launch Isaac Sim / Kit.
``Usd.Prim.GetAppliedSchemas`` filters out schemas the USD registry does not know about, so
token-authored schemas such as ``NewtonArticulationRootAPI`` are read back through
``Usd.PrimTypeInfo.GetAppliedAPISchemas`` instead.
"""

import math

import pytest

from pxr import Usd, UsdGeom, UsdPhysics

from isaaclab.sim.schemas import (
    MassCfg,
    MassPropertiesCfg,
    UsdPhysicsCollisionCfg,
    UsdPhysicsDriveCfg,
    UsdPhysicsRigidBodyCfg,
    apply_articulation_root_properties,
    apply_collision_properties,
    apply_joint_drive_properties,
    apply_mass_properties,
    apply_rigid_body_properties,
    define_articulation_root_properties,
    define_collision_properties,
    define_mass_properties,
    define_rigid_body_properties,
    modify_joint_drive_properties,
)

from isaaclab_newton.sim.schemas import NewtonArticulationCfg  # isort: skip
from isaaclab_physx.sim.schemas import (  # isort: skip
    PhysxArticulationCfg,
    PhysxArticulationRootPropertiesCfg,
    PhysxCollisionCfg,
    PhysxCollisionPropertiesCfg,
    PhysxJointCfg,
    PhysxJointDrivePropertiesCfg,
    PhysxRigidBodyCfg,
    PhysxRigidBodyPropertiesCfg,
)

LEGACY_PATH = "/World/Legacy"
FRAGMENT_PATH = "/World/Fragment"


def _make_sibling_prims(prim_type: str) -> tuple[Usd.Stage, Usd.Prim, Usd.Prim]:
    """Create an in-memory stage holding two identical sibling prims to author on.

    Args:
        prim_type: One of ``"cube"``, ``"xform"``, ``"revolute"`` or ``"prismatic"``.

    Returns:
        The stage, the prim reserved for the legacy writer, and the prim reserved for the
        fragment writer.
    """
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    definers = {
        "cube": UsdGeom.Cube.Define,
        "xform": UsdGeom.Xform.Define,
        "revolute": UsdPhysics.RevoluteJoint.Define,
        "prismatic": UsdPhysics.PrismaticJoint.Define,
    }
    define = definers[prim_type]
    for path in (LEGACY_PATH, FRAGMENT_PATH):
        define(stage, path)
    return stage, stage.GetPrimAtPath(LEGACY_PATH), stage.GetPrimAtPath(FRAGMENT_PATH)


def _applied_api_schemas(prim: Usd.Prim) -> set[str]:
    """Return the applied API schema names authored on a prim, including token schemas."""
    return set(prim.GetPrimTypeInfo().GetAppliedAPISchemas())


def _authored_attributes(prim: Usd.Prim) -> dict[str, object]:
    """Return every authored attribute on a prim as a name-to-value mapping."""
    return {attr.GetName(): attr.Get() for attr in prim.GetAttributes() if attr.IsAuthored()}


def _diff_authoring(legacy: Usd.Prim, fragment: Usd.Prim) -> list[str]:
    """Diff the applied schemas and authored attributes of two prims.

    Returns:
        A list of human-readable differences. Empty when both prims carry identical USD.
    """
    differences = []
    legacy_schemas, fragment_schemas = _applied_api_schemas(legacy), _applied_api_schemas(fragment)
    for name in sorted(legacy_schemas - fragment_schemas):
        differences.append(f"applied schema '{name}' missing on the fragment prim")
    for name in sorted(fragment_schemas - legacy_schemas):
        differences.append(f"applied schema '{name}' only on the fragment prim")

    legacy_attrs, fragment_attrs = _authored_attributes(legacy), _authored_attributes(fragment)
    for name in sorted(set(legacy_attrs) | set(fragment_attrs)):
        if name not in legacy_attrs:
            differences.append(f"attribute '{name}' only authored by the fragment writer")
            continue
        if name not in fragment_attrs:
            differences.append(f"attribute '{name}' not authored by the fragment writer")
            continue
        legacy_value, fragment_value = legacy_attrs[name], fragment_attrs[name]
        if isinstance(legacy_value, float) and isinstance(fragment_value, float):
            if math.isclose(legacy_value, fragment_value, rel_tol=1e-9, abs_tol=1e-12):
                continue
        if legacy_value != fragment_value:
            differences.append(f"attribute '{name}': legacy={legacy_value!r} fragment={fragment_value!r}")
    return differences


def test_rigid_body_fragments_match_legacy_authoring():
    """The UsdPhysics + PhysX rigid-body fragment pair reproduces the legacy rigid-body cfg."""
    stage, legacy, fragment = _make_sibling_prims("cube")
    physx_values = dict(
        disable_gravity=True,
        linear_damping=0.1,
        angular_damping=0.2,
        max_linear_velocity=123.0,
        max_angular_velocity=456.0,
        max_depenetration_velocity=1.5,
        max_contact_impulse=1.0e4,
        enable_gyroscopic_forces=True,
        retain_accelerations=False,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=1,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    )

    define_rigid_body_properties(
        LEGACY_PATH,
        PhysxRigidBodyPropertiesCfg(rigid_body_enabled=True, kinematic_enabled=False, **physx_values),
        stage,
    )
    apply_rigid_body_properties(
        FRAGMENT_PATH,
        [
            UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=False),
            PhysxRigidBodyCfg(**physx_values),
        ],
        create_if_missing=True,
        stage=stage,
    )

    assert _diff_authoring(legacy, fragment) == []
    # guard against both writers being a no-op
    assert fragment.GetAttribute("physxRigidBody:disableGravity").Get() is True


def test_collision_fragments_match_legacy_authoring():
    """The UsdPhysics + PhysX collision fragment pair reproduces the legacy collision cfg."""
    stage, legacy, fragment = _make_sibling_prims("cube")
    physx_values = dict(
        contact_offset=0.02, rest_offset=0.001, torsional_patch_radius=0.1, min_torsional_patch_radius=0.05
    )

    define_collision_properties(LEGACY_PATH, PhysxCollisionPropertiesCfg(collision_enabled=True, **physx_values), stage)
    apply_collision_properties(
        FRAGMENT_PATH,
        [UsdPhysicsCollisionCfg(collision_enabled=True), PhysxCollisionCfg(**physx_values)],
        create_if_missing=True,
        stage=stage,
    )

    assert _diff_authoring(legacy, fragment) == []
    assert fragment.GetAttribute("physxCollision:contactOffset").Get() == pytest.approx(0.02)


def test_mass_fragments_match_legacy_authoring():
    """The mass fragment reproduces the legacy mass cfg."""
    stage, legacy, fragment = _make_sibling_prims("cube")

    define_mass_properties(LEGACY_PATH, MassPropertiesCfg(mass=2.5), stage)
    apply_mass_properties(FRAGMENT_PATH, [MassCfg(mass=2.5)], create_if_missing=True, stage=stage)

    assert _diff_authoring(legacy, fragment) == []
    assert fragment.GetAttribute("physics:mass").Get() == pytest.approx(2.5)


def test_articulation_root_fragments_match_legacy_authoring():
    """The PhysX + Newton articulation fragment pair reproduces the legacy articulation cfg.

    The legacy writer mirrors ``enabled_self_collisions`` onto ``newton:selfCollisionEnabled``,
    so a faithful replacement needs both fragments.
    """
    stage, legacy, fragment = _make_sibling_prims("xform")
    physx_values = dict(
        articulation_enabled=True,
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=2,
        sleep_threshold=0.005,
        stabilization_threshold=0.001,
    )

    define_articulation_root_properties(LEGACY_PATH, PhysxArticulationRootPropertiesCfg(**physx_values), stage)
    UsdPhysics.ArticulationRootAPI.Apply(fragment)
    apply_articulation_root_properties(
        FRAGMENT_PATH,
        [PhysxArticulationCfg(**physx_values), NewtonArticulationCfg(self_collision_enabled=True)],
        stage=stage,
    )

    assert _diff_authoring(legacy, fragment) == []
    # both namespaces of the dual-namespace self-collision flag must be authored
    assert fragment.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is True
    assert fragment.GetAttribute("newton:selfCollisionEnabled").Get() is True
    assert "NewtonArticulationRootAPI" in _applied_api_schemas(fragment)


def test_parity_check_detects_dropped_newton_self_collision_mirror():
    """Negative control: a PhysX-only articulation mapping must be reported as a difference.

    This is the mapping mistake the parity check exists to catch -- replacing the legacy
    articulation cfg with :class:`PhysxArticulationCfg` alone silently drops the Newton
    self-collision attribute and reverts Newton to its schema default.
    """
    stage, legacy, fragment = _make_sibling_prims("xform")

    define_articulation_root_properties(
        LEGACY_PATH, PhysxArticulationRootPropertiesCfg(enabled_self_collisions=True), stage
    )
    UsdPhysics.ArticulationRootAPI.Apply(fragment)
    apply_articulation_root_properties(FRAGMENT_PATH, [PhysxArticulationCfg(enabled_self_collisions=True)], stage=stage)

    differences = _diff_authoring(legacy, fragment)
    assert differences, "the parity check failed to detect a dropped Newton self-collision mirror"
    assert any("newton:selfCollisionEnabled" in entry for entry in differences)
    assert any("NewtonArticulationRootAPI" in entry for entry in differences)
    # the PhysX half is identical, so the diff must be about the Newton namespace only
    assert not any("physxArticulation" in entry for entry in differences)


@pytest.mark.parametrize("joint_type", ["revolute", "prismatic"])
def test_joint_drive_fragments_match_legacy_authoring(joint_type):
    """The UsdPhysics drive + PhysX joint fragment pair reproduces the legacy joint-drive cfg.

    Running both joint types covers the two behaviours the joint-drive writer branches on: the
    ``DriveAPI:angular`` / ``DriveAPI:linear`` multi-instance selection, and the radian-to-degree
    conversion that applies to angular drives only.
    """
    stage, legacy, fragment = _make_sibling_prims(joint_type)
    drive_values = dict(drive_type="force", max_force=87.0, stiffness=100.0, damping=10.0)

    modify_joint_drive_properties(
        LEGACY_PATH, PhysxJointDrivePropertiesCfg(max_joint_velocity=3.0, **drive_values), stage
    )
    apply_joint_drive_properties(
        FRAGMENT_PATH,
        [UsdPhysicsDriveCfg(**drive_values), PhysxJointCfg(max_joint_velocity=3.0)],
        stage=stage,
    )

    assert _diff_authoring(legacy, fragment) == []

    instance = "angular" if joint_type == "revolute" else "linear"
    other_instance = "linear" if joint_type == "revolute" else "angular"
    authored = _authored_attributes(fragment)
    assert f"drive:{instance}:physics:stiffness" in authored
    assert f"drive:{other_instance}:physics:stiffness" not in authored

    # angular drives are stored in degree units, linear drives in the cfg's own units
    scale = math.pi / 180.0 if joint_type == "revolute" else 1.0
    assert authored[f"drive:{instance}:physics:stiffness"] == pytest.approx(100.0 * scale)
    assert authored[f"drive:{instance}:physics:damping"] == pytest.approx(10.0 * scale)
    assert authored[f"drive:{instance}:physics:maxForce"] == pytest.approx(87.0)
    velocity_scale = 180.0 / math.pi if joint_type == "revolute" else 1.0
    assert authored["physxJoint:maxJointVelocity"] == pytest.approx(3.0 * velocity_scale)


def test_migrated_consumers_select_joint_drive_fragments(source_checkout_root):
    """Joint-drive slots combine USD and PhysX fragments and cannot be read as one cfg."""
    import ast

    violations = []
    for backend in ("newton", "physx", "ov"):
        path = source_checkout_root / f"source/isaaclab_{backend}/test/assets/test_articulation.py"
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "joint_drive_props"
            ):
                violations.append(f"{path.relative_to(source_checkout_root)}:{node.lineno}: {ast.unparse(node)}")
    assert not violations, "Single-config access on fragment lists:\n" + "\n".join(violations)


def test_asset_and_task_configs_use_bare_single_fragments(source_checkout_root):
    """Single schema fragments stay bare; lists are reserved for multiple fragments."""
    import ast

    slots = {
        "rigid_props",
        "collision_props",
        "articulation_props",
        "joint_drive_props",
        "mass_props",
        "mesh_collision_props",
        "fixed_tendons_props",
        "spatial_tendons_props",
        "physics_material",
    }
    violations = []
    for package in ("isaaclab_assets", "isaaclab_tasks"):
        for path in (source_checkout_root / "source" / package / package).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if (
                    isinstance(node, ast.keyword)
                    and node.arg in slots
                    and isinstance(node.value, ast.List)
                    and len(node.value.elts) == 1
                ):
                    violations.append(f"{path.relative_to(source_checkout_root)}:{node.lineno}: {node.arg}")
    assert not violations, "Unnecessary single-fragment lists:\n" + "\n".join(violations)
