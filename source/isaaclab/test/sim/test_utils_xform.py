# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import math

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
    # Setup: Create a new stage
    sim_utils.create_new_stage()
    sim_utils.update_stage()

    # Yield for the test
    yield

    # Teardown: Clear stage after each test
    sim_utils.clear_stage()


def assert_vec3_close(v1: Gf.Vec3d | Gf.Vec3f, v2: tuple | Gf.Vec3d | Gf.Vec3f, eps: float = 1e-6):
    """Assert two 3D vectors are close."""
    if isinstance(v2, tuple):
        v2 = Gf.Vec3d(*v2)
    for i in range(3):
        assert math.isclose(v1[i], v2[i], abs_tol=eps), f"Vector mismatch at index {i}: {v1[i]} != {v2[i]}"


def assert_quat_close(q1: Gf.Quatf | Gf.Quatd, q2: Gf.Quatf | Gf.Quatd | tuple, eps: float = 1e-6):
    """Assert two quaternions are close, accounting for double-cover (q and -q represent same rotation)."""
    if isinstance(q2, tuple):
        q2 = Gf.Quatd(*q2)
    # Check if quaternions are close (either q1 ≈ q2 or q1 ≈ -q2)
    real_match = math.isclose(q1.GetReal(), q2.GetReal(), abs_tol=eps)
    imag_match = all(math.isclose(q1.GetImaginary()[i], q2.GetImaginary()[i], abs_tol=eps) for i in range(3))

    real_match_neg = math.isclose(q1.GetReal(), -q2.GetReal(), abs_tol=eps)
    imag_match_neg = all(math.isclose(q1.GetImaginary()[i], -q2.GetImaginary()[i], abs_tol=eps) for i in range(3))

    assert (real_match and imag_match) or (
        real_match_neg and imag_match_neg
    ), f"Quaternion mismatch: {q1} != {q2} (and not equal to negative either)"


def get_xform_ops(prim: Usd.Prim) -> list[str]:
    """Get the ordered list of xform operation names for a prim."""
    xformable = UsdGeom.Xformable(prim)
    return [op.GetOpName() for op in xformable.GetOrderedXformOps()]


"""
Tests.
"""


def test_standardize_xform_ops_basic():
    """Test basic functionality of standardize_xform_ops on a simple prim."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a simple xform prim with standard operations
    prim = sim_utils.create_prim(
        "/World/TestXform",
        "Xform",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),  # w, x, y, z
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)

    # Verify the operation succeeded
    assert result is True
    assert prim.IsValid()

    # Check that the xform operations are in the correct order
    xform_ops = get_xform_ops(prim)
    assert xform_ops == [
        "xformOp:translate",
        "xformOp:orient",
        "xformOp:scale",
    ], f"Expected standard xform order, got {xform_ops}"

    # Verify the transform values are preserved (approximately)
    assert_vec3_close(prim.GetAttribute("xformOp:translate").Get(), (1.0, 2.0, 3.0))
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), (1.0, 0.0, 0.0, 0.0))
    assert_vec3_close(prim.GetAttribute("xformOp:scale").Get(), (1.0, 1.0, 1.0))


def test_standardize_xform_ops_with_rotation_xyz():
    """Test standardize_xform_ops removes deprecated rotateXYZ operations."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim and manually add deprecated rotation operations
    prim_path = "/World/TestRotateXYZ"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)
    # Add deprecated rotateXYZ operation
    rotate_xyz_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_xyz_op.Set(Gf.Vec3d(45.0, 30.0, 60.0))
    # Add translate operation
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(1.0, 2.0, 3.0))

    # Verify the deprecated operation exists
    assert "xformOp:rotateXYZ" in prim.GetPropertyNames()

    # Get pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify world pose is preserved (may have small numeric differences due to rotation conversion)
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-4)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-4)

    # Verify the deprecated operation is removed
    assert "xformOp:rotateXYZ" not in prim.GetPropertyNames()
    # Verify standard operations exist
    assert "xformOp:translate" in prim.GetPropertyNames()
    assert "xformOp:orient" in prim.GetPropertyNames()
    assert "xformOp:scale" in prim.GetPropertyNames()
    # Check the xform operation order
    xform_ops = get_xform_ops(prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


def test_standardize_xform_ops_with_transform_matrix():
    """Test standardize_xform_ops removes transform matrix operations."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with a transform matrix
    prim_path = "/World/TestTransformMatrix"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add transform matrix operation
    transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    # Create a simple translation matrix
    matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(5.0, 10.0, 15.0))
    transform_op.Set(matrix)

    # Verify the transform operation exists
    assert "xformOp:transform" in prim.GetPropertyNames()

    # Get pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)

    # Verify the transform operation is removed
    assert "xformOp:transform" not in prim.GetPropertyNames()
    # Verify standard operations exist
    assert "xformOp:translate" in prim.GetPropertyNames()
    assert "xformOp:orient" in prim.GetPropertyNames()
    assert "xformOp:scale" in prim.GetPropertyNames()


def test_standardize_xform_ops_preserves_world_pose():
    """Test that standardize_xform_ops preserves the world-space pose of the prim."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with specific world pose
    translation = (10.0, 20.0, 30.0)
    # Rotation of 90 degrees around Z axis
    orientation = (0.7071068, 0.0, 0.0, 0.7071068)  # w, x, y, z
    scale = (2.0, 3.0, 4.0)

    prim = sim_utils.create_prim(
        "/World/TestPreservePose",
        "Xform",
        translation=translation,
        orientation=orientation,
        scale=scale,
        stage=stage,
    )

    # Get the world pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get the world pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify the world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)


def test_standardize_xform_ops_with_units_resolve():
    """Test standardize_xform_ops handles scale:unitsResolve attribute."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim
    prim_path = "/World/TestUnitsResolve"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add scale operation
    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))

    # Manually add a unitsResolve scale attribute (simulating imported asset with different units)
    units_resolve_attr = prim.CreateAttribute("xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Double3)
    units_resolve_attr.Set(Gf.Vec3d(100.0, 100.0, 100.0))  # e.g., cm to m conversion

    # Verify both attributes exist
    assert "xformOp:scale" in prim.GetPropertyNames()
    assert "xformOp:scale:unitsResolve" in prim.GetPropertyNames()

    # Get pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)

    # Verify unitsResolve is removed
    assert "xformOp:scale:unitsResolve" not in prim.GetPropertyNames()

    # Verify scale is updated (1.0 * 100.0 = 100.0)
    scale = prim.GetAttribute("xformOp:scale").Get()
    assert_vec3_close(scale, (100.0, 100.0, 100.0))


def test_standardize_xform_ops_with_hierarchy():
    """Test standardize_xform_ops works correctly with prim hierarchies."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create parent prim
    parent_prim = sim_utils.create_prim(
        "/World/Parent",
        "Xform",
        translation=(5.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    # Create child prim
    child_prim = sim_utils.create_prim(
        "/World/Parent/Child",
        "Xform",
        translation=(0.0, 3.0, 0.0),
        orientation=(0.7071068, 0.0, 0.7071068, 0.0),  # 90 deg around Y
        scale=(0.5, 0.5, 0.5),
        stage=stage,
    )

    # Get world poses before standardization
    parent_pos_before, parent_quat_before = sim_utils.resolve_prim_pose(parent_prim)
    child_pos_before, child_quat_before = sim_utils.resolve_prim_pose(child_prim)

    # Apply standardize_xform_ops to both
    sim_utils.standardize_xform_ops(parent_prim)
    sim_utils.standardize_xform_ops(child_prim)

    # Get world poses after standardization
    parent_pos_after, parent_quat_after = sim_utils.resolve_prim_pose(parent_prim)
    child_pos_after, child_quat_after = sim_utils.resolve_prim_pose(child_prim)

    # Verify world poses are preserved
    assert_vec3_close(Gf.Vec3d(*parent_pos_before), parent_pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*parent_quat_before), parent_quat_after, eps=1e-5)
    assert_vec3_close(Gf.Vec3d(*child_pos_before), child_pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*child_quat_before), child_quat_after, eps=1e-5)


def test_standardize_xform_ops_multiple_deprecated_ops():
    """Test standardize_xform_ops removes multiple deprecated operations."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with multiple deprecated operations
    prim_path = "/World/TestMultipleDeprecated"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add various deprecated rotation operations
    rotate_x_op = xformable.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_x_op.Set(45.0)
    rotate_y_op = xformable.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_y_op.Set(30.0)
    rotate_z_op = xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_z_op.Set(60.0)

    # Verify deprecated operations exist
    assert "xformOp:rotateX" in prim.GetPropertyNames()
    assert "xformOp:rotateY" in prim.GetPropertyNames()
    assert "xformOp:rotateZ" in prim.GetPropertyNames()

    # Obtain current local transformations
    pos, quat = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    sim_utils.standardize_xform_ops(prim)

    # Obtain current local transformations
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos), Gf.Vec3d(*pos_after), eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat), Gf.Quatd(*quat_after), eps=1e-5)

    # Verify all deprecated operations are removed
    assert "xformOp:rotateX" not in prim.GetPropertyNames()
    assert "xformOp:rotateY" not in prim.GetPropertyNames()
    assert "xformOp:rotateZ" not in prim.GetPropertyNames()
    # Verify standard operations exist
    xform_ops = get_xform_ops(prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


def test_standardize_xform_ops_with_existing_standard_ops():
    """Test standardize_xform_ops when prim already has standard operations."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with standard operations already in place
    prim = sim_utils.create_prim(
        "/World/TestExistingStandard",
        "Xform",
        translation=(7.0, 8.0, 9.0),
        orientation=(0.9238795, 0.3826834, 0.0, 0.0),  # rotation around X
        scale=(1.5, 2.5, 3.5),
        stage=stage,
    )

    # Get initial values
    initial_translate = prim.GetAttribute("xformOp:translate").Get()
    initial_orient = prim.GetAttribute("xformOp:orient").Get()
    initial_scale = prim.GetAttribute("xformOp:scale").Get()

    # Get world pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get world pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)

    # Verify operations still exist and are in correct order
    xform_ops = get_xform_ops(prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    # Verify values are approximately preserved
    final_translate = prim.GetAttribute("xformOp:translate").Get()
    final_orient = prim.GetAttribute("xformOp:orient").Get()
    final_scale = prim.GetAttribute("xformOp:scale").Get()

    assert_vec3_close(initial_translate, final_translate, eps=1e-5)
    assert_quat_close(initial_orient, final_orient, eps=1e-5)
    assert_vec3_close(initial_scale, final_scale, eps=1e-5)


def test_standardize_xform_ops_invalid_prim():
    """Test standardize_xform_ops raises error for invalid prim."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Get an invalid prim (non-existent path)
    invalid_prim = stage.GetPrimAtPath("/World/NonExistent")

    # Verify the prim is invalid
    assert not invalid_prim.IsValid()

    # Attempt to apply standardize_xform_ops and expect ValueError
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.standardize_xform_ops(invalid_prim)


def test_standardize_xform_ops_on_geometry_prim():
    """Test standardize_xform_ops on a geometry prim (Cube, Sphere, etc.)."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a cube with transform
    cube_prim = sim_utils.create_prim(
        "/World/TestCube",
        "Cube",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(2.0, 2.0, 2.0),
        attributes={"size": 1.0},
        stage=stage,
    )

    # Get world pose before
    pos_before, quat_before = sim_utils.resolve_prim_pose(cube_prim)

    # Apply standardize_xform_ops
    sim_utils.standardize_xform_ops(cube_prim)

    # Get world pose after
    pos_after, quat_after = sim_utils.resolve_prim_pose(cube_prim)
    # Verify world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)

    # Verify standard operations exist
    xform_ops = get_xform_ops(cube_prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


def test_standardize_xform_ops_with_non_uniform_scale():
    """Test standardize_xform_ops with non-uniform scale."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with non-uniform scale
    prim = sim_utils.create_prim(
        "/World/TestNonUniformScale",
        "Xform",
        translation=(5.0, 10.0, 15.0),
        orientation=(0.7071068, 0.0, 0.7071068, 0.0),  # 90 deg around Y
        scale=(1.0, 2.0, 3.0),  # Non-uniform scale
        stage=stage,
    )

    # Get initial scale
    initial_scale = prim.GetAttribute("xformOp:scale").Get()

    # Get world pose before standardization
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Get world pose after standardization
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    # Verify world pose is preserved
    assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)
    # Verify scale is preserved
    final_scale = prim.GetAttribute("xformOp:scale").Get()
    assert_vec3_close(initial_scale, final_scale, eps=1e-5)


def test_standardize_xform_ops_identity_transform():
    """Test standardize_xform_ops with identity transform (no translation, rotation, or scale)."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with identity transform
    prim = sim_utils.create_prim(
        "/World/TestIdentity",
        "Xform",
        translation=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # Apply standardize_xform_ops
    sim_utils.standardize_xform_ops(prim)

    # Verify standard operations exist
    xform_ops = get_xform_ops(prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]

    # Verify identity values
    assert_vec3_close(prim.GetAttribute("xformOp:translate").Get(), (0.0, 0.0, 0.0))
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), (1.0, 0.0, 0.0, 0.0))
    assert_vec3_close(prim.GetAttribute("xformOp:scale").Get(), (1.0, 1.0, 1.0))


def test_standardize_xform_ops_with_explicit_values():
    """Test standardize_xform_ops with explicit translation, orientation, and scale values."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with some initial transform
    prim = sim_utils.create_prim(
        "/World/TestExplicitValues",
        "Xform",
        translation=(10.0, 10.0, 10.0),
        orientation=(0.7071068, 0.7071068, 0.0, 0.0),
        scale=(5.0, 5.0, 5.0),
        stage=stage,
    )

    # Apply standardize_xform_ops with new explicit values
    new_translation = (1.0, 2.0, 3.0)
    new_orientation = (1.0, 0.0, 0.0, 0.0)
    new_scale = (2.0, 2.0, 2.0)

    result = sim_utils.standardize_xform_ops(
        prim, translation=new_translation, orientation=new_orientation, scale=new_scale
    )
    assert result is True

    # Verify the new values are set
    assert_vec3_close(prim.GetAttribute("xformOp:translate").Get(), new_translation)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), new_orientation)
    assert_vec3_close(prim.GetAttribute("xformOp:scale").Get(), new_scale)

    # Verify the prim is at the expected world location
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    assert_vec3_close(Gf.Vec3d(*pos_after), new_translation, eps=1e-5)
    assert_quat_close(Gf.Quatd(*quat_after), new_orientation, eps=1e-5)

    # Verify standard operation order
    xform_ops = get_xform_ops(prim)
    assert xform_ops == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


def test_standardize_xform_ops_with_partial_values():
    """Test standardize_xform_ops with only some values specified."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim
    prim = sim_utils.create_prim(
        "/World/TestPartialValues",
        "Xform",
        translation=(1.0, 2.0, 3.0),
        orientation=(0.9238795, 0.3826834, 0.0, 0.0),  # rotation around X
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    # Get initial local pose
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim, ref_prim=prim.GetParent())
    scale_before = prim.GetAttribute("xformOp:scale").Get()

    # Apply standardize_xform_ops with only translation specified
    new_translation = (10.0, 20.0, 30.0)
    result = sim_utils.standardize_xform_ops(prim, translation=new_translation)
    assert result is True

    # Verify translation is updated
    assert_vec3_close(prim.GetAttribute("xformOp:translate").Get(), new_translation)

    # Verify orientation and scale are preserved
    quat_after = prim.GetAttribute("xformOp:orient").Get()
    scale_after = prim.GetAttribute("xformOp:scale").Get()
    assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)
    assert_vec3_close(scale_before, scale_after, eps=1e-5)

    # Verify the prim's world orientation hasn't changed (only translation changed)
    _, quat_after_world = sim_utils.resolve_prim_pose(prim)
    assert_quat_close(Gf.Quatd(*quat_before), quat_after_world, eps=1e-5)


def test_standardize_xform_ops_non_xformable_prim():
    """Test standardize_xform_ops returns False for non-Xformable prims."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a Material prim (not Xformable)
    from pxr import UsdShade

    material_prim = UsdShade.Material.Define(stage, "/World/TestMaterial").GetPrim()

    # Verify the prim is valid but not Xformable
    assert material_prim.IsValid()
    assert not material_prim.IsA(UsdGeom.Xformable)

    # Attempt to apply standardize_xform_ops - should return False
    result = sim_utils.standardize_xform_ops(material_prim)
    assert result is False


def test_standardize_xform_ops_preserves_reset_xform_stack():
    """Test that standardize_xform_ops preserves the resetXformStack attribute."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim
    prim = sim_utils.create_prim("/World/TestResetStack", "Xform", stage=stage)
    xformable = UsdGeom.Xformable(prim)

    # Set resetXformStack to True
    xformable.SetResetXformStack(True)
    assert xformable.GetResetXformStack() is True

    # Apply standardize_xform_ops
    result = sim_utils.standardize_xform_ops(prim)
    assert result is True

    # Verify resetXformStack is preserved
    assert xformable.GetResetXformStack() is True


def test_standardize_xform_ops_with_complex_hierarchy():
    """Test standardize_xform_ops on deeply nested hierarchy."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a complex hierarchy
    root = sim_utils.create_prim("/World/Root", "Xform", translation=(1.0, 0.0, 0.0), stage=stage)
    child1 = sim_utils.create_prim("/World/Root/Child1", "Xform", translation=(0.0, 1.0, 0.0), stage=stage)
    child2 = sim_utils.create_prim("/World/Root/Child1/Child2", "Xform", translation=(0.0, 0.0, 1.0), stage=stage)
    child3 = sim_utils.create_prim("/World/Root/Child1/Child2/Child3", "Cube", translation=(1.0, 1.0, 1.0), stage=stage)

    # Get world poses before
    poses_before = {}
    for name, prim in [("root", root), ("child1", child1), ("child2", child2), ("child3", child3)]:
        poses_before[name] = sim_utils.resolve_prim_pose(prim)

    # Apply standardize_xform_ops to all prims
    assert sim_utils.standardize_xform_ops(root) is True
    assert sim_utils.standardize_xform_ops(child1) is True
    assert sim_utils.standardize_xform_ops(child2) is True
    assert sim_utils.standardize_xform_ops(child3) is True

    # Get world poses after
    poses_after = {}
    for name, prim in [("root", root), ("child1", child1), ("child2", child2), ("child3", child3)]:
        poses_after[name] = sim_utils.resolve_prim_pose(prim)

    # Verify all world poses are preserved
    for name in poses_before:
        pos_before, quat_before = poses_before[name]
        pos_after, quat_after = poses_after[name]
        assert_vec3_close(Gf.Vec3d(*pos_before), pos_after, eps=1e-5)
        assert_quat_close(Gf.Quatd(*quat_before), quat_after, eps=1e-5)


"""
Performance Benchmarking Tests
"""

import time


def test_standardize_xform_ops_performance_batch():
    """Benchmark standardize_xform_ops performance on multiple prims."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create many test prims
    num_prims = 1024
    prims = []

    for i in range(num_prims):
        prim = stage.DefinePrim(f"/World/PerfTestBatch/Prim_{i:03d}", "Xform")
        xformable = UsdGeom.Xformable(prim)
        # Add various deprecated operations
        xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(i * 1.0, i * 2.0, i * 3.0))
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(i, i, i))
        prims.append(prim)

    # Benchmark batch operation
    start_time = time.perf_counter()
    for prim in prims:
        result = sim_utils.standardize_xform_ops(prim)
        assert result is True
    end_time = time.perf_counter()

    # Print timing
    elapsed_ms = (end_time - start_time) * 1000
    avg_ms = elapsed_ms / num_prims
    print(f"\n  Batch standardization ({num_prims} prims): {elapsed_ms:.4f} ms total, {avg_ms:.4f} ms/prim")

    # Verify operation is reasonably fast
    assert avg_ms < 0.1, f"Average operation took {avg_ms:.2f}ms/prim, expected < 0.1ms/prim"
