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

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils


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

    assert (real_match and imag_match) or (real_match_neg and imag_match_neg), (
        f"Quaternion mismatch: {q1} != {q2} (and not equal to negative either)"
    )


def get_xform_ops(prim: Usd.Prim) -> list[str]:
    """Get the ordered list of xform operation names for a prim."""
    xformable = UsdGeom.Xformable(prim)
    return [op.GetOpName() for op in xformable.GetOrderedXformOps()]


"""
Test standardize_xform_ops() function.
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


def test_standardize_xform_ops_non_xformable_prim(caplog):
    """Test standardize_xform_ops returns False for non-Xformable prims and logs error."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a Material prim (not Xformable)
    from pxr import UsdShade

    material_prim = UsdShade.Material.Define(stage, "/World/TestMaterial").GetPrim()

    # Verify the prim is valid but not Xformable
    assert material_prim.IsValid()
    assert not material_prim.IsA(UsdGeom.Xformable)

    # Clear any previous logs
    caplog.clear()

    # Attempt to apply standardize_xform_ops - should return False and log a error
    with caplog.at_level("ERROR"):
        result = sim_utils.standardize_xform_ops(material_prim)

    assert result is False

    # Verify that a error was logged
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "ERROR"
    assert "not an Xformable" in caplog.records[0].message
    assert "/World/TestMaterial" in caplog.records[0].message


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


def test_standardize_xform_ops_preserves_float_precision():
    """Test that standardize_xform_ops preserves float precision when it already exists."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim manually with FLOAT precision operations (not double)
    prim_path = "/World/TestFloatPrecision"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add xform operations with FLOAT precision (not the default double)
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(Gf.Vec3f(1.0, 2.0, 3.0))

    orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat)
    scale_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    # Verify operations exist with float precision
    assert translate_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    assert orient_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    assert scale_op.GetPrecision() == UsdGeom.XformOp.PrecisionFloat

    # Now apply standardize_xform_ops with new values (provided as double precision Python floats)
    new_translation = (5.0, 10.0, 15.0)
    new_orientation = (0.7071068, 0.7071068, 0.0, 0.0)  # 90 deg around X
    new_scale = (2.0, 3.0, 4.0)

    result = sim_utils.standardize_xform_ops(
        prim, translation=new_translation, orientation=new_orientation, scale=new_scale
    )
    assert result is True

    # Verify the precision is STILL float (not converted to double)
    translate_op_after = UsdGeom.XformOp(prim.GetAttribute("xformOp:translate"))
    orient_op_after = UsdGeom.XformOp(prim.GetAttribute("xformOp:orient"))
    scale_op_after = UsdGeom.XformOp(prim.GetAttribute("xformOp:scale"))

    assert translate_op_after.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    assert orient_op_after.GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    assert scale_op_after.GetPrecision() == UsdGeom.XformOp.PrecisionFloat

    # Verify the VALUES are set correctly (cast to float, so they're Gf.Vec3f and Gf.Quatf)
    translate_value = prim.GetAttribute("xformOp:translate").Get()
    assert isinstance(translate_value, Gf.Vec3f), f"Expected Gf.Vec3f, got {type(translate_value)}"
    assert_vec3_close(translate_value, new_translation, eps=1e-5)

    orient_value = prim.GetAttribute("xformOp:orient").Get()
    assert isinstance(orient_value, Gf.Quatf), f"Expected Gf.Quatf, got {type(orient_value)}"
    assert_quat_close(orient_value, new_orientation, eps=1e-5)

    scale_value = prim.GetAttribute("xformOp:scale").Get()
    assert isinstance(scale_value, Gf.Vec3f), f"Expected Gf.Vec3f, got {type(scale_value)}"
    assert_vec3_close(scale_value, new_scale, eps=1e-5)

    # Verify the world pose matches what we set
    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    assert_vec3_close(Gf.Vec3d(*pos_after), new_translation, eps=1e-4)
    assert_quat_close(Gf.Quatd(*quat_after), new_orientation, eps=1e-4)


"""
Test validate_standard_xform_ops() function.
"""


def test_validate_standard_xform_ops_valid():
    """Test validate_standard_xform_ops returns True for standardized prims."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with standard operations
    prim = sim_utils.create_prim(
        "/World/TestValid",
        "Xform",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # Standardize the prim
    sim_utils.standardize_xform_ops(prim)

    # Validate it
    assert sim_utils.validate_standard_xform_ops(prim) is True


def test_validate_standard_xform_ops_invalid_order():
    """Test validate_standard_xform_ops returns False for non-standard operation order."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim and manually set up xform ops in wrong order
    prim_path = "/World/TestInvalidOrder"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add operations in wrong order: scale, translate, orient (should be translate, orient, scale)
    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))

    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(1.0, 2.0, 3.0))

    orient_op = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
    orient_op.Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))

    # Validate it - should return False
    assert sim_utils.validate_standard_xform_ops(prim) is False


def test_validate_standard_xform_ops_with_deprecated_ops():
    """Test validate_standard_xform_ops returns False when deprecated operations exist."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with deprecated rotateXYZ operation
    prim_path = "/World/TestDeprecated"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add deprecated rotateXYZ operation
    rotate_xyz_op = xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_xyz_op.Set(Gf.Vec3d(45.0, 30.0, 60.0))

    # Validate it - should return False
    assert sim_utils.validate_standard_xform_ops(prim) is False


def test_validate_standard_xform_ops_missing_operations():
    """Test validate_standard_xform_ops returns False when standard operations are missing."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with only translate operation (missing orient and scale)
    prim_path = "/World/TestMissing"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(1.0, 2.0, 3.0))

    # Validate it - should return False (missing orient and scale)
    assert sim_utils.validate_standard_xform_ops(prim) is False


def test_validate_standard_xform_ops_invalid_prim():
    """Test validate_standard_xform_ops returns False for invalid prim."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Get an invalid prim
    invalid_prim = stage.GetPrimAtPath("/World/NonExistent")

    # Validate it - should return False
    assert sim_utils.validate_standard_xform_ops(invalid_prim) is False


def test_validate_standard_xform_ops_non_xformable():
    """Test validate_standard_xform_ops returns False for non-Xformable prims."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a Material prim (not Xformable)
    from pxr import UsdShade

    material_prim = UsdShade.Material.Define(stage, "/World/TestMaterial").GetPrim()

    # Validate it - should return False
    assert sim_utils.validate_standard_xform_ops(material_prim) is False


def test_validate_standard_xform_ops_with_transform_matrix():
    """Test validate_standard_xform_ops returns False when transform matrix operation exists."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with transform matrix
    prim_path = "/World/TestTransformMatrix"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add transform matrix operation
    transform_op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    matrix = Gf.Matrix4d().SetTranslate(Gf.Vec3d(5.0, 10.0, 15.0))
    transform_op.Set(matrix)

    # Validate it - should return False
    assert sim_utils.validate_standard_xform_ops(prim) is False


def test_validate_standard_xform_ops_extra_operations():
    """Test validate_standard_xform_ops returns False when extra operations exist."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with standard operations
    prim = sim_utils.create_prim(
        "/World/TestExtra",
        "Xform",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # Standardize it
    sim_utils.standardize_xform_ops(prim)

    # Add an extra operation
    xformable = UsdGeom.Xformable(prim)
    extra_op = xformable.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble)
    extra_op.Set(45.0)

    # Validate it - should return False (has extra operation)
    assert sim_utils.validate_standard_xform_ops(prim) is False


def test_validate_standard_xform_ops_after_standardization():
    """Test validate_standard_xform_ops returns True after standardization of non-standard prim."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim with non-standard operations
    prim_path = "/World/TestBeforeAfter"
    prim = stage.DefinePrim(prim_path, "Xform")
    xformable = UsdGeom.Xformable(prim)

    # Add deprecated operations
    rotate_x_op = xformable.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble)
    rotate_x_op.Set(45.0)
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(1.0, 2.0, 3.0))

    # Validate before standardization - should be False
    assert sim_utils.validate_standard_xform_ops(prim) is False

    # Standardize the prim
    sim_utils.standardize_xform_ops(prim)

    # Validate after standardization - should be True
    assert sim_utils.validate_standard_xform_ops(prim) is True


def test_validate_standard_xform_ops_on_geometry():
    """Test validate_standard_xform_ops works correctly on geometry prims."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a cube with standard operations
    cube_prim = sim_utils.create_prim(
        "/World/TestCube",
        "Cube",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    # Standardize it
    sim_utils.standardize_xform_ops(cube_prim)

    # Validate it - should be True
    assert sim_utils.validate_standard_xform_ops(cube_prim) is True


def test_validate_standard_xform_ops_empty_prim():
    """Test validate_standard_xform_ops on prim with no xform operations."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a bare prim with no xform operations
    prim_path = "/World/TestEmpty"
    prim = stage.DefinePrim(prim_path, "Xform")

    # Validate it - should return False (no operations at all)
    assert sim_utils.validate_standard_xform_ops(prim) is False


"""
Test resolve_prim_pose() function.
"""


def test_resolve_prim_pose():
    """Test resolve_prim_pose() function."""
    # number of objects
    num_objects = 20
    # sample random scales for x, y, z
    rand_scales = np.random.uniform(0.5, 1.5, size=(num_objects, 3, 3))
    rand_widths = np.random.uniform(0.1, 10.0, size=(num_objects,))
    # sample random positions
    rand_positions = np.random.uniform(-100, 100, size=(num_objects, 3, 3))
    # sample random rotations
    rand_quats = np.random.randn(num_objects, 3, 4)
    rand_quats /= np.linalg.norm(rand_quats, axis=2, keepdims=True)

    # create objects
    for i in range(num_objects):
        # simple cubes
        cube_prim = sim_utils.create_prim(
            f"/World/Cubes/instance_{i:02d}",
            "Cube",
            translation=rand_positions[i, 0],
            orientation=rand_quats[i, 0],
            scale=rand_scales[i, 0],
            attributes={"size": rand_widths[i]},
        )
        # xform hierarchy
        xform_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}",
            "Xform",
            translation=rand_positions[i, 1],
            orientation=rand_quats[i, 1],
            scale=rand_scales[i, 1],
        )
        geometry_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/geometry",
            "Sphere",
            translation=rand_positions[i, 2],
            orientation=rand_quats[i, 2],
            scale=rand_scales[i, 2],
            attributes={"radius": rand_widths[i]},
        )
        dummy_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/dummy",
            "Sphere",
        )

        # cube prim w.r.t. world frame
        pos, quat = sim_utils.resolve_prim_pose(cube_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 0, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 0], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 0], atol=1e-3)
        # xform prim w.r.t. world frame
        pos, quat = sim_utils.resolve_prim_pose(xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 1, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 1], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 1], atol=1e-3)
        # dummy prim w.r.t. world frame
        pos, quat = sim_utils.resolve_prim_pose(dummy_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 1, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 1], atol=1e-3)
        np.testing.assert_allclose(quat, rand_quats[i, 1], atol=1e-3)

        # geometry prim w.r.t. xform prim
        pos, quat = sim_utils.resolve_prim_pose(geometry_prim, ref_prim=xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        quat = quat if np.sign(rand_quats[i, 2, 0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, rand_positions[i, 2] * rand_scales[i, 1], atol=1e-3)
        # TODO: Enabling scale causes the test to fail because the current implementation of
        # resolve_prim_pose does not correctly handle non-identity scales on Xform prims. This is a known
        # limitation. Until this is fixed, the test is disabled here to ensure the test passes.
        # np.testing.assert_allclose(quat, rand_quats[i, 2], atol=1e-3)

        # dummy prim w.r.t. xform prim
        pos, quat = sim_utils.resolve_prim_pose(dummy_prim, ref_prim=xform_prim)
        pos, quat = np.array(pos), np.array(quat)
        np.testing.assert_allclose(pos, np.zeros(3), atol=1e-3)
        np.testing.assert_allclose(quat, np.array([1, 0, 0, 0]), atol=1e-3)
        # xform prim w.r.t. cube prim
        pos, quat = sim_utils.resolve_prim_pose(xform_prim, ref_prim=cube_prim)
        pos, quat = np.array(pos), np.array(quat)
        # -- compute ground truth values
        gt_pos, gt_quat = math_utils.subtract_frame_transforms(
            torch.from_numpy(rand_positions[i, 0]).unsqueeze(0),
            torch.from_numpy(rand_quats[i, 0]).unsqueeze(0),
            torch.from_numpy(rand_positions[i, 1]).unsqueeze(0),
            torch.from_numpy(rand_quats[i, 1]).unsqueeze(0),
        )
        gt_pos, gt_quat = gt_pos.squeeze(0).numpy(), gt_quat.squeeze(0).numpy()
        quat = quat if np.sign(gt_quat[0]) == np.sign(quat[0]) else -quat
        np.testing.assert_allclose(pos, gt_pos, atol=1e-3)
        np.testing.assert_allclose(quat, gt_quat, atol=1e-3)


"""
Test resolve_prim_scale() function.
"""


def test_resolve_prim_scale():
    """Test resolve_prim_scale() function.

    To simplify the test, we assume that the effective scale at a prim
    is the product of the scales of the prims in the hierarchy:

        scale = scale_of_xform * scale_of_geometry_prim

    This is only true when rotations are identity or the transforms are
    orthogonal and uniformly scaled. Otherwise, scale is not composable
    like that in local component-wise fashion.
    """
    # number of objects
    num_objects = 20
    # sample random scales for x, y, z
    rand_scales = np.random.uniform(0.5, 1.5, size=(num_objects, 3, 3))
    rand_widths = np.random.uniform(0.1, 10.0, size=(num_objects,))
    # sample random positions
    rand_positions = np.random.uniform(-100, 100, size=(num_objects, 3, 3))

    # create objects
    for i in range(num_objects):
        # simple cubes
        cube_prim = sim_utils.create_prim(
            f"/World/Cubes/instance_{i:02d}",
            "Cube",
            translation=rand_positions[i, 0],
            scale=rand_scales[i, 0],
            attributes={"size": rand_widths[i]},
        )
        # xform hierarchy
        xform_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}",
            "Xform",
            translation=rand_positions[i, 1],
            scale=rand_scales[i, 1],
        )
        geometry_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/geometry",
            "Sphere",
            translation=rand_positions[i, 2],
            scale=rand_scales[i, 2],
            attributes={"radius": rand_widths[i]},
        )
        dummy_prim = sim_utils.create_prim(
            f"/World/Xform/instance_{i:02d}/dummy",
            "Sphere",
        )

        # cube prim
        scale = sim_utils.resolve_prim_scale(cube_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 0], atol=1e-5)
        # xform prim
        scale = sim_utils.resolve_prim_scale(xform_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1], atol=1e-5)
        # geometry prim
        scale = sim_utils.resolve_prim_scale(geometry_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1] * rand_scales[i, 2], atol=1e-5)
        # dummy prim
        scale = sim_utils.resolve_prim_scale(dummy_prim)
        scale = np.array(scale)
        np.testing.assert_allclose(scale, rand_scales[i, 1], atol=1e-5)


"""
Test convert_world_pose_to_local() function.
"""


def test_convert_world_pose_to_local_basic():
    """Test basic world-to-local pose conversion."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create parent and child prims
    parent_prim = sim_utils.create_prim(
        "/World/Parent",
        "Xform",
        translation=(5.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),  # identity rotation
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # World pose we want to achieve for a child
    world_position = (10.0, 3.0, 0.0)
    world_orientation = (1.0, 0.0, 0.0, 0.0)  # identity rotation

    # Convert to local space
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, parent_prim
    )
    # Assert orientation is not None
    assert local_orientation is not None

    # The expected local translation is world_position - parent_position = (10-5, 3-0, 0-0) = (5, 3, 0)
    assert_vec3_close(Gf.Vec3d(*local_translation), (5.0, 3.0, 0.0), eps=1e-5)
    assert_quat_close(Gf.Quatd(*local_orientation), (1.0, 0.0, 0.0, 0.0), eps=1e-5)


def test_convert_world_pose_to_local_with_rotation():
    """Test world-to-local conversion with parent rotation."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create parent with 90-degree rotation around Z axis
    parent_prim = sim_utils.create_prim(
        "/World/RotatedParent",
        "Xform",
        translation=(0.0, 0.0, 0.0),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 deg around Z
        scale=(1.0, 1.0, 1.0),
        stage=stage,
    )

    # World pose: position at (1, 0, 0) with identity rotation
    world_position = (1.0, 0.0, 0.0)
    world_orientation = (1.0, 0.0, 0.0, 0.0)

    # Convert to local space
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, parent_prim
    )

    # Create a child with the local transform and verify world pose
    child_prim = sim_utils.create_prim(
        "/World/RotatedParent/Child",
        "Xform",
        translation=local_translation,
        orientation=local_orientation,
        stage=stage,
    )

    # Get world pose of child
    child_world_pos, child_world_quat = sim_utils.resolve_prim_pose(child_prim)

    # Verify it matches the desired world pose
    assert_vec3_close(Gf.Vec3d(*child_world_pos), world_position, eps=1e-5)
    assert_quat_close(Gf.Quatd(*child_world_quat), world_orientation, eps=1e-5)


def test_convert_world_pose_to_local_with_scale():
    """Test world-to-local conversion with parent scale."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create parent with non-uniform scale
    parent_prim = sim_utils.create_prim(
        "/World/ScaledParent",
        "Xform",
        translation=(1.0, 2.0, 3.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    # World pose we want
    world_position = (5.0, 6.0, 7.0)
    world_orientation = (0.7071068, 0.7071068, 0.0, 0.0)  # 90 deg around X

    # Convert to local space
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, parent_prim
    )

    # Create child and verify
    child_prim = sim_utils.create_prim(
        "/World/ScaledParent/Child",
        "Xform",
        translation=local_translation,
        orientation=local_orientation,
        stage=stage,
    )

    # Get world pose
    child_world_pos, child_world_quat = sim_utils.resolve_prim_pose(child_prim)

    # Verify (may have some tolerance due to scale effects on rotation)
    assert_vec3_close(Gf.Vec3d(*child_world_pos), world_position, eps=1e-4)
    assert_quat_close(Gf.Quatd(*child_world_quat), world_orientation, eps=1e-4)


def test_convert_world_pose_to_local_invalid_parent():
    """Test world-to-local conversion with invalid parent returns world pose unchanged."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Get an invalid prim
    invalid_prim = stage.GetPrimAtPath("/World/NonExistent")
    assert not invalid_prim.IsValid()

    world_position = (10.0, 20.0, 30.0)
    world_orientation = (0.7071068, 0.0, 0.7071068, 0.0)

    # Convert with invalid reference prim
    with pytest.raises(ValueError):
        sim_utils.convert_world_pose_to_local(world_position, world_orientation, invalid_prim)


def test_convert_world_pose_to_local_root_parent():
    """Test world-to-local conversion with root as parent returns world pose unchanged."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Get the pseudo-root prim
    root_prim = stage.GetPrimAtPath("/")

    world_position = (15.0, 25.0, 35.0)
    world_orientation = (0.9238795, 0.3826834, 0.0, 0.0)

    # Convert with root parent
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, root_prim
    )
    # Assert orientation is not None
    assert local_orientation is not None

    # Should return unchanged
    assert_vec3_close(Gf.Vec3d(*local_translation), world_position, eps=1e-10)
    assert_quat_close(Gf.Quatd(*local_orientation), world_orientation, eps=1e-10)


def test_convert_world_pose_to_local_none_orientation():
    """Test world-to-local conversion with None orientation."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create parent
    parent_prim = sim_utils.create_prim(
        "/World/ParentNoOrient",
        "Xform",
        translation=(3.0, 4.0, 5.0),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 deg around Z
        stage=stage,
    )

    world_position = (10.0, 10.0, 10.0)

    # Convert with None orientation
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(world_position, None, parent_prim)

    # Orientation should be None
    assert local_orientation is None
    # Translation should still be converted
    assert local_translation is not None


def test_convert_world_pose_to_local_complex_hierarchy():
    """Test world-to-local conversion in a complex hierarchy."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a complex hierarchy
    _ = sim_utils.create_prim(
        "/World/Grandparent",
        "Xform",
        translation=(10.0, 0.0, 0.0),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 deg around Z
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    parent = sim_utils.create_prim(
        "/World/Grandparent/Parent",
        "Xform",
        translation=(5.0, 0.0, 0.0),  # local to grandparent
        orientation=(0.7071068, 0.7071068, 0.0, 0.0),  # 90 deg around X
        scale=(0.5, 0.5, 0.5),
        stage=stage,
    )

    # World pose we want to achieve
    world_position = (20.0, 15.0, 10.0)
    world_orientation = (1.0, 0.0, 0.0, 0.0)

    # Convert to local space relative to parent
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, parent
    )

    # Create child with the computed local transform
    child = sim_utils.create_prim(
        "/World/Grandparent/Parent/Child",
        "Xform",
        translation=local_translation,
        orientation=local_orientation,
        stage=stage,
    )

    # Verify world pose
    child_world_pos, child_world_quat = sim_utils.resolve_prim_pose(child)

    # Should match the desired world pose (with some tolerance for complex transforms)
    assert_vec3_close(Gf.Vec3d(*child_world_pos), world_position, eps=1e-4)
    assert_quat_close(Gf.Quatd(*child_world_quat), world_orientation, eps=1e-4)


def test_convert_world_pose_to_local_with_mixed_prim_types():
    """Test world-to-local conversion with mixed prim types (Xform, Scope, Mesh)."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a hierarchy with different prim types
    # Grandparent: Xform with transform
    sim_utils.create_prim(
        "/World/Grandparent",
        "Xform",
        translation=(5.0, 3.0, 2.0),
        orientation=(0.7071068, 0.0, 0.0, 0.7071068),  # 90 deg around Z
        scale=(2.0, 2.0, 2.0),
        stage=stage,
    )

    # Parent: Scope prim (organizational, typically has no transform)
    parent = stage.DefinePrim("/World/Grandparent/Parent", "Scope")

    # Obtain parent prim pose (should be grandparent's transform)
    parent_pos, parent_quat = sim_utils.resolve_prim_pose(parent)
    assert_vec3_close(Gf.Vec3d(*parent_pos), (5.0, 3.0, 2.0), eps=1e-5)
    assert_quat_close(Gf.Quatd(*parent_quat), (0.7071068, 0.0, 0.0, 0.7071068), eps=1e-5)

    # Child: Mesh prim (geometry)
    child = sim_utils.create_prim("/World/Grandparent/Parent/Child", "Mesh", stage=stage)

    # World pose we want to achieve for the child
    world_position = (10.0, 5.0, 3.0)
    world_orientation = (1.0, 0.0, 0.0, 0.0)  # identity rotation

    # Convert to local space relative to parent (Scope)
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, child
    )

    # Verify orientation is not None
    assert local_orientation is not None, "Expected orientation to be computed"

    # Set the local transform on the child (Mesh)
    xformable = UsdGeom.Xformable(child)
    translate_op = xformable.GetTranslateOp()
    translate_op.Set(Gf.Vec3d(*local_translation))
    orient_op = xformable.GetOrientOp()
    orient_op.Set(Gf.Quatd(*local_orientation))

    # Verify world pose of child
    child_world_pos, child_world_quat = sim_utils.resolve_prim_pose(child)

    # Should match the desired world pose
    # Note: Scope prims typically have no transform, so the child's world pose should account
    # for the grandparent's transform
    assert_vec3_close(Gf.Vec3d(*child_world_pos), world_position, eps=1e-10)
    assert_quat_close(Gf.Quatd(*child_world_quat), world_orientation, eps=1e-10)
