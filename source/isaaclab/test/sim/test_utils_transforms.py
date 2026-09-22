# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, UsdGeom, UsdShade

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

pytestmark = pytest.mark.unit

STANDARD_OPS = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
ROT_X_90 = (0.7071068, 0.0, 0.0, 0.7071068)
ROT_Y_90 = (0.0, 0.7071068, 0.0, 0.7071068)
ROT_Z_90 = (0.0, 0.0, 0.7071068, 0.7071068)
IDENTITY = (0.0, 0.0, 0.0, 1.0)


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def _quat_xyzw(quat) -> np.ndarray:
    if isinstance(quat, (Gf.Quatd, Gf.Quatf)):
        return np.array([*quat.GetImaginary(), quat.GetReal()])
    return np.asarray(quat, dtype=float)


def assert_quat_close(q1, q2, eps: float = 1e-6):
    """Quaternions are compared up to sign, since q and -q are the same rotation."""
    q1, q2 = _quat_xyzw(q1), _quat_xyzw(q2)
    assert np.allclose(q1, q2, atol=eps) or np.allclose(q1, -q2, atol=eps), f"{q1} != {q2}"


def xform_ops(prim) -> list[str]:
    return [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]


def add_rotate_xyz(xformable):
    xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(45.0, 30.0, 60.0))
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 2.0, 3.0))


def add_transform_matrix(xformable):
    xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(5.0, 10.0, 15.0)))


def add_rotate_xyz_separately(xformable):
    xformable.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(45.0)
    xformable.AddRotateYOp(UsdGeom.XformOp.PrecisionDouble).Set(30.0)
    xformable.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble).Set(60.0)


def add_scaled_units(xformable):
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.0, 1.0, 1.0))
    prim = xformable.GetPrim()
    prim.CreateAttribute("xformOp:scale:unitsResolve", Sdf.ValueTypeNames.Double3).Set(Gf.Vec3d(100.0, 100.0, 100.0))


def add_standard_ops(xformable):
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(7.0, 8.0, 9.0))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(0.9238795, 0.3826834, 0.0, 0.0))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(1.5, 2.5, 3.5))


"""
standardize_xform_ops()
"""


@pytest.mark.parametrize(
    ("prim_type", "author"),
    [
        ("Xform", add_rotate_xyz),
        ("Xform", add_transform_matrix),
        ("Xform", add_rotate_xyz_separately),
        ("Xform", add_scaled_units),
        ("Xform", add_standard_ops),
        ("Cube", add_standard_ops),
    ],
    ids=["rotate_xyz", "transform_matrix", "rotate_x_y_z", "units_resolve", "standard", "geometry"],
)
def test_standardize_xform_ops_preserves_pose(stage, prim_type, author):
    prim = stage.DefinePrim("/World/Prim", prim_type)
    author(UsdGeom.Xformable(prim))
    pos_before, quat_before = sim_utils.resolve_prim_pose(prim)
    scale_before = sim_utils.resolve_prim_scale(prim)

    assert sim_utils.standardize_xform_ops(prim) is True

    pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
    assert np.allclose(pos_before, pos_after, atol=1e-4)
    assert_quat_close(quat_before, quat_after, eps=1e-4)
    # unit resolution intentionally bakes the conversion into the scale
    expected_scale = (100.0,) * 3 if author is add_scaled_units else scale_before
    assert np.allclose(expected_scale, sim_utils.resolve_prim_scale(prim), atol=1e-5)
    assert xform_ops(prim) == STANDARD_OPS
    assert not set(prim.GetPropertyNames()) & {"xformOp:rotateXYZ", "xformOp:transform", "xformOp:rotateX"}
    assert "xformOp:scale:unitsResolve" not in prim.GetPropertyNames()
    assert sim_utils.validate_standard_xform_ops(prim) is True


def test_standardize_xform_ops_bakes_units_resolve(stage):
    prim = stage.DefinePrim("/World/Prim", "Xform")
    add_scaled_units(UsdGeom.Xformable(prim))
    sim_utils.standardize_xform_ops(prim)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(100.0, 100.0, 100.0)


def test_standardize_xform_ops_explicit_and_partial_values(stage):
    prim = sim_utils.create_prim(
        "/World/Prim", "Xform", translation=(10.0,) * 3, orientation=ROT_X_90, scale=(5.0,) * 3
    )

    assert sim_utils.standardize_xform_ops(prim, translation=(1.0, 2.0, 3.0), orientation=IDENTITY, scale=(2.0,) * 3)
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(1.0, 2.0, 3.0)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), IDENTITY)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(2.0, 2.0, 2.0)

    # only the provided value changes
    assert sim_utils.standardize_xform_ops(prim, orientation=ROT_Y_90)
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(1.0, 2.0, 3.0)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), ROT_Y_90)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(2.0, 2.0, 2.0)


def test_standardize_xform_ops_preserves_float_precision(stage):
    prim = stage.DefinePrim("/World/Prim", "Xform")
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1.0, 2.0, 3.0))
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionFloat).Set(Gf.Vec3f(1.0, 1.0, 1.0))

    assert sim_utils.standardize_xform_ops(
        prim, translation=(5.0, 10.0, 15.0), orientation=ROT_X_90, scale=(2.0, 3.0, 4.0)
    )

    for name in STANDARD_OPS:
        assert UsdGeom.XformOp(prim.GetAttribute(name)).GetPrecision() == UsdGeom.XformOp.PrecisionFloat
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3f(5.0, 10.0, 15.0)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), ROT_X_90)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3f(2.0, 3.0, 4.0)


def test_standardize_xform_ops_hierarchy_and_reset_stack(stage):
    root = sim_utils.create_prim("/World/Root", "Xform", translation=(1.0, 0.0, 0.0), scale=(2.0, 2.0, 2.0))
    child = sim_utils.create_prim("/World/Root/Child", "Xform", translation=(0.0, 1.0, 0.0), orientation=ROT_Y_90)
    leaf = sim_utils.create_prim("/World/Root/Child/Leaf", "Cube", translation=(1.0, 1.0, 1.0))
    UsdGeom.Xformable(child).SetResetXformStack(True)
    poses = [sim_utils.resolve_prim_pose(prim) for prim in (root, child, leaf)]

    for prim in (root, child, leaf):
        assert sim_utils.standardize_xform_ops(prim) is True

    for prim, (pos, quat) in zip((root, child, leaf), poses):
        pos_after, quat_after = sim_utils.resolve_prim_pose(prim)
        assert np.allclose(pos, pos_after, atol=1e-5)
        assert_quat_close(quat, quat_after, eps=1e-5)
    assert UsdGeom.Xformable(child).GetResetXformStack() is True


def test_standardize_xform_ops_rejects_invalid_and_non_xformable(stage, caplog):
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.standardize_xform_ops(stage.GetPrimAtPath("/World/Missing"))

    material = UsdShade.Material.Define(stage, "/World/Material").GetPrim()
    with caplog.at_level("ERROR"):
        assert sim_utils.standardize_xform_ops(material) is False
    assert len(caplog.records) == 1
    assert "not an Xformable" in caplog.records[0].message
    assert "/World/Material" in caplog.records[0].message


"""
validate_standard_xform_ops()
"""


def _author_wrong_order(xformable):
    xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)


def _author_extra_op(xformable):
    add_standard_ops(xformable)
    xformable.AddRotateXOp(UsdGeom.XformOp.PrecisionDouble).Set(45.0)


@pytest.mark.parametrize(
    "author",
    [
        _author_wrong_order,
        add_rotate_xyz,
        add_transform_matrix,
        _author_extra_op,
        lambda xformable: xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble),
        lambda xformable: None,
    ],
    ids=["wrong_order", "deprecated_op", "transform_matrix", "extra_op", "missing_ops", "no_ops"],
)
def test_validate_standard_xform_ops_rejects_non_standard(stage, author):
    prim = stage.DefinePrim("/World/Prim", "Xform")
    author(UsdGeom.Xformable(prim))
    assert sim_utils.validate_standard_xform_ops(prim) is False
    # standardization is exactly what validation checks for
    sim_utils.standardize_xform_ops(prim)
    assert sim_utils.validate_standard_xform_ops(prim) is True


def test_validate_standard_xform_ops_invalid_and_non_xformable(stage):
    assert sim_utils.validate_standard_xform_ops(stage.GetPrimAtPath("/World/Missing")) is False
    material = UsdShade.Material.Define(stage, "/World/Material").GetPrim()
    assert sim_utils.validate_standard_xform_ops(material) is False


"""
resolve_prim_pose() and resolve_prim_scale()
"""


def test_resolve_prim_pose_and_scale(stage):
    rng = np.random.default_rng(0)
    positions = rng.uniform(-100, 100, size=(3, 3))
    quats = rng.standard_normal(size=(3, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    scales = rng.uniform(0.5, 1.5, size=(3, 3))

    cube = sim_utils.create_prim("/World/Cube", "Cube", translation=positions[0], orientation=quats[0], scale=scales[0])
    xform = sim_utils.create_prim(
        "/World/Xform", "Xform", translation=positions[1], orientation=quats[1], scale=scales[1]
    )
    # a child without any transform of its own inherits the parent pose
    child = sim_utils.create_prim("/World/Xform/child", "Sphere")
    # a translated child inside a scaled parent (identity rotation keeps the scale composable)
    scaled_parent = sim_utils.create_prim("/World/Scaled", "Xform", translation=positions[2], scale=scales[1])
    geometry = sim_utils.create_prim("/World/Scaled/geometry", "Sphere", translation=positions[2], scale=scales[2])

    for prim, expected_pos, expected_quat in [(cube, positions[0], quats[0]), (xform, positions[1], quats[1])]:
        pos, quat = sim_utils.resolve_prim_pose(prim)
        assert np.allclose(pos, expected_pos, atol=1e-3)
        assert_quat_close(quat, expected_quat, eps=1e-3)
    pos, quat = sim_utils.resolve_prim_pose(child)
    assert np.allclose(pos, positions[1], atol=1e-3)
    assert_quat_close(quat, quats[1], eps=1e-3)

    # relative to a reference prim
    pos, quat = sim_utils.resolve_prim_pose(child, ref_prim=xform)
    assert np.allclose(pos, 0.0, atol=1e-3)
    assert_quat_close(quat, IDENTITY, eps=1e-3)
    pos, quat = sim_utils.resolve_prim_pose(geometry, ref_prim=scaled_parent)
    assert np.allclose(pos, positions[2] * scales[1], atol=1e-3)
    pos, quat = sim_utils.resolve_prim_pose(xform, ref_prim=cube)
    gt_pos, gt_quat = math_utils.subtract_frame_transforms(
        *(torch.from_numpy(v).unsqueeze(0) for v in (positions[0], quats[0], positions[1], quats[1]))
    )
    assert np.allclose(pos, gt_pos.squeeze(0).numpy(), atol=1e-3)
    assert_quat_close(quat, gt_quat.squeeze(0).numpy(), eps=1e-3)
    # the root reference frame is the world frame
    assert sim_utils.resolve_prim_pose(cube, ref_prim=stage.GetPseudoRoot()) == sim_utils.resolve_prim_pose(cube)

    # world scale composes down the hierarchy
    assert np.allclose(sim_utils.resolve_prim_scale(cube), scales[0], atol=1e-5)
    assert np.allclose(sim_utils.resolve_prim_scale(child), scales[1], atol=1e-5)
    assert np.allclose(sim_utils.resolve_prim_scale(geometry), scales[1] * scales[2], atol=1e-5)

    with pytest.raises(ValueError, match="not valid"):
        sim_utils.resolve_prim_pose(stage.GetPrimAtPath("/World/Missing"))
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.resolve_prim_scale(stage.GetPrimAtPath("/World/Missing"))


"""
convert_world_pose_to_local()
"""


@pytest.mark.parametrize(
    ("parents", "world_orientation"),
    [
        ([("/World/Parent", (5.0, 0.0, 0.0), IDENTITY, None)], IDENTITY),
        ([("/World/Parent", (0.0, 0.0, 0.0), ROT_Z_90, None)], IDENTITY),
        ([("/World/Parent", (1.0, 2.0, 3.0), IDENTITY, (2.0, 2.0, 2.0))], ROT_X_90),
        (
            [
                ("/World/Parent", (10.0, 0.0, 0.0), ROT_Z_90, (2.0, 2.0, 2.0)),
                ("/World/Parent/Parent", (5.0, 0.0, 0.0), ROT_X_90, (0.5, 0.5, 0.5)),
            ],
            IDENTITY,
        ),
    ],
    ids=["translated", "rotated", "scaled", "nested"],
)
def test_convert_world_pose_to_local_round_trip(stage, parents, world_orientation):
    for path, translation, orientation, scale in parents:
        parent = sim_utils.create_prim(path, "Xform", translation=translation, orientation=orientation, scale=scale)
    world_position = (20.0, 15.0, 10.0)

    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, parent
    )
    child = sim_utils.create_prim(
        f"{parent.GetPath()}/Child", "Xform", translation=local_translation, orientation=local_orientation
    )

    pos, quat = sim_utils.resolve_prim_pose(child)
    assert np.allclose(pos, world_position, atol=1e-4)
    assert_quat_close(quat, world_orientation, eps=1e-4)


def test_convert_world_pose_to_local_through_scope_parent(stage):
    """A Scope parent carries no transform, so the local pose accounts for the grandparent alone."""
    sim_utils.create_prim(
        "/World/Grandparent", "Xform", translation=(5.0, 3.0, 2.0), orientation=ROT_Z_90, scale=(2.0,) * 3
    )
    stage.DefinePrim("/World/Grandparent/Parent", "Scope")
    child = sim_utils.create_prim("/World/Grandparent/Parent/Child", "Mesh")

    world_position, world_orientation = (10.0, 5.0, 3.0), IDENTITY
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local(
        world_position, world_orientation, child
    )
    assert sim_utils.standardize_xform_ops(child, translation=local_translation, orientation=local_orientation)

    pos, quat = sim_utils.resolve_prim_pose(child)
    assert np.allclose(pos, world_position, atol=1e-10)
    assert_quat_close(quat, world_orientation, eps=1e-10)


def test_convert_world_pose_to_local_edge_cases(stage):
    parent = sim_utils.create_prim("/World/Parent", "Xform", translation=(3.0, 4.0, 5.0), orientation=ROT_Z_90)

    # the root parent returns the world pose unchanged
    world_position, world_orientation = (15.0, 25.0, 35.0), ROT_X_90
    assert sim_utils.convert_world_pose_to_local(world_position, world_orientation, stage.GetPseudoRoot()) == (
        world_position,
        world_orientation,
    )
    # an absent orientation stays absent while the translation is still converted
    local_translation, local_orientation = sim_utils.convert_world_pose_to_local((10.0, 10.0, 10.0), None, parent)
    assert local_orientation is None
    assert np.allclose(local_translation, (6.0, -7.0, 5.0), atol=1e-6)

    with pytest.raises(ValueError, match="not valid"):
        sim_utils.convert_world_pose_to_local(world_position, world_orientation, stage.GetPrimAtPath("/World/Missing"))
