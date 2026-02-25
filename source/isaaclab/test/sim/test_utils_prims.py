# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# note: need to enable cameras to be able to make replicator core available
simulation_app = AppLauncher(headless=True, enable_cameras=True).app

"""Rest everything follows."""

import math

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.prims import _to_tuple  # type: ignore[reportPrivateUsage]
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


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


def assert_quat_close(
    q1: Gf.Quatf | Gf.Quatd | tuple | list, q2: Gf.Quatf | Gf.Quatd | tuple | list, eps: float = 1e-6
):
    """Assert two quaternions are close."""
    if isinstance(q1, (tuple, list)):
        q1 = Gf.Quatd(q1[3], q1[0], q1[1], q1[2])
    if isinstance(q2, (tuple, list)):
        q2 = Gf.Quatd(q2[3], q2[0], q2[1], q2[2])
    assert math.isclose(q1.GetReal(), q2.GetReal(), abs_tol=eps)
    for i in range(3):
        assert math.isclose(q1.GetImaginary()[i], q2.GetImaginary()[i], abs_tol=eps)


"""
General Utils
"""


def test_create_prim():
    """Test create_prim() function."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create scene
    prim = sim_utils.create_prim(prim_path="/World/Test", prim_type="Xform", stage=stage)
    # check prim created
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test"
    assert prim.GetTypeName() == "Xform"

    # check recreation of prim
    with pytest.raises(ValueError, match="already exists"):
        sim_utils.create_prim(prim_path="/World/Test", prim_type="Xform", stage=stage)

    # check attribute setting
    prim = sim_utils.create_prim(prim_path="/World/Test/Cube", prim_type="Cube", stage=stage, attributes={"size": 100})
    # check attribute set
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test/Cube"
    assert prim.GetTypeName() == "Cube"
    assert prim.GetAttribute("size").Get() == 100

    # check adding USD reference
    franka_usd = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    prim = sim_utils.create_prim("/World/Test/USDReference", usd_path=franka_usd, stage=stage)
    # check USD reference set
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test/USDReference"
    assert prim.GetTypeName() == "Xform"
    # get the reference of the prim
    references = []
    for prim_spec in prim.GetPrimStack():
        references.extend(prim_spec.referenceList.prependedItems)
    assert len(references) == 1
    assert str(references[0].assetPath) == franka_usd

    # check adding semantic label
    prim = sim_utils.create_prim(
        "/World/Test/Sphere", "Sphere", stage=stage, semantic_label="sphere", attributes={"radius": 10.0}
    )
    # check semantic label set
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test/Sphere"
    assert prim.GetTypeName() == "Sphere"
    assert prim.GetAttribute("radius").Get() == 10.0
    assert sim_utils.get_labels(prim)["class"] == ["sphere"]

    # check setting transform
    pos = (1.0, 2.0, 3.0)
    quat = (0.0, 0.0, 1.0, 0.0)
    scale = (1.0, 0.5, 0.5)
    prim = sim_utils.create_prim(
        "/World/Test/Xform", "Xform", stage=stage, translation=pos, orientation=quat, scale=scale
    )
    # check transform set
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test/Xform"
    assert prim.GetTypeName() == "Xform"
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(pos)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), quat)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(scale)
    # check xform operation order
    op_names = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    assert op_names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


@pytest.mark.parametrize(
    "input_type",
    ["list", "tuple", "numpy", "torch_cpu", "torch_cuda"],
    ids=["list", "tuple", "numpy", "torch_cpu", "torch_cuda"],
)
def test_create_prim_with_different_input_types(input_type: str):
    """Test create_prim() with different input types (list, tuple, numpy array, torch tensor)."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Define test values
    translation_vals = [1.0, 2.0, 3.0]
    orientation_vals = [0.0, 0.0, 0.0, 1.0]  # x, y, z, w
    scale_vals = [2.0, 3.0, 4.0]

    # Convert to the specified input type
    if input_type == "list":
        translation = translation_vals
        orientation = orientation_vals
        scale = scale_vals
    elif input_type == "tuple":
        translation = tuple(translation_vals)
        orientation = tuple(orientation_vals)
        scale = tuple(scale_vals)
    elif input_type == "numpy":
        translation = np.array(translation_vals)
        orientation = np.array(orientation_vals)
        scale = np.array(scale_vals)
    elif input_type == "torch_cpu":
        translation = torch.tensor(translation_vals)
        orientation = torch.tensor(orientation_vals)
        scale = torch.tensor(scale_vals)
    elif input_type == "torch_cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        translation = torch.tensor(translation_vals, device="cuda")
        orientation = torch.tensor(orientation_vals, device="cuda")
        scale = torch.tensor(scale_vals, device="cuda")

    # Create prim with translation (local space)
    prim = sim_utils.create_prim(
        f"/World/Test/Xform_{input_type}",
        "Xform",
        stage=stage,
        translation=translation,
        orientation=orientation,
        scale=scale,
    )

    # Verify prim was created correctly
    assert prim.IsValid()
    assert prim.GetPrimPath() == f"/World/Test/Xform_{input_type}"

    # Verify transform values
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(*translation_vals)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), orientation_vals)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(*scale_vals)

    # Verify xform operation order
    op_names = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    assert op_names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


@pytest.mark.parametrize(
    "input_type",
    ["list", "tuple", "numpy", "torch_cpu", "torch_cuda"],
    ids=["list", "tuple", "numpy", "torch_cpu", "torch_cuda"],
)
def test_create_prim_with_world_position_different_types(input_type: str):
    """Test create_prim() with world position using different input types."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a parent prim
    _ = sim_utils.create_prim(
        "/World/Parent",
        "Xform",
        stage=stage,
        translation=(5.0, 10.0, 15.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
    )

    # Define world position and orientation values
    world_pos_vals = [10.0, 20.0, 30.0]
    world_orient_vals = [0.0, 0.7071068, 0.0, 0.7071068]  # 90 deg around Y

    # Convert to the specified input type
    if input_type == "list":
        world_pos = world_pos_vals
        world_orient = world_orient_vals
    elif input_type == "tuple":
        world_pos = tuple(world_pos_vals)
        world_orient = tuple(world_orient_vals)
    elif input_type == "numpy":
        world_pos = np.array(world_pos_vals)
        world_orient = np.array(world_orient_vals)
    elif input_type == "torch_cpu":
        world_pos = torch.tensor(world_pos_vals)
        world_orient = torch.tensor(world_orient_vals)
    elif input_type == "torch_cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        world_pos = torch.tensor(world_pos_vals, device="cuda")
        world_orient = torch.tensor(world_orient_vals, device="cuda")

    # Create child prim with world position
    child = sim_utils.create_prim(
        f"/World/Parent/Child_{input_type}",
        "Xform",
        stage=stage,
        position=world_pos,  # Using position (world space)
        orientation=world_orient,
    )

    # Verify prim was created
    assert child.IsValid()

    # Verify world pose matches what we specified
    world_pose = sim_utils.resolve_prim_pose(child)
    pos_result, quat_result = world_pose

    # Check position (should be close to world_pos_vals)
    for i in range(3):
        assert math.isclose(pos_result[i], world_pos_vals[i], abs_tol=1e-4)

    # Check orientation (quaternions may have sign flipped)
    quat_match = all(math.isclose(quat_result[i], world_orient_vals[i], abs_tol=1e-4) for i in range(4))
    quat_match_neg = all(math.isclose(quat_result[i], -world_orient_vals[i], abs_tol=1e-4) for i in range(4))
    assert quat_match or quat_match_neg


def test_create_prim_non_xformable():
    """Test create_prim() with non-Xformable prim types (Material, Shader, Scope).

    This test verifies that prims which are not Xformable (like Material, Shader, Scope)
    are created successfully but transform operations are not applied to them.
    This is expected behavior as documented in the create_prim function.
    """
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Test with Material prim (not Xformable)
    material_prim = sim_utils.create_prim(
        "/World/TestMaterial",
        "Material",
        stage=stage,
        translation=(1.0, 2.0, 3.0),  # These should be ignored
        orientation=(0.0, 0.0, 0.0, 1.0),  # These should be ignored
        scale=(2.0, 2.0, 2.0),  # These should be ignored
    )

    # Verify prim was created
    assert material_prim.IsValid()
    assert material_prim.GetPrimPath() == "/World/TestMaterial"
    assert material_prim.GetTypeName() == "Material"

    # Verify that it's not Xformable
    assert not material_prim.IsA(UsdGeom.Xformable)

    # Verify that no xform operations were applied (Material prims don't support these)
    assert not material_prim.HasAttribute("xformOp:translate")
    assert not material_prim.HasAttribute("xformOp:orient")
    assert not material_prim.HasAttribute("xformOp:scale")

    # Test with Scope prim (not Xformable)
    scope_prim = sim_utils.create_prim(
        "/World/TestScope",
        "Scope",
        stage=stage,
        translation=(5.0, 6.0, 7.0),  # These should be ignored
    )

    # Verify prim was created
    assert scope_prim.IsValid()
    assert scope_prim.GetPrimPath() == "/World/TestScope"
    assert scope_prim.GetTypeName() == "Scope"

    # Verify that it's not Xformable
    assert not scope_prim.IsA(UsdGeom.Xformable)

    # Verify that no xform operations were applied (Scope prims don't support these)
    assert not scope_prim.HasAttribute("xformOp:translate")
    assert not scope_prim.HasAttribute("xformOp:orient")
    assert not scope_prim.HasAttribute("xformOp:scale")


def test_delete_prim():
    """Test delete_prim() function."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create scene
    prim = sim_utils.create_prim("/World/Test/Xform", "Xform", stage=stage)
    # delete prim
    sim_utils.delete_prim("/World/Test/Xform")
    # check prim deleted
    assert not prim.IsValid()

    # check for usd reference
    prim = sim_utils.create_prim(
        "/World/Test/USDReference",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        stage=stage,
    )
    # delete prim
    sim_utils.delete_prim("/World/Test/USDReference", stage=stage)
    # check prim deleted
    assert not prim.IsValid()

    # check deleting multiple prims
    prim1 = sim_utils.create_prim("/World/Test/Xform1", "Xform", stage=stage)
    prim2 = sim_utils.create_prim("/World/Test/Xform2", "Xform", stage=stage)
    sim_utils.delete_prim(("/World/Test/Xform1", "/World/Test/Xform2"), stage=stage)
    # check prims deleted
    assert not prim1.IsValid()
    assert not prim2.IsValid()


"""
USD references and variants.
"""


def test_get_usd_references():
    """Test get_usd_references() function."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # Create a prim without USD reference
    sim_utils.create_prim("/World/NoReference", "Xform", stage=stage)
    # Check that it has no references
    refs = sim_utils.get_usd_references("/World/NoReference", stage=stage)
    assert len(refs) == 0

    # Create a prim with a USD reference
    franka_usd = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    sim_utils.create_prim("/World/WithReference", usd_path=franka_usd, stage=stage)
    # Check that it has the expected reference
    refs = sim_utils.get_usd_references("/World/WithReference", stage=stage)
    assert len(refs) == 1
    assert refs == [franka_usd]

    # Test with invalid prim path
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.get_usd_references("/World/NonExistent", stage=stage)


def test_select_usd_variants():
    """Test select_usd_variants() function."""
    stage = sim_utils.get_current_stage()

    # Create a dummy prim
    prim: Usd.Prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    stage.SetDefaultPrim(prim)

    # Create the variant set and add your variants to it.
    variants = ["red", "blue", "green"]
    variant_set = prim.GetVariantSets().AddVariantSet("colors")
    for variant in variants:
        variant_set.AddVariant(variant)

    # Set the variant selection
    sim_utils.utils.select_usd_variants("/World", {"colors": "red"}, stage)

    # Check if the variant selection is correct
    assert variant_set.GetVariantSelection() == "red"


def test_select_usd_variants_in_usd_file():
    """Test select_usd_variants() function in USD file."""
    stage = sim_utils.get_current_stage()

    prim = sim_utils.create_prim(
        "/World/Test", "Xform", usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/UniversalRobots/ur10e/ur10e.usd", stage=stage
    )

    variant_sets = prim.GetVariantSets()

    # show all variants
    for name in variant_sets.GetNames():
        vs = variant_sets.GetVariantSet(name)
        options = vs.GetVariantNames()
        selected = vs.GetVariantSelection()

        print(f"{name}: {selected} / {options}")

    print("Setting variant 'Gripper' to 'Robotiq_2f_140'.")
    # The following performs the operations done internally
    # in Isaac Lab. This should be removed in favor of 'select_usd_variants'.
    target_vs = variant_sets.GetVariantSet("Gripper")
    target_vs.SetVariantSelection("Robotiq_2f_140")

    # show again all variants
    variant_sets = prim.GetVariantSets()

    for name in variant_sets.GetNames():
        vs = variant_sets.GetVariantSet(name)
        options = vs.GetVariantNames()
        selected = vs.GetVariantSelection()

        print(f"{name}: {selected} / {options}")

    # Uncomment the following once resolved

    # Set the variant selection
    # sim_utils.select_usd_variants(prim.GetPath(), {"Gripper": "Robotiq_2f_140"}, stage)

    # Obtain variant set
    # variant_set = prim.GetVariantSet("Gripper")
    # # Check if the variant selection is correct
    # assert variant_set.GetVariantSelection() == "Robotiq_2f_140"


"""
Property Management.
"""


def test_change_prim_property_basic():
    """Test change_prim_property() with existing property."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a cube prim
    prim = sim_utils.create_prim("/World/Cube", "Cube", stage=stage, attributes={"size": 1.0})

    # check initial value
    assert prim.GetAttribute("size").Get() == 1.0

    # change the property
    result = sim_utils.change_prim_property(
        prop_path="/World/Cube.size",
        value=2.0,
        stage=stage,
    )

    # check that the change was successful
    assert result is True
    assert prim.GetAttribute("size").Get() == 2.0


def test_change_prim_property_create_new():
    """Test change_prim_property() creates new property when it doesn't exist."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a prim
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)

    # check that the property doesn't exist
    assert prim.GetAttribute("customValue").Get() is None

    # create a new property
    result = sim_utils.change_prim_property(
        prop_path="/World/Test.customValue",
        value=42,
        stage=stage,
        type_to_create_if_not_exist=Sdf.ValueTypeNames.Int,
        is_custom=True,
    )

    # check that the property was created successfully
    assert result is True
    assert prim.GetAttribute("customValue").Get() == 42


def test_change_prim_property_clear_value():
    """Test change_prim_property() clears property value when value is None."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a cube with an attribute
    prim = sim_utils.create_prim("/World/Cube", "Cube", stage=stage, attributes={"size": 1.0})

    # check initial value
    assert prim.GetAttribute("size").Get() == 1.0

    # clear the property value
    result = sim_utils.change_prim_property(
        prop_path="/World/Cube.size",
        value=None,
        stage=stage,
    )

    # check that the value was cleared
    assert result is True
    # Note: After clearing, the attribute should go its default value
    assert prim.GetAttribute("size").Get() == 2.0


@pytest.mark.parametrize(
    "attr_name,value,value_type,expected",
    [
        ("floatValue", 3.14, Sdf.ValueTypeNames.Float, 3.14),
        ("boolValue", True, Sdf.ValueTypeNames.Bool, True),
        ("intValue", 42, Sdf.ValueTypeNames.Int, 42),
        ("stringValue", "test", Sdf.ValueTypeNames.String, "test"),
        ("vec3Value", Gf.Vec3f(1.0, 2.0, 3.0), Sdf.ValueTypeNames.Float3, Gf.Vec3f(1.0, 2.0, 3.0)),
        ("colorValue", Gf.Vec3f(1.0, 0.0, 0.5), Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1.0, 0.0, 0.5)),
    ],
    ids=["float", "bool", "int", "string", "vec3", "color"],
)
def test_change_prim_property_different_types(attr_name: str, value, value_type, expected):
    """Test change_prim_property() with different value types."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a prim
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)

    # change the property
    result = sim_utils.change_prim_property(
        prop_path=f"/World/Test.{attr_name}",
        value=value,
        stage=stage,
        type_to_create_if_not_exist=value_type,
        is_custom=True,
    )

    # check that the change was successful
    assert result is True
    actual_value = prim.GetAttribute(attr_name).Get()

    # handle float comparison separately for precision
    if isinstance(expected, float):
        assert math.isclose(actual_value, expected, abs_tol=1e-6)
    else:
        assert actual_value == expected


@pytest.mark.parametrize(
    "prop_path_input",
    ["/World/Cube.size", Sdf.Path("/World/Cube.size")],
    ids=["str_path", "sdf_path"],
)
def test_change_prim_property_path_types(prop_path_input):
    """Test change_prim_property() with different path input types."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a cube prim
    prim = sim_utils.create_prim("/World/Cube", "Cube", stage=stage, attributes={"size": 1.0})

    # change property using different path types
    result = sim_utils.change_prim_property(
        prop_path=prop_path_input,
        value=3.0,
        stage=stage,
    )

    # check that the change was successful
    assert result is True
    assert prim.GetAttribute("size").Get() == 3.0


def test_change_prim_property_error_invalid_prim():
    """Test change_prim_property() raises error for invalid prim path."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()

    # try to change property on non-existent prim
    with pytest.raises(ValueError, match="Prim does not exist"):
        sim_utils.change_prim_property(
            prop_path="/World/NonExistent.property",
            value=1.0,
            stage=stage,
        )


def test_change_prim_property_error_missing_type():
    """Test change_prim_property() returns False when property doesn't exist and type not provided."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create a prim
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)

    # try to create property without providing type
    result = sim_utils.change_prim_property(
        prop_path="/World/Test.nonExistentProperty",
        value=42,
        stage=stage,
    )

    # should return False since type was not provided
    assert result is False
    # property should not have been created
    assert prim.GetAttribute("nonExistentProperty").Get() is None


"""
Internal Helpers.
"""


def test_to_tuple_basic():
    """Test _to_tuple() with basic input types."""
    # Test with list
    result = _to_tuple([1.0, 2.0, 3.0])
    assert result == (1.0, 2.0, 3.0)
    assert isinstance(result, tuple)

    # Test with tuple
    result = _to_tuple((1.0, 2.0, 3.0))
    assert result == (1.0, 2.0, 3.0)

    # Test with numpy array
    result = _to_tuple(np.array([1.0, 2.0, 3.0]))
    assert result == (1.0, 2.0, 3.0)

    # Test with torch tensor (CPU)
    result = _to_tuple(torch.tensor([1.0, 2.0, 3.0]))
    assert result == (1.0, 2.0, 3.0)

    # Test squeezing first dimension (batch size 1)
    result = _to_tuple(torch.tensor([[1.0, 2.0]]))
    assert result == (1.0, 2.0)

    result = _to_tuple(np.array([[1.0, 2.0, 3.0]]))
    assert result == (1.0, 2.0, 3.0)


def test_to_tuple_raises_error():
    """Test _to_tuple() raises an error for N-dimensional arrays."""

    with pytest.raises(ValueError, match="not one dimensional"):
        _to_tuple(np.array([[1.0, 2.0], [3.0, 4.0]]))

    with pytest.raises(ValueError, match="not one dimensional"):
        _to_tuple(torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]))

    with pytest.raises(ValueError, match="only one element tensors can be converted"):
        _to_tuple((torch.tensor([1.0, 2.0]), 3.0))


def test_to_tuple_mixed_sequences():
    """Test _to_tuple() with mixed type sequences."""

    # Mixed list with numpy and floats
    result = _to_tuple([np.float32(1.0), 2.0, 3.0])
    assert len(result) == 3
    assert all(isinstance(x, float) for x in result)

    # Mixed tuple with torch tensor items and floats
    result = _to_tuple([torch.tensor(1.0), 2.0, 3.0])
    assert result == (1.0, 2.0, 3.0)

    # Mixed tuple with numpy array items and torch tensor
    result = _to_tuple((np.float32(1.0), 2.0, torch.tensor(3.0)))
    assert result == (1.0, 2.0, 3.0)


def test_to_tuple_precision():
    """Test _to_tuple() maintains numerical precision."""
    from isaaclab.sim.utils.prims import _to_tuple

    # Test with high precision values
    high_precision = [1.123456789, 2.987654321, 3.141592653]
    result = _to_tuple(torch.tensor(high_precision, dtype=torch.float64))

    # Check that precision is maintained reasonably well
    for i, val in enumerate(high_precision):
        assert math.isclose(result[i], val, abs_tol=1e-6)
