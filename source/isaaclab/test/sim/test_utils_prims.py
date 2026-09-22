# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import numpy as np
import pytest
import torch

from pxr import Gf, Sdf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.sim.utils.prims import _to_tuple  # type: ignore[reportPrivateUsage]
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR, retrieve_file_path

pytestmark = [pytest.mark.unit, pytest.mark.isaacsim_ci]

FRANKA_USD = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/Legacy/panda_instanceable.usd"
STANDARD_OPS = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


@pytest.fixture
def stage():
    return sim_utils.create_new_stage()


def assert_quat_close(quat: Gf.Quatd, expected_xyzw, eps: float = 1e-6):
    got = np.array([*quat.GetImaginary(), quat.GetReal()])
    assert np.allclose(got, expected_xyzw, atol=eps) or np.allclose(got, -np.asarray(expected_xyzw), atol=eps)


def _as_type(values: list[float], input_type: str):
    if input_type == "tuple":
        return tuple(values)
    if input_type == "numpy":
        return np.array(values)
    if input_type == "torch_cpu":
        return torch.tensor(values)
    if input_type == "torch_cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.tensor(values, device="cuda")
    return values


"""
create_prim() and delete_prim()
"""


def test_create_prim(stage):
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)
    assert prim.GetPrimPath() == "/World/Test"
    assert prim.GetTypeName() == "Xform"
    with pytest.raises(ValueError, match="already exists"):
        sim_utils.create_prim("/World/Test", "Xform", stage=stage)

    # attributes and semantic labels
    prim = sim_utils.create_prim(
        "/World/Test/Sphere", "Sphere", stage=stage, semantic_label="sphere", attributes={"radius": 10.0}
    )
    assert prim.GetAttribute("radius").Get() == 10.0
    assert sim_utils.get_labels(prim)["class"] == ["sphere"]

    # local transform authored in the standard op order
    pos, quat, scale = (1.0, 2.0, 3.0), (0.0, 0.0, 1.0, 0.0), (1.0, 0.5, 0.5)
    prim = sim_utils.create_prim("/World/Test/Xform", stage=stage, translation=pos, orientation=quat, scale=scale)
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(pos)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), quat)
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(scale)
    assert [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()] == STANDARD_OPS

    with pytest.raises(ValueError, match="both position and translation"):
        sim_utils.create_prim("/World/Test/Both", stage=stage, position=pos, translation=pos)


def test_create_prim_with_usd_reference(stage):
    prim = sim_utils.create_prim("/World/Franka", usd_path=FRANKA_USD, stage=stage)
    assert prim.GetTypeName() == "Xform"
    # remote paths are resolved to the local cache
    assert sim_utils.get_usd_references("/World/Franka", stage=stage) == [retrieve_file_path(FRANKA_USD)]
    assert sim_utils.get_usd_references("/World", stage=stage) == []
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.get_usd_references("/World/Missing", stage=stage)

    sim_utils.delete_prim("/World/Franka", stage=stage)
    assert not prim.IsValid()


@pytest.mark.parametrize("input_type", ["list", "tuple", "numpy", "torch_cpu", "torch_cuda"])
def test_create_prim_input_types(stage, input_type):
    parent = sim_utils.create_prim("/World/Parent", "Xform", stage=stage, translation=(5.0, 10.0, 15.0))
    translation, orientation, scale = [1.0, 2.0, 3.0], [0.0, 0.7071068, 0.0, 0.7071068], [2.0, 3.0, 4.0]

    # local pose is authored as given
    local = sim_utils.create_prim(
        "/World/Local",
        "Xform",
        stage=stage,
        translation=_as_type(translation, input_type),
        orientation=_as_type(orientation, input_type),
        scale=_as_type(scale, input_type),
    )
    assert local.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(*translation)
    assert_quat_close(local.GetAttribute("xformOp:orient").Get(), orientation)
    assert local.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(*scale)

    # world pose is converted into the parent frame
    child = sim_utils.create_prim(
        "/World/Parent/Child",
        "Xform",
        stage=stage,
        position=_as_type(translation, input_type),
        orientation=_as_type(orientation, input_type),
    )
    pos, quat = sim_utils.resolve_prim_pose(child)
    assert np.allclose(pos, translation, atol=1e-4)
    assert np.allclose(quat, orientation, atol=1e-4) or np.allclose(quat, -np.asarray(orientation), atol=1e-4)
    assert child.GetParent() == parent


@pytest.mark.parametrize("prim_type", ["Material", "Scope"])
def test_create_prim_non_xformable_skips_transforms(stage, prim_type):
    prim = sim_utils.create_prim(f"/World/{prim_type}", prim_type, stage=stage, translation=(1.0, 2.0, 3.0))
    assert prim.GetTypeName() == prim_type
    assert not prim.IsA(UsdGeom.Xformable)
    assert not any(prim.HasAttribute(name) for name in STANDARD_OPS)


def test_delete_prim(stage):
    prims = [sim_utils.create_prim(f"/World/Xform{i}", "Xform", stage=stage) for i in range(3)]
    assert sim_utils.delete_prim("/World/Xform0", stage=stage)
    assert sim_utils.delete_prim(("/World/Xform1", "/World/Xform2"), stage=stage)
    assert not any(prim.IsValid() for prim in prims)


"""
USD variants.
"""


def test_select_usd_variants(stage):
    prim = UsdGeom.Xform.Define(stage, Sdf.Path("/World")).GetPrim()
    variant_set = prim.GetVariantSets().AddVariantSet("colors")
    for variant in ["red", "blue"]:
        variant_set.AddVariant(variant)

    sim_utils.select_usd_variants("/World", {"colors": "red"}, stage)
    assert variant_set.GetVariantSelection() == "red"

    # USD would silently accept an unknown variant and compose the prim as if nothing were selected
    with pytest.raises(ValueError, match="does not offer variant"):
        sim_utils.select_usd_variants("/World", {"colors": "chartreuse"}, stage)
    assert variant_set.GetVariantSelection() == "red"
    # a variant set the prim does not have is skipped so one config can serve assets with different options
    sim_utils.select_usd_variants("/World", {"absent_set": "anything"}, stage)
    with pytest.raises(ValueError, match="not valid"):
        sim_utils.select_usd_variants("/Missing", {"colors": "red"}, stage)


"""
change_prim_property()
"""


@pytest.mark.parametrize("prop_path", ["/World/Cube.size", Sdf.Path("/World/Cube.size")], ids=["str", "sdf_path"])
def test_change_prim_property_existing(stage, prop_path):
    prim = sim_utils.create_prim("/World/Cube", "Cube", stage=stage, attributes={"size": 1.0})
    assert sim_utils.change_prim_property(prop_path, 3.0, stage=stage) is True
    assert prim.GetAttribute("size").Get() == 3.0
    # clearing falls back to the schema default
    assert sim_utils.change_prim_property(prop_path, None, stage=stage) is True
    assert prim.GetAttribute("size").Get() == 2.0


@pytest.mark.parametrize(
    ("value", "value_type"),
    [
        (3.14, Sdf.ValueTypeNames.Float),
        (True, Sdf.ValueTypeNames.Bool),
        (42, Sdf.ValueTypeNames.Int),
        ("test", Sdf.ValueTypeNames.String),
        (Gf.Vec3f(1.0, 2.0, 3.0), Sdf.ValueTypeNames.Float3),
        (Gf.Vec3f(1.0, 0.0, 0.5), Sdf.ValueTypeNames.Color3f),
    ],
    ids=["float", "bool", "int", "string", "vec3", "color"],
)
def test_change_prim_property_creates_custom(stage, value, value_type):
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)
    assert sim_utils.change_prim_property(
        "/World/Test.custom", value, stage=stage, type_to_create_if_not_exist=value_type, is_custom=True
    )
    assert prim.GetAttribute("custom").Get() == pytest.approx(value)


def test_change_prim_property_errors(stage):
    prim = sim_utils.create_prim("/World/Test", "Xform", stage=stage)
    with pytest.raises(ValueError, match="Prim does not exist"):
        sim_utils.change_prim_property("/World/Missing.property", 1.0, stage=stage)
    # a missing property needs a type to be created
    assert sim_utils.change_prim_property("/World/Test.missing", 42, stage=stage) is False
    assert not prim.HasAttribute("missing")


"""
_to_tuple()
"""


@pytest.mark.parametrize(
    "value",
    [
        [1.0, 2.0, 3.0],
        (1.0, 2.0, 3.0),
        np.array([1.0, 2.0, 3.0]),
        np.array([[1.0, 2.0, 3.0]]),
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([[1.0, 2.0, 3.0]]),
        [np.float32(1.0), 2.0, torch.tensor(3.0)],
    ],
    ids=["list", "tuple", "numpy", "numpy_batched", "torch", "torch_batched", "mixed"],
)
def test_to_tuple(value):
    result = _to_tuple(value)
    assert result == (1.0, 2.0, 3.0)
    assert all(isinstance(x, float) for x in result)


def test_to_tuple_precision():
    values = [1.123456789, 2.987654321, 3.141592653]
    assert all(
        math.isclose(a, b, abs_tol=1e-6) for a, b in zip(_to_tuple(torch.tensor(values, dtype=torch.float64)), values)
    )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.array([[1.0, 2.0], [3.0, 4.0]]), "not one dimensional"),
        (torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]), "not one dimensional"),
        ((torch.tensor([1.0, 2.0]), 3.0), "only one element tensors can be converted"),
    ],
    ids=["numpy_2d", "torch_3d", "nested_tensor"],
)
def test_to_tuple_rejects_non_1d(value, message):
    with pytest.raises(ValueError, match=message):
        _to_tuple(value)
