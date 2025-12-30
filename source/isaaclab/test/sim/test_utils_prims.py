# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


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


def assert_quat_close(q1: Gf.Quatf | Gf.Quatd, q2: Gf.Quatf | Gf.Quatd, eps: float = 1e-6):
    """Assert two quaternions are close."""
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
    quat = (0.0, 0.0, 0.0, 1.0)
    scale = (1.0, 0.5, 0.5)
    prim = sim_utils.create_prim(
        "/World/Test/Xform", "Xform", stage=stage, translation=pos, orientation=quat, scale=scale
    )
    # check transform set
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/Test/Xform"
    assert prim.GetTypeName() == "Xform"
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d(pos)
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd(*quat))
    assert prim.GetAttribute("xformOp:scale").Get() == Gf.Vec3d(scale)
    # check xform operation order
    op_names = [op.GetOpName() for op in UsdGeom.Xformable(prim).GetOrderedXformOps()]
    assert op_names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]


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


def test_move_prim():
    """Test move_prim() function."""
    # obtain stage handle
    stage = sim_utils.get_current_stage()
    # create scene
    sim_utils.create_prim("/World/Test", "Xform", stage=stage)
    prim = sim_utils.create_prim(
        "/World/Test/Xform",
        "Xform",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd",
        translation=(1.0, 2.0, 3.0),
        orientation=(0.0, 0.0, 0.0, 1.0),
        stage=stage,
    )

    # move prim
    sim_utils.create_prim("/World/TestMove", "Xform", stage=stage, translation=(1.0, 1.0, 1.0))
    sim_utils.move_prim("/World/Test/Xform", "/World/TestMove/Xform", stage=stage)
    # check prim moved
    prim = stage.GetPrimAtPath("/World/TestMove/Xform")
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/TestMove/Xform"
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d((0.0, 1.0, 2.0))
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd(0.0, 0.0, 0.0, 1.0))

    # check moving prim with keep_world_transform=False
    # it should preserve the local transform from last move
    sim_utils.create_prim(
        "/World/TestMove2", "Xform", stage=stage, translation=(2.0, 2.0, 2.0), orientation=(0.0, 0.7071, 0.0, 0.7071)
    )
    sim_utils.move_prim("/World/TestMove/Xform", "/World/TestMove2/Xform", keep_world_transform=False, stage=stage)
    # check prim moved
    prim = stage.GetPrimAtPath("/World/TestMove2/Xform")
    assert prim.IsValid()
    assert prim.GetPrimPath() == "/World/TestMove2/Xform"
    assert prim.GetAttribute("xformOp:translate").Get() == Gf.Vec3d((0.0, 1.0, 2.0))
    assert_quat_close(prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd(0.0, 0.0, 0.0, 1.0))


"""
USD references and variants.
"""


def test_select_usd_variants():
    """Test select_usd_variants() function."""
    stage = sim_utils.get_current_stage()
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
