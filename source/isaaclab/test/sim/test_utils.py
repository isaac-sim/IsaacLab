# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
simulation_app = AppLauncher(headless=True).app

"""Rest everything follows."""

import numpy as np

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import pytest
from pxr import Sdf, Usd, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR


@pytest.fixture(autouse=True)
def test_setup_teardown():
    """Create a blank new stage for each test."""
    # Setup: Create a new stage
    stage_utils.create_new_stage()
    stage_utils.update_stage()

    # Yield for the test
    yield

    # Teardown: Clear stage after each test
    stage_utils.clear_stage()


def test_get_all_matching_child_prims():
    """Test get_all_matching_child_prims() function."""
    # create scene
    prim_utils.create_prim("/World/Floor")
    prim_utils.create_prim(
        "/World/Floor/thefloor", "Cube", position=np.array([75, 75, -150.1]), attributes={"size": 300}
    )
    prim_utils.create_prim("/World/Room", "Sphere", attributes={"radius": 1e3})

    # test
    isaac_sim_result = prim_utils.get_all_matching_child_prims("/World")
    isaaclab_result = sim_utils.get_all_matching_child_prims("/World")
    assert isaac_sim_result == isaaclab_result

    # test valid path
    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Room")


def test_find_matching_prim_paths():
    """Test find_matching_prim_paths() function."""
    # create scene
    for index in range(2048):
        random_pos = np.random.uniform(-100, 100, size=3)
        prim_utils.create_prim(f"/World/Floor_{index}", "Cube", position=random_pos, attributes={"size": 2.0})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere", "Sphere", attributes={"radius": 10})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere", "Sphere", attributes={"radius": 1})
        prim_utils.create_prim(f"/World/Floor_{index}/Sphere/childSphere2", "Sphere", attributes={"radius": 1})

    # test leaf paths
    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere")
    assert isaac_sim_result == isaaclab_result

    # test non-leaf paths
    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*")
    assert isaac_sim_result == isaaclab_result

    # test child-leaf paths
    isaac_sim_result = prim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
    isaaclab_result = sim_utils.find_matching_prim_paths("/World/Floor_.*/Sphere/childSphere.*")
    assert isaac_sim_result == isaaclab_result

    # test valid path
    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Floor_.*")


def test_find_global_fixed_joint_prim():
    """Test find_global_fixed_joint_prim() function."""
    # create scene
    prim_utils.create_prim("/World")
    prim_utils.create_prim("/World/ANYmal", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd")
    prim_utils.create_prim(
        "/World/Franka", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    if "4.5" in ISAAC_NUCLEUS_DIR:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"
    else:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    prim_utils.create_prim("/World/Franka_Isaac", usd_path=franka_usd)

    # test
    assert sim_utils.find_global_fixed_joint_prim("/World/ANYmal") is None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka_Isaac") is not None

    # make fixed joint disabled manually
    joint_prim = sim_utils.find_global_fixed_joint_prim("/World/Franka")
    joint_prim.GetJointEnabledAttr().Set(False)
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka", check_enabled_only=True) is None


def test_select_usd_variants():
    """Test select_usd_variants() function."""
    stage = stage_utils.get_current_stage()
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
