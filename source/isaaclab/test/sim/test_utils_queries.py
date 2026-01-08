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

import pytest

from pxr import UsdPhysics

import isaaclab.sim as sim_utils
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


"""
USD Stage Querying.
"""


def test_get_next_free_prim_path():
    """Test get_next_free_prim_path() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Wall", "Sphere", attributes={"radius": 1e3})

    # test
    isaaclab_result = sim_utils.get_next_free_prim_path("/World/Floor")
    assert isaaclab_result == "/World/Floor_01"

    # create another prim
    sim_utils.create_prim("/World/Floor/Box_01", "Cube", position=[75, 75, -150.1], attributes={"size": 300})

    # test again
    isaaclab_result = sim_utils.get_next_free_prim_path("/World/Floor/Box")
    assert isaaclab_result == "/World/Floor/Box_02"


def test_get_first_matching_ancestor_prim():
    """Test get_first_matching_ancestor_prim() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Floor/Box/Sphere", "Sphere", attributes={"radius": 1e3})

    # test with input prim not having the predicate
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box/Sphere", predicate=lambda x: x.GetTypeName() == "Cube"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/Floor/Box"

    # test with input prim having the predicate
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box", predicate=lambda x: x.GetTypeName() == "Cube"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/Floor/Box"

    # test with no predicate match
    isaaclab_result = sim_utils.get_first_matching_ancestor_prim(
        "/World/Floor/Box/Sphere", predicate=lambda x: x.GetTypeName() == "Cone"
    )
    assert isaaclab_result is None


def test_get_all_matching_child_prims():
    """Test get_all_matching_child_prims() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim("/World/Floor/Box", "Cube", position=[75, 75, -150.1], attributes={"size": 300})
    sim_utils.create_prim("/World/Wall", "Sphere", attributes={"radius": 1e3})

    # add articulation root prim -- this asset has instanced prims
    # note: isaac sim function does not support instanced prims so we add it here
    #  after the above test for the above test to still pass.
    sim_utils.create_prim(
        "/World/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )

    # test with predicate
    isaaclab_result = sim_utils.get_all_matching_child_prims("/World", predicate=lambda x: x.GetTypeName() == "Cube")
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Floor/Box"

    # test with predicate and instanced prims
    isaaclab_result = sim_utils.get_all_matching_child_prims(
        "/World/Franka/panda_hand/visuals", predicate=lambda x: x.GetTypeName() == "Mesh"
    )
    assert len(isaaclab_result) == 1
    assert isaaclab_result[0].GetPrimPath() == "/World/Franka/panda_hand/visuals/panda_hand"

    # test valid path
    with pytest.raises(ValueError):
        sim_utils.get_all_matching_child_prims("World/Room")


def test_get_first_matching_child_prim():
    """Test get_first_matching_child_prim() function."""
    # create scene
    sim_utils.create_prim("/World/Floor")
    sim_utils.create_prim(
        "/World/env_1/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    sim_utils.create_prim(
        "/World/env_2/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )
    sim_utils.create_prim(
        "/World/env_0/Franka", "Xform", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd"
    )

    # test
    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World", predicate=lambda prim: prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka"

    # test with instanced prims
    isaaclab_result = sim_utils.get_first_matching_child_prim(
        "/World/env_1/Franka", predicate=lambda prim: prim.GetTypeName() == "Mesh"
    )
    assert isaaclab_result is not None
    assert isaaclab_result.GetPrimPath() == "/World/env_1/Franka/panda_link0/visuals/panda_link0"


def test_find_global_fixed_joint_prim():
    """Test find_global_fixed_joint_prim() function."""
    # create scene
    sim_utils.create_prim("/World")
    sim_utils.create_prim("/World/ANYmal", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd")
    sim_utils.create_prim("/World/Franka", usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/panda_instanceable.usd")
    if "4.5" in ISAAC_NUCLEUS_DIR:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/Franka/franka.usd"
    else:
        franka_usd = f"{ISAAC_NUCLEUS_DIR}/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    sim_utils.create_prim("/World/Franka_Isaac", usd_path=franka_usd)

    # test
    assert sim_utils.find_global_fixed_joint_prim("/World/ANYmal") is None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka_Isaac") is not None

    # make fixed joint disabled manually
    joint_prim = sim_utils.find_global_fixed_joint_prim("/World/Franka")
    joint_prim.GetJointEnabledAttr().Set(False)
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka") is not None
    assert sim_utils.find_global_fixed_joint_prim("/World/Franka", check_enabled_only=True) is None
