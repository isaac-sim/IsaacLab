# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates reference count of the robot view in Isaac Sim.
"""

"""Launch Isaac Sim Simulator first."""


import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from omni.isaac.kit import SimulationApp

# launch omniverse app
simulation_app = SimulationApp({"headless": True})

"""Rest everything follows."""

import ctypes
import gc

import carb

try:
    import omni.isaac.nucleus as nucleus_utils
except ModuleNotFoundError:
    import omni.isaac.core.utils.nucleus as nucleus_utils

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.world import World

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    carb.log_error(msg)
    raise RuntimeError(msg)


ISAAC_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac"
"""Path to the `Isaac` directory on the NVIDIA Nucleus Server."""

ISAACLAB_NUCLEUS_DIR = f"{ISAAC_NUCLEUS_DIR}/IsaacLab"
"""Path to the `Isaac/IsaacLab` directory on the NVIDIA Nucleus Server."""


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""

    # Load kit helper
    world = World(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(world._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # -- Robot
    # resolve asset
    usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    # add asset
    print("Loading robot from: ", usd_path)
    prim_utils.create_prim("/World/Robot", usd_path=usd_path, translation=(0.0, 0.0, 0.6))

    # Resolve robot prim paths
    root_prim_path = "/World/Robot/base"
    # Setup robot
    robot_view = ArticulationView(root_prim_path, name="ANYMAL")

    # Check the reference count
    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    world.scene.add(robot_view)

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Play the simulator
    world.reset()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Stop the simulator
    world.stop()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Clean up
    world.clear()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
