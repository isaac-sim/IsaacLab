# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates reference count of the robot view in Isaac Sim.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows the issue in Isaac Sim with reference count of the robot view."
)
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import ctypes
import gc

from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
import omni.isaac.core.utils.prims as prim_utils

from omni.isaac.lab.utils.assets import ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.assets import Articulation

##
# Predefined configurations.
##
from omni.isaac.lab_assets import ANYMAL_C_CFG  # isort:skip


"""
Main
"""


def main():
    """Spawns the ANYmal robot and clones it using Isaac Sim Cloner API."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.005, rendering_dt=0.005, backend="torch", device="cuda:0")

    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # resolve asset
    # usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
    # # add asset
    # print("Loading robot from: ", usd_path)
    # prim_utils.create_prim("/World/Robot", usd_path=usd_path, translation=(0.0, 0.0, 0.6))

    # Spawn things into stage
    # -- Robot
    cfg = ANYMAL_C_CFG.replace(prim_path="/World/Robot")

    # cfg.spawn.func("/World/Robot", cfg.spawn)
    # cfg.spawn = None

    robot_view = Articulation(cfg)

    # Check the reference count
    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Play the simulator
    sim.reset()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Stop the simulator
    sim.stop()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)

    # Clean up
    sim.clear()

    print("Reference count of the robot view: ", ctypes.c_long.from_address(id(robot_view)).value)
    print("Referrers of the robot view: ", gc.get_referrers(robot_view))
    print("---" * 10)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
