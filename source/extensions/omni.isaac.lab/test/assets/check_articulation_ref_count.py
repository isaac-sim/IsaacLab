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

import torch
import ctypes
import gc

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner

import omni.isaac.lab.sim as sim_utils
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
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Lights-1
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
    # Lights-2
    prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))
    # -- Robot
    robot_view = Articulation(ANYMAL_C_CFG.replace(prim_path="/World/envs/env_.*/Robot"))

    # Clone the scene
    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    envs_positions = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True
    )
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=sim.device)
    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )

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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
