# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the cloner API from Isaac Sim.

Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows the issue in Isaac Sim with GPU simulation of floating robots."
)
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
parser.add_argument(
    "--asset",
    type=str,
    default="orbit",
    help="The asset source location for the robot. Can be: orbit, oige, custom asset path.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import os
import torch
import traceback

import carb
import omni.isaac.core.utils.nucleus as nucleus_utils
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner
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

ISAAC_ORBIT_NUCLEUS_DIR = f"{nucleus_utils.get_assets_root_path()}/Isaac/Samples/Orbit"
"""Path to the `Isaac/Samples/Orbit` directory on the NVIDIA Nucleus Server."""


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

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Ground-plane
    world.scene.add_default_ground_plane(prim_path="/World/defaultGroundPlane", z_position=0.0)
    # Lights-1
    prim_utils.create_prim("/World/Light/GreySphere", "SphereLight", translation=(4.5, 3.5, 10.0))
    # Lights-2
    prim_utils.create_prim("/World/Light/WhiteSphere", "SphereLight", translation=(-4.5, 3.5, 10.0))
    # -- Robot
    # resolve asset
    if args_cli.asset == "orbit":
        usd_path = f"{ISAAC_ORBIT_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd"
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: orbit, oige.")
    # add asset
    print("Loading robot from: ", usd_path)
    prim_utils.create_prim(
        "/World/envs/env_0/Robot",
        usd_path=usd_path,
        translation=(0.0, 0.0, 0.6),
    )

    # Clone the scene
    num_envs = args_cli.num_robots
    cloner.define_base_env("/World/envs")
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    envs_positions = cloner.clone(
        source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths, replicate_physics=True
    )
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=world.device)
    # filter collisions within each environment instance
    physics_scene_path = world.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )

    # Resolve robot prim paths
    if args_cli.asset == "orbit":
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
        root_prim_path = "/World/envs/env_.*/Robot"
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: orbit, oige.")
    # Setup robot
    robot_view = ArticulationView(root_prim_path, name="ANYMAL")
    world.scene.add(robot_view)
    # Play the simulator
    world.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    # actions = torch.zeros(robot.count, robot.num_actions, device=robot.device)

    # Define simulation stepping
    sim_dt = world.get_physics_dt()
    # episode counter
    sim_time = 0.0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if world.is_stopped():
            break
        # If simulation is paused, then skip.
        if not world.is_playing():
            world.step(render=False)
            continue
        # perform step
        world.step()
        # update sim-time
        sim_time += sim_dt


if __name__ == "__main__":
    try:
        # Run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
