# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the cloner API from Isaac Sim.

Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html
"""

"""Launch Isaac Sim Simulator first."""


import argparse
import contextlib

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from isaacsim import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script shows the issue in Isaac Sim with GPU simulation of floating robots."
)
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
parser.add_argument(
    "--asset",
    type=str,
    default="isaaclab",
    help="The asset source location for the robot. Can be: isaaclab, oige, custom asset path.",
)
parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
simulation_app = SimulationApp({"headless": args_cli.headless})

"""Rest everything follows."""

import os
import torch

import omni.log

try:
    import isaacsim.storage.native as nucleus_utils
except ModuleNotFoundError:
    import isaacsim.core.utils.nucleus as nucleus_utils

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.world import World
from isaacsim.core.cloner import GridCloner
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.carb import set_carb_setting
from isaacsim.core.utils.viewports import set_camera_view

# check nucleus connection
if nucleus_utils.get_assets_root_path() is None:
    msg = (
        "Unable to perform Nucleus login on Omniverse. Assets root path is not set.\n"
        "\tPlease check: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html#omniverse-nucleus"
    )
    omni.log.error(msg)
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
    if args_cli.asset == "isaaclab":
        usd_path = f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd"
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        usd_path = f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/anymal_instanceable.usd"
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: isaaclab, oige.")
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
    if args_cli.asset == "isaaclab":
        root_prim_path = "/World/envs/env_.*/Robot/base"
    elif args_cli.asset == "oige":
        root_prim_path = "/World/envs/env_.*/Robot"
    elif os.path.exists(args_cli.asset):
        usd_path = args_cli.asset
        root_prim_path = "/World/envs/env_.*/Robot"
    else:
        raise ValueError(f"Invalid asset: {args_cli.asset}. Must be one of: isaaclab, oige.")
    # Setup robot
    robot_view = Articulation(root_prim_path, name="ANYMAL")
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
    # run the main function
    main()
    # close sim app
    simulation_app.close()
