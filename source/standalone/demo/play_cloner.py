# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the cloner API from Isaac Sim.

Reference: https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/tutorial_gym_cloner.html
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config)


"""Rest everything follows."""


import torch

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.carb import set_carb_setting
from omni.isaac.core.utils.viewports import set_camera_view

import omni.isaac.orbit.utils.kit as kit_utils
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

"""
Main
"""


def main():
    """Spawns the Franka robot and clones it using Isaac Gym Cloner API."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda:0")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Enable flatcache which avoids passing data over to USD structure
    # this speeds up the read-write operation of GPU buffers
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    # this is needed to visualize the scene when flatcache is enabled
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Ground-plane
    kit_utils.create_ground_plane("/World/defaultGroundPlane", z_position=-1.05)
    # Lights-1
    prim_utils.create_prim(
        "/World/Light/GreySphere",
        "SphereLight",
        translation=(4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (0.75, 0.75, 0.75)},
    )
    # Lights-2
    prim_utils.create_prim(
        "/World/Light/WhiteSphere",
        "SphereLight",
        translation=(-4.5, 3.5, 10.0),
        attributes={"radius": 2.5, "intensity": 600.0, "color": (1.0, 1.0, 1.0)},
    )
    # -- Table
    table_usd_path = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    prim_utils.create_prim("/World/envs/env_0/Table", usd_path=table_usd_path)
    # -- Robot
    # resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/env_0/Robot", translation=(0.0, 0.0, 0.0))

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

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/envs/env_.*/Robot")
    # Reset states
    robot.reset_buffers()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
    has_gripper = robot.cfg.meta_info.tool_num_dof > 0

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step(render=not args_cli.headless)
            continue
        # reset
        if ep_step_count % 100 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset command
            actions = torch.rand(robot.count, robot.num_actions, device=robot.device)
            # reset gripper
            if has_gripper:
                actions[:, -1] = -1
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 50 == 0 and has_gripper:
            # flip command for the gripper
            actions[:, -1] = -actions[:, -1]
        # apply actions
        robot.apply_action(actions)
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
