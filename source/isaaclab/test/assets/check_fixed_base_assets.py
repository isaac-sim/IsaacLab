# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates fixed-base API for different robots.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/isaaclab/test/assets/check_fixed_base_assets.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script checks the fixed-base API for different robots.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

##
# Pre-defined configs
##
from isaaclab_assets import ANYMAL_C_CFG, FRANKA_PANDA_CFG  # isort:skip


def define_origins(num_origins: int, spacing: float) -> list[list[float]]:
    """Defines the origins of the the scene."""
    # create tensor based on number of environments
    env_origins = torch.zeros(num_origins, 3)
    # create a grid of origins
    num_cols = np.floor(np.sqrt(num_origins))
    num_rows = np.ceil(num_origins / num_cols)
    xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="xy")
    env_origins[:, 0] = spacing * xx.flatten()[:num_origins] - spacing * (num_rows - 1) / 2
    env_origins[:, 1] = spacing * yy.flatten()[:num_origins] - spacing * (num_cols - 1) / 2
    env_origins[:, 2] = 0.0
    # return the origins
    return env_origins.tolist()


def design_scene() -> tuple[dict, list[list[float]]]:
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2", "Origin3"
    # Each group will have a mount and a robot on top of it
    origins = define_origins(num_origins=4, spacing=2.0)

    # Origin 1 with Franka Panda
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # -- Robot
    franka = Articulation(FRANKA_PANDA_CFG.replace(prim_path="/World/Origin1/Robot"))

    # Origin 2 with Anymal C
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])
    # -- Robot
    robot_cfg = ANYMAL_C_CFG.replace(prim_path="/World/Origin2/Robot")
    robot_cfg.spawn.articulation_props.fix_root_link = True
    anymal_c = Articulation(robot_cfg)

    # return the scene information
    scene_entities = {
        "franka": franka,
        "anymal_c": anymal_c,
    }
    return scene_entities, origins


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation], origins: torch.Tensor):
    """Runs the simulation loop."""
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if count % 200 == 0:
            # reset counters
            sim_time = 0.0
            count = 0
            # reset robots
            for index, robot in enumerate(entities.values()):
                # root state
                root_state = robot.data.default_root_state.clone()
                root_state[:, :3] += origins[index]
                root_state[:, :2] += torch.randn_like(root_state[:, :2]) * 0.25
                robot.write_root_pose_to_sim(root_state[:, :7])
                robot.write_root_velocity_to_sim(root_state[:, 7:])
                # joint state
                joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
                robot.write_joint_state_to_sim(joint_pos, joint_vel)
                # reset the internal state
                robot.reset()
            print("[INFO]: Resetting robots state...")
        # apply default actions to the quadrupedal robots
        for name, robot in entities.items():
            if count % 200 == 0:
                print("Name: ", name, "is_fixed_base: ", robot.is_fixed_base)
            # generate random joint positions
            joint_pos_target = robot.data.default_joint_pos + torch.randn_like(robot.data.joint_pos) * 0.1
            # apply action to the robot
            robot.set_joint_position_target(joint_pos_target)
            # write data to sim
            robot.write_data_to_sim()
        # perform step
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        # update buffers
        for robot in entities.values():
            robot.update(sim_dt)


def main():
    """Main function."""

    # Initialize the simulation context
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01))
    # Set main camera
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])
    # design scene
    scene_entities, scene_origins = design_scene()
    scene_origins = torch.tensor(scene_origins, device=sim.device)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities, scene_origins)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
