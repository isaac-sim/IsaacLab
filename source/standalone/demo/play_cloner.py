# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
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
parser = argparse.ArgumentParser(description="This script demonstrates how to use the cloner API from Isaac Sim.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_robots", type=int, default=128, help="Number of robots to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""


import torch
import traceback

import carb
import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.cloner import GridCloner

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.assets import Articulation
from omni.isaac.orbit.assets.config import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG, UR10_CFG
from omni.isaac.orbit.sim import SimulationContext
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cuda:0", use_gpu_pipeline=True))
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg(height=-1.05)
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights-1
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(0.75, 0.75, 0.75), radius=2.5)
    cfg.func("/World/Light/greyLight", cfg, translation=(4.5, 3.5, 10.0))
    # Lights-2
    cfg = sim_utils.SphereLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0), radius=2.5)
    cfg.func("/World/Light/whiteSphere", cfg, translation=(-4.5, 3.5, 10.0))
    # Table
    cfg = sim_utils.UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd")
    cfg.func("/World/envs/env_0/Table", cfg)
    # Robot
    # -- resolve robot config from command-line arguments
    if args_cli.robot == "franka_panda":
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
    elif args_cli.robot == "ur10":
        robot_cfg = UR10_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # -- spawn internally and create interface
    robot = Articulation(cfg=robot_cfg.replace(prim_path="/World/envs/env_.*/Robot"))

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

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # dummy actions
    actions = torch.rand(robot.root_view.count, robot.num_joints, device=robot.device) + robot.data.default_joint_pos
    has_gripper = args_cli.robot == "franka_panda"

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    ep_step_count = 0
    # Simulate physics
    while simulation_app.is_running():
        # reset
        if ep_step_count % 100 == 0:
            sim_time = 0.0
            ep_step_count = 0
            # reset dof state
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset command
            actions = (
                torch.rand(robot.root_view.count, robot.num_joints, device=robot.device) + robot.data.default_joint_pos
            )
            # reset gripper
            if has_gripper:
                actions[:, -2:] = 0.04
            print("[INFO]: Resetting robots state...")
        # change the gripper action
        if ep_step_count % 50 == 0 and has_gripper:
            # flip command for the gripper
            actions[:, -2:] = 0.0 if actions[0, -2] > 0.0 else 0.04
        # apply action to the robot
        robot.set_joint_position_target(actions)
        robot.write_data_to_sim()
        # perform step
        sim.step(render=app_launcher.RENDER)
        # update sim-time
        sim_time += sim_dt
        ep_step_count += 1
        # update buffers
        robot.update(sim_dt)


if __name__ == "__main__":
    try:
        # run the main execution
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
