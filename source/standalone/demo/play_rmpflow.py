# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the RMPFlow controller with the simulator.

The RMP-Flow can be configured in different modes. It uses the LULA library for motion generation.
"""

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--robot", type=str, default="ur10", help="Name of the robot. Options: franka_panda, ur10.")
parser.add_argument("--num_envs", type=int, default=5, help="Number of environments to spawn.")
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
from omni.isaac.orbit.controllers.config.rmp_flow import FRANKA_RMPFLOW_CFG, UR10_RMPFLOW_CFG
from omni.isaac.orbit.controllers.rmp_flow import RmpFlowController
from omni.isaac.orbit.markers import StaticMarker
from omni.isaac.orbit.robots.config.franka import FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
from omni.isaac.orbit.robots.config.universal_robots import UR10_CFG
from omni.isaac.orbit.robots.single_arm import SingleArmManipulator
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR

"""
Main
"""


def main():
    """Spawns a single-arm manipulator and applies commands through RMPFlow kinematics control."""

    # Load kit helper
    sim = SimulationContext(physics_dt=0.01, rendering_dt=0.01, backend="torch", device="cuda:0")
    # Set main camera
    set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Enable GPU pipeline and flatcache
    if sim.get_physics_context().use_gpu_pipeline:
        sim.get_physics_context().enable_flatcache(True)
    # Enable hydra scene-graph instancing
    set_carb_setting(sim._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)

    # Create interface to clone the scene
    cloner = GridCloner(spacing=2.0)
    cloner.define_base_env("/World/envs")
    # Everything under the namespace "/World/envs/env_0" will be cloned
    prim_utils.define_prim("/World/envs/env_0")

    # Spawn things into stage
    # Markers
    ee_marker = StaticMarker("/Visuals/ee_current", count=args_cli.num_envs, scale=(0.1, 0.1, 0.1))
    goal_marker = StaticMarker("/Visuals/ee_goal", count=args_cli.num_envs, scale=(0.1, 0.1, 0.1))
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
        # robot config
        robot_cfg = FRANKA_PANDA_ARM_WITH_PANDA_HAND_CFG
        robot_cfg.actuator_groups["panda_shoulder"].control_cfg.command_types = ["p_abs", "v_abs"]
        robot_cfg.actuator_groups["panda_forearm"].control_cfg.command_types = ["p_abs", "v_abs"]
        # rmpflow controller config
        rmpflow_cfg = FRANKA_RMPFLOW_CFG
    elif args_cli.robot == "ur10":
        # robot config
        robot_cfg = UR10_CFG
        robot_cfg.actuator_groups["arm"].control_cfg.command_types = ["p_abs", "v_abs"]
        # rmpflow controller config
        rmpflow_cfg = UR10_RMPFLOW_CFG
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
    # configure robot settings to use IK controller
    robot_cfg.data_info.enable_jacobian = True
    robot_cfg.rigid_props.disable_gravity = True
    # spawn robot
    robot = SingleArmManipulator(cfg=robot_cfg)
    robot.spawn("/World/envs/env_0/Robot", translation=(0.0, 0.0, 0.0))

    # Clone the scene
    num_envs = args_cli.num_envs
    envs_prim_paths = cloner.generate_paths("/World/envs/env", num_paths=num_envs)
    envs_positions = cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=envs_prim_paths)
    # convert environment positions to torch tensor
    envs_positions = torch.tensor(envs_positions, dtype=torch.float, device=sim.device)
    # filter collisions within each environment instance
    physics_scene_path = sim.get_physics_context().prim_path
    cloner.filter_collisions(
        physics_scene_path, "/World/collisions", envs_prim_paths, global_paths=["/World/defaultGroundPlane"]
    )

    # Create controller
    # the controller takes as command type: {position/pose}_{abs/rel}
    rmp_controller = RmpFlowController(cfg=rmpflow_cfg, device=sim.device)

    # Play the simulator
    sim.reset()
    # Acquire handles
    # Initialize handles
    robot.initialize("/World/envs/env_.*/Robot")
    rmp_controller.initialize("/World/envs/env_.*/Robot")
    # Reset states
    robot.reset_buffers()
    rmp_controller.reset_idx()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Create buffers to store actions
    rmp_commands = torch.zeros(robot.count, rmp_controller.num_actions, device=robot.device)
    robot_actions = torch.ones(robot.count, robot.num_actions, device=robot.device)
    has_gripper = robot.cfg.meta_info.tool_num_dof > 0

    # Set end effector goals
    # Define goals for the arm
    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    # Track the given command
    current_goal_idx = 0
    rmp_commands[:] = ee_goals[current_goal_idx]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # episode counter
    sim_time = 0.0
    count = 0
    # Note: We need to update buffers before the first step for the controller.
    robot.update_buffers(sim_dt)

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
        if count % 350 == 0:
            # reset time
            count = 0
            sim_time = 0.0
            # reset dof state
            dof_pos, dof_vel = robot.get_default_dof_state()
            robot.set_dof_state(dof_pos, dof_vel)
            robot.reset_buffers()
            # reset controller
            rmp_controller.reset_idx()
            # reset actions
            rmp_commands[:] = ee_goals[current_goal_idx]
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            # step simulation
            # FIXME: this is needed for lula to update the buffers!
            # the bug has been reported to the lula team
            sim.step(render=not args_cli.headless)
        # set the controller commands
        rmp_controller.set_command(rmp_commands)
        # compute the joint commands
        desired_joint_pos, desired_joint_vel = rmp_controller.compute()
        # in some cases the zero action correspond to offset in actuators
        # so we need to subtract these over here so that they can be added later on
        arm_command_offset = robot.data.actuator_pos_offset[:, : robot.arm_num_dof]
        # offset actuator command with position offsets
        # note: valid only when doing position control of the robot
        desired_joint_pos -= arm_command_offset
        # set the desired joint position and velocity into buffers
        # note: this is a hack for now to deal with the fact that the robot has two different
        #   command types. We need to split and interleave the commands for the robot.
        #   This will be fixed in the future.
        # get all the actuator groups
        group_num_actuators = [group.num_actuators for group in robot.actuator_groups.values()]
        if has_gripper:
            # remove gripper from the list
            group_num_actuators = group_num_actuators[:-1]
        # set command for the arm
        group_desired_joint_pos = desired_joint_pos.split(group_num_actuators, dim=-1)
        group_desired_joint_vel = desired_joint_vel.split(group_num_actuators, dim=-1)
        # interleave the commands
        robot_actions_list = [None] * (len(group_desired_joint_pos) + len(group_desired_joint_vel))
        robot_actions_list[::2] = group_desired_joint_pos
        robot_actions_list[1::2] = group_desired_joint_vel
        # concatenate the list
        robot_actions[:, : 2 * sum(group_num_actuators)] = torch.cat(robot_actions_list, dim=-1)
        # apply actions
        robot.apply_action(robot_actions)
        # perform step
        sim.step(not args_cli.headless)
        # update sim-time
        sim_time += sim_dt
        count += 1
        # note: to deal with timeline events such as stopping, we need to check if the simulation is playing
        if sim.is_playing():
            # update buffers
            robot.update_buffers(sim_dt)
            # update marker positions
            ee_marker.set_world_poses(robot.data.ee_state_w[:, 0:3], robot.data.ee_state_w[:, 3:7])
            goal_marker.set_world_poses(rmp_commands[:, 0:3] + envs_positions, rmp_commands[:, 3:7])


if __name__ == "__main__":
    # Run IK example with Manipulator
    main()
    # Close the simulator
    simulation_app.close()
