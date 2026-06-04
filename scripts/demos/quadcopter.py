# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to simulate a quadcopter.

.. code-block:: bash

    # Usage with default PhysX physics and default kit visualizer.
    ./isaaclab.sh -p scripts/demos/quadcopter.py

    # Usage with Newton visualizer and default PhysX physics.
    ./isaaclab.sh -p scripts/demos/quadcopter.py --visualizer newton

    # Usage with Newton (MJWarp) physics and default kit visualizer.
    ./isaaclab.sh -p scripts/demos/quadcopter.py --physics newton_mjwarp

    # Usage with Newton visualizer and Newton (MJWarp) physics.
    ./isaaclab.sh -p scripts/demos/quadcopter.py --visualizer newton --physics newton_mjwarp

"""

"""Parse CLI first so we can decide whether to launch Isaac Sim Kit."""

import argparse

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(
    description="This script demonstrates how to simulate a quadcopter.",
    conflict_handler="resolve",
)
parser.add_argument("--physics", default="physx", choices=["physx", "newton_mjwarp"], help="Physics backend.")
add_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()

import torch

import isaaclab.sim as sim_utils

##
# Pre-defined configs
##
from isaaclab.physics import PhysicsCfg

from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def main():
    """Main function."""
    with launch_simulation(cfg=PhysicsCfg(), launcher_args=args_cli) as physics_cfg:
        # Load kit helper
        sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device, physics=physics_cfg)
        sim = sim_utils.SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view(eye=[0.25, -0.25, 0.7], target=[0.0, 0.0, 0.5])

        # Spawn things into stage
        # Ground-plane
        cfg = sim_utils.GroundPlaneCfg()
        cfg.func("/World/defaultGroundPlane", cfg)
        # Lights
        cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
        cfg.func("/World/Light", cfg)

        # Robots
        robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
        robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)

        # create handles for the robots
        robot = robot_cfg.class_type(robot_cfg)

        # Play the simulator
        sim.reset()

        # Fetch relevant parameters to make the quadcopter hover in place
        prop_body_ids = robot.find_bodies("m.*_prop")[0]
        robot_mass = robot.data.body_mass.torch[0].sum()
        gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0
        count = 0
        # Step while a visualizer window is still open (or none exist, e.g. headless); works for kit and newton.
        while sim.is_headless_or_exist_active_visualizer():
            # reset
            if count % 2000 == 0:
                # reset counters
                sim_time = 0.0
                count = 0
                # reset dof state
                joint_pos, joint_vel = robot.data.default_joint_pos.torch, robot.data.default_joint_vel.torch
                robot.write_joint_position_to_sim_index(position=joint_pos)
                robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
                default_root_pose = robot.data.default_root_pose.torch
                robot.write_root_pose_to_sim_index(root_pose=default_root_pose)
                default_root_vel = robot.data.default_root_vel.torch
                robot.write_root_velocity_to_sim_index(root_velocity=default_root_vel)
                robot.reset()
                # reset command
                print(">>>>>>>> Reset!")
            # apply action to the robot (make the robot float in place)
            forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
            torques = torch.zeros_like(forces)
            forces[..., 2] = robot_mass * gravity / 4.0
            robot.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces,
                torques=torques,
                body_ids=prop_body_ids,
            )
            robot.write_data_to_sim()
            # perform step
            sim.step()
            # update sim-time
            sim_time += sim_dt
            count += 1
            # update buffers
            robot.update(sim_dt)


if __name__ == "__main__":
    # run the main function
    main()
