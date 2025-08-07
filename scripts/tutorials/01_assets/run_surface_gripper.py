# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn a pick-and-place robot equipped with a surface gripper and interact with it.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/01_assets/run_surface_gripper.py --device=cpu

When running this script make sure the --device flag is set to cpu. This is because the surface gripper is
currently only supported on the CPU.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a Surface Gripper.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, SurfaceGripper, SurfaceGripperCfg
from isaaclab.sim import SimulationContext

##
# Pre-defined configs
##
from isaaclab_assets import PICK_AND_PLACE_CFG  # isort:skip


def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Create separate groups called "Origin1", "Origin2"
    # Each group will have a robot in it
    origins = [[2.75, 0.0, 0.0], [-2.75, 0.0, 0.0]]
    # Origin 1
    prim_utils.create_prim("/World/Origin1", "Xform", translation=origins[0])
    # Origin 2
    prim_utils.create_prim("/World/Origin2", "Xform", translation=origins[1])

    # Articulation: First we define the robot config
    pick_and_place_robot_cfg = PICK_AND_PLACE_CFG.copy()
    pick_and_place_robot_cfg.prim_path = "/World/Origin.*/Robot"
    pick_and_place_robot = Articulation(cfg=pick_and_place_robot_cfg)

    # Surface Gripper: Next we define the surface gripper config
    surface_gripper_cfg = SurfaceGripperCfg()
    # We need to tell the View which prim to use for the surface gripper
    surface_gripper_cfg.prim_expr = "/World/Origin.*/Robot/picker_head/SurfaceGripper"
    # We can then set different parameters for the surface gripper, note that if these parameters are not set,
    # the View will try to read them from the prim.
    surface_gripper_cfg.max_grip_distance = 0.1  # [m] (Maximum distance at which the gripper can grasp an object)
    surface_gripper_cfg.shear_force_limit = 500.0  # [N] (Force limit in the direction perpendicular direction)
    surface_gripper_cfg.coaxial_force_limit = 500.0  # [N] (Force limit in the direction of the gripper's axis)
    surface_gripper_cfg.retry_interval = 0.1  # seconds (Time the gripper will stay in a grasping state)
    # We can now spawn the surface gripper
    surface_gripper = SurfaceGripper(cfg=surface_gripper_cfg)

    # return the scene information
    scene_entities = {"pick_and_place_robot": pick_and_place_robot, "surface_gripper": surface_gripper}
    return scene_entities, origins


def run_simulator(
    sim: sim_utils.SimulationContext, entities: dict[str, Articulation | SurfaceGripper], origins: torch.Tensor
):
    """Runs the simulation loop."""
    # Extract scene entities
    robot: Articulation = entities["pick_and_place_robot"]
    surface_gripper: SurfaceGripper = entities["surface_gripper"]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 500 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            # root state
            # we offset the root state by the origin since the states are written in simulation world frame
            # if this is not done, then the robots will be spawned at the (0, 0, 0) of the simulation world
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            # set joint positions with some noise
            joint_pos, joint_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            robot.reset()
            print("[INFO]: Resetting robot state...")
            # Opens the gripper and makes sure the gripper is in the open state
            surface_gripper.reset()
            print("[INFO]: Resetting gripper state...")

        # Sample a random command between -1 and 1.
        gripper_commands = torch.rand(surface_gripper.num_instances) * 2.0 - 1.0
        # The gripper behavior is as follows:
        # -1 < command < -0.3 --> Gripper is Opening
        # -0.3 < command < 0.3 --> Gripper is Idle
        # 0.3 < command < 1 --> Gripper is Closing
        print(f"[INFO]: Gripper commands: {gripper_commands}")
        mapped_commands = [
            "Opening" if command < -0.3 else "Closing" if command > 0.3 else "Idle" for command in gripper_commands
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")
        # Set the gripper command
        surface_gripper.set_grippers_command(gripper_commands)
        # Write data to sim
        surface_gripper.write_data_to_sim()
        # Perform step
        sim.step()
        # Increment counter
        count += 1
        # Read the gripper state from the simulation
        surface_gripper.update(sim_dt)
        # Read the gripper state from the buffer
        surface_gripper_state = surface_gripper.state
        # The gripper state is a list of integers that can be mapped to the following:
        # -1 --> Open
        # 0 --> Closing
        # 1 --> Closed
        # Print the gripper state
        print(f"[INFO]: Gripper state: {surface_gripper_state}")
        mapped_commands = [
            "Open" if state == -1 else "Closing" if state == 0 else "Closed" for state in surface_gripper_state.tolist()
        ]
        print(f"[INFO]: Mapped commands: {mapped_commands}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.75, 7.5, 10.0], [2.75, 0.0, 0.0])
    # Design scene
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
