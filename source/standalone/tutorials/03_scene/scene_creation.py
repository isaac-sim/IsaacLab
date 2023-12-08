# Copyright (c) 2022-2023, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the scene interface to quickly setup a scene with multiple
articulated robots and sensors.
"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""


import argparse

from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import math
import traceback

import carb

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.scene import InteractiveScene
from omni.isaac.orbit.sim import SimulationContext

from omni.isaac.orbit_tasks.classic.cartpole.cartpole_scene import CartpoleSceneCfg


# Main
def main():
    # Load kit helper
    sim = SimulationContext(sim_utils.SimulationCfg(device="cpu", use_gpu_pipeline=False))
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 4.5], [0.0, 0.0, 2.0])

    # Spawn things into stage
    scene = InteractiveScene(CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0))

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0

    # Extract cartpole from InteractiveScene
    cartpole = scene.articulations["robot"]

    # Simulation loop
    while simulation_app.is_running():
        # reset
        if count % 1000 == 0:
            # reset counter
            count = 0

            # Get default joint positions and velocities and set them as targets
            joint_pos, joint_vel = cartpole.data.default_joint_pos, cartpole.data.default_joint_vel

            joint_ids, _ = cartpole.find_joints("cart_to_pole")

            # Set joint position to be pi/8 so the pole will move
            joint_pos[:, joint_ids[0]] = math.pi / 8.0

            cartpole.set_joint_position_target(joint_pos)
            cartpole.set_joint_velocity_target(joint_vel)

            scene.write_data_to_sim()

            print("[INFO]: Resetting robot state...")

        # perform step
        sim.step()

        count += 1

        # update buffers
        scene.update(sim_dt)

    # End simulation loop


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
