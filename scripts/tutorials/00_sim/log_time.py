# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/00_sim/log_time.py

"""

"""Parse the command-line arguments first."""


import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    # Specify that the logs must be in logs/docker_tutorial
    log_dir_path = os.path.join("logs")
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    # In the container, the absolute path will be
    # /workspace/isaaclab/logs/docker_tutorial, because
    # all Python execution is done through the uv-managed workspace
    # and the calling process' path will be /workspace/isaaclab
    log_dir_path = os.path.abspath(os.path.join(log_dir_path, "docker_tutorial"))
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")

    # Configure the simulation
    sim_cfg = SimulationCfg(dt=0.01, device=args_cli.device)
    # Launch the simulator runtime that the configuration needs
    with launch_simulation(sim_cfg, args_cli):
        # Initialize the simulation context
        sim = SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

        # Play the simulator
        sim.reset()
        # Now we are ready!
        print("[INFO]: Setup complete...")

        # Prepare to count sim_time
        sim_dt = sim.get_physics_dt()
        sim_time = 0.0

        # Open logging file
        with open(os.path.join(log_dir_path, "log.txt"), "w") as log_file:
            # Simulate physics
            while sim.is_headless_or_exist_active_visualizer():
                log_file.write(f"{sim_time}" + "\n")
                # perform step
                sim.step()
                sim_time += sim_dt


if __name__ == "__main__":
    # run the main function
    main()
