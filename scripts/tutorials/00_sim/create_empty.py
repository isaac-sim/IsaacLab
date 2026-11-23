# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    uv run python scripts/tutorials/00_sim/create_empty.py

"""

"""Parse the command-line arguments first."""


import argparse

from isaaclab.app import add_launcher_args, launch_simulation

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append simulation launcher cli args
add_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

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

        # Simulate physics
        while sim.is_running():
            # perform step
            sim.step()


if __name__ == "__main__":
    # run the main function
    main()
