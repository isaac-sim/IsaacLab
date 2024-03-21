# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/00_sim/create_empty.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse

from omni.isaac.orbit.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
AppLauncher.add_app_launcher_args(parser)  # appends some our AppLauncher cli args
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from omni.isaac.orbit.sim import SimulationCfg, SimulationContext


def main():
    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
    sim = SimulationContext(sim_cfg)

    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])  # (optional)

    sim.reset()  # plays the simulator
    print("[INFO]: Setup complete...")

    while simulation_app.is_running():
        sim.step()  # simulates physics


if __name__ == "__main__":
    main()
    simulation_app.close()
