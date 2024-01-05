# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to generate log outputs while the simulation plays.
It accompanies the tutorial on docker usage.

.. code-block:: bash

    # Usage
    ./orbit.sh -p source/standalone/tutorials/00_sim/log_time.py

"""

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""
import argparse
import os

from omni.isaac.orbit.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating logs from within the docker container.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import traceback

import carb

from omni.isaac.orbit.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""
    # Specify that the logs must be in logs/docker_tutorial
    log_dir_path = os.path.join("logs", "docker_tutorial")
    # In the container, the absolute path will be
    # /workspace/orbit/logs/docker_tutorial, because
    # all python execution is done through /workspace/orbit/orbit.sh
    # and the calling process' path will be /workspace/orbit
    log_dir_path = os.path.abspath(log_dir_path)
    if not os.path.isdir(log_dir_path):
        os.mkdir(log_dir_path)
    print(f"[INFO] Logging experiment to directory: {log_dir_path}")

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01, substeps=1)
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
        while simulation_app.is_running():
            log_file.write(f"{sim_time}" + "\n")
            # perform step
            sim.step()
            sim_time += sim_dt


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
