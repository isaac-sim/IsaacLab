# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage with standalone OvPhysX.

.. code-block:: bash

    # Usage
    uv run --extra ov python scripts/tutorials/00_sim/create_empty.py

"""

"""Parse the command-line arguments first."""


import argparse
from types import SimpleNamespace

from isaaclab_ov.physics import OvPhysxCfg

from isaaclab.app import add_launcher_args, launch_simulation
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import instantiate

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append simulation launcher cli args
add_launcher_args(parser)
parser.set_defaults(visualizer=["newton_rtx"])
# parse the arguments
args_cli = parser.parse_args()

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext


def main():
    """Main function."""

    # Configure the simulation
    sim_cfg = SimulationCfg(
        physics=OvPhysxCfg(),
        dt=0.01,
        device=args_cli.device,
    )
    # Launch the simulator runtime that the configuration needs
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=0.0)
    scene_cfg.world = AssetBaseCfg(prim_path="/World")
    with launch_simulation(SimpleNamespace(sim=sim_cfg, scene=scene_cfg), args_cli):
        # Initialize the simulation context
        sim = SimulationContext(sim_cfg)
        # Set main camera
        sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

        _scene = instantiate(scene_cfg)
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
