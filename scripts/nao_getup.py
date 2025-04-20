# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to create a simple stage in Isaac Sim.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parser.add_argument(
#     "--/isaac/startup/ros_bridge_extension=isaacsim.ros2.bridge", help="Width of the viewport and generated images. Defaults to 1280"
# )

# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli, renderer="Hydra")  # or "RayTracedLighting", "PathTracing", etc.
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab.sim import SimulationCfg, SimulationContext
import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

def design_scene():
    """Designs the scene by spawning ground plane, light, objects and meshes from usd files."""
    # Ground-plane
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # spawn distant light
    cfg_light_distant = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    cfg_light_distant.func("/World/lightDistant", cfg_light_distant, translation=(1, 0, 10))

    # create a new xform prim for all objects to be spawned under
    prim_utils.create_prim("/World/Objects", "Xform")

    # spawn a usd file of nao
    cfg = sim_utils.UsdFileCfg(usd_path=f"nao2/nao/nao_nohands.usd")
    cfg.func("/World/Objects/Nao", cfg, translation=(0.0, 0.0, 0.345))

from isaacsim.core.utils.extensions import enable_extension

def setup_extensions():
    # UI Windows
    enable_extension("omni.kit.widget.stage")
    enable_extension("omni.kit.widget.layers")
    enable_extension("omni.kit.widget.graph")

    # OmniGraph Core
    enable_extension("omni.graph.core")
    enable_extension("omni.graph.action")
    enable_extension("omni.graph.action_nodes")
    enable_extension("omni.graph.bundle.action")
    enable_extension("omni.graph.window.core")
    enable_extension("omni.graph.window.action")

    # ROS 2 Bridge (optional)
    enable_extension("omni.isaac.ros2_bridge")  # The correct name for most recent Isaac Sim builds



def main():
    """Main function."""

    # Initialize the simulation context
    sim_cfg = SimulationCfg(dt=0.01)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])

    # Design scene by adding assets to it
    design_scene()

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Simulate physics
    while simulation_app.is_running():
        # perform step
        sim.step()


if __name__ == "__main__":
    # Setup required extensions
    setup_extensions()
    # run the main function
    main()
    # close sim app
    simulation_app.close()
