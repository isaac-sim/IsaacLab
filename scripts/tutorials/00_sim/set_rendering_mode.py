# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn prims into the scene.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/tutorials/00_sim/set_rendering_mode.py

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# create argparser
parser = argparse.ArgumentParser(
    description="Tutorial on viewing a warehouse scene with a given rendering mode preset."
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from isaaclab_physx.visualizers import KitVisualizerCfg

import isaaclab.sim as sim_utils
from isaaclab.rendering_mode import RenderingModeCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def main():
    """Main function."""

    # rendering modes include performance, balanced, and quality
    # note: the CLI argument (--rendering_mode) takes precedence over this visualizer setting
    rendering_mode = "performance"

    # RTX tuning uses :class:`RenderingModeCfg` profiles on ``SimulationCfg.rendering_mode_cfgs``
    # (not raw carb paths). Here we extend the ``performance`` preset with reflections for this scene.
    sim_cfg = sim_utils.SimulationCfg(
        rendering_mode_cfgs={
            "performance": RenderingModeCfg(
                rendering_mode_preset="performance",
                kit_enable_reflections=True,
            ),
            "balanced": RenderingModeCfg(rendering_mode_preset="balanced"),
            "quality": RenderingModeCfg(rendering_mode_preset="quality"),
        },
        visualizer_cfgs=[
            KitVisualizerCfg(rendering_mode=rendering_mode),
        ],
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # Pose camera in the hospital lobby area
    sim.set_camera_view([-11, -0.5, 2], [0, 0, 0.5])

    # Load hospital scene
    hospital_usd_path = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
    cfg = sim_utils.UsdFileCfg(usd_path=hospital_usd_path)
    cfg.func("/Scene", cfg)

    # Play the simulator
    sim.reset()

    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Run simulation and view scene
    while simulation_app.is_running():
        sim.step()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
