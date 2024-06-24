# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
# note: we only need to do this because of `TerrainImporter` which uses Omniverse functions
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import shutil

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.terrain_generator import TerrainGenerator


def main():
    # Create directory to dump results
    test_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(test_dir, "output", "generator")
    # remove directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    # create directory
    os.makedirs(output_dir, exist_ok=True)
    # modify the config to cache
    ROUGH_TERRAINS_CFG.use_cache = True
    ROUGH_TERRAINS_CFG.cache_dir = output_dir
    ROUGH_TERRAINS_CFG.curriculum = False
    # generate terrains
    terrain_generator = TerrainGenerator(cfg=ROUGH_TERRAINS_CFG)  # noqa: F841


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
