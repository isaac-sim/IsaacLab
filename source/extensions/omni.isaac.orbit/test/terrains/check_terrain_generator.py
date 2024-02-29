# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

"""Launch Isaac Sim Simulator first."""

import os
import traceback

from omni.isaac.orbit.app import AppLauncher

# launch omniverse app
# note: we only need to do this because of `TerrainImporter` which uses Omniverse functions
app_experience = f"{os.environ['EXP_PATH']}/omni.isaac.sim.python.gym.headless.kit"
app_launcher = AppLauncher(headless=True, experience=app_experience)
simulation_app = app_launcher.app

"""Rest everything follows."""

import shutil

import carb

from omni.isaac.orbit.terrains.config.rough import ROUGH_TERRAINS_CFG
from omni.isaac.orbit.terrains.terrain_generator import TerrainGenerator


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
    try:
        # Run the main function
        main()
    except Exception as err:
        carb.log_error(err)
        carb.log_error(traceback.format_exc())
        raise
    finally:
        # close sim app
        simulation_app.close()
